# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""grpo.dump_train_samples: the advantage stage keeps each row, the step writes them."""

from __future__ import annotations

import asyncio
import gzip
import json
from types import SimpleNamespace

import torch
from tensordict import TensorDict

from nemo_rl.algorithms.async_utils.replay_buffer import DataPlaneCheckpointBarrier
from nemo_rl.algorithms.grpo import GRPOConfig
from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.algorithms.single_controller_utils.config import AdvantageConfig
from nemo_rl.algorithms.single_controller_utils.sample_dump import (
    sample_dump_rows,
    write_sample_dump,
)
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.experience.rollout_reassembler import FinalizedRollout, capture_identity


class _DataPlane:
    def __init__(self, data: TensorDict) -> None:
        self._data = data
        self.selects: list[list[str]] = []

    def get_samples(self, *, select_fields, **kwargs):
        del kwargs
        self.selects.append(list(select_fields))
        return self._data

    def put_samples(self, **kwargs) -> None:
        del kwargs


class _ZeroEstimator:
    def compute_advantage(self, *, rewards, mask, **kwargs) -> torch.Tensor:
        del rewards, kwargs
        return torch.zeros_like(mask)


def _controller(data: TensorDict, *, dump: bool):
    ctrl = object.__new__(SingleControllerActor.__ray_metadata__.modified_class)
    ctrl._dp_client = _DataPlane(data)
    ctrl._advantage_cfg = AdvantageConfig()
    ctrl._advantage_estimator = _ZeroEstimator()
    ctrl._data_plane_checkpoint_barrier = DataPlaneCheckpointBarrier()
    ctrl._policy_logprobs_required = True
    ctrl._reference_logprobs_required = False
    ctrl._teacher_logprobs_required = False
    ctrl._is_ppo = False
    ctrl._master_config = SimpleNamespace(
        grpo=GRPOConfig(dump_train_samples=dump, seq_logprob_error_threshold=None)
    )
    ctrl._algo_cfg = ctrl._master_config.grpo
    ctrl._message_level_advantage_penalties_enabled = False
    ctrl._step_log_dict = {
        "rewards": [],
        "sample_masks": [],
        "masked_advantages": [],
        "sequence_lengths": [],
        "num_mask_sample_filtered": [],
        "seq_logprob_error_metrics": [],
    }
    ctrl._sample_dump_rows = []
    return ctrl


def _batch() -> tuple[TensorDict, KVBatchMeta]:
    # Row 0 trains, row 1 is a placeholder (sample_mask 0, length 1), row 2 env-masked.
    n, s = 3, 6
    token_mask = torch.tensor(
        [[0, 0, 1, 1, 0, 1], [0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0]],
        dtype=torch.float32,
    )
    data = TensorDict(
        {
            "input_ids": torch.arange(n * s).reshape(n, s),
            "input_lengths": torch.tensor([6, 1, 4]),
            "prompt_ids_for_adv": torch.zeros(n, 2, dtype=torch.long),
            "total_reward": torch.tensor([1.0, 0.0, 0.0]),
            "token_mask": token_mask,
            "sample_mask": torch.tensor([1.0, 0.0, 1.0]),
            "mask_sample": torch.tensor([False, False, True]),
            "truncated": torch.tensor([False, False, False]),
            "generation_logprobs": -0.25 * token_mask,
            "prev_logprobs": torch.full((n, s), -0.1234567),
        },
        batch_size=[n],
    )
    capture = {"rollout_id": "grp_g0_a1", "cum_len": 6, "cumulative_hash": "ab"}
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["grp_g0", "grp_g1", "grp_g2"],
        fields=list(data.keys()),
        tags=[
            {"weight_version": 2, "prompt_idx": 7, "capture": capture},
            {"weight_version": 2, "prompt_idx": 7},
            {"weight_version": 2, "prompt_idx": 7},
        ],
    )
    return data, meta


def test_advantage_stage_keeps_dump_rows_only_when_enabled() -> None:
    data, meta = _batch()
    off = _controller(data, dump=False)
    asyncio.run(off._advantage_stage(meta))
    assert off._sample_dump_rows == []
    assert all("input_ids" not in select for select in off._dp_client.selects)

    ctrl = _controller(data, dump=True)
    asyncio.run(ctrl._advantage_stage(meta))
    rows = ctrl._sample_dump_rows
    assert [row["sample_id"] for row in rows] == ["grp_g0", "grp_g1", "grp_g2"]
    assert [row["length"] for row in rows] == [6, 1, 4]
    assert rows[0]["input_ids"] == [0, 1, 2, 3, 4, 5]
    assert rows[2]["input_ids"] == [12, 13, 14, 15]
    assert rows[0]["token_mask"] == [0, 0, 1, 1, 0, 1]
    assert rows[0]["generation_logprobs"] == [0.0, 0.0, -0.25, -0.25, 0.0, -0.25]
    assert rows[0]["prev_logprobs"] == [-0.123457] * 6
    # Finalizer validity and what finally trains after the env mask.
    assert [row["sample_mask"] for row in rows] == [1.0, 0.0, 1.0]
    assert [row["final_sample_mask"] for row in rows] == [1.0, 0.0, 0.0]
    assert [row["mask_sample"] for row in rows] == [False, False, True]
    assert rows[0]["capture"]["rollout_id"] == "grp_g0_a1"
    assert rows[1]["capture"] is None
    assert rows[0]["reward"] == 1.0 and rows[0]["prompt_idx"] == 7


def test_write_sample_dump_round_trips(tmp_path) -> None:
    data, meta = _batch()
    rows = sample_dump_rows(
        sample_ids=list(meta.sample_ids),
        tags=list(meta.tags),
        input_ids=data["input_ids"],
        input_lengths=data["input_lengths"],
        token_mask=data["token_mask"],
        generation_logprobs=data["generation_logprobs"],
        prev_logprobs=None,
        rewards=data["total_reward"],
        sample_mask=data["sample_mask"],
        final_sample_mask=data["sample_mask"],
        mask_sample=data["mask_sample"],
        truncated=data["truncated"],
    )
    path = write_sample_dump(str(tmp_path / "train_samples"), 3, 2, rows)
    assert path.endswith("train_samples/step_00003.jsonl.gz")
    with gzip.open(path, "rt") as handle:
        read = [json.loads(line) for line in handle]
    assert [row["step"] for row in read] == [3, 3, 3]
    assert [row["trainer_version"] for row in read] == [2, 2, 2]
    assert read[1]["input_ids"] == [6]
    assert read[0]["prev_logprobs"] is None
    assert read[2]["generation_logprobs"] == [0.0, -0.25, -0.25, 0.0]


def test_capture_identity_reads_the_terminal_record() -> None:
    row = FinalizedRollout(
        rollout_id="grp_g0_a1",
        valid=True,
        rejection_reason=None,
        token_ids=[1, 2, 3],
        token_mask=[0, 1, 1],
        logprobs=[0.0, -0.1, -0.2],
        prompt_len=1,
        reward=1.0,
        staging_keys=["k0", "k1"],
    )
    receipt = {
        "terminal_model_call_id": "c1",
        "manifest": [
            {"model_call_id": "c0", "cum_len": 2, "cumulative_hash": "h0"},
            {
                "model_call_id": "c1",
                "cum_len": 3,
                "cumulative_hash": "h1",
                "chain_hash": "x",
            },
        ],
    }
    assert capture_identity(row, receipt) == {
        "rollout_id": "grp_g0_a1",
        "rejection_reason": None,
        "manifest_calls": 2,
        "terminal_model_call_id": "c1",
        "cum_len": 3,
        "cumulative_hash": "h1",
        "chain_hash": "x",
    }
    lost = capture_identity(
        FinalizedRollout(
            rollout_id="grp_g1_a2",
            valid=False,
            rejection_reason="missing_receipt",
            token_ids=[],
            token_mask=[],
            logprobs=[],
            prompt_len=0,
            reward=0.0,
            staging_keys=[],
        ),
        None,
    )
    assert lost["rejection_reason"] == "missing_receipt"
    assert lost["cumulative_hash"] is None
