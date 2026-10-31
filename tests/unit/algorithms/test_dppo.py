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

import math
from types import SimpleNamespace

import pytest
import torch

from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn
from nemo_rl.algorithms.grpo import _validate_dppo_sampling
from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.experience.rollout_reassembler import has_trainable_reward_variation


def _dppo_loss() -> ClippedPGLossFn:
    return ClippedPGLossFn(
        ClippedPGLossConfig(reference_policy_kl_penalty=0, dppo_tv_threshold=0.1)
    )


def _batch() -> dict[str, torch.Tensor]:
    # Positive/negative advantages at both sides of the rollout distribution.
    return {
        "input_ids": torch.zeros((1, 5), dtype=torch.long),
        "token_mask": torch.tensor([[0, 1, 1, 1, 1]]),
        "sample_mask": torch.tensor([1]),
        "advantages": torch.tensor([[0, 1, -1, -1, 1]], dtype=torch.float32),
        "prev_logprobs": torch.full((1, 5), -3.0),
        "generation_logprobs": torch.tensor(
            [[0, math.log(0.2), math.log(0.2), math.log(0.4), math.log(0.4)]]
        ),
    }


def test_dppo_uses_rollout_logprobs_and_directional_binary_tv_mask() -> None:
    policy = torch.tensor(
        [[math.log(0.4), math.log(0.4), math.log(0.2), math.log(0.2)]],
        requires_grad=True,
    )
    loss, metrics = _dppo_loss()(
        next_token_logprobs=policy,
        data=_batch(),
        global_valid_seqs=torch.tensor(1),
        global_valid_toks=torch.tensor(4),
    )
    # Out of region: positive/up and negative/down are blocked; the two
    # directions returning toward the rollout policy keep their gradients.
    torch.testing.assert_close(loss, torch.tensor((2.0 - 0.5) / 4))
    assert metrics["dppo_masked_fraction"] == pytest.approx(0.5)
    assert metrics["dppo_binary_tv"] == pytest.approx(0.2)
    loss.backward()
    torch.testing.assert_close(policy.grad, torch.tensor([[0.0, 0.5, 0.0, -0.125]]))


def test_dppo_rejects_invalid_behavior_logprobs() -> None:
    batch = _batch()
    batch["generation_logprobs"][0, 2] = float("nan")
    with pytest.raises(ValueError, match="finite rollout logprobs"):
        _dppo_loss()(
            next_token_logprobs=torch.full((1, 4), -1.0),
            data=batch,
            global_valid_seqs=torch.tensor(1),
            global_valid_toks=torch.tensor(4),
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"reference_policy_kl_penalty": 0.01},
        {"use_importance_sampling_correction": True},
        {"force_on_policy_ratio": True},
        {"token_level_loss": False},
        {"use_cispo": True},
        {"score_centering": True},
    ],
)
def test_dppo_rejects_incompatible_objective_settings(overrides: dict) -> None:
    values = {"reference_policy_kl_penalty": 0, "dppo_tv_threshold": 0.1}
    values.update(overrides)
    cfg = ClippedPGLossConfig(**values)
    with pytest.raises(ValueError, match="DPPO binary-TV requires"):
        ClippedPGLossFn(cfg)


@pytest.mark.parametrize(
    "overrides",
    [
        {"top_p": 0.95},
        {"top_k": 50},
        {"temperature": 0.8},
        {"backend": "sglang"},
    ],
)
def test_dppo_sampling_rejects_probability_mismatch(overrides: dict) -> None:
    generation = {"backend": "vllm", "top_p": 1.0, "top_k": None, "temperature": 1.0}
    generation.update(overrides)
    config = SimpleNamespace(
        loss_fn=ClippedPGLossConfig(
            reference_policy_kl_penalty=0, dppo_tv_threshold=0.1
        ),
        policy={"generation": generation},
    )
    with pytest.raises(ValueError, match="DPPO"):
        _validate_dppo_sampling(config)


@pytest.mark.parametrize(
    ("rewards", "masked", "expected"),
    [
        ([0.0, 1.0], [False, False], True),
        ([0.0, 0.0], [False, False], False),
        ([0.0, 1.0], [False, True], False),
    ],
)
def test_dynamic_sampling_filters_only_trainable_constant_reward_groups(
    rewards: list[float], masked: list[bool], expected: bool
) -> None:
    rows = [SimpleNamespace(valid=True, reward=reward) for reward in rewards]
    assert has_trainable_reward_variation(rows, masked, 1.0) is expected
    assert has_trainable_reward_variation(rows, masked, 0.0) is False


def test_dynamic_sampling_does_not_train_a_short_step() -> None:
    controller = object.__new__(SingleControllerActor.__ray_metadata__.modified_class)
    controller._algo_cfg = SimpleNamespace(num_prompts_per_step=8)
    controller._async_cfg = SimpleNamespace(
        rollout_failure=SimpleNamespace(min_step_batch_fraction=1.0)
    )
    controller._batch_shortfall = {0: 1}
    with pytest.raises(RuntimeError, match="below the floor of 8"):
        controller._target_groups_for_step(0)
