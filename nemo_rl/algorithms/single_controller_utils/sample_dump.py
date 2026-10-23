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
"""Per-sample dump of what a GRPO step trained on (``grpo.dump_train_samples``).

One gzip JSON-lines file per step, ``<logger.log_dir>/train_samples/step_<n>.jsonl.gz``,
one line per row of the step: its sample id, prompt index, reward, masks,
token ids, token mask, vLLM generation log-probs and the trainer's
``prev_logprobs``, plus the receipt identity the finalizer stamped on the row
(``capture`` tag: physical rollout id, terminal call, its ``cum_len`` and
``cumulative_hash``) so the rows can be checked against Gym's capture ledger
offline. Plain JSON so the check script needs no torch.
"""

from __future__ import annotations

import gzip
import json
import os
from typing import Any, Optional

import torch

# Log-probs are rounded to this many decimals in the dump.
LOGPROB_DECIMALS = 6


def _logprobs(values: Optional[torch.Tensor], length: int) -> Optional[list[float]]:
    if values is None:
        return None
    return [round(v, LOGPROB_DECIMALS) for v in values[:length].float().tolist()]


def sample_dump_rows(
    *,
    sample_ids: list[str],
    tags: list[dict[str, Any]],
    input_ids: torch.Tensor,
    input_lengths: torch.Tensor,
    token_mask: torch.Tensor,
    generation_logprobs: torch.Tensor,
    prev_logprobs: Optional[torch.Tensor],
    rewards: torch.Tensor,
    sample_mask: torch.Tensor,
    final_sample_mask: torch.Tensor,
    mask_sample: torch.Tensor,
    truncated: torch.Tensor,
) -> list[dict[str, Any]]:
    """One JSON-ready dict per row; token columns are cut to the row's length."""
    rows = []
    for i, sample_id in enumerate(sample_ids):
        length = int(input_lengths[i])
        tag = tags[i] if i < len(tags) else {}
        rows.append(
            {
                "sample_id": sample_id,
                "prompt_idx": tag.get("prompt_idx"),
                "weight_version": tag.get("weight_version"),
                "capture": tag.get("capture"),
                "reward": float(rewards[i]),
                # Finalizer validity (0 = placeholder row) and what finally trains.
                "sample_mask": float(sample_mask[i]),
                "final_sample_mask": float(final_sample_mask[i]),
                "mask_sample": bool(mask_sample[i]),
                "truncated": bool(truncated[i]),
                "length": length,
                "input_ids": input_ids[i, :length].tolist(),
                "token_mask": [int(m) for m in token_mask[i, :length].tolist()],
                "generation_logprobs": _logprobs(generation_logprobs[i], length),
                "prev_logprobs": _logprobs(
                    None if prev_logprobs is None else prev_logprobs[i], length
                ),
            }
        )
    return rows


def write_sample_dump(
    dump_dir: str, step: int, trainer_version: int, rows: list[dict[str, Any]]
) -> str:
    """Write one step's rows; returns the file path."""
    os.makedirs(dump_dir, exist_ok=True)
    path = os.path.join(dump_dir, f"step_{step:05d}.jsonl.gz")
    tmp = f"{path}.tmp"
    with gzip.open(tmp, "wt", compresslevel=3) as handle:
        for row in rows:
            handle.write(
                json.dumps({"step": step, "trainer_version": trainer_version, **row})
                + "\n"
            )
    os.replace(tmp, path)
    return path
