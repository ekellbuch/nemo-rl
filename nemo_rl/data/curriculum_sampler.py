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

"""Online pass-rate curriculum for GRPO prompt sampling.

A GRPO group whose rollouts all pass or all fail has zero advantage, so drawing
such a task wastes its rollouts. ``PassRateCurriculumSampler`` keeps a decayed
pass/attempt count per dataset row, fed back from finished groups, and draws
rows with replacement in proportion to how likely their next group is to carry
signal under the current policy.
"""

import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Optional

import torch
from pydantic import BaseModel, Field
from torch.utils.data import Sampler


class PassRateCurriculumConfig(BaseModel, extra="forbid"):
    """``grpo.curriculum``: pass-rate curriculum for the SingleController dataloader.

    Off by default. When enabled, it replaces ``data.shuffle``: prompts are drawn
    with replacement, weighted by each task's estimated pass rate ``p``.
    """

    enabled: bool = False
    # mixed_group: 1 - p^n - (1-p)^n with n = num_generations_per_prompt, the chance a
    # group has reward spread. variance: p(1-p). uniform: equal weights (A/B baseline).
    weighting: Literal["mixed_group", "variance", "uniform"] = "mixed_group"
    # A task's counts are multiplied by this before each new group of it is added.
    decay: float = Field(default=0.7, gt=0, le=1)
    # Beta prior: an unseen task's pass rate, worth prior_strength rollouts of evidence.
    prior_pass_rate: float = Field(default=0.5, ge=0, le=1)
    prior_strength: float = Field(default=2.0, gt=0)
    # Weight floor as a fraction of the curve's peak, so no task is starved.
    min_weight: float = Field(default=0.05, gt=0, le=1)
    # A rollout counts as a pass when its reward is above this.
    pass_threshold: float = 0.5
    # Optional JSON/JSONL of per-task prior counts {task_name, passes, attempts} (e.g. from
    # a profiling run) that seed the stats; rows match on their Gym row's task_name.
    init_stats_path: Optional[str] = None


def load_init_stats(path: str | Path) -> dict[str, tuple[float, float]]:
    """Read ``init_stats_path``: task_name -> (passes, attempts).

    Accepts JSONL with one ``{"task_name", "passes", "attempts"}`` record per line,
    a JSON list of such records, or a JSON object mapping task_name to
    ``{"passes", "attempts"}``.
    """
    text = Path(path).read_text()
    if str(path).endswith(".jsonl"):
        records: Any = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        records = json.loads(text)
    if isinstance(records, Mapping):
        records = [{"task_name": name, **counts} for name, counts in records.items()]
    stats: dict[str, tuple[float, float]] = {}
    for record in records:
        name = record["task_name"]
        passes, attempts = float(record["passes"]), float(record["attempts"])
        if not 0 <= passes <= attempts:
            raise ValueError(
                f"{path}: task {name!r} needs 0 <= passes <= attempts, got "
                f"{passes}/{attempts}"
            )
        if name in stats:
            raise ValueError(f"{path}: task {name!r} appears more than once")
        stats[name] = (passes, attempts)
    return stats


class _CurriculumIterator(Iterator[int]):
    """One epoch of draws; carries the checkpoint state torchdata saves.

    The state lives on the iterator rather than the sampler because torchdata
    restores a sampler's own state only into a sampler that has one, so a
    checkpoint written with the curriculum on could not resume with it off.
    """

    def __init__(self, sampler: "PassRateCurriculumSampler") -> None:
        self._sampler = sampler
        self.yielded = 0

    def __iter__(self) -> "_CurriculumIterator":
        return self

    def __next__(self) -> int:
        if self.yielded >= len(self._sampler):
            raise StopIteration
        self.yielded += 1
        return self._sampler.draw()

    def state_dict(self) -> dict[str, Any]:
        return {"yielded": self.yielded, **self._sampler.export_state()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.yielded = state_dict["yielded"]
        # Absent when resuming a checkpoint written by the default shuffle sampler.
        if "curriculum" in state_dict:
            self._sampler.import_state(state_dict)


class PassRateCurriculumSampler(Sampler[int]):
    """Draws dataset indices with replacement, weighted by estimated pass rate.

    One epoch is ``len(dataset)`` draws, so epoch and step accounting match the
    shuffled sampler it replaces. Deterministic given ``seed`` and the sequence
    of ``update`` calls.
    """

    def __init__(
        self,
        num_tasks: int,
        config: PassRateCurriculumConfig,
        *,
        group_size: int,
        seed: int,
    ) -> None:
        """Initialize an empty curriculum.

        Args:
            num_tasks: Dataset length; indices are drawn from ``range(num_tasks)``.
            config: The ``grpo.curriculum`` block.
            group_size: Rollouts per prompt (the ``n`` in ``mixed_group``).
            seed: Seed for the draw generator.
        """
        if num_tasks < 1:
            raise ValueError(f"curriculum needs a non-empty dataset, got {num_tasks}")
        if config.weighting == "mixed_group" and group_size < 2:
            raise ValueError(
                "curriculum weighting 'mixed_group' needs num_generations_per_prompt "
                f">= 2: a single rollout never has reward spread (got {group_size})."
            )
        self._num_tasks = num_tasks
        self._config = config
        self._group_size = group_size
        self._passes = torch.zeros(num_tasks, dtype=torch.float64)
        self._attempts = torch.zeros(num_tasks, dtype=torch.float64)
        self._groups_observed = 0
        self._generator = torch.Generator().manual_seed(seed)
        # Cumulative weights for draws; None after an update until the next draw.
        self._cdf: torch.Tensor | None = None

    def set_initial_counts(
        self, task_index: int, passes: float, attempts: float
    ) -> None:
        """Seed a task's counts from a prior profile (``init_stats_path``).

        Seeded counts decay like observed ones and do not count as observed groups.
        A checkpoint resume replaces them with the saved stats.
        """
        if not 0 <= task_index < self._num_tasks:
            raise IndexError(
                f"curriculum task index {task_index} is outside the dataset "
                f"(len {self._num_tasks})"
            )
        if not 0 <= passes <= attempts:
            raise ValueError(f"need 0 <= passes <= attempts, got {passes}/{attempts}")
        self._passes[task_index] = passes
        self._attempts[task_index] = attempts
        self._cdf = None

    def __len__(self) -> int:
        return self._num_tasks

    def __iter__(self) -> Iterator[int]:
        return _CurriculumIterator(self)

    def update(self, task_index: int, rewards: Sequence[float]) -> None:
        """Fold one finished group's valid rewards into its task's estimate.

        Args:
            task_index: Dataset index the group was generated from.
            rewards: One reward per valid rollout; masked rollouts must be left out.
        """
        if not 0 <= task_index < self._num_tasks:
            raise IndexError(
                f"curriculum task index {task_index} is outside the dataset "
                f"(len {self._num_tasks})"
            )
        if not rewards:
            return
        passes = sum(reward > self._config.pass_threshold for reward in rewards)
        decay = self._config.decay
        self._passes[task_index] = decay * self._passes[task_index] + passes
        self._attempts[task_index] = decay * self._attempts[task_index] + len(rewards)
        self._groups_observed += 1
        self._cdf = None

    def pass_rates(self) -> torch.Tensor:
        """Posterior-mean pass rate per task under the Beta prior."""
        strength = self._config.prior_strength
        return (self._passes + strength * self._config.prior_pass_rate) / (
            self._attempts + strength
        )

    def weights(self) -> torch.Tensor:
        """Unnormalized draw weight per task, in ``[min_weight, 1]``."""
        if self._config.weighting == "uniform":
            return torch.ones(self._num_tasks, dtype=torch.float64)
        p = self.pass_rates()
        if self._config.weighting == "variance":
            curve = 4.0 * p * (1.0 - p)
        else:
            n = self._group_size
            # Divided by the value at p=0.5 so the peak is 1, as for variance.
            curve = (1.0 - p**n - (1.0 - p) ** n) / (1.0 - 2.0 * 0.5**n)
        return curve.clamp(min=self._config.min_weight)

    def draw(self) -> int:
        """Draw one task index from the current weights."""
        if self._cdf is None:
            self._cdf = torch.cumsum(self.weights(), dim=0)
        target = (
            torch.rand((), dtype=torch.float64, generator=self._generator)
            * self._cdf[-1]
        )
        index = int(torch.searchsorted(self._cdf, target, right=True))
        return min(index, self._num_tasks - 1)

    def metrics(self) -> dict[str, float]:
        """Summary of the curriculum state for the step log."""
        weights = self.weights()
        ess = float(weights.sum() ** 2 / (weights**2).sum())
        tracked = self._attempts > 0
        num_tracked = int(tracked.sum())
        metrics = {
            "curriculum/tracked_tasks": float(num_tracked),
            "curriculum/groups_observed_total": float(self._groups_observed),
            "curriculum/effective_sample_size": ess,
            "curriculum/effective_sample_fraction": ess / self._num_tasks,
        }
        if num_tracked:
            passes = self._passes[tracked]
            attempts = self._attempts[tracked]
            metrics["curriculum/mean_pass_rate"] = float(
                self.pass_rates()[tracked].mean()
            )
            # Every rollout seen so far passed (failed); decay keeps these exact.
            metrics["curriculum/all_pass_fraction"] = float(
                (passes == attempts).double().mean()
            )
            metrics["curriculum/all_fail_fraction"] = float(
                (passes == 0).double().mean()
            )
        return metrics

    def export_state(self) -> dict[str, Any]:
        """Stats and generator state, as plain tensors so ``weights_only`` loads work."""
        return {
            "generator": self._generator.get_state(),
            "curriculum": {
                "num_tasks": self._num_tasks,
                "passes": self._passes.clone(),
                "attempts": self._attempts.clone(),
                "groups_observed": self._groups_observed,
            },
        }

    def import_state(self, state: dict[str, Any]) -> None:
        """Restore what ``export_state`` saved."""
        stats = state["curriculum"]
        if stats["num_tasks"] != self._num_tasks:
            raise ValueError(
                f"curriculum checkpoint was saved for {stats['num_tasks']} tasks but "
                f"the dataset has {self._num_tasks}; per-task stats cannot be mapped."
            )
        self._passes = stats["passes"].clone()
        self._attempts = stats["attempts"].clone()
        self._groups_observed = stats["groups_observed"]
        self._generator.set_state(state["generator"])
        self._cdf = None
