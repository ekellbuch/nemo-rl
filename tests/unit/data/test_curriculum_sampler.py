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

"""Unit tests for the pass-rate curriculum sampler."""

import io
from collections import Counter

import pytest
import torch
from pydantic import ValidationError
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_rl.data.curriculum_sampler import (
    PassRateCurriculumConfig,
    PassRateCurriculumSampler,
)


def _sampler(
    num_tasks: int = 10, *, group_size: int = 4, seed: int = 0, **config
) -> PassRateCurriculumSampler:
    return PassRateCurriculumSampler(
        num_tasks,
        PassRateCurriculumConfig(enabled=True, **config),
        group_size=group_size,
        seed=seed,
    )


def _loader(sampler: PassRateCurriculumSampler, batch_size: int = 2):
    return StatefulDataLoader(
        list(range(len(sampler))),
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
    )


class TestEstimates:
    def test_unseen_task_sits_at_the_prior(self):
        sampler = _sampler(prior_pass_rate=0.3, prior_strength=4.0)
        assert sampler.pass_rates()[0].item() == pytest.approx(0.3)

    def test_prior_is_mixed_with_observed_rollouts(self):
        sampler = _sampler(prior_pass_rate=0.5, prior_strength=2.0)
        sampler.update(0, [1.0, 1.0, 1.0, 0.0])
        # (3 + 2 * 0.5) / (4 + 2)
        assert sampler.pass_rates()[0].item() == pytest.approx(4 / 6)

    def test_old_evidence_decays_per_update_of_the_task(self):
        sampler = _sampler(decay=0.5, prior_strength=1.0, prior_pass_rate=0.0)
        sampler.update(0, [1.0, 1.0, 1.0, 1.0])
        sampler.update(1, [0.0, 0.0])  # another task's update does not decay task 0
        sampler.update(0, [0.0, 0.0, 0.0, 0.0])
        # passes = 0.5 * 4 + 0, attempts = 0.5 * 4 + 4
        assert sampler.pass_rates()[0].item() == pytest.approx(2 / (6 + 1))

    def test_pass_threshold_is_strict(self):
        sampler = _sampler(pass_threshold=0.5, prior_strength=1.0, prior_pass_rate=0.0)
        sampler.update(0, [0.5, 0.51])
        assert sampler.pass_rates()[0].item() == pytest.approx(1 / 3)

    def test_empty_group_is_ignored(self):
        sampler = _sampler()
        sampler.update(0, [])
        assert sampler.metrics()["curriculum/tracked_tasks"] == 0.0

    def test_out_of_range_task_is_rejected(self):
        with pytest.raises(IndexError, match="outside the dataset"):
            _sampler(num_tasks=3).update(3, [1.0])


class TestWeights:
    @staticmethod
    def _with_pass_rate(**config) -> PassRateCurriculumSampler:
        # Task 0 at p = 0.25 (prior-free), task 1 unseen at the prior 0.5.
        sampler = _sampler(
            num_tasks=2, prior_pass_rate=0.5, prior_strength=1e-9, **config
        )
        sampler.update(0, [1.0, 0.0, 0.0, 0.0])
        return sampler

    def test_mixed_group_is_the_chance_a_group_has_spread(self):
        sampler = self._with_pass_rate(weighting="mixed_group", min_weight=1e-6)
        n, p = 4, 0.25
        expected = (1 - p**n - (1 - p) ** n) / (1 - 2 * 0.5**n)
        weights = sampler.weights()
        assert weights[0].item() == pytest.approx(expected, rel=1e-6)
        assert weights[1].item() == pytest.approx(1.0)

    def test_variance_is_p_times_one_minus_p_scaled_to_peak_one(self):
        sampler = self._with_pass_rate(weighting="variance", min_weight=1e-6)
        weights = sampler.weights()
        assert weights[0].item() == pytest.approx(4 * 0.25 * 0.75, rel=1e-6)
        assert weights[1].item() == pytest.approx(1.0)

    def test_uniform_ignores_the_estimates(self):
        sampler = self._with_pass_rate(weighting="uniform")
        assert sampler.weights().tolist() == [1.0, 1.0]

    @pytest.mark.parametrize("weighting", ["mixed_group", "variance"])
    def test_solved_and_failed_tasks_keep_the_floor(self, weighting):
        sampler = _sampler(
            num_tasks=3, weighting=weighting, min_weight=0.1, prior_strength=1e-9
        )
        sampler.update(0, [1.0] * 4)
        sampler.update(1, [0.0] * 4)
        weights = sampler.weights()
        assert weights[0].item() == pytest.approx(0.1)
        assert weights[1].item() == pytest.approx(0.1)
        assert weights[2].item() == pytest.approx(1.0)

    def test_mixed_group_needs_two_generations(self):
        with pytest.raises(ValueError, match="num_generations_per_prompt"):
            _sampler(group_size=1, weighting="mixed_group")
        _sampler(group_size=1, weighting="variance")

    def test_config_rejects_out_of_range_values(self):
        with pytest.raises(ValidationError):
            PassRateCurriculumConfig(decay=0.0)
        with pytest.raises(ValidationError):
            PassRateCurriculumConfig(min_weight=0.0)
        with pytest.raises(ValidationError):
            PassRateCurriculumConfig(weighting="thompson")


class TestDraws:
    def test_draw_frequencies_follow_the_weights(self):
        sampler = _sampler(num_tasks=2, min_weight=0.1, prior_strength=1e-9)
        sampler.update(0, [1.0] * 4)  # all-pass: floor weight 0.1 vs peak 1.0
        counts = Counter(sampler.draw() for _ in range(20_000))
        assert counts[0] / 20_000 == pytest.approx(0.1 / 1.1, abs=0.01)

    def test_same_seed_and_updates_give_the_same_draws(self):
        def run(seed: int) -> list[int]:
            sampler = _sampler(num_tasks=50, seed=seed)
            draws = [sampler.draw() for _ in range(20)]
            for index in draws[:10]:
                sampler.update(index, [1.0, 1.0, 1.0, 1.0])
            return draws + [sampler.draw() for _ in range(20)]

        assert run(7) == run(7)
        assert run(7) != run(8)

    def test_an_epoch_is_dataset_length_draws(self):
        sampler = _sampler(num_tasks=7)
        loader = _loader(sampler, batch_size=2)
        assert len(sampler) == 7
        assert len(loader) == 3
        for _ in range(2):
            batches = list(loader)
            assert len(batches) == 3
            assert all(0 <= int(i) < 7 for batch in batches for i in batch)

    def test_draws_are_with_replacement(self):
        sampler = _sampler(num_tasks=2, min_weight=0.01, prior_strength=1e-9)
        sampler.update(0, [1.0] * 4)
        epochs = [list(iter(sampler)) for _ in range(20)]
        assert all(len(epoch) == 2 for epoch in epochs)
        # At weight 0.01 vs 1.0, task 1 is drawn twice in nearly every epoch.
        assert [1, 1] in epochs


class TestMetrics:
    def test_metrics_summarize_tracked_tasks(self):
        sampler = _sampler(num_tasks=4, prior_strength=1e-9, min_weight=0.05)
        sampler.update(0, [1.0, 1.0])
        sampler.update(1, [0.0, 0.0])
        sampler.update(2, [1.0, 0.0])

        metrics = sampler.metrics()

        assert metrics["curriculum/tracked_tasks"] == 3.0
        assert metrics["curriculum/groups_observed_total"] == 3.0
        assert metrics["curriculum/all_pass_fraction"] == pytest.approx(1 / 3)
        assert metrics["curriculum/all_fail_fraction"] == pytest.approx(1 / 3)
        assert metrics["curriculum/mean_pass_rate"] == pytest.approx(0.5, abs=1e-6)
        # Weights 0.05, 0.05, 1, 1 (task 3 unseen at the prior peak).
        ess = 2.1**2 / (2 * 0.05**2 + 2)
        assert metrics["curriculum/effective_sample_size"] == pytest.approx(ess)
        assert metrics["curriculum/effective_sample_fraction"] == pytest.approx(ess / 4)

    def test_uniform_weights_have_full_effective_sample_size(self):
        metrics = _sampler(num_tasks=5).metrics()
        assert metrics["curriculum/effective_sample_fraction"] == pytest.approx(1.0)
        assert "curriculum/mean_pass_rate" not in metrics


class TestCheckpointing:
    def test_dataloader_state_round_trip_resumes_the_same_stream(self):
        sampler = _sampler(num_tasks=12, seed=3)
        loader = _loader(sampler, batch_size=3)
        iterator = iter(loader)
        first = next(iterator)
        for index in first:
            sampler.update(int(index), [1.0, 0.0, 0.0, 0.0])
        state = loader.state_dict()
        expected_rest = [batch.tolist() for batch in iterator]
        expected_metrics = sampler.metrics()

        # weights_only, as load_dataloader_state does by default.
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        restored_state = torch.load(buffer, weights_only=True)

        resumed_sampler = _sampler(num_tasks=12, seed=99)
        resumed = _loader(resumed_sampler, batch_size=3)
        resumed.load_state_dict(restored_state)
        resumed_rest = [batch.tolist() for batch in resumed]

        assert resumed_rest == expected_rest
        assert resumed_sampler.metrics() == expected_metrics

    def test_dataset_size_change_is_rejected(self):
        loader = _loader(_sampler(num_tasks=4))
        next(iter(loader))
        state = loader.state_dict()

        resumed = _loader(_sampler(num_tasks=6))
        resumed.load_state_dict(state)
        with pytest.raises(ValueError, match="saved for 4 tasks"):
            next(iter(resumed))

    def test_shuffle_checkpoint_resumes_into_the_curriculum(self):
        shuffled = StatefulDataLoader(
            list(range(8)), batch_size=2, shuffle=True, drop_last=True
        )
        iterator = iter(shuffled)
        next(iterator)
        state = shuffled.state_dict()

        sampler = _sampler(num_tasks=8)
        resumed = _loader(sampler, batch_size=2)
        resumed.load_state_dict(state)

        # The epoch position carries over; the stats start empty.
        assert len(list(resumed)) == 3
        assert sampler.metrics()["curriculum/tracked_tasks"] == 0.0

    def test_curriculum_checkpoint_resumes_with_the_curriculum_off(self):
        loader = _loader(_sampler(num_tasks=8), batch_size=2)
        iterator = iter(loader)
        next(iterator)
        state = loader.state_dict()

        shuffled = StatefulDataLoader(
            list(range(8)), batch_size=2, shuffle=True, drop_last=True
        )
        shuffled.load_state_dict(state)

        assert len(list(shuffled)) == 3
