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

"""Tests for vLLM worker helper functions."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from nemo_rl.models.generation.vllm.worker_utils import (
    find_tokenizer_required_architectures,
    resolve_data_parallel_local_rank,
    resolve_distributed_executor_backend,
)


@pytest.mark.parametrize(
    ("architectures", "expected"),
    [
        (None, []),
        ([], []),
        (["Gemma4ForCausalLM"], []),
        (
            ["Gemma4ForConditionalGeneration"],
            ["Gemma4ForConditionalGeneration"],
        ),
        (
            [
                "Gemma4ForCausalLM",
                "Gemma4UnifiedForConditionalGeneration",
                "Mistral3ForConditionalGeneration",
            ],
            [
                "Gemma4UnifiedForConditionalGeneration",
                "Mistral3ForConditionalGeneration",
            ],
        ),
    ],
)
def test_find_tokenizer_required_architectures(architectures, expected):
    assert find_tokenizer_required_architectures(architectures) == expected


@pytest.mark.parametrize(
    ("tp", "pp", "ep", "expected"),
    [
        (2, 1, 2, "ray"),
        (1, 2, 2, "ray"),
        (1, 1, 8, "uni"),
        (1, 1, 1, None),
    ],
)
def test_resolve_distributed_executor_backend(tp, pp, ep, expected):
    assert resolve_distributed_executor_backend(tp, pp, ep) == expected


@pytest.mark.parametrize(
    ("rank", "model_parallel_size", "executor_backend", "expected"),
    [
        (7, 1, "uni", 0),
        (6, 2, "ray", 3),
    ],
)
def test_resolve_data_parallel_local_rank(
    rank, model_parallel_size, executor_backend, expected
):
    assert (
        resolve_data_parallel_local_rank(rank, model_parallel_size, executor_backend)
        == expected
    )


@pytest.mark.vllm
def test_qwen35_fp32_head_uses_native_vllm_override():
    from nemo_rl.models.generation.vllm.vllm_worker import (
        _configure_qwen35_fp32_lm_head,
    )

    kwargs = {"hf_overrides": {"rope_theta": 1000000}}
    _configure_qwen35_fp32_lm_head(kwargs, "0.26.0")

    assert kwargs["hf_overrides"] == {
        "rope_theta": 1000000,
        "head_dtype": "float32",
    }


@pytest.mark.vllm
@pytest.mark.parametrize(
    ("kwargs", "version", "error"),
    [
        ({}, "0.25.1", RuntimeError),
        ({"hf_overrides": {"head_dtype": "bfloat16"}}, "0.26.0", ValueError),
        ({"hf_overrides": "invalid"}, "0.26.0", ValueError),
    ],
)
def test_qwen35_fp32_head_fails_closed(kwargs, version, error):
    from nemo_rl.models.generation.vllm.vllm_worker import (
        _configure_qwen35_fp32_lm_head,
    )

    with pytest.raises(error):
        _configure_qwen35_fp32_lm_head(kwargs, version)


@pytest.mark.vllm
@pytest.mark.parametrize(
    ("model_type", "nemotron_patch"),
    [("qwen3_5", False), ("nemotron_h", True)],
)
def test_fp32_head_selects_model_specific_vllm_path(model_type, nemotron_patch):
    from nemo_rl.models.generation.vllm.vllm_worker import (
        BaseVllmGenerationWorker,
    )

    config = {
        "model_name": "test-model",
        "vllm_cfg": {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": 1,
            "gpu_memory_utilization": 0.8,
            "precision": "bfloat16",
            "fp32_lm_head": True,
        },
    }
    worker = BaseVllmGenerationWorker.__new__(BaseVllmGenerationWorker)
    with (
        patch(
            "nemo_rl.models.generation.vllm.vllm_worker.AutoConfig.from_pretrained",
            return_value=SimpleNamespace(model_type=model_type),
        ),
        patch(
            "nemo_rl.models.generation.vllm.vllm_worker._apply_vllm_patches"
        ) as apply_patches,
    ):
        worker._init_config(
            config,
            bundle_indices=None,
            fraction_of_gpus=1.0,
            seed=None,
            extra_env_vars=None,
        )

    assert worker._qwen35_fp32_lm_head is (not nemotron_patch)
    assert apply_patches.call_args.kwargs["nemotron_h_fp32_lm_head"] is nemotron_patch


@pytest.mark.vllm
def test_fp32_head_rejects_unsupported_model_before_patching():
    from nemo_rl.models.generation.vllm.vllm_worker import (
        BaseVllmGenerationWorker,
    )

    worker = BaseVllmGenerationWorker.__new__(BaseVllmGenerationWorker)
    config = {
        "model_name": "test-model",
        "vllm_cfg": {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": 1,
            "gpu_memory_utilization": 0.8,
            "precision": "bfloat16",
            "fp32_lm_head": True,
        },
    }
    with (
        patch(
            "nemo_rl.models.generation.vllm.vllm_worker.AutoConfig.from_pretrained",
            return_value=SimpleNamespace(model_type="qwen2"),
        ),
        patch(
            "nemo_rl.models.generation.vllm.vllm_worker._apply_vllm_patches"
        ) as apply_patches,
        pytest.raises(ValueError, match="supports only Nemotron-H or Qwen3.5"),
    ):
        worker._init_config(
            config,
            bundle_indices=None,
            fraction_of_gpus=1.0,
            seed=None,
            extra_env_vars=None,
        )
    apply_patches.assert_not_called()
