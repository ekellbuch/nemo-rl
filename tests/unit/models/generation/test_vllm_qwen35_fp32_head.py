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

"""Check the native vLLM projection used by Qwen3.5's LogitsProcessor."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


@pytest.mark.vllm
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_pinned_vllm_head_dtype_projects_bf16_inputs_to_fp32(device):
    vllm = pytest.importorskip("vllm")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA GPU unavailable")

    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )

    assert vllm.__version__ == "0.26.0"

    processor = LogitsProcessor.__new__(LogitsProcessor)
    processor.head_dtype = torch.float32
    hidden = torch.tensor([[1.25, -0.75]], dtype=torch.bfloat16, device=device)
    weight = torch.tensor(
        [[0.5, 1.5], [-1.25, 0.25]], dtype=torch.bfloat16, device=device
    )
    head = SimpleNamespace(weight=weight, quant_method=UnquantizedEmbeddingMethod())

    logits = processor._apply_head(head, hidden, None)
    expected = F.linear(hidden.float(), weight.float())

    assert logits.dtype == torch.float32
    torch.testing.assert_close(logits, expected)
