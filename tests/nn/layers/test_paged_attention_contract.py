"""Tests for the AR paged-attention backend contract.

Every test here is the real dataclass rejecting or accepting real tensors, but
nothing in this file reaches a kernel, so the whole module carries one
``real_cover`` label naming the gpu-lane test that does.
"""

from __future__ import annotations

import pytest
import torch

from vrl.nn.layers.attention.paged import (
    ARAttentionPrefillInput,
    ARAttentionStepInput,
)

pytestmark = pytest.mark.real_cover(
    "tests/nn/kernels/test_vllm_paged_attention_real_ops.py"
    "::test_vllm_paged_attention_writes_real_cuda_kv_cache",
    why=(
        "these are shape and identity contracts only — no test in this file allocates a KV "
        "cache, builds a block table or calls a paged-attention kernel, all of which need "
        "CUDA plus vLLM's worker internals; the gpu-lane test drives the real ops"
    ),
)


def test_paged_attention_prefill_validates_batch_shape() -> None:
    """Embeds and mask must agree on the batch dimension."""
    with pytest.raises(ValueError, match="batch sizes"):
        ARAttentionPrefillInput(
            inputs_embeds=torch.zeros(2, 3, 4),
            attention_mask=torch.ones(1, 3),
            branch="cond",
        )


def test_paged_attention_prefill_requires_positive_token_budget() -> None:
    """A zero token budget cannot reserve blocks, so it is refused up front."""
    with pytest.raises(ValueError, match="max_new_tokens"):
        ARAttentionPrefillInput(
            inputs_embeds=torch.zeros(1, 3, 4),
            attention_mask=torch.ones(1, 3),
            branch="cond",
            max_new_tokens=0,
        )


def test_paged_attention_step_validates_state_shape() -> None:
    """One sequence state per batch row, or construction fails."""
    with pytest.raises(ValueError, match="sequence_states"):
        ARAttentionStepInput(
            input_embeds=torch.zeros(2, 1, 4),
            attention_mask=torch.ones(2, 4),
            sequence_states=("row-0",),
        )
