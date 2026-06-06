"""Real vLLM paged-attention integration test."""

from __future__ import annotations

import pytest
import torch

from vrl.nn.kernels.attention.vllm_paged import VllmPagedAttentionKernels
from vrl.nn.layers.attention.paged import (
    ARAttentionConfig,
    ARAttentionUnavailable,
)


@pytest.mark.gpu
def test_vllm_paged_attention_writes_real_cuda_kv_cache() -> None:
    try:
        kernels = VllmPagedAttentionKernels(
            ARAttentionConfig(family="janus_pro", model_key="probe")
        )
    except ARAttentionUnavailable as exc:
        pytest.skip(f"vLLM paged-attention internals are unavailable: {exc}")

    assert kernels.get_kv_cache_shape(
        num_blocks=1,
        num_kv_heads=2,
        head_size=8,
    ) == (2, 1, 16, 2, 8)

    block_table = kernels.new_block_table(
        max_num_reqs=1,
        max_num_blocks_per_req=1,
        max_num_batched_tokens=1,
        device=torch.device("cuda"),
    )
    block_table.add_row([0], row_idx=0)
    block_table.commit_block_table(1)
    slot_mapping = kernels.compute_slot_mapping(
        block_table=block_table,
        num_reqs=1,
        query_start_loc=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        positions=torch.tensor([0], device="cuda", dtype=torch.int64),
    )

    kv_cache = torch.zeros((2, 1, 16, 2, 8), device="cuda", dtype=torch.float16)
    key_cache, value_cache = kernels.split_kv_cache(
        kv_cache,
        num_kv_heads=2,
        head_size=8,
    )
    key = torch.arange(16, device="cuda", dtype=torch.float16).reshape(1, 2, 8)
    value = torch.arange(16, 32, device="cuda", dtype=torch.float16).reshape(1, 2, 8)
    scale = torch.ones(1, device="cuda", dtype=torch.float32)

    kernels.write_to_paged_cache(
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_mapping=slot_mapping,
        kv_cache_dtype="auto",
        k_scale=scale,
        v_scale=scale,
    )
    torch.cuda.synchronize()

    assert key_cache.sum().item() == pytest.approx(120.0)
    assert value_cache.sum().item() == pytest.approx(376.0)
