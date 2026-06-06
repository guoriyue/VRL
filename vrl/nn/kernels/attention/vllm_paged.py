"""vLLM internal paged-attention kernel wrapper for AR decoder layers."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from typing import Any

import torch

from vrl.nn.layers.attention.paged import (
    ARAttentionConfig,
    ARAttentionUnavailable,
)


class VllmPagedAttentionKernels:
    """Thin, real-call wrapper around vLLM's internal paged-attention APIs.

    This class deliberately is not an ``ARAttentionBackend``. It does not
    pretend to run a Janus/NextStep transformer by itself; family runners still
    need to patch their attention layers so Q/K/V projection, output projection,
    residuals, and MLP stay model-specific. The boundary here is only the part
    worth borrowing from vLLM: block-table layout, slot mapping, paged KV writes,
    and FlashAttention forward over a vLLM KV cache.
    """

    _REQUIRED_MODULES = (
        "vllm",
        "vllm.v1.worker.block_table",
        "vllm.v1.attention.backend",
        "vllm.v1.attention.backends.flash_attn",
        "vllm.v1.attention.ops.paged_attn",
    )

    def __init__(
        self,
        config: ARAttentionConfig,
        *,
        import_module: Callable[[str], Any] | None = None,
    ) -> None:
        self.config = config
        loader = import_module or importlib.import_module
        self.modules: dict[str, Any] = {}
        for module_name in self._REQUIRED_MODULES:
            try:
                self.modules[module_name] = loader(module_name)
            except Exception as exc:
                raise ARAttentionUnavailable(
                    "vLLM paged-attention initialization failed while importing "
                    f"{module_name!r}. This is a real internal API import, so a "
                    "failure usually means the installed vLLM wheel does not match "
                    "the active PyTorch/CUDA ABI.",
                ) from exc

    @property
    def block_table_module(self) -> Any:
        return self.modules["vllm.v1.worker.block_table"]

    @property
    def attention_backend_module(self) -> Any:
        return self.modules["vllm.v1.attention.backend"]

    @property
    def flash_attn_module(self) -> Any:
        return self.modules["vllm.v1.attention.backends.flash_attn"]

    @property
    def paged_attn_module(self) -> Any:
        return self.modules["vllm.v1.attention.ops.paged_attn"]

    def get_kv_cache_shape(
        self,
        *,
        num_blocks: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype: str = "auto",
    ) -> tuple[int, ...]:
        return self.flash_attn_module.FlashAttentionBackend.get_kv_cache_shape(
            num_blocks=num_blocks,
            block_size=self.config.block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            cache_dtype_str=cache_dtype,
        )

    def split_kv_cache(
        self,
        kv_cache: torch.Tensor,
        *,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.paged_attn_module.PagedAttention.split_kv_cache(
            kv_cache,
            num_kv_heads,
            head_size,
        )

    def new_block_table(
        self,
        *,
        max_num_reqs: int,
        max_num_blocks_per_req: int,
        max_num_batched_tokens: int,
        device: torch.device,
        pin_memory: bool = False,
        kernel_block_size: int | None = None,
        cp_kv_cache_interleave_size: int = 1,
    ) -> Any:
        return self.block_table_module.BlockTable(
            block_size=self.config.block_size,
            max_num_reqs=max_num_reqs,
            max_num_blocks_per_req=max_num_blocks_per_req,
            max_num_batched_tokens=max_num_batched_tokens,
            pin_memory=pin_memory,
            device=device,
            kernel_block_size=kernel_block_size or self.config.block_size,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        )

    def compute_slot_mapping(
        self,
        *,
        block_table: Any,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        block_table.compute_slot_mapping(num_reqs, query_start_loc, positions)
        return block_table.slot_mapping.gpu[: positions.shape[0]]

    def make_flash_attention_impl(
        self,
        *,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: Any | None = None,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> Any:
        if attn_type is None:
            attn_type = self.attention_backend_module.AttentionType.DECODER
        return self.flash_attn_module.FlashAttentionImpl(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            kv_cache_dtype=kv_cache_dtype,
            logits_soft_cap=logits_soft_cap,
            attn_type=attn_type,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            sinks=sinks,
        )

    def make_flash_attention_metadata(
        self,
        *,
        num_actual_tokens: int,
        max_query_len: int,
        query_start_loc: torch.Tensor,
        max_seq_len: int,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
        use_cascade: bool = False,
        common_prefix_len: int = 0,
        cu_prefix_query_lens: torch.Tensor | None = None,
        prefix_kv_lens: torch.Tensor | None = None,
        suffix_kv_lens: torch.Tensor | None = None,
        max_dcp_context_kv_len: int | None = None,
        dcp_context_kv_lens: torch.Tensor | None = None,
        scheduler_metadata: torch.Tensor | None = None,
        prefix_scheduler_metadata: torch.Tensor | None = None,
        max_num_splits: int = 0,
        causal: bool = True,
    ) -> Any:
        return self.flash_attn_module.FlashAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table,
            slot_mapping=slot_mapping,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            max_dcp_context_kv_len=max_dcp_context_kv_len,
            dcp_context_kv_lens=dcp_context_kv_lens,
            scheduler_metadata=scheduler_metadata,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            max_num_splits=max_num_splits,
            causal=causal,
        )

    def write_to_paged_cache(
        self,
        *,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> None:
        self.paged_attn_module.PagedAttention.write_to_paged_cache(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )

    def update_flash_kv_cache(
        self,
        *,
        impl: Any,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        impl.do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)

    def run_flash_attention(
        self,
        *,
        impl: Any,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        metadata: Any,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output is None:
            output = torch.empty(
                query.shape[0],
                query.shape[1],
                query.shape[2],
                dtype=query.dtype,
                device=query.device,
            )
        result = impl.forward(
            layer,
            query,
            key,
            value,
            kv_cache,
            metadata,
            output=output,
        )
        return result.reshape(result.shape[0], -1)

    def debug_info(self) -> Mapping[str, Any]:
        vllm = self.modules.get("vllm")
        return {
            "family": self.config.family,
            "model_key": self.config.model_key,
            "backend": "vllm_paged_attention_kernels",
            "cache_layout_version": self.config.cache_layout_version,
            "vllm_version": getattr(vllm, "__version__", None),
            "required_modules": self._REQUIRED_MODULES,
        }


__all__ = ["VllmPagedAttentionKernels"]
