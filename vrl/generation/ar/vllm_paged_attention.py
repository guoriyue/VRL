"""vLLM internal paged-attention kernel wrapper for AR generation."""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Mapping
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from typing import Any

import torch

from vrl.generation.ar.paged_attention import (
    ARPagedAttentionBackend,
    ARPagedAttentionConfig,
    ARPagedAttentionPrefillInput,
    ARPagedAttentionPrefillOutput,
    ARPagedAttentionStepInput,
    ARPagedAttentionStepOutput,
    ARPagedAttentionUnavailable,
)


class VllmPagedAttentionKernels:
    """Thin, real-call wrapper around vLLM's internal paged-attention APIs.

    This class deliberately is not an ``ARPagedAttentionBackend``. It does not
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
        config: ARPagedAttentionConfig,
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
                raise ARPagedAttentionUnavailable(
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


@dataclass(frozen=True, slots=True)
class VllmDecoderPagedSequenceState:
    """Per-sequence physical vLLM KV page ownership for decoder-only AR."""

    sequence_id: str
    branch: str
    row: int
    length: int
    block_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _PackedPrefill:
    inputs_embeds: torch.Tensor
    positions: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    last_token_offsets: torch.Tensor
    states: tuple[VllmDecoderPagedSequenceState, ...]


@dataclass(frozen=True, slots=True)
class _PackedStep:
    inputs_embeds: torch.Tensor
    positions: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    states: tuple[VllmDecoderPagedSequenceState, ...]


@dataclass(frozen=True, slots=True)
class _PagedForwardContext:
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int


class _VllmAttentionScaleShim(torch.nn.Module):
    """Small shim exposing the quant-scale attrs required by vLLM attention."""

    def __init__(self, *, device: torch.device) -> None:
        super().__init__()
        scale = torch.ones(1, device=device, dtype=torch.float32)
        self.register_buffer("_q_scale", scale, persistent=False)
        self.register_buffer("_k_scale", scale.clone(), persistent=False)
        self.register_buffer("_v_scale", scale.clone(), persistent=False)


class VllmDecoderPagedAttentionBackend(ARPagedAttentionBackend):
    """Run a HF-style decoder trunk with vLLM block tables and paged KV cache."""

    def __init__(
        self,
        *,
        trunk: Any,
        config: ARPagedAttentionConfig,
        kernels: VllmPagedAttentionKernels | None = None,
    ) -> None:
        super().__init__(config)
        self.trunk = trunk
        self.kernels = kernels or VllmPagedAttentionKernels(config)
        self.cache_dtype = str(config.extra.get("cache_dtype", "auto"))
        self.backend_label = str(
            config.extra.get("backend_label", f"{config.family}_vllm_paged_attention")
        )
        self._next_sequence_id = 0
        self._next_block_id = 0
        self._kv_caches: list[torch.Tensor] = []
        self._attention_impls: list[Any] = []
        self._scale_shim: _VllmAttentionScaleShim | None = None
        self._kv_cache_num_blocks = 0

    @torch.no_grad()
    def prefill(
        self,
        request: ARPagedAttentionPrefillInput,
    ) -> ARPagedAttentionPrefillOutput:
        packed = self._pack_prefill(request)
        last_hidden_states = self._forward_paged_trunk(
            packed.inputs_embeds,
            positions=packed.positions,
            query_start_loc=packed.query_start_loc,
            seq_lens=packed.seq_lens,
            states=packed.states,
        )
        return ARPagedAttentionPrefillOutput(
            last_hidden=last_hidden_states.index_select(0, packed.last_token_offsets),
            sequence_states=packed.states,
            metrics={
                "backend": self.backend_label,
                "prefill_tokens": int(packed.inputs_embeds.shape[0]),
                "allocated_blocks": self._next_block_id,
            },
        )

    @torch.no_grad()
    def step(self, request: ARPagedAttentionStepInput) -> ARPagedAttentionStepOutput:
        packed = self._pack_step(request)
        last_hidden_states = self._forward_paged_trunk(
            packed.inputs_embeds,
            positions=packed.positions,
            query_start_loc=packed.query_start_loc,
            seq_lens=packed.seq_lens,
            states=packed.states,
        )
        return ARPagedAttentionStepOutput(
            last_hidden=last_hidden_states,
            sequence_states=packed.states,
            metrics={
                "backend": self.backend_label,
                "decode_tokens": int(packed.inputs_embeds.shape[0]),
                "allocated_blocks": self._next_block_id,
            },
        )

    def debug_info(self) -> dict[str, Any]:
        return {
            **dict(super().debug_info()),
            "backend": self.backend_label,
            "allocated_blocks": self._next_block_id,
            "kv_cache_num_blocks": self._kv_cache_num_blocks,
            "vllm": self.kernels.debug_info(),
        }

    def _pack_prefill(self, request: ARPagedAttentionPrefillInput) -> _PackedPrefill:
        embeds = request.inputs_embeds
        mask = request.attention_mask
        self._validate_runtime_tensor(embeds)
        mask_bool = mask.to(dtype=torch.bool)
        lengths = mask_bool.sum(dim=1).to(dtype=torch.long)
        if bool((lengths <= 0).any()):
            raise ValueError(f"{self.backend_label} prefill requires non-empty prompts")
        self._validate_right_padded_mask(mask_bool, lengths)

        packed_embeds: list[torch.Tensor] = []
        packed_positions: list[torch.Tensor] = []
        query_starts = [0]
        last_offsets: list[int] = []
        states: list[VllmDecoderPagedSequenceState] = []
        max_new_tokens = self._max_new_tokens_from_metadata(request.metadata)
        sequence_ids = request.sequence_ids or tuple(
            f"{request.branch}:{row}:{self._next_sequence_id + row}"
            for row in range(embeds.shape[0])
        )
        for row, length_tensor in enumerate(lengths):
            length = int(length_tensor.item())
            packed_embeds.append(embeds[row, :length])
            packed_positions.append(
                torch.arange(length, dtype=torch.long, device=embeds.device)
            )
            query_starts.append(query_starts[-1] + length)
            last_offsets.append(query_starts[-1] - 1)
            states.append(
                VllmDecoderPagedSequenceState(
                    sequence_id=str(sequence_ids[row]),
                    branch=request.branch,
                    row=row,
                    length=length,
                    block_ids=self._allocate_blocks(length + max_new_tokens),
                )
            )
        self._next_sequence_id += embeds.shape[0]
        return _PackedPrefill(
            inputs_embeds=torch.cat(packed_embeds, dim=0),
            positions=torch.cat(packed_positions, dim=0),
            query_start_loc=torch.tensor(
                query_starts,
                dtype=torch.int32,
                device=embeds.device,
            ),
            seq_lens=lengths.to(device=embeds.device, dtype=torch.int32),
            last_token_offsets=torch.tensor(
                last_offsets,
                dtype=torch.long,
                device=embeds.device,
            ),
            states=tuple(states),
        )

    def _pack_step(self, request: ARPagedAttentionStepInput) -> _PackedStep:
        embeds = request.input_embeds
        self._validate_runtime_tensor(embeds)
        if embeds.shape[1] != 1:
            raise ValueError(f"{self.backend_label} step expects one token per sequence")
        states = self._typed_states(request.sequence_states)
        if len(states) != embeds.shape[0]:
            raise ValueError(f"{self.backend_label} state count must match batch size")
        for state in states:
            if state.length >= len(state.block_ids) * self.config.block_size:
                raise RuntimeError(
                    f"{self.backend_label} sequence exhausted its allocated KV blocks: "
                    f"sequence_id={state.sequence_id!r}",
                )

        next_states = tuple(
            VllmDecoderPagedSequenceState(
                sequence_id=state.sequence_id,
                branch=branch,
                row=state.row,
                length=state.length + 1,
                block_ids=state.block_ids,
            )
            for state, branch in zip(states, request.branch_names, strict=True)
        )
        positions = torch.tensor(
            [state.length for state in states],
            dtype=torch.long,
            device=embeds.device,
        )
        batch = embeds.shape[0]
        return _PackedStep(
            inputs_embeds=embeds[:, 0, :],
            positions=positions,
            query_start_loc=torch.arange(
                batch + 1,
                dtype=torch.int32,
                device=embeds.device,
            ),
            seq_lens=torch.tensor(
                [state.length for state in next_states],
                dtype=torch.int32,
                device=embeds.device,
            ),
            states=next_states,
        )

    def _forward_paged_trunk(
        self,
        inputs_embeds: torch.Tensor,
        *,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        states: SequenceABC[VllmDecoderPagedSequenceState],
    ) -> torch.Tensor:
        self._ensure_runtime_objects(
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        context = self._build_paged_forward_context(
            positions=positions,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            states=states,
        )

        hidden_states = inputs_embeds
        position_ids = positions.unsqueeze(0)
        position_embeddings = self.trunk.rotary_emb(
            hidden_states.unsqueeze(0),
            position_ids,
        )
        for layer_idx, decoder_layer in enumerate(self.trunk.layers):
            residual = hidden_states
            hidden_states = decoder_layer.input_layernorm(hidden_states)
            hidden_states = self._forward_vllm_attention(
                layer_idx,
                decoder_layer.self_attn,
                hidden_states,
                position_embeddings=position_embeddings,
                context=context,
            )
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
            hidden_states = decoder_layer.mlp(hidden_states)
            hidden_states = residual + hidden_states
        return self.trunk.norm(hidden_states)

    def _forward_vllm_attention(
        self,
        layer_idx: int,
        attention: Any,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        context: _PagedForwardContext,
    ) -> torch.Tensor:
        num_heads = self._num_attention_heads(attention)
        num_kv_heads = self._num_key_value_heads(attention)
        head_dim = self._head_dim(attention, num_heads)
        query_states = attention.q_proj(hidden_states).view(-1, num_heads, head_dim)
        key_states = attention.k_proj(hidden_states).view(-1, num_kv_heads, head_dim)
        value_states = attention.v_proj(hidden_states).view(-1, num_kv_heads, head_dim)
        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb(
            query_states.unsqueeze(0),
            key_states.unsqueeze(0),
            cos,
            sin,
            unsqueeze_dim=2,
        )
        query_states = query_states.squeeze(0)
        key_states = key_states.squeeze(0)

        impl = self._attention_impls[layer_idx]
        kv_cache = self._kv_caches[layer_idx]
        layer = self._require_scale_shim(hidden_states.device)
        self.kernels.update_flash_kv_cache(
            impl=impl,
            layer=layer,
            key=key_states,
            value=value_states,
            kv_cache=kv_cache,
            slot_mapping=context.slot_mapping,
        )
        metadata = self.kernels.make_flash_attention_metadata(
            num_actual_tokens=context.num_actual_tokens,
            max_query_len=context.max_query_len,
            query_start_loc=context.query_start_loc,
            max_seq_len=context.max_seq_len,
            seq_lens=context.seq_lens,
            block_table=context.block_table,
            slot_mapping=context.slot_mapping,
        )
        output = self.kernels.run_flash_attention(
            impl=impl,
            layer=layer,
            query=query_states,
            key=key_states,
            value=value_states,
            kv_cache=kv_cache,
            metadata=metadata,
        )
        return attention.o_proj(output)

    def _build_paged_forward_context(
        self,
        *,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        states: SequenceABC[VllmDecoderPagedSequenceState],
    ) -> _PagedForwardContext:
        max_blocks = max(len(state.block_ids) for state in states)
        block_table = self.kernels.new_block_table(
            max_num_reqs=len(states),
            max_num_blocks_per_req=max_blocks,
            max_num_batched_tokens=int(positions.shape[0]),
            device=positions.device,
        )
        for row_idx, state in enumerate(states):
            block_table.add_row(list(state.block_ids), row_idx=row_idx)
        block_table.commit_block_table(len(states))
        slot_mapping = self.kernels.compute_slot_mapping(
            block_table=block_table,
            num_reqs=len(states),
            query_start_loc=query_start_loc,
            positions=positions,
        )
        return _PagedForwardContext(
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            block_table=block_table.get_device_tensor(len(states))[:, :max_blocks],
            slot_mapping=slot_mapping,
            num_actual_tokens=int(positions.shape[0]),
            max_query_len=int((query_start_loc[1:] - query_start_loc[:-1]).max().item()),
            max_seq_len=int(seq_lens.max().item()),
        )

    def _ensure_runtime_objects(self, *, device: torch.device, dtype: torch.dtype) -> None:
        if device.type != "cuda":
            raise ARPagedAttentionUnavailable(
                f"{self.backend_label} requires CUDA tensors because vLLM "
                "BlockTable slot mapping and FlashAttention are CUDA kernels.",
            )
        if dtype not in (torch.float16, torch.bfloat16):
            raise ARPagedAttentionUnavailable(
                f"{self.backend_label} requires float16 or bfloat16 model embeddings, "
                f"got {dtype}.",
            )
        layers = list(self.trunk.layers)
        if not layers:
            raise RuntimeError(f"{self.backend_label} trunk has no decoder layers")
        first_attention = layers[0].self_attn
        num_kv_heads = self._num_key_value_heads(first_attention)
        head_dim = self._head_dim(first_attention, self._num_attention_heads(first_attention))
        self._ensure_kv_caches(
            num_layers=len(layers),
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            device=device,
            dtype=dtype,
        )
        if not self._attention_impls:
            self._attention_impls = [
                self.kernels.make_flash_attention_impl(
                    num_heads=self._num_attention_heads(layer.self_attn),
                    head_size=self._head_dim(
                        layer.self_attn,
                        self._num_attention_heads(layer.self_attn),
                    ),
                    scale=float(
                        getattr(
                            layer.self_attn,
                            "scaling",
                            self._head_dim(
                                layer.self_attn,
                                self._num_attention_heads(layer.self_attn),
                            )
                            ** -0.5,
                        )
                    ),
                    num_kv_heads=self._num_key_value_heads(layer.self_attn),
                    sliding_window=self._sliding_window_for_layer(layer),
                    kv_cache_dtype=self.cache_dtype,
                )
                for layer in layers
            ]

    def _ensure_kv_caches(
        self,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        required_blocks = max(self._next_block_id, 1)
        if required_blocks <= self._kv_cache_num_blocks and self._kv_caches:
            return
        target_blocks = max(required_blocks, self._kv_cache_num_blocks * 2, 1)
        shape = self.kernels.get_kv_cache_shape(
            num_blocks=target_blocks,
            num_kv_heads=num_kv_heads,
            head_size=head_dim,
            cache_dtype=self.cache_dtype,
        )
        next_caches: list[torch.Tensor] = []
        for layer_idx in range(num_layers):
            cache = torch.zeros(shape, dtype=dtype, device=device)
            if self._kv_caches:
                old = self._kv_caches[layer_idx]
                cache[:, : old.shape[1]].copy_(old)
            next_caches.append(cache)
        self._kv_caches = next_caches
        self._kv_cache_num_blocks = target_blocks

    def _require_scale_shim(self, device: torch.device) -> _VllmAttentionScaleShim:
        if self._scale_shim is None or self._scale_shim._q_scale.device != device:
            self._scale_shim = _VllmAttentionScaleShim(device=device)
        return self._scale_shim

    def _allocate_blocks(self, max_tokens: int) -> tuple[int, ...]:
        blocks_needed = max(1, math.ceil(max_tokens / self.config.block_size))
        start = self._next_block_id
        self._next_block_id += blocks_needed
        return tuple(range(start, start + blocks_needed))

    def _validate_runtime_tensor(self, tensor: torch.Tensor) -> None:
        if tensor.ndim != 3:
            raise ValueError(f"{self.backend_label} input embeddings must be [B, T, H]")

    @staticmethod
    def _validate_right_padded_mask(mask: torch.Tensor, lengths: torch.Tensor) -> None:
        positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
        expected = positions < lengths.unsqueeze(1)
        if not torch.equal(mask, expected):
            raise ValueError("vLLM paged attention requires right-padded prompt masks")

    @staticmethod
    def _typed_states(
        states: SequenceABC[Any],
    ) -> tuple[VllmDecoderPagedSequenceState, ...]:
        typed = tuple(states)
        if not all(isinstance(state, VllmDecoderPagedSequenceState) for state in typed):
            got = sorted({type(state).__name__ for state in typed})
            raise TypeError(
                "vLLM paged attention received incompatible sequence states: "
                f"{got}",
            )
        return typed  # type: ignore[return-value]

    @staticmethod
    def _max_new_tokens_from_metadata(metadata: Any) -> int:
        if isinstance(metadata, dict) and "image_token_num" in metadata:
            return max(1, int(metadata["image_token_num"]))
        return 1

    def _num_attention_heads(self, attention: Any) -> int:
        return int(getattr(attention, "num_heads", self.trunk.config.num_attention_heads))

    def _num_key_value_heads(self, attention: Any) -> int:
        return int(
            getattr(
                attention,
                "num_key_value_heads",
                self.trunk.config.num_key_value_heads,
            )
        )

    def _head_dim(self, attention: Any, num_heads: int) -> int:
        return int(
            getattr(
                attention,
                "head_dim",
                self.trunk.config.hidden_size // num_heads,
            )
        )

    @staticmethod
    def _sliding_window_for_layer(layer: Any) -> int | None:
        if getattr(layer, "attention_type", None) != "sliding_attention":
            return None
        sliding_window = getattr(layer.self_attn, "sliding_window", None)
        if sliding_window is None:
            sliding_window = getattr(layer.self_attn.config, "sliding_window", None)
        return None if sliding_window is None else int(sliding_window)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    unsqueeze_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (query * cos) + (_rotate_half(query) * sin), (key * cos) + (
        _rotate_half(key) * sin
    )


__all__ = [
    "VllmDecoderPagedAttentionBackend",
    "VllmDecoderPagedSequenceState",
    "VllmPagedAttentionKernels",
]
