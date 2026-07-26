"""AR decoder module backed by reusable NN paged-attention layers."""

from __future__ import annotations

import math
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from typing import Any

import torch

from vrl.nn.kernels.attention.vllm_paged import VllmPagedAttentionKernels
from vrl.nn.layers.attention.paged import (
    ARAttentionBackend,
    ARAttentionPrefillInput,
    ARAttentionPrefillOutput,
    ARAttentionStepInput,
    ARAttentionStepOutput,
    ARAttentionUnavailable,
    VllmPagedAttentionConfig,
)


@dataclass(frozen=True, slots=True)
class VllmDecoderPagedSequenceState:
    """Per-sequence physical vLLM KV page ownership for decoder-only AR."""

    sequence_id: str
    length: int
    next_position_id: int
    block_ids: tuple[int, ...]


class _VllmAttentionScaleShim(torch.nn.Module):
    """Small shim exposing the quant-scale attrs required by vLLM attention."""

    def __init__(self, *, device: torch.device) -> None:
        super().__init__()
        scale = torch.ones(1, device=device, dtype=torch.float32)
        self.register_buffer("_q_scale", scale, persistent=False)
        self.register_buffer("_k_scale", scale.clone(), persistent=False)
        self.register_buffer("_v_scale", scale.clone(), persistent=False)


class VllmDecoderPagedAttentionBackend(ARAttentionBackend):
    """Run a HF-style decoder trunk with vLLM block tables and paged KV cache."""

    def __init__(
        self,
        *,
        trunk: Any,
        config: VllmPagedAttentionConfig,
        kernels: VllmPagedAttentionKernels | None = None,
    ) -> None:
        super().__init__(config)
        self.trunk = trunk
        self.kernels = kernels or VllmPagedAttentionKernels(config)
        self.cache_dtype = config.cache_dtype
        self.backend_label = f"{config.family}_vllm_paged_attention"
        self._next_sequence_id = 0
        self._next_block_id = 0
        self._kv_caches: list[torch.Tensor] = []
        self._attention_impls: list[Any] = []
        self._scale_shim: _VllmAttentionScaleShim | None = None
        self._kv_cache_num_blocks = 0

    @torch.no_grad()
    def prefill(
        self,
        request: ARAttentionPrefillInput,
    ) -> ARAttentionPrefillOutput:
        (
            inputs_embeds,
            cache_positions,
            position_ids,
            query_start_loc,
            seq_lens,
            last_token_offsets,
            states,
        ) = self._pack_prefill(request)
        last_hidden_states = self._forward_paged_trunk(
            inputs_embeds,
            cache_positions=cache_positions,
            position_ids=position_ids,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            states=states,
        )
        return ARAttentionPrefillOutput(
            last_hidden=last_hidden_states.index_select(0, last_token_offsets),
            sequence_states=states,
        )

    @torch.no_grad()
    def step(self, request: ARAttentionStepInput) -> ARAttentionStepOutput:
        (
            inputs_embeds,
            cache_positions,
            position_ids,
            query_start_loc,
            seq_lens,
            states,
        ) = self._pack_step(request)
        last_hidden_states = self._forward_paged_trunk(
            inputs_embeds,
            cache_positions=cache_positions,
            position_ids=position_ids,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            states=states,
        )
        return ARAttentionStepOutput(
            last_hidden=last_hidden_states,
            sequence_states=states,
        )

    def _pack_prefill(
        self,
        request: ARAttentionPrefillInput,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[VllmDecoderPagedSequenceState, ...],
    ]:
        embeds = request.inputs_embeds
        mask = request.attention_mask
        self._validate_runtime_tensor(embeds)
        mask_bool = mask.to(dtype=torch.bool)
        lengths = mask_bool.sum(dim=1).to(dtype=torch.long)
        if bool((lengths <= 0).any()):
            raise ValueError(f"{self.backend_label} prefill requires non-empty prompts")

        packed_embeds: list[torch.Tensor] = []
        packed_cache_positions: list[torch.Tensor] = []
        packed_position_ids: list[torch.Tensor] = []
        query_starts = [0]
        last_offsets: list[int] = []
        states: list[VllmDecoderPagedSequenceState] = []
        sequence_ids = tuple(
            f"{request.branch}:{row}:{self._next_sequence_id + row}"
            for row in range(embeds.shape[0])
        )
        for row, length_tensor in enumerate(lengths):
            length = int(length_tensor.item())
            start, end = self._contiguous_valid_token_span(mask_bool[row], length)
            packed_embeds.append(embeds[row, start:end])
            packed_cache_positions.append(
                torch.arange(length, dtype=torch.long, device=embeds.device)
            )
            packed_position_ids.append(
                torch.arange(start, end, dtype=torch.long, device=embeds.device)
            )
            query_starts.append(query_starts[-1] + length)
            last_offsets.append(query_starts[-1] - 1)
            states.append(
                VllmDecoderPagedSequenceState(
                    sequence_id=sequence_ids[row],
                    length=length,
                    next_position_id=end,
                    block_ids=self._allocate_blocks(length + request.max_new_tokens),
                )
            )
        self._next_sequence_id += embeds.shape[0]
        return (
            torch.cat(packed_embeds, dim=0),
            torch.cat(packed_cache_positions, dim=0),
            torch.cat(packed_position_ids, dim=0),
            torch.tensor(
                query_starts,
                dtype=torch.int32,
                device=embeds.device,
            ),
            lengths.to(device=embeds.device, dtype=torch.int32),
            torch.tensor(
                last_offsets,
                dtype=torch.long,
                device=embeds.device,
            ),
            tuple(states),
        )

    def _pack_step(
        self,
        request: ARAttentionStepInput,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[VllmDecoderPagedSequenceState, ...],
    ]:
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
                length=state.length + 1,
                next_position_id=state.next_position_id + 1,
                block_ids=state.block_ids,
            )
            for state in states
        )
        cache_positions = torch.tensor(
            [state.length for state in states],
            dtype=torch.long,
            device=embeds.device,
        )
        position_ids = torch.tensor(
            [state.next_position_id for state in states],
            dtype=torch.long,
            device=embeds.device,
        )
        batch = embeds.shape[0]
        return (
            embeds[:, 0, :],
            cache_positions,
            position_ids,
            torch.arange(
                batch + 1,
                dtype=torch.int32,
                device=embeds.device,
            ),
            torch.tensor(
                [state.length for state in next_states],
                dtype=torch.int32,
                device=embeds.device,
            ),
            next_states,
        )

    def _forward_paged_trunk(
        self,
        inputs_embeds: torch.Tensor,
        *,
        cache_positions: torch.Tensor,
        position_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        states: SequenceABC[VllmDecoderPagedSequenceState],
    ) -> torch.Tensor:
        self._ensure_runtime_objects(
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        (
            block_table,
            slot_mapping,
            num_actual_tokens,
            max_query_len,
            max_seq_len,
        ) = self._build_paged_forward_inputs(
            cache_positions=cache_positions,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            states=states,
        )

        hidden_states = inputs_embeds
        position_ids = position_ids.unsqueeze(0)
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
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                block_table=block_table,
                slot_mapping=slot_mapping,
                num_actual_tokens=num_actual_tokens,
                max_query_len=max_query_len,
                max_seq_len=max_seq_len,
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
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
        num_actual_tokens: int,
        max_query_len: int,
        max_seq_len: int,
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
            slot_mapping=slot_mapping,
        )
        metadata = self.kernels.make_flash_attention_metadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table,
            slot_mapping=slot_mapping,
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

    def _build_paged_forward_inputs(
        self,
        *,
        cache_positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        states: SequenceABC[VllmDecoderPagedSequenceState],
    ) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
        max_blocks = max(len(state.block_ids) for state in states)
        block_table = self.kernels.new_block_table(
            max_num_reqs=len(states),
            max_num_blocks_per_req=max_blocks,
            max_num_batched_tokens=int(cache_positions.shape[0]),
            device=cache_positions.device,
        )
        for row_idx, state in enumerate(states):
            block_table.add_row(list(state.block_ids), row_idx=row_idx)
        block_table.commit_block_table(len(states))
        slot_mapping = self.kernels.compute_slot_mapping(
            block_table=block_table,
            num_reqs=len(states),
            query_start_loc=query_start_loc,
            positions=cache_positions,
        )
        return (
            block_table.get_device_tensor(len(states))[:, :max_blocks],
            slot_mapping,
            int(cache_positions.shape[0]),
            int((query_start_loc[1:] - query_start_loc[:-1]).max().item()),
            int(seq_lens.max().item()),
        )

    def _ensure_runtime_objects(self, *, device: torch.device, dtype: torch.dtype) -> None:
        if device.type != "cuda":
            raise ARAttentionUnavailable(
                f"{self.backend_label} requires CUDA tensors because vLLM "
                "BlockTable slot mapping and FlashAttention are CUDA kernels.",
            )
        if dtype not in (torch.float16, torch.bfloat16):
            raise ARAttentionUnavailable(
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

    def _contiguous_valid_token_span(
        self,
        mask: torch.Tensor,
        length: int,
    ) -> tuple[int, int]:
        valid_positions = torch.nonzero(mask, as_tuple=False).flatten()
        if int(valid_positions.numel()) != int(length):
            raise ValueError(f"{self.backend_label} prompt mask length mismatch")
        start = int(valid_positions[0].item())
        end = int(valid_positions[-1].item()) + 1
        if end - start != int(length):
            raise ValueError(
                f"{self.backend_label} requires prompt masks with one contiguous valid-token span",
            )
        return start, end

    @staticmethod
    def _typed_states(
        states: SequenceABC[Any],
    ) -> tuple[VllmDecoderPagedSequenceState, ...]:
        typed = tuple(states)
        if not all(isinstance(state, VllmDecoderPagedSequenceState) for state in typed):
            got = sorted({type(state).__name__ for state in typed})
            raise TypeError(
                f"vLLM paged attention received incompatible sequence states: {got}",
            )
        return typed  # type: ignore[return-value]

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
    return (query * cos) + (_rotate_half(query) * sin), (key * cos) + (_rotate_half(key) * sin)


__all__ = [
    "VllmDecoderPagedAttentionBackend",
    "VllmDecoderPagedSequenceState",
]
