"""Janus-Pro AR model runner executed by the AR engine."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from vrl.generation.ar.decode_loop import (
    ARStepBatch,
    ARStepOutput,
    ARStepResult,
    ARTokenLoopInit,
    ar_concat_rows,
    ar_split_rows,
)
from vrl.generation.ar.paged_attention import (
    ARPagedAttentionBackend,
    ARPagedAttentionConfig,
    ARPagedAttentionPrefillInput,
    ARPagedAttentionPrefillOutput,
    ARPagedAttentionStepInput,
    ARPagedAttentionStepOutput,
    ARPagedAttentionUnavailable,
)
from vrl.generation.ar.vllm_paged_attention import (
    VllmDecoderPagedAttentionBackend,
    VllmPagedAttentionKernels,
)
from vrl.models.ar.janus_pro.model import image_token_logits_from_hidden


@dataclass(slots=True)
class JanusProARState:
    """Mutable Janus state owned by one scheduled AR token loop."""

    token_ids: torch.Tensor
    logprobs: torch.Tensor
    cfg_weight: float
    temperature: float
    image_token_num: int
    paged_cond_states: list[Any] | None = None
    paged_uncond_states: list[Any] | None = None
    prefill_forwards: int = 0
    decode_forwards: int = 0
    decode_tokens: int = 0


@dataclass(frozen=True, slots=True)
class JanusVllmPagedSequenceState:
    """Per-sequence physical vLLM KV page ownership for Janus AR."""

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
    states: tuple[JanusVllmPagedSequenceState, ...]


@dataclass(frozen=True, slots=True)
class _PackedStep:
    inputs_embeds: torch.Tensor
    positions: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    states: tuple[JanusVllmPagedSequenceState, ...]


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


class JanusVllmPagedAttentionBackend(ARPagedAttentionBackend):
    """Run Janus Llama trunk attention with vLLM block tables and paged KV cache."""

    def __init__(
        self,
        model: Any,
        *,
        config: ARPagedAttentionConfig | None = None,
        kernels: VllmPagedAttentionKernels | None = None,
    ) -> None:
        config = config or ARPagedAttentionConfig(
            family="janus_pro",
            model_key=str(getattr(model.config, "model_path", "janus_pro")),
            block_size=16,
        )
        super().__init__(config)
        self.model = model
        self.trunk = model._lm_trunk()
        self.kernels = kernels or VllmPagedAttentionKernels(config)
        self.cache_dtype = str(config.extra.get("cache_dtype", "auto"))
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
                "backend": "janus_vllm_paged_attention",
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
                "backend": "janus_vllm_paged_attention",
                "decode_tokens": int(packed.inputs_embeds.shape[0]),
                "allocated_blocks": self._next_block_id,
            },
        )

    def debug_info(self) -> dict[str, Any]:
        return {
            **dict(super().debug_info()),
            "backend": "janus_vllm_paged_attention",
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
            raise ValueError("Janus vLLM paged prefill requires non-empty prompts")
        self._validate_right_padded_mask(mask_bool, lengths)

        packed_embeds: list[torch.Tensor] = []
        packed_positions: list[torch.Tensor] = []
        query_starts = [0]
        last_offsets: list[int] = []
        states: list[JanusVllmPagedSequenceState] = []
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
                JanusVllmPagedSequenceState(
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
            raise ValueError("Janus vLLM paged step expects one token per sequence")
        states = self._typed_states(request.sequence_states)
        if len(states) != embeds.shape[0]:
            raise ValueError("Janus vLLM step state count must match batch size")
        for state in states:
            if state.length >= len(state.block_ids) * self.config.block_size:
                raise RuntimeError(
                    "Janus vLLM paged attention sequence exhausted its allocated "
                    f"KV blocks: sequence_id={state.sequence_id!r}",
                )

        next_states = tuple(
            JanusVllmPagedSequenceState(
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
        states: Sequence[JanusVllmPagedSequenceState],
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
        query_states, key_states = apply_rotary_pos_emb(
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
        states: Sequence[JanusVllmPagedSequenceState],
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
                "Janus vLLM paged attention requires CUDA tensors because "
                "vLLM BlockTable slot mapping and FlashAttention are CUDA kernels.",
            )
        if dtype not in (torch.float16, torch.bfloat16):
            raise ARPagedAttentionUnavailable(
                "Janus vLLM paged attention requires float16 or bfloat16 model "
                f"embeddings, got {dtype}.",
            )
        layers = list(self.trunk.layers)
        if not layers:
            raise RuntimeError("Janus Llama trunk has no decoder layers")
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
            raise ValueError("Janus vLLM paged input embeddings must be [B, T, H]")

    @staticmethod
    def _validate_right_padded_mask(mask: torch.Tensor, lengths: torch.Tensor) -> None:
        positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
        expected = positions < lengths.unsqueeze(1)
        if not torch.equal(mask, expected):
            raise ValueError(
                "Janus vLLM paged attention requires right-padded prompt masks",
            )

    @staticmethod
    def _typed_states(states: Sequence[Any]) -> tuple[JanusVllmPagedSequenceState, ...]:
        typed = tuple(states)
        if not all(isinstance(state, JanusVllmPagedSequenceState) for state in typed):
            got = sorted({type(state).__name__ for state in typed})
            raise TypeError(
                "Janus vLLM paged attention received incompatible sequence states: "
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


def build_janus_vllm_paged_attention_backend(
    model: Any,
    *,
    block_size: int = 16,
    cache_dtype: str = "auto",
) -> VllmDecoderPagedAttentionBackend:
    """Construct the explicit Janus backend that borrows vLLM paged attention."""

    config = ARPagedAttentionConfig(
        family="janus_pro",
        model_key=str(getattr(model.config, "model_path", "janus_pro")),
        block_size=block_size,
        dtype=str(getattr(model, "dtype", "")) or None,
        device=str(getattr(model, "device", "")) or None,
        extra={
            "cache_dtype": cache_dtype,
            "backend_label": "janus_vllm_paged_attention",
        },
    )
    return VllmDecoderPagedAttentionBackend(
        trunk=model._lm_trunk(),
        config=config,
    )


class JanusProARModelRunner:
    """Family model runner that lets the AR engine schedule Janus token steps."""

    def __init__(
        self,
        model: Any,
        *,
        paged_attention_backend: ARPagedAttentionBackend | None = None,
    ) -> None:
        self.model = model
        self.paged_attention_backend = paged_attention_backend

    @torch.no_grad()
    def init_ar(
        self,
        cond_inputs_embeds: torch.Tensor,
        uncond_inputs_embeds: torch.Tensor,
        cond_attention_mask: torch.Tensor,
        uncond_attention_mask: torch.Tensor,
        *,
        cfg_weight: float | None = None,
        temperature: float | None = None,
        image_token_num: int | None = None,
    ) -> ARTokenLoopInit:
        cfg = cfg_weight if cfg_weight is not None else self.model.config.cfg_weight
        temp = temperature if temperature is not None else self.model.config.temperature
        image_token_num = image_token_num or self.model.config.image_token_num
        batch_size = cond_inputs_embeds.shape[0]
        device = cond_inputs_embeds.device
        if self.paged_attention_backend is None:
            cond_past, cond_last_hidden = self._prefill_ar_prompt(
                cond_inputs_embeds,
                cond_attention_mask,
            )
            uncond_past, uncond_last_hidden = self._prefill_ar_prompt(
                uncond_inputs_embeds,
                uncond_attention_mask,
            )
            cache_lanes = {
                "cond_past": cond_past,
                "uncond_past": uncond_past,
            }
            cache_lane_owners = {
                "cond_past": "janus.cond_past",
                "uncond_past": "janus.uncond_past",
            }
            paged_cond_states = None
            paged_uncond_states = None
        else:
            cond_prefill = self._prefill_ar_prompt_paged(
                cond_inputs_embeds,
                cond_attention_mask,
                branch="cond",
                image_token_num=int(image_token_num),
            )
            uncond_prefill = self._prefill_ar_prompt_paged(
                uncond_inputs_embeds,
                uncond_attention_mask,
                branch="uncond",
                image_token_num=int(image_token_num),
            )
            cond_last_hidden = cond_prefill.last_hidden
            uncond_last_hidden = uncond_prefill.last_hidden
            cache_lanes = {}
            cache_lane_owners = {}
            paged_cond_states = list(cond_prefill.sequence_states)
            paged_uncond_states = list(uncond_prefill.sequence_states)

        return ARTokenLoopInit(
            state=JanusProARState(
                token_ids=torch.empty(
                    batch_size, image_token_num, dtype=torch.long, device=device
                ),
                logprobs=torch.empty(
                    batch_size, image_token_num, dtype=torch.float32, device=device
                ),
                cfg_weight=float(cfg),
                temperature=float(temp),
                image_token_num=int(image_token_num),
                paged_cond_states=paged_cond_states,
                paged_uncond_states=paged_uncond_states,
                prefill_forwards=2,
            ),
            cache_lanes=cache_lanes,
            row_lanes={
                "cond_last_hidden": cond_last_hidden,
                "uncond_last_hidden": uncond_last_hidden,
                "cond_attn": cond_attention_mask,
                "uncond_attn": uncond_attention_mask,
            },
            cache_lane_owners=cache_lane_owners,
            row_lane_owners={
                "cond_last_hidden": "janus.cond_last_hidden",
                "uncond_last_hidden": "janus.uncond_last_hidden",
                "cond_attn": "janus.cond_attn",
                "uncond_attn": "janus.uncond_attn",
            },
        )

    @torch.no_grad()
    def step_ar(
        self,
        state: JanusProARState,
        batch: ARStepBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> ARStepOutput:
        del generator
        token, log_prob, cache_updates, row_updates = self._sample_ar_step(state, batch)
        return ARStepOutput(
            result=ARStepResult(
                sequence_ids=batch.sequence_ids,
                positions=batch.positions,
                token=token,
                log_prob=log_prob,
                debug_counters={
                    "ar_kv_cache_enabled": True,
                    "ar_paged_attention_enabled": state.paged_cond_states is not None,
                    "ar_prefill_forwards": state.prefill_forwards,
                    "ar_decode_forwards": state.decode_forwards,
                    "ar_decode_tokens": state.decode_tokens,
                },
            ),
            updated_cache_lanes=cache_updates,
            updated_row_lanes=row_updates,
        )

    @torch.no_grad()
    def finalize_ar(self, state: JanusProARState) -> tuple[torch.Tensor, torch.Tensor]:
        return state.token_ids, state.logprobs

    def _prefill_ar_prompt(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[Any, torch.Tensor]:
        outputs = self.model._lm_trunk()(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
        )
        past = getattr(outputs, "past_key_values", None)
        last_hidden = self.model._last_token_hidden(outputs)
        return past, last_hidden

    def _prefill_ar_prompt_paged(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        branch: str,
        image_token_num: int,
    ) -> Any:
        return self._require_paged_attention_backend().prefill(
            ARPagedAttentionPrefillInput(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                branch=branch,
                metadata={"family": "janus_pro", "image_token_num": image_token_num},
            )
        )

    def _sample_ar_step(
        self,
        state: JanusProARState,
        batch: ARStepBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], dict[str, Any]]:
        self._validate_ar_step_batch(state, batch)
        return self._sample_ar_step_kv(state, batch)

    @staticmethod
    def _validate_ar_step_batch(
        state: JanusProARState,
        batch: ARStepBatch,
    ) -> None:
        row_indices = batch.row_indices
        if not row_indices:
            raise ValueError("row_indices must be non-empty")
        if any(row < 0 or row >= state.token_ids.shape[0] for row in row_indices):
            raise ValueError(f"invalid Janus row indices: {row_indices}")
        if len(set(batch.positions)) != 1:
            raise ValueError("ActiveSequence positions must match within one AR step")
        if batch.position >= state.image_token_num:
            raise ValueError("JanusProARState has already finished sampling")

    def _sample_ar_step_kv(
        self,
        state: JanusProARState,
        batch: ARStepBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], dict[str, Any]]:
        row_indices = batch.row_indices
        position = batch.position
        batch_size = len(row_indices)
        rows = torch.tensor(row_indices, dtype=torch.long, device=state.token_ids.device)
        cond_hidden = batch.row_lanes["cond_last_hidden"]
        uncond_hidden = batch.row_lanes["uncond_last_hidden"]
        hidden = torch.cat([cond_hidden, uncond_hidden], dim=0).unsqueeze(1)
        sampled, lp = self._sample_cfg_image_token(state, hidden)
        state.token_ids[rows, position] = sampled
        state.logprobs[rows, position] = lp

        cache_updates: dict[str, Any] = {}
        row_updates: dict[str, Any] = {}
        if position + 1 < state.image_token_num:
            cache_updates, row_updates = self._advance_kv_cache_after_sample(
                state,
                batch=batch,
                sampled=sampled,
            )

        state.decode_tokens += batch_size
        return sampled, lp, cache_updates, row_updates

    def _sample_cfg_image_token(
        self,
        state: JanusProARState,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = image_token_logits_from_hidden(self.model.mmgpt, hidden).squeeze(1)
        cond_logits, uncond_logits = logits.chunk(2, dim=0)
        cond_logits = cond_logits.float()
        uncond_logits = uncond_logits.float()
        guided = uncond_logits + state.cfg_weight * (cond_logits - uncond_logits)

        probs = F.softmax(guided / state.temperature, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)

        log_probs = F.log_softmax(cond_logits / state.temperature, dim=-1)
        lp = log_probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        return sampled, lp

    def _advance_kv_cache_after_sample(
        self,
        state: JanusProARState,
        *,
        batch: ARStepBatch,
        sampled: torch.Tensor,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.paged_attention_backend is not None:
            return self._advance_paged_attention_after_sample(
                state,
                batch=batch,
                sampled=sampled,
            )

        batch_size = len(batch.row_indices)
        token_embed = self.model._base().prepare_gen_img_embeds(sampled.unsqueeze(-1))
        inputs_embeds = torch.cat([token_embed, token_embed], dim=0)

        cond_attn = batch.row_lanes["cond_attn"]
        uncond_attn = batch.row_lanes["uncond_attn"]
        cond_next_attn = torch.cat(
            [
                cond_attn,
                torch.ones(
                    batch_size,
                    1,
                    dtype=cond_attn.dtype,
                    device=cond_attn.device,
                ),
            ],
            dim=1,
        )
        uncond_next_attn = torch.cat(
            [
                uncond_attn,
                torch.ones(
                    batch_size,
                    1,
                    dtype=uncond_attn.dtype,
                    device=uncond_attn.device,
                ),
            ],
            dim=1,
        )

        past = ar_concat_rows(
            [batch.cache_lanes["cond_past"], batch.cache_lanes["uncond_past"]]
        )
        outputs = self.model._lm_trunk()(
            inputs_embeds=inputs_embeds,
            attention_mask=torch.cat([cond_next_attn, uncond_next_attn], dim=0),
            past_key_values=past,
            use_cache=True,
        )

        updated_past_rows = ar_split_rows(
            getattr(outputs, "past_key_values", None),
            2 * batch_size,
        )
        updated_hidden_rows = ar_split_rows(
            self.model._last_token_hidden(outputs),
            2 * batch_size,
        )
        state.decode_forwards += 1
        return (
            {
                "cond_past": ar_concat_rows(updated_past_rows[:batch_size]),
                "uncond_past": ar_concat_rows(updated_past_rows[batch_size:]),
            },
            {
                "cond_last_hidden": ar_concat_rows(updated_hidden_rows[:batch_size]),
                "uncond_last_hidden": ar_concat_rows(updated_hidden_rows[batch_size:]),
                "cond_attn": cond_next_attn,
                "uncond_attn": uncond_next_attn,
            },
        )

    def _advance_paged_attention_after_sample(
        self,
        state: JanusProARState,
        *,
        batch: ARStepBatch,
        sampled: torch.Tensor,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        batch_size = len(batch.row_indices)
        cond_states = self._select_paged_states(state.paged_cond_states, batch.row_indices)
        uncond_states = self._select_paged_states(
            state.paged_uncond_states,
            batch.row_indices,
        )
        token_embed = self.model._base().prepare_gen_img_embeds(sampled.unsqueeze(-1))
        inputs_embeds = torch.cat([token_embed, token_embed], dim=0)

        cond_next_attn = self._append_attention_token(batch.row_lanes["cond_attn"])
        uncond_next_attn = self._append_attention_token(batch.row_lanes["uncond_attn"])
        output = self._require_paged_attention_backend().step(
            ARPagedAttentionStepInput(
                input_embeds=inputs_embeds,
                attention_mask=torch.cat([cond_next_attn, uncond_next_attn], dim=0),
                sequence_states=tuple(cond_states + uncond_states),
                branch_names=tuple(["cond"] * batch_size + ["uncond"] * batch_size),
                position=batch.position,
                row_indices=tuple(batch.row_indices + batch.row_indices),
                metadata={"family": "janus_pro"},
            )
        )

        updated_states = list(output.sequence_states)
        if len(updated_states) != 2 * batch_size:
            raise ValueError(
                "paged attention step returned "
                f"{len(updated_states)} states for batch={batch_size}",
            )
        self._scatter_paged_states(
            state.paged_cond_states,
            batch.row_indices,
            updated_states[:batch_size],
        )
        self._scatter_paged_states(
            state.paged_uncond_states,
            batch.row_indices,
            updated_states[batch_size:],
        )
        hidden = self._normalize_paged_last_hidden(output.last_hidden)
        state.decode_forwards += 1
        return (
            {},
            {
                "cond_last_hidden": hidden[:batch_size],
                "uncond_last_hidden": hidden[batch_size:],
                "cond_attn": cond_next_attn,
                "uncond_attn": uncond_next_attn,
            },
        )

    def _require_paged_attention_backend(self) -> ARPagedAttentionBackend:
        if self.paged_attention_backend is None:
            raise RuntimeError("Janus paged-attention path requires a backend")
        return self.paged_attention_backend

    @staticmethod
    def _append_attention_token(attention_mask: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                attention_mask,
                torch.ones(
                    attention_mask.shape[0],
                    1,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ],
            dim=1,
        )

    @staticmethod
    def _select_paged_states(
        states: list[Any] | None,
        row_indices: list[int],
    ) -> list[Any]:
        if states is None:
            raise RuntimeError("paged attention state is not initialized")
        return [states[index] for index in row_indices]

    @staticmethod
    def _scatter_paged_states(
        states: list[Any] | None,
        row_indices: list[int],
        values: list[Any],
    ) -> None:
        if states is None:
            raise RuntimeError("paged attention state is not initialized")
        if len(row_indices) != len(values):
            raise ValueError("paged attention state updates must match row indices")
        for row_index, value in zip(row_indices, values, strict=True):
            states[row_index] = value

    @staticmethod
    def _normalize_paged_last_hidden(last_hidden: torch.Tensor) -> torch.Tensor:
        if last_hidden.ndim == 3:
            if last_hidden.shape[1] != 1:
                raise ValueError("paged attention last_hidden must be [B, H] or [B, 1, H]")
            return last_hidden[:, 0, :]
        if last_hidden.ndim != 2:
            raise ValueError("paged attention last_hidden must be [B, H] or [B, 1, H]")
        return last_hidden

__all__ = [
    "JanusProARModelRunner",
    "JanusProARState",
    "build_janus_vllm_paged_attention_backend",
]
