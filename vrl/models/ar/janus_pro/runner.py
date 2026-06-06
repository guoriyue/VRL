"""Janus-Pro AR model runner executed by the AR engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from vrl.generation.ar.decode_loop import (
    ARStepBatch,
    ARStepOutput,
    ARStepResult,
    ARTokenLoopInit,
    ar_concat_rows,
    ar_split_rows,
)
from vrl.models.ar.janus_pro.model import image_token_logits_from_hidden
from vrl.models.ar.paged_attention_helpers import (
    append_attention_token,
    normalize_paged_last_hidden,
    require_attention_backend,
    scatter_paged_states,
    select_paged_states,
)
from vrl.nn.layers.attention.paged import (
    ARAttentionBackend,
    ARAttentionConfig,
    ARAttentionPrefillInput,
    ARAttentionStepInput,
)
from vrl.nn.modules.ar_decoder import (
    VllmDecoderPagedAttentionBackend,
)


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


def build_janus_vllm_attention_backend(
    model: Any,
    *,
    block_size: int = 16,
    cache_dtype: str = "auto",
) -> VllmDecoderPagedAttentionBackend:
    """Construct the explicit Janus backend that borrows vLLM paged attention."""

    config = ARAttentionConfig(
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
        attention_backend: ARAttentionBackend | None = None,
    ) -> None:
        self.model = model
        self.attention_backend = attention_backend

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
        if self.attention_backend is None:
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
        return require_attention_backend(
            self.attention_backend, family="Janus"
        ).prefill(
            ARAttentionPrefillInput(
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
        if self.attention_backend is not None:
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
        cond_next_attn = append_attention_token(cond_attn)
        uncond_next_attn = append_attention_token(uncond_attn)

        past = ar_concat_rows([batch.cache_lanes["cond_past"], batch.cache_lanes["uncond_past"]])
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
        cond_states = select_paged_states(state.paged_cond_states, batch.row_indices)
        uncond_states = select_paged_states(
            state.paged_uncond_states,
            batch.row_indices,
        )
        token_embed = self.model._base().prepare_gen_img_embeds(sampled.unsqueeze(-1))
        inputs_embeds = torch.cat([token_embed, token_embed], dim=0)

        cond_next_attn = append_attention_token(batch.row_lanes["cond_attn"])
        uncond_next_attn = append_attention_token(batch.row_lanes["uncond_attn"])
        output = require_attention_backend(
            self.attention_backend, family="Janus"
        ).step(
            ARAttentionStepInput(
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
        scatter_paged_states(
            state.paged_cond_states,
            batch.row_indices,
            updated_states[:batch_size],
        )
        scatter_paged_states(
            state.paged_uncond_states,
            batch.row_indices,
            updated_states[batch_size:],
        )
        hidden = normalize_paged_last_hidden(output.last_hidden)
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

__all__ = [
    "JanusProARModelRunner",
    "JanusProARState",
    "build_janus_vllm_attention_backend",
]
