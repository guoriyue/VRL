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
)
from vrl.models.ar.janus_pro.model import image_token_logits_from_hidden
from vrl.models.ar.paged_attention_helpers import (
    append_attention_token,
    normalize_paged_last_hidden,
    scatter_paged_states,
    select_paged_states,
)
from vrl.nn.layers.attention.paged import (
    ARAttentionBackend,
    ARAttentionPrefillInput,
    ARAttentionStepInput,
)


@dataclass(slots=True)
class JanusProARState:
    """Mutable Janus state owned by one scheduled AR token loop."""

    token_ids: torch.Tensor
    logprobs: torch.Tensor
    guidance_scale: float
    temperature: float
    image_token_num: int
    paged_cond_states: list[Any] | None = None
    paged_uncond_states: list[Any] | None = None
    prefill_forwards: int = 0
    decode_forwards: int = 0
    decode_tokens: int = 0


class JanusProARModelRunner:
    """Family model runner that lets the AR engine schedule Janus token steps."""

    def __init__(
        self,
        model: Any,
        *,
        attention_backend: ARAttentionBackend,
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
        guidance_scale: float | None = None,
        temperature: float | None = None,
        image_token_num: int | None = None,
    ) -> ARTokenLoopInit:
        cfg = guidance_scale if guidance_scale is not None else self.model.config.guidance_scale
        temp = temperature if temperature is not None else self.model.config.temperature
        image_token_num = image_token_num or self.model.config.image_token_num
        batch_size = cond_inputs_embeds.shape[0]
        device = cond_inputs_embeds.device
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
        cache_lanes: dict[str, Any] = {}
        cache_lane_owners: dict[str, str] = {}
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
                guidance_scale=float(cfg),
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
        cache_updates, row_updates = self._sample_ar_step(state, batch)
        return ARStepOutput(
            result=ARStepResult(
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

    def _prefill_ar_prompt_paged(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        branch: str,
        image_token_num: int,
    ) -> Any:
        return self.attention_backend.prefill(
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
    ) -> tuple[dict[str, Any], dict[str, Any]]:
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
    ) -> tuple[dict[str, Any], dict[str, Any]]:
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
            cache_updates, row_updates = self._advance_after_sample(
                state,
                batch=batch,
                sampled=sampled,
            )

        state.decode_tokens += batch_size
        return cache_updates, row_updates

    def _sample_cfg_image_token(
        self,
        state: JanusProARState,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = image_token_logits_from_hidden(self.model.mmgpt, hidden).squeeze(1)
        cond_logits, uncond_logits = logits.chunk(2, dim=0)
        cond_logits = cond_logits.float()
        uncond_logits = uncond_logits.float()
        guided = uncond_logits + state.guidance_scale * (cond_logits - uncond_logits)

        probs = F.softmax(guided / state.temperature, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # RL-correctness contract — do NOT "align to upstream" by scoring `guided`.
        # We SAMPLE from the CFG-`guided` distribution but score the log-prob under
        # `cond_logits`: the conditional model is the policy GRPO optimizes, and CFG
        # is only the sampling/importance distribution. Scoring `guided` here would
        # make lp the behavior policy and silently break the old_log_prob invariant
        # (train/infer logprob parity). Upstream Janus' inference script computes no
        # log-prob, so there is no upstream line to copy. Locked by
        # tests/models/ar/janus_pro/test_upstream_reconcile_contracts.py.
        log_probs = F.log_softmax(cond_logits / state.temperature, dim=-1)
        lp = log_probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        return sampled, lp

    def _advance_after_sample(
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
        output = self.attention_backend.step(
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
]
