"""GLM-Image AR model runner executed by the AR engine.

Follows the Emu3 token-loop pattern with two GLM-specific deviations:

  * **Single-sequence sampling (no AR CFG).** The reference generation is a
    plain ``vision_language_encoder.generate(..., do_sample=True)`` with
    temperature/top_p from the checkpoint generation_config — no guidance
    logits processor (diffusers pipeline_glm_image.py lines 439-443). There
    is no uncond branch to prefill; CFG exists only inside the frozen DiT
    decode segment.

  * **Native trunk decode with explicit 3-axis mrope positions.** Generated
    image tokens carry 2D spatial position ids (``glm_image_decode_position_
    schedule``); the shared attention backends cannot drive this — the
    ``vllm_paged`` backend hand-walks LLaMA-style blocks (2 layernorms, 1D
    rope; GLM's block has post_self_attn/post_mlp layernorms and a 3-axis
    mrope), and ``torch_native`` cannot inject position ids. Like the
    LlamaGen runner, this runner drives the HF trunk + ``DynamicCache``
    directly; the family executor overrides ``_ar_runner`` accordingly.
"""

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
from vrl.math.ar.logprob import top_k_top_p_filtering
from vrl.models.ar.glm_image.model import (
    glm_image_decode_position_schedule,
    glm_image_prefill_position_ids,
)
from vrl.models.ar.paged_attention_helpers import (
    append_attention_token,
    normalize_paged_last_hidden,
    scatter_paged_states,
    select_paged_states,
)
from vrl.nn.layers.attention.cache_rows import ar_concat_rows, ar_split_rows


@dataclass(slots=True)
class GlmImageARState:
    """Mutable GLM-Image state owned by one scheduled AR token loop."""

    token_ids: torch.Tensor
    logprobs: torch.Tensor
    temperature: float
    top_p: float
    total_token_num: int
    # [3, L_decode] mrope positions for generated token j fed as input,
    # relative to prompt_len == 0 (per-row valid lengths added at use time).
    base_position_schedule: torch.Tensor
    # [B] per-row valid prompt token count (left-padded prompts).
    prompt_valid_lens: torch.Tensor
    kv_rows: list[Any] | None = None
    prefill_forwards: int = 0
    decode_forwards: int = 0
    decode_tokens: int = 0


class GlmImageTokenRunner:
    """Family model runner that lets the AR engine schedule GLM-Image steps."""

    def __init__(self, model: Any) -> None:
        self.model = model

    @torch.no_grad()
    def init_ar(
        self,
        cond_inputs_embeds: torch.Tensor,
        cond_attention_mask: torch.Tensor,
        *,
        token_h: int,
        token_w: int,
        prev_h: int,
        prev_w: int,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> ARTokenLoopInit:
        temp = temperature if temperature is not None else self.model.config.temperature
        nucleus = top_p if top_p is not None else self.model.config.top_p
        grids = ((int(token_h), int(token_w)), (int(prev_h), int(prev_w)))
        # Generation order matches the schedule: preview raster first, then
        # the large raster (schedule walks the grid list back-to-front).
        total_token_num = prev_h * prev_w + token_h * token_w
        batch_size = cond_inputs_embeds.shape[0]
        device = cond_inputs_embeds.device

        base_schedule = glm_image_decode_position_schedule(0, grids).to(device)
        prompt_valid_lens = cond_attention_mask.sum(dim=1).to(
            device=device, dtype=torch.long,
        )

        last_hidden, kv_rows = self._prefill_prompt(
            cond_inputs_embeds, cond_attention_mask,
        )

        return ARTokenLoopInit(
            state=GlmImageARState(
                token_ids=torch.empty(
                    batch_size, total_token_num, dtype=torch.long, device=device,
                ),
                logprobs=torch.empty(
                    batch_size, total_token_num, dtype=torch.float32, device=device,
                ),
                temperature=float(temp),
                top_p=float(nucleus),
                total_token_num=int(total_token_num),
                base_position_schedule=base_schedule,
                prompt_valid_lens=prompt_valid_lens,
                kv_rows=kv_rows,
                prefill_forwards=1,
            ),
            cache_lanes={},
            row_lanes={
                "cond_last_hidden": last_hidden,
                "cond_attn": cond_attention_mask,
            },
            cache_lane_owners={},
            row_lane_owners={
                "cond_last_hidden": "glm_image.cond_last_hidden",
                "cond_attn": "glm_image.cond_attn",
            },
        )

    @torch.no_grad()
    def step_ar(
        self,
        state: GlmImageARState,
        batch: ARStepBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> ARStepOutput:
        del generator
        self._validate_ar_step_batch(state, batch)
        row_updates = self._sample_ar_step(state, batch)
        return ARStepOutput(
            result=ARStepResult(
                debug_counters={
                    "ar_kv_cache_enabled": True,
                    "ar_paged_attention_enabled": False,
                    "ar_prefill_forwards": state.prefill_forwards,
                    "ar_decode_forwards": state.decode_forwards,
                    "ar_decode_tokens": state.decode_tokens,
                },
            ),
            updated_row_lanes=row_updates,
        )

    @torch.no_grad()
    def finalize_ar(self, state: GlmImageARState) -> tuple[torch.Tensor, torch.Tensor]:
        return state.token_ids, state.logprobs

    # -- internals -----------------------------------------------------

    def _prefill_prompt(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, list[Any]]:
        """Prefill the t2i prompt with explicit text-position mrope ids."""
        trunk = self.model._lm_trunk()
        position_ids = glm_image_prefill_position_ids(attention_mask)
        outputs = trunk(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        past = outputs.past_key_values
        if past is None:
            raise RuntimeError(
                "GLM-Image trunk prefill returned no past_key_values; "
                "use_cache must be enabled",
            )
        last_hidden = outputs.last_hidden_state[:, -1, :]
        return last_hidden, ar_split_rows(past, inputs_embeds.shape[0])

    @staticmethod
    def _validate_ar_step_batch(
        state: GlmImageARState,
        batch: ARStepBatch,
    ) -> None:
        row_indices = batch.row_indices
        if not row_indices:
            raise ValueError("row_indices must be non-empty")
        if any(row < 0 or row >= state.token_ids.shape[0] for row in row_indices):
            raise ValueError(f"invalid GLM-Image row indices: {row_indices}")
        if len(set(batch.positions)) != 1:
            raise ValueError("ActiveSequence positions must match within one AR step")
        if batch.position >= state.total_token_num:
            raise ValueError("GlmImageARState has already finished sampling")

    def _sample_ar_step(
        self,
        state: GlmImageARState,
        batch: ARStepBatch,
    ) -> dict[str, Any]:
        row_indices = batch.row_indices
        position = batch.position
        rows = torch.tensor(row_indices, dtype=torch.long, device=state.token_ids.device)
        hidden = batch.row_lanes["cond_last_hidden"].unsqueeze(1)  # [B, 1, H]
        sampled, lp = self._sample_image_token(state, hidden)
        state.token_ids[rows, position] = sampled
        state.logprobs[rows, position] = lp
        state.decode_tokens += len(row_indices)

        if position + 1 >= state.total_token_num:
            return {}
        return self._advance_after_sample(state, batch=batch, sampled=sampled)

    def _sample_image_token(
        self,
        state: GlmImageARState,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ``image_gen_logits`` restricts to the codebook prefix — every
        # generated position is a free codebook draw (no structural tokens
        # inside the raster), so no per-position mask is needed.
        logits = self.model.image_gen_logits(hidden).squeeze(1).float()  # [B, V]

        temp = max(state.temperature, 1e-5)
        filtered = logits / temp
        if state.top_p < 1.0:
            filtered = top_k_top_p_filtering(filtered.clone(), top_p=state.top_p)
        probs = F.softmax(filtered, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # RL-correctness contract (same as janus/llamagen): SAMPLE from the
        # top_p-filtered behavior distribution (matching the reference
        # generation_config) but SCORE the log-prob under the plain
        # temperature-scaled conditional over the restricted vocab — the
        # conditional model is the policy GRPO optimizes; nucleus filtering
        # is only the behavior distribution.
        log_probs = F.log_softmax(logits / temp, dim=-1)
        lp = log_probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        return sampled, lp

    def _advance_after_sample(
        self,
        state: GlmImageARState,
        *,
        batch: ARStepBatch,
        sampled: torch.Tensor,
    ) -> dict[str, Any]:
        batch_size = len(batch.row_indices)
        trunk = self.model._lm_trunk()
        rows = torch.tensor(
            batch.row_indices, dtype=torch.long,
            device=state.prompt_valid_lens.device,
        )

        token_embed = self.model.embed_gen_tokens(sampled.unsqueeze(-1))  # [B, 1, H]
        next_attn = append_attention_token(batch.row_lanes["cond_attn"])
        # Input token j==batch.position gets schedule position j, offset by
        # each row's valid prompt length (schedule is translation-equivariant
        # in the prompt length).
        step_positions = (
            state.base_position_schedule[:, batch.position]
            .view(3, 1, 1)
            .expand(3, batch_size, 1)
            + state.prompt_valid_lens[rows].view(1, batch_size, 1)
        )

        kv = ar_concat_rows(select_paged_states(state.kv_rows, batch.row_indices))
        outputs = trunk(
            inputs_embeds=token_embed,
            attention_mask=next_attn,
            position_ids=step_positions,
            past_key_values=kv,
            use_cache=True,
        )
        past = outputs.past_key_values
        if past is None:
            raise RuntimeError(
                "GLM-Image trunk step returned no past_key_values; "
                "use_cache must be enabled",
            )
        scatter_paged_states(
            state.kv_rows,
            batch.row_indices,
            ar_split_rows(past, batch_size),
        )
        hidden = normalize_paged_last_hidden(outputs.last_hidden_state)
        state.decode_forwards += 1
        return {
            "cond_last_hidden": hidden,
            "cond_attn": next_attn,
        }


__all__ = [
    "GlmImageARState",
    "GlmImageTokenRunner",
]
