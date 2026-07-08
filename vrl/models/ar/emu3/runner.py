"""Emu3 AR model runner executed by the AR engine.

Mirrors ``vrl.models.ar.janus_pro.runner.JanusProARModelRunner`` with one
Emu3-specific addition: a per-position structural logits mask (EOL forced at
column ``width`` of every grid row, then the EOF/EOI/EOS tail — the official
``prefix_allowed_tokens_fn`` re-expressed as a precomputed schedule).
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
from vrl.models.ar.emu3.model import (
    emu3_allowed_token_mask,
    emu3_forced_token_schedule,
    emu3_grid_token_num,
)
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
class Emu3ARState:
    """Mutable Emu3 state owned by one scheduled AR token loop."""

    token_ids: torch.Tensor
    logprobs: torch.Tensor
    guidance_scale: float
    temperature: float
    total_token_num: int
    # [L_total] generation-vocab index forced at each position, -1 = free.
    forced_gen_index: torch.Tensor
    paged_cond_states: list[Any] | None = None
    paged_uncond_states: list[Any] | None = None
    prefill_forwards: int = 0
    decode_forwards: int = 0
    decode_tokens: int = 0


class Emu3TokenRunner:
    """Family model runner that lets the AR engine schedule Emu3 token steps."""

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
        height: int,
        width: int,
        guidance_scale: float | None = None,
        temperature: float | None = None,
    ) -> ARTokenLoopInit:
        cfg = guidance_scale if guidance_scale is not None else self.model.config.guidance_scale
        temp = temperature if temperature is not None else self.model.config.temperature
        total_token_num = emu3_grid_token_num(int(height), int(width))
        batch_size = cond_inputs_embeds.shape[0]
        device = cond_inputs_embeds.device
        forced = emu3_forced_token_schedule(
            int(height), int(width), self.model.image_vocab_size,
        ).to(device)
        cond_prefill = self._prefill_ar_prompt_paged(
            cond_inputs_embeds,
            cond_attention_mask,
            branch="cond",
            image_token_num=total_token_num,
        )
        uncond_prefill = self._prefill_ar_prompt_paged(
            uncond_inputs_embeds,
            uncond_attention_mask,
            branch="uncond",
            image_token_num=total_token_num,
        )

        return ARTokenLoopInit(
            state=Emu3ARState(
                token_ids=torch.empty(
                    batch_size, total_token_num, dtype=torch.long, device=device,
                ),
                logprobs=torch.empty(
                    batch_size, total_token_num, dtype=torch.float32, device=device,
                ),
                guidance_scale=float(cfg),
                temperature=float(temp),
                total_token_num=total_token_num,
                forced_gen_index=forced,
                paged_cond_states=list(cond_prefill.sequence_states),
                paged_uncond_states=list(uncond_prefill.sequence_states),
                prefill_forwards=2,
            ),
            cache_lanes={},
            row_lanes={
                "cond_last_hidden": cond_prefill.last_hidden,
                "uncond_last_hidden": uncond_prefill.last_hidden,
                "cond_attn": cond_attention_mask,
                "uncond_attn": uncond_attention_mask,
            },
            cache_lane_owners={},
            row_lane_owners={
                "cond_last_hidden": "emu3.cond_last_hidden",
                "uncond_last_hidden": "emu3.uncond_last_hidden",
                "cond_attn": "emu3.cond_attn",
                "uncond_attn": "emu3.uncond_attn",
            },
        )

    @torch.no_grad()
    def step_ar(
        self,
        state: Emu3ARState,
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
    def finalize_ar(self, state: Emu3ARState) -> tuple[torch.Tensor, torch.Tensor]:
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
                metadata={"family": "emu3", "image_token_num": image_token_num},
            )
        )

    def _sample_ar_step(
        self,
        state: Emu3ARState,
        batch: ARStepBatch,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self._validate_ar_step_batch(state, batch)
        return self._sample_ar_step_kv(state, batch)

    @staticmethod
    def _validate_ar_step_batch(
        state: Emu3ARState,
        batch: ARStepBatch,
    ) -> None:
        row_indices = batch.row_indices
        if not row_indices:
            raise ValueError("row_indices must be non-empty")
        if any(row < 0 or row >= state.token_ids.shape[0] for row in row_indices):
            raise ValueError(f"invalid Emu3 row indices: {row_indices}")
        if len(set(batch.positions)) != 1:
            raise ValueError("ActiveSequence positions must match within one AR step")
        if batch.position >= state.total_token_num:
            raise ValueError("Emu3ARState has already finished sampling")

    def _sample_ar_step_kv(
        self,
        state: Emu3ARState,
        batch: ARStepBatch,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        row_indices = batch.row_indices
        position = batch.position
        batch_size = len(row_indices)
        rows = torch.tensor(row_indices, dtype=torch.long, device=state.token_ids.device)
        cond_hidden = batch.row_lanes["cond_last_hidden"]
        uncond_hidden = batch.row_lanes["uncond_last_hidden"]
        hidden = torch.cat([cond_hidden, uncond_hidden], dim=0).unsqueeze(1)
        sampled, lp = self._sample_cfg_image_token(state, hidden, position)
        state.token_ids[rows, position] = sampled
        state.logprobs[rows, position] = lp

        cache_updates: dict[str, Any] = {}
        row_updates: dict[str, Any] = {}
        if position + 1 < state.total_token_num:
            cache_updates, row_updates = self._advance_after_sample(
                state,
                batch=batch,
                sampled=sampled,
            )

        state.decode_tokens += batch_size
        return cache_updates, row_updates

    def _sample_cfg_image_token(
        self,
        state: Emu3ARState,
        hidden: torch.Tensor,
        position: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.model.image_gen_logits(hidden).squeeze(1)
        cond_logits, uncond_logits = logits.chunk(2, dim=0)
        cond_logits = cond_logits.float()
        uncond_logits = uncond_logits.float()
        guided = uncond_logits + state.guidance_scale * (cond_logits - uncond_logits)

        # Structural constraint: within the grid only image tokens are legal;
        # column `width` forces EOL; the tail forces EOF/EOI/EOS. Combine CFG
        # BEFORE masking so -inf never enters the guidance arithmetic.
        allowed = emu3_allowed_token_mask(
            state.forced_gen_index[position : position + 1],
            self.model.image_vocab_size,
        )[0]
        guided = guided.masked_fill(~allowed, float("-inf"))
        probs = F.softmax(guided / state.temperature, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # RL-correctness contract (copied from janus_pro): SAMPLE from the
        # CFG-guided masked distribution but score the log-prob under the
        # masked, renormalized COND logits — the conditional model is the
        # policy GRPO optimizes; CFG is only the behavior distribution.
        # Forced positions renormalize to a single legal token -> lp == 0.
        cond_masked = cond_logits.masked_fill(~allowed, float("-inf"))
        log_probs = F.log_softmax(cond_masked / state.temperature, dim=-1)
        lp = log_probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        return sampled, lp

    def _advance_after_sample(
        self,
        state: Emu3ARState,
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
        token_embed = self.model.embed_gen_tokens(sampled.unsqueeze(-1))
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
                metadata={"family": "emu3"},
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
    "Emu3ARState",
    "Emu3TokenRunner",
]
