"""Emu3 AR model runner executed by the AR engine.

Shares the cond/uncond paged-CFG token loop with Janus-Pro
(``PagedCFGTokenRunner``) with one Emu3-specific addition: a per-position
structural logits mask (EOL forced at column ``width`` of every grid row, then
the EOF/EOI/EOS tail — the official ``prefix_allowed_tokens_fn`` re-expressed
as a precomputed schedule).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from vrl.generation.ar.decode_loop import ARTokenLoopInit
from vrl.models.ar.emu3.model import (
    emu3_allowed_token_mask,
    emu3_forced_token_schedule,
    emu3_grid_token_num,
)
from vrl.models.ar.paged_attention_helpers import (
    PagedCFGARState,
    PagedCFGTokenRunner,
)


@dataclass(slots=True, kw_only=True)
class Emu3ARState(PagedCFGARState):
    """Mutable Emu3 state owned by one scheduled AR token loop."""

    # [L_total] generation-vocab index forced at each position, -1 = free.
    forced_gen_index: torch.Tensor


class Emu3TokenRunner(PagedCFGTokenRunner):
    """Family model runner that lets the AR engine schedule Emu3 token steps."""

    family = "emu3"

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

    def _sample_cfg_image_token(
        self,
        state: PagedCFGARState,
        hidden: torch.Tensor,
        position: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(state, Emu3ARState)
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

    def _embed_sampled_token(self, sampled: torch.Tensor) -> torch.Tensor:
        return self.model.embed_gen_tokens(sampled.unsqueeze(-1))


__all__ = [
    "Emu3ARState",
    "Emu3TokenRunner",
]
