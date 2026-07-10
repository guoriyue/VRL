"""Janus-Pro AR model runner executed by the AR engine."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vrl.generation.ar.decode_loop import ARTokenLoopInit
from vrl.models.ar.janus_pro.model import image_token_logits_from_hidden
from vrl.models.ar.paged_attention_helpers import (
    PagedCFGARState,
    PagedCFGTokenRunner,
)


@dataclass(slots=True, kw_only=True)
class JanusProARState(PagedCFGARState):
    """Mutable Janus state owned by one scheduled AR token loop."""


class JanusProARModelRunner(PagedCFGTokenRunner):
    """Family model runner that lets the AR engine schedule Janus token steps."""

    family = "janus_pro"
    lane_owner_prefix = "janus"

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
        return self._init_paged_cfg(
            state_cls=JanusProARState,
            cond_inputs_embeds=cond_inputs_embeds,
            uncond_inputs_embeds=uncond_inputs_embeds,
            cond_attention_mask=cond_attention_mask,
            uncond_attention_mask=uncond_attention_mask,
            total_token_num=int(image_token_num),
            guidance_scale=float(cfg),
            temperature=float(temp),
        )

    def _sample_cfg_image_token(
        self,
        state: PagedCFGARState,
        hidden: torch.Tensor,
        position: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del position  # Janus has no per-position structural constraint.
        logits = image_token_logits_from_hidden(self.model.mmgpt, hidden).squeeze(1)
        cond_logits, uncond_logits = logits.chunk(2, dim=0)

        # RL-correctness contract — do NOT "align to upstream" by scoring `guided`.
        # We SAMPLE from the CFG-`guided` distribution but score the log-prob under
        # `cond_logits`: the conditional model is the policy GRPO optimizes, and CFG
        # is only the sampling/importance distribution. Scoring `guided` here would
        # make lp the behavior policy and silently break the old_log_prob invariant
        # (train/infer logprob parity). Upstream Janus' inference script computes no
        # log-prob, so there is no upstream line to copy. Locked by
        # tests/models/ar/janus_pro/test_upstream_reconcile_contracts.py.
        return self._sample_cfg_logits(state, cond_logits, uncond_logits)

    def _embed_sampled_token(self, sampled: torch.Tensor) -> torch.Tensor:
        return self.model._base().prepare_gen_img_embeds(sampled.unsqueeze(-1))


__all__ = [
    "JanusProARModelRunner",
    "JanusProARState",
]
