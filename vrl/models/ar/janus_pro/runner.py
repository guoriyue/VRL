"""Janus-Pro AR model runner executed by the AR engine."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

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
                total_token_num=int(image_token_num),
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
                "cond_last_hidden": "janus.cond_last_hidden",
                "uncond_last_hidden": "janus.uncond_last_hidden",
                "cond_attn": "janus.cond_attn",
                "uncond_attn": "janus.uncond_attn",
            },
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

    def _embed_sampled_token(self, sampled: torch.Tensor) -> torch.Tensor:
        return self.model._base().prepare_gen_img_embeds(sampled.unsqueeze(-1))


__all__ = [
    "JanusProARModelRunner",
    "JanusProARState",
]
