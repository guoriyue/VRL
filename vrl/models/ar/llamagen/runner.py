"""LlamaGen AR model runner executed by the AR engine.

Follows the Janus dual-sequence CFG token-loop pattern, with one structural
difference: the vendored LlamaGen GPT owns a *static, in-place* KV cache
(``vendor/gpt.py::KVCache`` keyed by ``input_pos``) that does not speak the HF
``past_key_values`` protocol required by the shared attention backends in
``vrl.nn.modules.ar_attention_backends``. The runner therefore drives the
native cache directly — no ``attention_backend`` is injected — and the family
executor overrides ``_ar_runner`` accordingly.

The cond/uncond pair runs as one combined ``2B`` batch (upstream
``generate.py`` layout: ``[cond rows | uncond rows]``), so the per-layer cache
rows are positional. Because cache advancement is an in-place buffer write for
the whole combined batch, every scheduled step must cover all rows; the runner
validates this instead of supporting partial-row scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from vrl.generation.ar.decode_loop import (
    ARStepBatch,
    ARTokenLoopInit,
)
from vrl.math.ar.logprob import (
    gather_categorical_log_probs,
    require_positive_temperature,
    top_k_top_p_filtering,
)
from vrl.models.ar.base import ARDiscreteTokenRunner, ARDiscreteTokenState


@dataclass(slots=True, kw_only=True)
class LlamaGenARState(ARDiscreteTokenState):
    """Mutable LlamaGen state owned by one scheduled AR token loop."""

    guidance_scale: float
    temperature: float
    top_k: int
    top_p: float
    caption_len: int


class LlamaGenARModelRunner(ARDiscreteTokenRunner):
    """Family model runner that lets the AR engine schedule LlamaGen token steps."""

    family = "llamagen"

    def __init__(self, model: Any) -> None:
        self.model = model

    @torch.no_grad()
    def init_ar(
        self,
        cond_embeds: torch.Tensor,          # [B, T, caption_dim] T5 features
        uncond_embeds: torch.Tensor,        # [B, T, caption_dim] learned null caption
        cond_attention_mask: torch.Tensor,  # [B, T] left-pad caption mask
        uncond_attention_mask: torch.Tensor,
        *,
        guidance_scale: float | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        image_token_num: int | None = None,
    ) -> ARTokenLoopInit:
        config = self.model.config
        cfg = guidance_scale if guidance_scale is not None else config.guidance_scale
        temp = require_positive_temperature(
            temperature if temperature is not None else config.temperature,
        )
        top_k = top_k if top_k is not None else config.top_k
        top_p = top_p if top_p is not None else config.top_p
        image_token_num = image_token_num or config.image_token_num

        batch_size, caption_len = cond_attention_mask.shape
        device = self.model.device
        dtype = self.model.dtype
        trunk = self.model._gpt_trunk()
        # Generation is inference-only; eval() keeps the vendored forward on
        # its deterministic branches (freqs_cis[input_pos], no logits slice).
        trunk.eval()

        # Upstream generate(): one combined [cond | uncond] batch with a
        # per-request static cache. setup_caches allocates on the current
        # default device, hence the torch.device scope (as upstream does).
        total_len = caption_len + int(image_token_num)
        with torch.device(device):
            trunk.setup_caches(
                max_batch_size=2 * batch_size,
                max_seq_length=total_len,
                dtype=dtype,
            )
        # Patch the causal mask with the caption pad mask exactly like
        # upstream generate() (uncond rows reuse the cond mask by contract of
        # the caller; the runner just consumes what it is handed).
        masks = torch.cat(
            [cond_attention_mask, uncond_attention_mask], dim=0
        ).to(device)
        trunk.causal_mask[:, :, :caption_len] = (
            trunk.causal_mask[:, :, :caption_len] * masks.unsqueeze(1)
        )
        eye = torch.eye(
            trunk.causal_mask.size(1), trunk.causal_mask.size(2), device=device
        )
        trunk.causal_mask[:] = trunk.causal_mask * (1 - eye) + eye

        cond_combined = torch.cat([cond_embeds, uncond_embeds], dim=0).to(
            device=device, dtype=dtype
        )
        input_pos = torch.arange(caption_len, device=device)
        logits, _ = trunk(idx=None, cond_idx=cond_combined, input_pos=input_pos)
        last = logits[:, -1].float()  # [2B, V]

        return ARTokenLoopInit(
            state=LlamaGenARState(
                token_ids=torch.empty(
                    batch_size, image_token_num, dtype=torch.long, device=device
                ),
                logprobs=torch.empty(
                    batch_size, image_token_num, dtype=torch.float32, device=device
                ),
                guidance_scale=float(cfg),
                temperature=temp,
                top_k=int(top_k),
                top_p=float(top_p),
                total_token_num=int(image_token_num),
                caption_len=int(caption_len),
                prefill_forwards=1,
            ),
            row_lanes={
                "cond_logits": last[:batch_size],
                "uncond_logits": last[batch_size:],
            },
            row_lane_owners={
                "cond_logits": "llamagen.cond_logits",
                "uncond_logits": "llamagen.uncond_logits",
            },
        )

    # -- internals -----------------------------------------------------

    def _validate_family_step(
        self,
        state: ARDiscreteTokenState,
        batch: ARStepBatch,
    ) -> None:
        assert isinstance(state, LlamaGenARState)
        batch_size = state.token_ids.shape[0]
        # The vendored static KV cache advances in-place for the whole
        # combined [cond | uncond] batch; partial-row steps would desync it.
        if batch.row_indices != list(range(batch_size)):
            raise ValueError(
                "LlamaGen requires full-batch AR steps in row order "
                f"(cache rows are positional); got row_indices={batch.row_indices} "
                f"for batch_size={batch_size}. Schedule with "
                "scheduler_batch_size == chunk sample count."
            )

    def _sample_ar_step(
        self,
        state: ARDiscreteTokenState,
        batch: ARStepBatch,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        assert isinstance(state, LlamaGenARState)
        position = batch.position
        sampled, lp = self._sample_cfg_image_token(
            state,
            cond_logits=batch.row_lanes["cond_logits"],
            uncond_logits=batch.row_lanes["uncond_logits"],
        )
        state.token_ids[:, position] = sampled
        state.logprobs[:, position] = lp
        state.decode_tokens += len(batch.row_indices)

        if position + 1 >= state.total_token_num:
            return {}, {}
        return {}, self._advance_after_sample(state, position=position, sampled=sampled)

    def _sample_cfg_image_token(
        self,
        state: LlamaGenARState,
        *,
        cond_logits: torch.Tensor,    # [B, V]
        uncond_logits: torch.Tensor,  # [B, V]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cond_logits = cond_logits.float()
        uncond_logits = uncond_logits.float()
        # Upstream CFG combine: uncond + (cond - uncond) * cfg_scale.
        guided = uncond_logits + (cond_logits - uncond_logits) * state.guidance_scale

        filtered = guided / state.temperature
        if state.top_k > 0 or state.top_p < 1.0:
            filtered = top_k_top_p_filtering(
                filtered.clone(), top_k=state.top_k, top_p=state.top_p
            )
        probs = F.softmax(filtered, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # RL-correctness contract (same as Janus — do NOT "align to upstream"
        # by scoring `guided`): we SAMPLE from the CFG-guided/filtered
        # distribution but score the log-prob under the plain conditional
        # policy, which is what GRPO optimizes; CFG and top-k/top-p are only
        # the behavior/sampling distribution. Upstream computes no log-prob,
        # so there is no upstream line to copy.
        lp = gather_categorical_log_probs(cond_logits, sampled, temperature=state.temperature)
        return sampled, lp

    def _advance_after_sample(
        self,
        state: LlamaGenARState,
        *,
        position: int,
        sampled: torch.Tensor,  # [B]
    ) -> dict[str, Any]:
        batch_size = sampled.shape[0]
        trunk = self.model._gpt_trunk()
        # Combined [cond | uncond] batch: both branches consume the token
        # sampled from the guided distribution (upstream decode_one_token).
        x = sampled.view(-1, 1)
        x_combined = torch.cat([x, x], dim=0)
        input_pos = torch.tensor(
            [state.caption_len + position],
            device=sampled.device,
            dtype=torch.int,
        )
        logits, _ = trunk(idx=x_combined, cond_idx=None, input_pos=input_pos)
        last = logits[:, -1].float()
        state.decode_forwards += 1
        return {
            "cond_logits": last[:batch_size],
            "uncond_logits": last[batch_size:],
        }


__all__ = [
    "LlamaGenARModelRunner",
    "LlamaGenARState",
]
