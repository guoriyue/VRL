"""Token-level GRPO for autoregressive image / text generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vrl.algorithms.grpo.continuous import GRPO, GRPOConfig
from vrl.algorithms.logprob_mismatch import (
    apply_rejection_sample_mask,
    apply_truncated_importance_weight,
    combine_keep_masks,
)
from vrl.algorithms.trajectory import AlgorithmInput
from vrl.algorithms.types import TrainStepMetrics


@dataclass(slots=True)
class TokenGRPOConfig(GRPOConfig):
    """Token-level GRPO hyper-parameters."""

    kl_estimator: str = "k3"


class TokenGRPO(GRPO):
    """GRPO with per-token PPO loss for autoregressive policies."""

    def __init__(self, config: TokenGRPOConfig | None = None) -> None:
        cfg = config or TokenGRPOConfig()
        super().__init__(cfg)
        self.config: TokenGRPOConfig = cfg

    def compute_loss(
        self,
        inputs: AlgorithmInput,
    ) -> tuple[Any, TrainStepMetrics]:
        cfg = self.config

        # signals presence + required_signal_keys are enforced upstream by
        # AlgorithmAdapter.validate_inputs (inherited ("log_prob","old_log_prob")).
        if inputs.advantages is None:
            raise RuntimeError("AlgorithmInput.advantages is required for TokenGRPO")
        signals = inputs.signals.primary
        new_lp: torch.Tensor = signals.log_prob
        old_lp: torch.Tensor = signals.old_log_prob
        advantages = inputs.advantages
        if new_lp.shape != old_lp.shape:
            raise ValueError(
                f"log_prob shape mismatch: new={tuple(new_lp.shape)} old={tuple(old_lp.shape)}"
            )

        mask = signals.mask.to(dtype=new_lp.dtype, device=new_lp.device)

        if advantages.dim() == 1:
            adv_bL = advantages.unsqueeze(-1).expand_as(new_lp)
        else:
            adv_bL = advantages

        raw_ratio = torch.exp(new_lp - old_lp)
        # TIS on the rollout->replay weight (old_lp is the rollout behavior logprob),
        # folded into the per-token mask so 'mask' mode drops drift-rejected tokens.
        # RS rejects whole sequences whose log-ratio drift is out of band (a per-
        # sequence (B,1) mask that broadcasts over the token axis); restrict its
        # sequence reduction to valid tokens via the token mask.
        pc = self.precision_correction
        ratio, tis_keep = apply_truncated_importance_weight(raw_ratio, pc)
        rs_keep = apply_rejection_sample_mask(new_lp - old_lp, pc, mask=mask)
        eff_mask = combine_keep_masks(mask, tis_keep, rs_keep)
        clipped_ratio = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)
        unclipped_loss = -adv_bL * ratio
        clipped_loss = -adv_bL * clipped_ratio
        per_token_loss = torch.maximum(unclipped_loss, clipped_loss)

        denom = eff_mask.sum().clamp_min(1.0)
        policy_loss = (per_token_loss * eff_mask).sum() / denom

        if cfg.init_kl_coef > 0:
            if signals.ref_log_prob is None:
                raise RuntimeError(
                    f"TokenGRPOConfig.init_kl_coef={cfg.init_kl_coef} > 0 but "
                    "signals.ref_log_prob is None. Check: (1) OnlineTrainer "
                    "is passing SignalRequest(need_ref=True), (2) the AR evaluator "
                    "implements the reference log-prob path, and (3) either a "
                    "frozen ref_model is passed or the train model has a real "
                    "adapter that can be disabled for the reference pass."
                )
            ref_lp: torch.Tensor = signals.ref_log_prob
            log_ratio = new_lp - ref_lp
            kl_per_tok = _token_kl_per_token(log_ratio, cfg.kl_estimator)
            kl_loss = (kl_per_tok * eff_mask).sum() / denom
            loss = policy_loss + cfg.init_kl_coef * kl_loss
        else:
            kl_loss = torch.zeros((), device=new_lp.device)
            loss = policy_loss

        with torch.no_grad():
            valid = mask > 0
            if valid.any():
                ratio_valid = ratio[valid]
                clip_fraction = (torch.abs(ratio_valid - 1.0) > cfg.clip_ratio).float().mean().item()
                approx_kl = 0.5 * ((new_lp - old_lp) ** 2)[valid].mean().item()
                if tis_keep is None:
                    tis_clip_fraction = (
                        0.0
                        if pc.tis_mode == "off"
                        else (ratio[valid] != raw_ratio[valid]).float().mean().item()
                    )
                else:
                    tis_clip_fraction = (1.0 - tis_keep[valid]).mean().item()
            else:
                clip_fraction = 0.0
                approx_kl = 0.0
                tis_clip_fraction = 0.0
            # RS keep is per-sequence (B,1); its rejected fraction is over whole
            # sequences, independent of the per-token valid count above.
            rs_seq_masked_fraction = (
                0.0 if rs_keep is None else (1.0 - rs_keep.mean()).item()
            )

        metrics = TrainStepMetrics(
            loss=loss.item(),
            policy_loss=policy_loss.item(),
            kl_penalty=kl_loss.item(),
            clip_fraction=clip_fraction,
            approx_kl=approx_kl,
            tis_clip_fraction=tis_clip_fraction,
            rs_seq_masked_fraction=rs_seq_masked_fraction,
        )
        return loss, metrics


def _token_kl_per_token(log_ratio: torch.Tensor, estimator: str) -> torch.Tensor:
    if estimator == "k1":
        return log_ratio
    if estimator == "k2":
        # Quadratic log-ratio penalty avoids the exp(log_ratio) spikes that can
        # dominate token RL when the current policy briefly outruns the reference.
        return 0.5 * log_ratio.square()
    if estimator == "k3":
        return torch.exp(log_ratio) * (log_ratio - 1.0) + 1.0
    raise ValueError(f"unknown kl_estimator: {estimator}")
