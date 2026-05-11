"""Token-level GRPO for autoregressive image / text generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vrl.algorithms.grpo.continuous import GRPO, GRPOConfig
from vrl.algorithms.types import TrainStepMetrics
from vrl.rollouts.evaluators.types import SignalBatch


@dataclass(slots=True)
class TokenGRPOConfig(GRPOConfig):
    """Token-level GRPO hyper-parameters."""

    mask_key: str = "token_mask"
    kl_estimator: str = "k3"


class TokenGRPO(GRPO):
    """GRPO with per-token PPO loss for autoregressive policies."""

    def __init__(self, config: TokenGRPOConfig | None = None) -> None:
        cfg = config or TokenGRPOConfig()
        super().__init__(cfg)
        self.config: TokenGRPOConfig = cfg

    def compute_signal_loss(
        self,
        signals: SignalBatch,
        advantages: Any,
        old_log_probs: Any,
    ) -> tuple[Any, TrainStepMetrics]:
        cfg = self.config

        new_lp: torch.Tensor = signals.log_prob
        old_lp: torch.Tensor = old_log_probs
        if new_lp.shape != old_lp.shape:
            raise ValueError(
                f"log_prob shape mismatch: new={tuple(new_lp.shape)} old={tuple(old_lp.shape)}"
            )

        mask = signals.aux.get(cfg.mask_key) if signals.aux else None
        if mask is None:
            mask = torch.ones_like(new_lp)
        mask = mask.to(dtype=new_lp.dtype, device=new_lp.device)

        if advantages.dim() == 1:
            adv_bL = advantages.unsqueeze(-1).expand_as(new_lp)
        else:
            adv_bL = advantages

        ratio = torch.exp(new_lp - old_lp)
        clipped_ratio = torch.clamp(ratio, 1.0 - cfg.eps_clip, 1.0 + cfg.eps_clip)
        unclipped_loss = -adv_bL * ratio
        clipped_loss = -adv_bL * clipped_ratio
        per_token_loss = torch.maximum(unclipped_loss, clipped_loss)

        denom = mask.sum().clamp_min(1.0)
        policy_loss = (per_token_loss * mask).sum() / denom

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
            if cfg.kl_estimator == "k3":
                kl_per_tok = torch.exp(log_ratio) * (log_ratio - 1.0) + 1.0
            elif cfg.kl_estimator == "k1":
                kl_per_tok = log_ratio
            else:
                raise ValueError(f"unknown kl_estimator: {cfg.kl_estimator}")
            kl_loss = (kl_per_tok * mask).sum() / denom
            loss = policy_loss + cfg.init_kl_coef * kl_loss
        else:
            kl_loss = torch.zeros((), device=new_lp.device)
            loss = policy_loss

        with torch.no_grad():
            valid = mask > 0
            if valid.any():
                ratio_valid = ratio[valid]
                clip_fraction = (torch.abs(ratio_valid - 1.0) > cfg.eps_clip).float().mean().item()
                approx_kl = 0.5 * ((new_lp - old_lp) ** 2)[valid].mean().item()
            else:
                clip_fraction = 0.0
                approx_kl = 0.0

        metrics = TrainStepMetrics(
            loss=loss.item(),
            policy_loss=policy_loss.item(),
            kl_penalty=kl_loss.item(),
            clip_fraction=clip_fraction,
            approx_kl=approx_kl,
        )
        return loss, metrics
