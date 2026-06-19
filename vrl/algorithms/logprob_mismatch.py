"""Rollout-vs-replay log-probability drift: measure + correct.

When the rollout (behavior) policy and the training (replay) policy diverge — e.g.
a bf16/fp8 rollout transformer vs an fp32 replay forward — the collection-time
logprob no longer equals the freshly recomputed replay logprob, and the GRPO
importance ratio drifts from 1. This module is the shared, algorithm-agnostic
toolkit for that drift:

- :func:`compute_logprob_mismatch_stats` MEASURES it (the source of truth for both
  the per-step training metrics and the precision drift guard).
- :class:`PrecisionCorrectionConfig` + :func:`apply_truncated_importance_weight`
  CORRECT it via truncated importance sampling (TIS) — the counterpart to the
  drift guard's gate. The config lives at the trainer level
  (``trainer.precision_correction``), not in any algorithm's hyperparameters,
  because bounding a quantized/backend rollout's drift is a precision concern
  shared across importance-ratio algorithms; the trainer injects it into the
  algorithm, which applies the weight inside its own surrogate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True, slots=True)
class LogprobMismatchStats:
    """Reduced stats on ``fresh_log_prob - old_log_prob`` (all fp32 scalars)."""

    logprob_abs_diff_mean: float = 0.0
    logprob_abs_diff_max: float = 0.0
    ratio_abs_dev_mean: float = 0.0
    ratio_abs_dev_max: float = 0.0
    mismatch_kl: float = 0.0
    mismatch_k3_kl: float = 0.0
    finite: bool = True


def compute_logprob_mismatch_stats(
    fresh_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
) -> LogprobMismatchStats:
    """Stats on replay-vs-rollout-behavior log-probability drift.

    ``fresh_log_prob`` is the freshly recomputed replay logprob (compute dtype);
    ``old_log_prob`` is the rollout behavior logprob (rollout dtype). Reductions run
    in fp32 so a bf16 input does not itself add noise to the measurement. Mirrors
    slime's ``train_rollout_logprob_abs_diff`` / ``mismatch_kl`` / ``mismatch_k3_kl``.
    """

    fresh = fresh_log_prob.detach().to(torch.float32)
    old = old_log_prob.detach().to(torch.float32)
    if fresh.numel() == 0:
        return LogprobMismatchStats()

    delta = fresh - old  # replay - rollout
    abs_diff = delta.abs()
    ratio = torch.exp(delta)
    ratio_dev = (ratio - 1.0).abs()
    finite = bool(torch.isfinite(delta).all() and torch.isfinite(ratio).all())
    return LogprobMismatchStats(
        logprob_abs_diff_mean=float(abs_diff.mean()),
        logprob_abs_diff_max=float(abs_diff.max()),
        ratio_abs_dev_mean=float(ratio_dev.mean()),
        ratio_abs_dev_max=float(ratio_dev.max()),
        mismatch_kl=float((-delta).mean()),  # mean(old - fresh)
        mismatch_k3_kl=float((ratio - delta - 1.0).mean()),  # mean(exp(d) - d - 1)
        finite=finite,
    )


@dataclass(slots=True)
class PrecisionCorrectionConfig:
    """Truncated importance sampling (TIS) knobs for rollout->replay drift.

    The correction counterpart to ``PrecisionDriftGuardConfig``: the guard
    measures/fails on drift; this bounds it in the loss. ``old_log_prob`` IS the
    rollout (behavior) logprob in this codebase, so the surrogate ratio
    ``exp(replay - rollout)`` is exactly the importance weight TIS truncates.
    A quantized rollout (fp8/fp4 vs bf16 replay) inflates that weight on a few
    samples and, on negative advantages, drives a large unclipped PPO gradient.

    Modes: ``off`` keeps legacy behavior; ``truncate`` = one-sided upper cap (keep
    the gradient, bound it); ``clip`` = two-sided; ``mask`` = drop out-of-range
    samples entirely (catastrophic-drift rejection).
    """

    tis_mode: str = field(default="off")  # "off" | "truncate" | "clip" | "mask"
    tis_imp_weight_cap: float = field(default=2.0)  # upper bound C on the weight
    tis_clip_low: float = field(default=0.0)  # lower bound (clip/mask modes)

    def __post_init__(self) -> None:
        if self.tis_mode not in ("off", "truncate", "clip", "mask"):
            raise ValueError(
                "precision_correction.tis_mode must be off/truncate/clip/mask; "
                f"got {self.tis_mode!r}",
            )
        if float(self.tis_imp_weight_cap) <= 0:
            raise ValueError("precision_correction.tis_imp_weight_cap must be > 0")
        if float(self.tis_clip_low) < 0:
            raise ValueError("precision_correction.tis_clip_low must be >= 0")
        if self.tis_mode != "off" and float(self.tis_clip_low) >= float(self.tis_imp_weight_cap):
            raise ValueError(
                "precision_correction.tis_clip_low must be < tis_imp_weight_cap "
                f"(got low={self.tis_clip_low}, cap={self.tis_imp_weight_cap})",
            )


def apply_truncated_importance_weight(
    ratio: torch.Tensor,
    config: PrecisionCorrectionConfig,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Bound the rollout->replay importance weight (TIS).

    ``ratio = exp(replay_logprob - rollout_logprob)`` is the importance weight
    ``pi_replay / pi_rollout``. Returns ``(weighted_ratio, keep_mask)``: ``keep_mask``
    is a 0/1 tensor for ``'mask'`` mode (so the caller can drop rejected samples
    from the mean) and ``None`` otherwise. ``'off'`` returns the ratio unchanged.
    """

    mode = config.tis_mode
    if mode == "off":
        return ratio, None
    cap, low = config.tis_imp_weight_cap, config.tis_clip_low
    if mode == "mask":
        keep = (ratio <= cap) & (ratio >= low)
        return ratio, keep.to(ratio.dtype)
    if mode == "truncate":
        return ratio.clamp(max=cap), None
    if mode == "clip":
        return ratio.clamp(min=low, max=cap), None
    raise ValueError(f"unknown tis_mode: {mode!r}")


__all__ = [
    "LogprobMismatchStats",
    "PrecisionCorrectionConfig",
    "apply_truncated_importance_weight",
    "compute_logprob_mismatch_stats",
]
