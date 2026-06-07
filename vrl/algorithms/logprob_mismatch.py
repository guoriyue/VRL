"""Rollout-vs-replay log-probability mismatch statistics.

When the rollout (behavior) policy and the training (replay) policy diverge — e.g.
a bf16 rollout transformer vs an fp32 replay forward — the collection-time logprob
no longer equals the freshly recomputed replay logprob, and the GRPO importance
ratio drifts from 1. These statistics surface that drift as numbers so it can be
measured (and later gated or importance-corrected) instead of silently biasing
training. They are the shared source of truth for both the per-step training
metrics (P1) and the precision drift guard (P0).
"""

from __future__ import annotations

from dataclasses import dataclass

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


__all__ = ["LogprobMismatchStats", "compute_logprob_mismatch_stats"]
