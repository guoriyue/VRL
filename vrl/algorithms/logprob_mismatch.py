"""Rollout-vs-replay log-probability drift: measure + correct.

When the rollout (behavior) policy and the training (replay) policy diverge — e.g.
a bf16/fp8 rollout transformer vs an fp32 replay forward — the collection-time
logprob no longer equals the freshly recomputed replay logprob, and the GRPO
importance ratio drifts from 1. This module is the shared, algorithm-agnostic
toolkit for that drift:

- :func:`LogprobMismatchStats` MEASURES it (the source of truth for both
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

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

# Torch is a call-time dependency, not an import-time one: the two dataclasses
# below are the trainer's public precision contract (``TrainerConfig`` carries
# PrecisionCorrectionConfig, ``TrainStepMetrics`` carries LogprobMismatchStats),
# so config parsing must be able to reach them without loading torch.
if TYPE_CHECKING:
    import torch


@dataclass(frozen=True, slots=True)
class LogprobMismatchStats:
    """Reduced stats on ``fresh_log_prob - old_log_prob`` (all fp32 scalars).

    Pass-zero ``finite`` and max values feed the pre-optimizer parity gate. The
    remaining values are observation/provenance metrics flattened only at the CSV
    boundary.
    """

    # Display/provenance-only: CSV and offline precision probes.
    logprob_abs_diff_mean: float = 0.0
    # Behavior-consumed by the unchanged-policy parity gate.
    logprob_abs_diff_max: float = 0.0
    # Display/provenance-only: CSV and offline precision probes.
    ratio_abs_dev_mean: float = 0.0
    # Behavior-consumed by the precision drift guard.
    ratio_abs_dev_max: float = 0.0
    # Display/provenance-only divergence diagnostics.
    mismatch_kl: float = 0.0
    mismatch_k3_kl: float = 0.0
    # Behavior-consumed by parity and precision drift gates.
    finite: bool = True

    @classmethod
    def aggregate(
        cls,
        values: list[LogprobMismatchStats],
        weights: list[float] | None = None,
    ) -> LogprobMismatchStats:
        """Aggregate weighted observations without losing max/finite semantics."""

        if not values:
            return cls()
        resolved_weights = [1.0] * len(values) if weights is None else list(weights)
        if len(resolved_weights) != len(values):
            raise ValueError("LogprobMismatchStats values/weights length mismatch")
        total_weight = sum(resolved_weights)
        if total_weight <= 0:
            return cls()

        def mean(name: str) -> float:
            return (
                sum(
                    float(getattr(value, name)) * weight
                    for value, weight in zip(values, resolved_weights, strict=True)
                )
                / total_weight
            )

        return cls(
            logprob_abs_diff_mean=mean("logprob_abs_diff_mean"),
            logprob_abs_diff_max=max(value.logprob_abs_diff_max for value in values),
            ratio_abs_dev_mean=mean("ratio_abs_dev_mean"),
            ratio_abs_dev_max=max(value.ratio_abs_dev_max for value in values),
            mismatch_kl=mean("mismatch_kl"),
            mismatch_k3_kl=mean("mismatch_k3_kl"),
            finite=all(value.finite for value in values),
        )

    @classmethod
    def compute(
        cls,
        fresh_log_prob: torch.Tensor,
        old_log_prob: torch.Tensor,
    ) -> LogprobMismatchStats:
        """Compute drift statistics between replay and rollout-behavior log-probabilities.

        ``fresh_log_prob`` is the freshly recomputed replay logprob (compute dtype);
        ``old_log_prob`` is the rollout behavior logprob (rollout dtype). Reductions run
        in fp32 so a bf16 input does not itself add noise to the measurement.
        """

        import torch

        fresh = fresh_log_prob.detach().to(torch.float32)
        old = old_log_prob.detach().to(torch.float32)
        if fresh.numel() == 0:
            return cls()

        delta = fresh - old  # replay - rollout
        abs_diff = delta.abs()
        ratio = torch.exp(delta)
        ratio_dev = (ratio - 1.0).abs()
        finite = bool(torch.isfinite(delta).all() and torch.isfinite(ratio).all())
        return cls(
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
    """Rollout->replay drift correction knobs (TIS + RS), all at the trainer level.

    The correction counterpart to ``PrecisionDriftGuardConfig``: the guard
    measures/fails on drift; this bounds it in the loss. ``old_log_prob`` IS the
    rollout (behavior) logprob in this codebase, so the surrogate ratio
    ``exp(replay - rollout)`` is exactly the importance weight TIS truncates and
    its log is what RS rejects on. Two orthogonal axes:

    **TIS** (per-element weight magnitude). ``off`` keeps legacy behavior;
    ``truncate`` = one-sided upper cap (keep the gradient, bound it); ``clip`` =
    two-sided; ``mask`` = drop out-of-range elements entirely. Judges the IS
    *weight* ``exp(replay - rollout)``: catches a single inflated weight.

    **RS** (whole-sequence off-policy rejection). Judges the *log-ratio*
    ``replay - rollout`` aggregated over the sequence and rejects the entire
    sample (weight 0) when its trajectory drift is out of band — the complement
    to TIS's "keep but clamp". Modes are ``seq_mean_k1`` / ``seq_max_k1`` only:
    on diffusion the SDE window is short (``sde_window_size`` is usually 2), so a
    per-token statistic has near-zero power (each per-step stat is already a mean
    over thousands of latent dims). ``token_*`` RS is intentionally *not*
    offered; it would only invite misuse. The continuous-diffusion evaluator
    exposes one per-sample scalar for the current denoise timestep, so the two
    seq modes coincide there (there is no sequence axis to reduce).

    **Bypass vs recompute** (``recompute_old_logprob``). This codebase already
    uses the rollout-recorded ``old_log_prob`` directly as the PPO "old" (bypass:
    the PPO ratio ``exp(replay - rollout)`` itself acts as the importance
    weight), skipping a separate full-precision old-recompute forward. ``off``
    (default) is that bypass and is the only implemented path. ``on`` is a
    reserved interface for a decoupled recompute that this codebase does not
    implement; constructing it raises rather than silently behaving like a no-op.

    **Combination contract.** Under fp8/bf16 rollout + bypass, drift must be
    bounded: run the drift guard (``auto``/``fail``, which checks parity before
    the first step) together with RS (``seq_mean_k1``), so the guard fail-stops
    on a catastrophic precision split while RS keeps out-of-band trajectories out
    of the per-step gradient. Both read the same ``LogprobMismatchStats``
    so their drift judgement cannot diverge. TIS and RS are orthogonal and may be
    enabled together.
    """

    tis_mode: str = field(default="off")  # "off" | "truncate" | "clip" | "mask"
    tis_imp_weight_cap: float = field(default=2.0)  # upper bound C on the weight
    tis_clip_low: float = field(default=0.0)  # lower bound (clip/mask modes)
    # RS axis (orthogonal to TIS). Whole-sequence rejection on the log-ratio.
    rs_mode: str = field(default="off")  # "off" | "seq_mean_k1" | "seq_max_k1"
    # Log-ratio acceptance band [low, high]. Stored as log-ratio numbers (not the
    # verl-omni "low_high" string), defaulting to ln(0.5)/ln(2.0) ~= [-0.69, 0.69]
    # to match verl-omni's recommended LLM preset.
    rs_log_ratio_low: float = field(default=math.log(0.5))
    rs_log_ratio_high: float = field(default=math.log(2.0))
    # Bypass (off, the only implemented path) vs a reserved decoupled-recompute
    # interface (on, unimplemented — raises on construction, never a no-op knob).
    recompute_old_logprob: str = field(default="off")  # "off" | "on"

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
        if self.rs_mode not in ("off", "seq_mean_k1", "seq_max_k1"):
            if str(self.rs_mode).startswith("token"):
                raise ValueError(
                    "precision_correction.rs_mode does not support token-level RS: "
                    "on a short SDE window (usually 2 steps) a per-token statistic "
                    "has near-zero power. Use 'seq_mean_k1' or 'seq_max_k1' "
                    f"(got {self.rs_mode!r}).",
                )
            raise ValueError(
                "precision_correction.rs_mode must be off/seq_mean_k1/seq_max_k1; "
                f"got {self.rs_mode!r}",
            )
        if self.rs_mode != "off" and float(self.rs_log_ratio_low) >= float(self.rs_log_ratio_high):
            raise ValueError(
                "precision_correction.rs_log_ratio_low must be < rs_log_ratio_high "
                f"(got low={self.rs_log_ratio_low}, high={self.rs_log_ratio_high})",
            )
        if self.recompute_old_logprob not in ("off", "on"):
            raise ValueError(
                "precision_correction.recompute_old_logprob must be off/on; "
                f"got {self.recompute_old_logprob!r}",
            )
        if self.recompute_old_logprob == "on":
            raise NotImplementedError(
                "precision_correction.recompute_old_logprob='on' (decoupled "
                "full-precision old-recompute) is not implemented in this codebase: "
                "old_log_prob is the rollout-recorded behavior logprob (bypass). "
                "Keep 'off' (bypass) and bound drift with the drift guard + RS.",
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


def apply_rejection_sample_mask(
    log_ratio: torch.Tensor,
    config: PrecisionCorrectionConfig,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Whole-sequence reject-sampling (RS) mask on the rollout->replay log-ratio.

    ``log_ratio = replay_logprob - rollout_logprob`` (NOT the IS weight: RS judges
    the log-ratio, TIS judges ``exp(log_ratio)``). Returns a 0/1 keep mask whose
    rejected entries the caller drops from the loss denominator, or ``None`` when
    ``rs_mode == 'off'``. An entire sample is kept or rejected as one unit:

    - ``seq_mean_k1``: the mean log-ratio over the sequence axis must lie in
      ``[low, high]``.
    - ``seq_max_k1``: every step must lie in band (a single out-of-band step
      rejects the whole sequence) — the strictest mode.

    Shapes: per-sample log-ratio ``(B,)`` (one continuous-diffusion timestep)
    has no sequence axis, so both modes reduce to the same per-sample band check
    and the returned mask is ``(B,)``. Per-token log-ratio ``(B, L)`` reduces
    over ``L`` and the returned mask is ``(B, 1)`` so it broadcasts over the
    token axis. ``mask`` (token validity) restricts the reduction to real tokens
    when given.
    """

    import torch

    mode = config.rs_mode
    if mode == "off":
        return None
    low, high = config.rs_log_ratio_low, config.rs_log_ratio_high

    if log_ratio.dim() <= 1:
        # Continuous diffusion supplies one scalar per sample for the current
        # denoise timestep; seq_mean and seq_max therefore coincide here.
        keep = (log_ratio >= low) & (log_ratio <= high)
        return keep.to(log_ratio.dtype)

    if mode == "seq_mean_k1":
        if mask is not None:
            m = mask.to(log_ratio.dtype)
            denom = m.sum(dim=-1).clamp_min(1.0)
            agg = (log_ratio * m).sum(dim=-1) / denom
        else:
            agg = log_ratio.mean(dim=-1)
        keep = (agg >= low) & (agg <= high)
    else:  # seq_max_k1: reject if ANY in-band step is violated
        if mask is not None:
            valid = mask.to(torch.bool)
            # Neutralize padding: fill with an in-band value so it never drives
            # the min below low or the max above high.
            for_min = torch.where(valid, log_ratio, torch.full_like(log_ratio, high))
            for_max = torch.where(valid, log_ratio, torch.full_like(log_ratio, low))
        else:
            for_min = for_max = log_ratio
        keep = (for_min.min(dim=-1).values >= low) & (for_max.max(dim=-1).values <= high)

    return keep.unsqueeze(-1).to(log_ratio.dtype)


def combine_keep_masks(*masks: torch.Tensor | None) -> torch.Tensor | None:
    """Intersect optional 0/1 keep masks by multiplication; ``None`` entries skip.

    Returns ``None`` only if every input is ``None``. Shapes may broadcast (e.g.
    a per-token ``(B, L)`` validity mask times a per-sequence ``(B, 1)`` RS mask).
    """

    result: torch.Tensor | None = None
    for m in masks:
        if m is None:
            continue
        result = m if result is None else result * m
    return result


__all__ = [
    "LogprobMismatchStats",
    "LogprobMismatchStats",
    "PrecisionCorrectionConfig",
    "apply_rejection_sample_mask",
    "apply_truncated_importance_weight",
    "combine_keep_masks",
]
