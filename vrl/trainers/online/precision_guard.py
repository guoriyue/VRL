"""Precision drift guard: enforce/observe rollout-vs-replay logprob parity.

When rollout and replay use different forward precision policies, the collection
time logprob no longer equals the freshly recomputed replay logprob, so the GRPO
importance ratio is != 1 at the very first step. This guard recomputes parity on
the first training step (before any optimizer update) and either warns or fails,
using the shared :func:`compute_logprob_mismatch_stats`.

It is the enforcement side of the same fact the per-step mismatch metrics (P1)
report; both read the one shared stats helper so debug, metrics, and gate never
diverge on how drift is measured.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from vrl.algorithms.logprob_mismatch import (
    LogprobMismatchStats,
    compute_logprob_mismatch_stats,
)
from vrl.trainers.core.types import PrecisionDriftGuardConfig

_logger = logging.getLogger(__name__)


class PrecisionDriftError(RuntimeError):
    """Raised when rollout-vs-replay logprob drift exceeds the guard threshold."""


def resolve_guard_mode(
    mode: str,
    *,
    compute_precision: str,
    rollout_precision: str,
) -> str:
    """Resolve ``auto`` into an effective ``off``/``warn``/``fail`` mode.

    ``auto`` enables the guard only when rollout/compute precision differ — same-dtype
    first-step parity is already the debug probe's job, so ``auto`` stays off there to
    avoid a redundant replay forward. On a mismatch it resolves to ``fail``: explicit
    ``warn`` is the measurement escape hatch for calibration runs, not the default.
    Explicit ``off``/``warn``/``fail`` always apply regardless of precision.
    """

    if mode in ("off", "warn", "fail"):
        return mode
    if mode != "auto":
        raise ValueError(
            f"precision drift guard mode must be auto/off/warn/fail; got {mode!r}",
        )
    if _normalize(rollout_precision) != _normalize(compute_precision):
        return "fail"
    return "off"


def select_guard_timesteps(timestep_indices: Sequence[int], max_checks: int) -> list[int]:
    """Pick first / last / evenly-spaced-middle timesteps, up to ``max_checks``."""

    ordered = list(dict.fromkeys(int(t) for t in timestep_indices))
    if max_checks <= 0 or not ordered:
        return []
    if len(ordered) <= max_checks:
        return ordered
    if max_checks == 1:
        return [ordered[0]]
    last = len(ordered) - 1
    picks = sorted({round(i * last / (max_checks - 1)) for i in range(max_checks)})
    return [ordered[i] for i in picks]


def run_precision_drift_guard(
    config: PrecisionDriftGuardConfig,
    *,
    compute_precision: str,
    rollout_precision: str,
    math_precision: str,
    timestep_indices: Sequence[int],
    evaluate_fn: Callable[[int], Any],
    metadata: Mapping[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any] | None:
    """Check first-step parity across a few timesteps; warn or fail on drift.

    ``evaluate_fn(timestep_idx)`` returns a ``TrajectorySignalBatch``-like object with
    ``.primary.log_prob`` (fresh replay) and ``.primary.old_log_prob`` (rollout
    behavior). Returns a record dict (also suitable for jsonl) or ``None`` when off.
    """

    mode = resolve_guard_mode(
        config.mode,
        compute_precision=compute_precision,
        rollout_precision=rollout_precision,
    )
    if mode == "off":
        return None

    compute_label = _normalize(compute_precision)
    rollout_label = _normalize(rollout_precision)
    math_label = _normalize(math_precision)
    log = logger or _logger
    worst: LogprobMismatchStats | None = None
    worst_timestep = -1
    violated = False
    for timestep in select_guard_timesteps(timestep_indices, config.max_timestep_checks):
        primary = evaluate_fn(timestep).primary
        stats = compute_logprob_mismatch_stats(primary.log_prob, primary.old_log_prob)
        if (
            stats.ratio_abs_dev_max > config.max_ratio_abs_dev
            or stats.logprob_abs_diff_max > config.max_abs_log_ratio
            or (config.fail_on_nonfinite and not stats.finite)
        ):
            violated = True
        if worst is None or stats.ratio_abs_dev_max > worst.ratio_abs_dev_max:
            worst, worst_timestep = stats, timestep

    record: dict[str, Any] = {
        "event": "precision_drift_guard",
        "mode": mode,
        "violated": violated,
        "compute_precision": compute_label,
        "rollout_precision": rollout_label,
        "math_precision": math_label,
        "forward_precision_match": rollout_label == compute_label,
        "worst_timestep": worst_timestep,
        "max_abs_log_ratio": config.max_abs_log_ratio,
        "max_ratio_abs_dev": config.max_ratio_abs_dev,
        "worst_stats": dataclasses.asdict(worst) if worst is not None else None,
    }
    if metadata:
        metadata_dict = dict(metadata)
        record["metadata"] = metadata_dict
        # Promote scalar precision fields to top-level jsonl columns. Derive the
        # set from the payload itself so new precision fields added by the
        # trainer's metadata producer surface automatically, instead of silently
        # dropping out of a stale hand-maintained key list. setdefault keeps the
        # normalized compute/rollout/math labels already set above.
        for key, value in metadata_dict.items():
            if isinstance(value, (str, bool, int, float)) or value is None:
                record.setdefault(key, value)
    if violated:
        message = (
            "precision drift guard: rollout-vs-replay logprob parity exceeded "
            f"threshold (compute={compute_label}, rollout={rollout_label}, "
            f"math={math_label}); worst timestep={worst_timestep} "
            f"abs_log_diff_max={worst.logprob_abs_diff_max:.3e} "
            f"ratio_abs_dev_max={worst.ratio_abs_dev_max:.3e} "
            f"(limits abs_log_ratio={config.max_abs_log_ratio:.1e} "
            f"ratio_abs_dev={config.max_ratio_abs_dev:.1e}, finite={worst.finite})"
            if worst is not None
            else "precision drift guard: parity exceeded threshold"
        )
        if mode == "fail":
            raise PrecisionDriftError(message)
        log.warning(message)
    return record


def _normalize(precision: str) -> str:
    token = str(precision or "").strip().lower()
    return "fp32" if token in ("", "no") else token


__all__ = [
    "PrecisionDriftError",
    "resolve_guard_mode",
    "run_precision_drift_guard",
    "select_guard_timesteps",
]
