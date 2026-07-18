"""Precision drift guard: enforce/observe rollout-vs-replay logprob parity.

When rollout and replay use different role precision policies, the collection
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
import math
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
    training_precision: str,
    rollout_precision: str,
) -> str:
    """Resolve ``auto`` into an effective ``off``/``warn``/``fail`` mode.

    ``auto`` enables the guard when the role execution label differs, including
    rollout-only quantization.
    """

    if mode in ("off", "warn", "fail"):
        return mode
    if mode != "auto":
        raise ValueError(
            f"precision drift guard mode must be auto/off/warn/fail; got {mode!r}",
        )
    role_match = _normalize_precision_label(rollout_precision) == _normalize_precision_label(
        training_precision,
    )
    return "off" if role_match else "fail"


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


def _drift_record_key(record: Mapping[str, Any]) -> tuple[bool, bool, float, float, float]:
    """Order complete records without combining fields from different producers."""

    def relative(value: Any, limit: Any) -> float:
        measured = float(value)
        threshold = float(limit)
        if not math.isfinite(measured):
            return float("inf")
        if threshold > 0:
            return measured / threshold
        return float("inf") if measured > 0 else 0.0

    stats = record.get("worst_stats") or {}
    abs_score = relative(
        stats.get("logprob_abs_diff_max", 0.0),
        record.get("max_abs_log_ratio", 0.0),
    )
    ratio_score = relative(
        stats.get("ratio_abs_dev_max", 0.0),
        record.get("max_ratio_abs_dev", 0.0),
    )
    finite = bool(stats.get("finite", True))
    return (
        bool(record.get("violated", False)),
        not finite,
        max(abs_score, ratio_score),
        abs_score,
        ratio_score,
    )


def _select_distributed_worst_record(record: dict[str, Any]) -> dict[str, Any]:
    """All-gather and select one whole rank record on every process.

    A field-wise max would fabricate a record whose timestep and mean statistics
    came from one rank while its maxima came from another. Ranking complete records
    keeps provenance coherent; ties retain the lower-rank all-gather order.
    """

    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return record

    world_size = int(dist.get_world_size())
    local = dict(record)
    local["worst_rank"] = int(dist.get_rank())
    local["world_size"] = world_size
    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, local)
    if not all(isinstance(item, Mapping) for item in gathered):
        raise RuntimeError("precision drift all-gather returned a non-record payload")
    selected = max(gathered, key=_drift_record_key)
    result = dict(selected)
    result["violated"] = any(bool(item.get("violated", False)) for item in gathered)
    return result


def measure_precision_drift(
    config: PrecisionDriftGuardConfig,
    *,
    training_precision: str,
    rollout_precision: str,
    math_precision: str,
    timestep_indices: Sequence[int],
    evaluate_fn: Callable[[int], Any],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Measure first-step precision drift without choosing process-wide failure.

    ``evaluate_fn(timestep_idx)`` returns a ``TrajectorySignalBatch``-like object with
    ``.primary.log_prob`` (fresh replay) and ``.primary.old_log_prob`` (rollout
    behavior). Under distributed training every rank must call this function; it
    all-gathers complete records and returns the same worst-rank record everywhere
    before the caller applies warn/fail enforcement.
    """

    mode = resolve_guard_mode(
        config.mode,
        training_precision=training_precision,
        rollout_precision=rollout_precision,
    )
    if mode == "off":
        return None

    training_label = _normalize_precision_label(training_precision)
    rollout_label = _normalize_precision_label(rollout_precision)
    math_label = _normalize_precision_label(math_precision)
    worst: LogprobMismatchStats | None = None
    worst_timestep = -1
    worst_key: tuple[bool, bool, float, float, float] | None = None
    violated = False
    for timestep in select_guard_timesteps(timestep_indices, config.max_timestep_checks):
        primary = evaluate_fn(timestep).primary
        stats = compute_logprob_mismatch_stats(primary.log_prob, primary.old_log_prob)
        stats_violated = (
            stats.ratio_abs_dev_max > config.max_ratio_abs_dev
            or stats.logprob_abs_diff_max > config.max_abs_log_ratio
            or (config.fail_on_nonfinite and not stats.finite)
        )
        if stats_violated:
            violated = True
        candidate = {
            "violated": stats_violated,
            "max_abs_log_ratio": config.max_abs_log_ratio,
            "max_ratio_abs_dev": config.max_ratio_abs_dev,
            "worst_stats": dataclasses.asdict(stats),
        }
        candidate_key = _drift_record_key(candidate)
        if worst_key is None or candidate_key > worst_key:
            worst, worst_timestep = stats, timestep
            worst_key = candidate_key

    record: dict[str, Any] = {
        "event": "precision_drift_guard",
        "mode": mode,
        "violated": violated,
        "training_precision": training_label,
        "rollout_precision": rollout_label,
        "math_precision": math_label,
        "role_precision_match": rollout_label == training_label,
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
        # normalized train/rollout/math labels already set above.
        for key, value in metadata_dict.items():
            if isinstance(value, (str, bool, int, float)) or value is None:
                record.setdefault(key, value)
    return _select_distributed_worst_record(record)


def enforce_precision_drift(
    record: Mapping[str, Any] | None,
    *,
    logger: logging.Logger | None = None,
) -> None:
    """Apply one already-resolved precision-drift verdict."""

    if record is None or not bool(record.get("violated", False)):
        return
    worst = record.get("worst_stats") or {}
    message = (
        "precision drift guard: rollout-vs-replay logprob parity exceeded "
        f"threshold (training_role={record.get('training_precision')}, "
        f"rollout_role={record.get('rollout_precision')}, "
        f"math={record.get('math_precision')}); "
        f"worst rank={record.get('worst_rank', 0)} "
        f"worst timestep={record.get('worst_timestep')} "
        f"abs_log_diff_max={float(worst.get('logprob_abs_diff_max', float('nan'))):.3e} "
        f"ratio_abs_dev_max={float(worst.get('ratio_abs_dev_max', float('nan'))):.3e} "
        f"(limits abs_log_ratio={float(record.get('max_abs_log_ratio', 0.0)):.1e} "
        f"ratio_abs_dev={float(record.get('max_ratio_abs_dev', 0.0)):.1e}, "
        f"finite={worst.get('finite')})"
    )
    if record.get("mode") == "fail":
        raise PrecisionDriftError(message)
    (logger or _logger).warning(message)


def run_precision_drift_guard(
    config: PrecisionDriftGuardConfig,
    *,
    training_precision: str,
    rollout_precision: str,
    math_precision: str,
    timestep_indices: Sequence[int],
    evaluate_fn: Callable[[int], Any],
    metadata: Mapping[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any] | None:
    """Single-process facade: measure, then immediately warn or fail."""

    record = measure_precision_drift(
        config,
        training_precision=training_precision,
        rollout_precision=rollout_precision,
        math_precision=math_precision,
        timestep_indices=timestep_indices,
        evaluate_fn=evaluate_fn,
        metadata=metadata,
    )
    enforce_precision_drift(record, logger=logger)
    return record


def _normalize_precision_label(precision: str) -> str:
    token = str(precision or "").strip().lower()
    return "fp32" if token in ("", "no") else token


__all__ = [
    "PrecisionDriftError",
    "enforce_precision_drift",
    "measure_precision_drift",
    "resolve_guard_mode",
    "run_precision_drift_guard",
    "select_guard_timesteps",
]
