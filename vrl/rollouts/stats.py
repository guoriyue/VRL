"""Per-request rollout stats accumulator and its per-step emitter.

One typed object (``RolloutStats``) carries a request's phase wall-clock
timings and its reward-inference timings as it flows
generation -> reward -> trainer, replacing the hand-threaded
``phase_times: dict[str, float]`` that was passed through ~12 files.

Two design choices make this robust:

* The accumulator travels *on* the rollout item/iteration, so per-request
  timings live with the request they describe instead of in a side dict on
  the scheduler. Concurrent collects never share mutable state, and the
  timings serialize naturally with the item.
* Recording is decoupled from emitting: stats are accumulated into the typed
  object, and ``record_step_stats`` writes a step's log line and JSONL row
  without touching the accumulation sites.

Metric keys are a *dynamic* namespace (``collect.*``, ``continuous.*``,
``advantage``, ``backward``, ``optim_step``, plus model-family phases), so they
stay string-keyed maps. The typed object keeps their reduction semantics
explicit: phase durations and counters sum, while gauge observations retain
their peak across merged microbatches.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class RolloutStats:
    """Typed per-request stats that travel with a rollout item/iteration.

    ``phase_seconds`` accumulates namespaced wall-clock phase timings.
    Reward-inference timings are folded in as typed millisecond fields so the
    already-typed ``RewardInferenceResult`` numbers stop being a separate,
    disconnected channel.
    """

    phase_seconds: dict[str, float] = field(default_factory=dict)
    counters: dict[str, float] = field(default_factory=dict)
    gauges: dict[str, float] = field(default_factory=dict)
    reward_queue_wait_ms: float | None = None
    reward_inference_ms: float | None = None
    reward_extra_ms: dict[str, float] = field(default_factory=dict)
    _reward_latency_samples_ms: list[float] = field(
        default_factory=list,
        repr=False,
    )

    def add_phase(self, name: str, seconds: float) -> None:
        """Accumulate ``seconds`` under phase ``name`` (sums on repeat)."""

        self.phase_seconds[name] = self.phase_seconds.get(name, 0.0) + float(seconds)

    def add_phases(self, phases: Mapping[str, float]) -> None:
        """Accumulate every ``(name, seconds)`` pair from ``phases``."""

        for name, seconds in phases.items():
            self.add_phase(name, seconds)

    @contextlib.contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Time the enclosed block into phase ``name``.

        Recorded in a ``finally`` so a failing phase still reports the wall
        clock it burned before raising.
        """

        start = time.perf_counter()
        try:
            yield
        finally:
            self.add_phase(name, time.perf_counter() - start)

    def add_counter(self, name: str, value: float = 1.0) -> None:
        """Accumulate a unitless count without treating it as phase time."""

        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError(f"counter {name!r} must be finite")
        self.counters[name] = self.counters.get(name, 0.0) + normalized

    def observe_gauge(self, name: str, value: float) -> None:
        """Record a point-in-time value, retaining the peak when merged.

        Continuous rollout diagnostics are state snapshots, not elapsed time or
        deltas. Keeping the peak makes a streamed optimizer update report its
        worst observed staleness, queue depth, and producer gap instead of
        summing snapshots into impossible values.
        """

        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError(f"gauge {name!r} must be finite")
        self.gauges[name] = max(self.gauges.get(name, normalized), normalized)

    def observe_gauges(self, gauges: Mapping[str, float]) -> None:
        """Record every gauge observation in ``gauges``."""

        for name, value in gauges.items():
            self.observe_gauge(name, value)

    def merge(self, other: RolloutStats) -> None:
        """Fold another accumulator into this one without losing concurrent calls."""

        self.add_phases(other.phase_seconds)
        for name, value in other.counters.items():
            self.add_counter(name, value)
        self.observe_gauges(other.gauges)
        self.reward_queue_wait_ms = _sum_optional(
            self.reward_queue_wait_ms,
            other.reward_queue_wait_ms,
        )
        self.reward_inference_ms = _sum_optional(
            self.reward_inference_ms,
            other.reward_inference_ms,
        )
        self._reward_latency_samples_ms.extend(other._reward_latency_samples_ms)
        for name, milliseconds in other.reward_extra_ms.items():
            self.reward_extra_ms[name] = self.reward_extra_ms.get(name, 0.0) + float(
                milliseconds,
            )

    def fold_reward_timing(
        self,
        *,
        latency_ms: float | None = None,
        queue_wait_ms: float | None = None,
        inference_ms: float | None = None,
        extra_ms: Mapping[str, float] | None = None,
    ) -> None:
        """Accumulate one reward call's timings (primitives, no reward import).

        ``RewardOutput`` already validated every value as finite and
        non-negative, and the collector passes only non-standard ``*_ms``
        names in ``extra_ms``.
        """

        extra = dict(extra_ms or {})
        if any(value is not None for value in (latency_ms, queue_wait_ms, inference_ms)) or extra:
            self.add_counter("reward.call_count")
        if latency_ms is not None:
            self._reward_latency_samples_ms.append(float(latency_ms))
        self.reward_queue_wait_ms = _sum_optional(self.reward_queue_wait_ms, queue_wait_ms)
        self.reward_inference_ms = _sum_optional(self.reward_inference_ms, inference_ms)
        for name, milliseconds in extra.items():
            self.reward_extra_ms[name] = self.reward_extra_ms.get(name, 0.0) + float(milliseconds)

    def as_metrics_dict(self) -> dict[str, float]:
        """Flat metric view combining durations, counters and peak gauges.

        Reward millisecond fields surface as ``reward.<name>_s`` so existing
        metric consumers see them alongside the wall-clock phases without
        a second mechanism.
        """

        out = dict(self.phase_seconds)
        out.update(self.counters)
        out.update(self.gauges)
        for key, ms in (
            ("reward.queue_wait_s", self.reward_queue_wait_ms),
            ("reward.inference_s", self.reward_inference_ms),
        ):
            if ms is not None:
                out[key] = float(ms) / 1000.0
        if self._reward_latency_samples_ms:
            ordered = sorted(self._reward_latency_samples_ms)
            out["reward.latency_s"] = sum(ordered) / 1000.0
            out["reward.latency_p50_s"] = ordered[math.ceil(0.50 * len(ordered)) - 1] / 1000.0
            out["reward.latency_p95_s"] = ordered[math.ceil(0.95 * len(ordered)) - 1] / 1000.0
        for name, milliseconds in self.reward_extra_ms.items():
            out[f"reward.{name[:-3]}_s"] = float(milliseconds) / 1000.0
        return out


def _sum_optional(left: float | None, right: float | None) -> float | None:
    if right is None:
        return left
    return float(right) if left is None else float(left) + float(right)


def record_step_stats(
    step: int,
    stats: RolloutStats,
    *,
    jsonl_path: str | Path,
    logger: logging.Logger,
) -> None:
    """Log one step's phase timings and append them as one JSONL row.

    The log line is the human-facing view: ``collect.*`` phases are excluded
    from its percentage base. The JSONL row is the complete machine-readable
    view; ``metrics.csv`` exposes only the stable continuous-health subset,
    not arbitrary ``collect.*`` phases. Empty stats emit nothing.
    """

    metrics = stats.as_metrics_dict()
    if not metrics:
        return
    percentage_phases = {
        name: seconds
        for name, seconds in stats.phase_seconds.items()
        if not name.startswith("collect.")
    }
    total = sum(percentage_phases.values())
    if total <= 0:
        total = 0.0
        percentage_phases = {}
    parts = " | ".join(
        (
            f"{name}={value:.3f}s ({100 * value / total:.1f}%)"
            if name in percentage_phases
            else f"{name}={value:.3f}"
        )
        for name, value in metrics.items()
    )
    logger.info("phase_times[step=%d] total=%.3fs | %s", step, total, parts)

    path = Path(jsonl_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": int(step), **metrics}, sort_keys=True) + "\n")


__all__ = [
    "RolloutStats",
    "record_step_stats",
]
