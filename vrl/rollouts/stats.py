"""Per-request rollout stats accumulator + pluggable sinks.

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
  object, and a pluggable ``StatsSink`` decides where they go (a log line
  today, a jsonl/Prometheus sink later) without touching the accumulation
  sites.

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
from typing import Protocol


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

        if not name:
            raise ValueError("phase name must be non-empty")
        self.phase_seconds[name] = self.phase_seconds.get(name, 0.0) + float(seconds)

    def add_phases(self, phases: Mapping[str, float]) -> None:
        """Accumulate every ``(name, seconds)`` pair from ``phases``."""

        for name, seconds in phases.items():
            self.add_phase(str(name), float(seconds))

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

        if not name:
            raise ValueError("counter name must be non-empty")
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

        if not name:
            raise ValueError("gauge name must be non-empty")
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError(f"gauge {name!r} must be finite")
        self.gauges[name] = max(self.gauges.get(name, normalized), normalized)

    def observe_gauges(self, gauges: Mapping[str, float]) -> None:
        """Record every gauge observation in ``gauges``."""

        for name, value in gauges.items():
            self.observe_gauge(str(name), float(value))

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
        """Accumulate one reward call's timings (primitives, no reward import)."""

        latency = _timing_value("latency_ms", latency_ms)
        queue_wait = _timing_value("queue_wait_ms", queue_wait_ms)
        inference = _timing_value("inference_ms", inference_ms)
        normalized_extra: dict[str, float] = {}
        for name, milliseconds in dict(extra_ms or {}).items():
            if not name or not name.endswith("_ms"):
                raise ValueError("extra reward timing names must end with '_ms'")
            value = _timing_value(name, milliseconds)
            assert value is not None
            normalized_extra[name] = value

        if any(value is not None for value in (latency, queue_wait, inference)) or (
            normalized_extra
        ):
            self.add_counter("reward.call_count")
        if latency is not None:
            self._reward_latency_samples_ms.append(latency)
        if queue_wait is not None:
            self.reward_queue_wait_ms = _sum_optional(
                self.reward_queue_wait_ms,
                queue_wait,
            )
        if inference is not None:
            self.reward_inference_ms = _sum_optional(
                self.reward_inference_ms,
                inference,
            )
        for name, value in normalized_extra.items():
            self.reward_extra_ms[name] = self.reward_extra_ms.get(name, 0.0) + value

    def as_phase_dict(self) -> dict[str, float]:
        """Flat phase->seconds view, including reward timings as seconds.

        Reward millisecond fields surface as ``reward.<name>_s`` so existing
        phase-line consumers see them alongside the wall-clock phases without
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


def _timing_value(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"reward timing {name!r} must be finite and non-negative")
    return normalized


class StatsSink(Protocol):
    """Where recorded stats go. Recording is decoupled from emitting."""

    def record(self, step: int, stats: RolloutStats) -> None: ...


class LoggingStatsSink:
    """Emit the per-step phase-timing log line (the historical behavior).

    Replaces the inline ``logger.info("phase_times[step=...]")`` in the
    trainer. ``collect.*`` phases are excluded from the percentage base so the
    breakdown matches the previous output exactly.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger("vrl.stats")

    def record(self, step: int, stats: RolloutStats) -> None:
        phases = stats.as_phase_dict()
        if not phases:
            return
        percentage_phases = {
            name: seconds
            for name, seconds in stats.phase_seconds.items()
            if not name.startswith("collect.")
        }
        total = sum(percentage_phases.values())
        if total <= 0:
            self._logger.info("phase_times[step=%d] total=0.000s", step)
            return
        parts = " | ".join(
            (
                f"{name}={value:.3f}s ({100 * value / total:.1f}%)"
                if name in percentage_phases
                else f"{name}={value:.3f}"
            )
            for name, value in phases.items()
        )
        self._logger.info("phase_times[step=%d] total=%.3fs | %s", step, total, parts)


class JsonlStatsSink:
    """Append one JSON object per step to a JSONL file.

    ``metrics.csv`` exposes a stable subset of continuous-health diagnostics,
    but not arbitrary ``collect.*`` phases. A benchmark comparing collection
    arms needs the complete phase mapping per step as data, not as text to
    re-parse out of a log.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def record(self, step: int, stats: RolloutStats) -> None:
        phases = stats.as_phase_dict()
        if not phases:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": int(step), **phases}, sort_keys=True) + "\n")


class MultiStatsSink:
    """Fan one record out to several sinks, keeping one sink per owner."""

    def __init__(self, *sinks: StatsSink) -> None:
        self._sinks = sinks

    def record(self, step: int, stats: RolloutStats) -> None:
        for sink in self._sinks:
            sink.record(step, stats)


__all__ = [
    "JsonlStatsSink",
    "LoggingStatsSink",
    "MultiStatsSink",
    "RolloutStats",
    "StatsSink",
]
