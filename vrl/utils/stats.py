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

Phase keys are a *dynamic* namespace (``collect.*``, ``continuous.*``,
``advantage``, ``backward``, ``optim_step``, plus model-family phases), so the
timings stay a string-keyed map — but one that lives *inside* the typed object
with explicit ``add_phase`` / ``merge`` semantics, not a bare dict the caller
mutates directly.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
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
    reward_latency_ms: float | None = None
    reward_queue_wait_ms: float | None = None
    reward_inference_ms: float | None = None

    def add_phase(self, name: str, seconds: float) -> None:
        """Accumulate ``seconds`` under phase ``name`` (sums on repeat)."""

        if not name:
            raise ValueError("phase name must be non-empty")
        self.phase_seconds[name] = self.phase_seconds.get(name, 0.0) + float(seconds)

    def add_phases(self, phases: Mapping[str, float]) -> None:
        """Accumulate every ``(name, seconds)`` pair from ``phases``."""

        for name, seconds in phases.items():
            self.add_phase(str(name), float(seconds))

    def merge(self, other: RolloutStats) -> None:
        """Fold another accumulator into this one (phase sums; reward last-wins).

        Reward timings are call-level (one score_many per collect), so the
        last non-None value wins rather than summing — matching how the
        collector records reward_score on the first group only to avoid
        double-counting one wall time across a group.
        """

        self.add_phases(other.phase_seconds)
        if other.reward_latency_ms is not None:
            self.reward_latency_ms = other.reward_latency_ms
        if other.reward_queue_wait_ms is not None:
            self.reward_queue_wait_ms = other.reward_queue_wait_ms
        if other.reward_inference_ms is not None:
            self.reward_inference_ms = other.reward_inference_ms

    def fold_reward_timing(
        self,
        *,
        latency_ms: float | None = None,
        queue_wait_ms: float | None = None,
        inference_ms: float | None = None,
    ) -> None:
        """Record reward-inference timings (primitives, no reward-type import)."""

        if latency_ms is not None:
            self.reward_latency_ms = float(latency_ms)
        if queue_wait_ms is not None:
            self.reward_queue_wait_ms = float(queue_wait_ms)
        if inference_ms is not None:
            self.reward_inference_ms = float(inference_ms)

    def as_phase_dict(self) -> dict[str, float]:
        """Flat phase->seconds view, including reward timings as seconds.

        Reward millisecond fields surface as ``reward.<name>_s`` so existing
        phase-line consumers see them alongside the wall-clock phases without
        a second mechanism.
        """

        out = dict(self.phase_seconds)
        for key, ms in (
            ("reward.latency_s", self.reward_latency_ms),
            ("reward.queue_wait_s", self.reward_queue_wait_ms),
            ("reward.inference_s", self.reward_inference_ms),
        ):
            if ms is not None:
                out[key] = float(ms) / 1000.0
        return out

    @classmethod
    def from_phase_dict(cls, phases: Mapping[str, float] | None) -> RolloutStats:
        """Build a RolloutStats from a flat ``phase_times`` mapping."""

        stats = cls()
        if phases:
            stats.add_phases(phases)
        return stats


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
        total = sum(v for k, v in phases.items() if not k.startswith("collect."))
        if total <= 0:
            self._logger.info("phase_times[step=%d] total=0.000s", step)
            return
        parts = " | ".join(
            f"{k}={v:.3f}s ({100 * v / total:.1f}%)" for k, v in phases.items()
        )
        self._logger.info("phase_times[step=%d] total=%.3fs | %s", step, total, parts)


__all__ = [
    "LoggingStatsSink",
    "RolloutStats",
    "StatsSink",
]
