"""Per-request rollout stats accumulator.

One typed object (``RolloutStats``) carries a request's phase wall-clock
timings and its reward-inference timings as it flows
generation -> reward -> trainer, replacing the hand-threaded
``phase_times: dict[str, float]`` that was passed through ~12 files.

Two design choices make this robust:

* The accumulator travels *on* the rollout item/iteration, so per-request
  timings live with the request they describe instead of in a side dict on
  the scheduler. Concurrent collects never share mutable state, and the
  timings serialize naturally with the item.
* Stats are accumulated into the typed object and emitted by the trainer at
  the end of a step, without changing the collection sites.

Metric keys are a *dynamic* namespace (``collect.*``, ``continuous.*``,
``advantage``, ``backward``, ``optim_step``, plus model-family phases), so they
stay string-keyed maps. The typed object keeps their reduction semantics
explicit: phase durations and counters sum, while gauge observations retain
their peak across merged microbatches.
"""

from __future__ import annotations

import contextlib
import math
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field


@dataclass(slots=True)
class RolloutStats:
    """Typed per-request stats that travel with a rollout item/iteration.

    ``phase_seconds`` accumulates namespaced wall-clock phase timings; a
    reward call's millisecond timings enter it as ``reward.<name>_s`` phases.
    """

    phase_seconds: dict[str, float] = field(default_factory=dict)
    counters: dict[str, float] = field(default_factory=dict)
    gauges: dict[str, float] = field(default_factory=dict)
    # ``(name, wall start, wall end)`` of every profiled phase, for the
    # trainer's phase_events.jsonl; taken and cleared when written.
    phase_events: list[tuple[str, float, float]] = field(default_factory=list, repr=False)

    def add_phase(self, name: str, seconds: float) -> None:
        """Accumulate ``seconds`` under phase ``name`` (sums on repeat)."""

        self.phase_seconds[name] = self.phase_seconds.get(name, 0.0) + float(seconds)

    def add_phases(self, phases: Mapping[str, float]) -> None:
        """Accumulate every ``(name, seconds)`` pair from ``phases``."""

        for name, seconds in phases.items():
            self.add_phase(name, seconds)

    @contextlib.contextmanager
    def phase(self, name: str, *, profile: bool = False) -> Iterator[None]:
        """Time the enclosed block into phase ``name``.

        Recorded in a ``finally`` so a failing phase still reports the wall
        clock it burned before raising. ``profile`` is the trainer's GPU-phase
        mode: CUDA is synchronized on both ends so asynchronous kernels are
        counted, and the phase's wall-clock interval lands in ``phase_events``.
        """

        if profile:
            _synchronize_cuda()
        start = time.perf_counter()
        wall_start = time.time()
        try:
            yield
        finally:
            if profile:
                _synchronize_cuda()
            self.add_phase(name, time.perf_counter() - start)
            if profile:
                self.phase_events.append((name, wall_start, time.time()))

    def take_phase_events(self) -> list[tuple[str, float, float]]:
        """Hand over the recorded phase intervals and forget them."""

        events, self.phase_events = self.phase_events, []
        return events

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
        self.phase_events.extend(other.phase_events)
        for name, value in other.counters.items():
            self.add_counter(name, value)
        self.observe_gauges(other.gauges)

    def fold_reward_timing(self, timing_ms: Mapping[str, float]) -> None:
        """Accumulate one reward call's ``*_ms`` timings as ``reward.<name>_s`` phases.

        ``RewardOutput`` already validated every value as finite and
        non-negative; a key that does not end in ``_ms`` is not a timing and
        is ignored.
        """

        timings = {name: value for name, value in timing_ms.items() if name.endswith("_ms")}
        if not timings:
            return
        self.add_counter("reward.call_count")
        for name, milliseconds in timings.items():
            self.add_phase(f"reward.{name[:-3]}_s", float(milliseconds) / 1000.0)

    def as_metrics_dict(self) -> dict[str, float]:
        """Flat metric view combining durations, counters and peak gauges."""

        return {**self.phase_seconds, **self.counters, **self.gauges}


def _synchronize_cuda() -> None:
    """Wait for the card's queued kernels, when there is a card."""

    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


__all__ = ["RolloutStats"]
