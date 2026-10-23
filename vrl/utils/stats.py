"""Stats: one accumulator per scope, written to from wherever the work runs.

``Stats`` holds wall-clock phases, counters and peak gauges. Metric keys are a
dynamic namespace (``collect.*``, ``reward.*``, ``continuous.*``,
``advantage``, ``backward``, ...), so they stay string-keyed maps; the object
keeps the reduction explicit: phases and counters sum, gauges keep their peak
across merges.

A measurement site does not take an accumulator argument and no class or
result type grows a stats field. The owner of a scope binds one with
``recording()``; below it, ``timed`` (a block, or a whole function as a
decorator), ``elapsed``, ``count`` and ``gauge`` write to whatever accumulator
is bound, and do nothing when none is. The binding is a ``ContextVar``: it
follows asyncio tasks created inside the scope and ``asyncio.to_thread``, but
not a plain thread or ``run_in_executor``, whose measurements are dropped
unless that thread binds its own scope. The accumulator is a plain object: it
takes one writer at a time, which the event loop gives tasks for free.

A boundary the binding cannot cross carries the accumulator itself: a Ray
actor method decorated with ``recorded`` binds a fresh one around its work and
returns ``Recorded(value, stats)``, and the caller's ``unwrap`` folds the stats
into its own scope and hands back the value. ``absorb`` folds in an
accumulator that was kept as data, such as a continuous rollout item's.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import math
import time
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Stats:
    """Namespaced phase seconds, counters and peak gauges of one scope."""

    phase_seconds: dict[str, float] = field(default_factory=dict)
    counters: dict[str, float] = field(default_factory=dict)
    gauges: dict[str, float] = field(default_factory=dict)
    # ``(name, wall start, wall end)`` of every profiled phase, for the
    # trainer's phase_events.jsonl; taken and cleared when written.
    phase_events: list[tuple[str, float, float]] = field(default_factory=list, repr=False)

    def add_phase(self, name: str, seconds: float) -> None:
        """Accumulate ``seconds`` under phase ``name`` (sums on repeat)."""

        self.phase_seconds[name] = self.phase_seconds.get(name, 0.0) + float(seconds)

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

    def merge(self, other: Stats) -> None:
        """Fold another accumulator into this one without losing concurrent calls."""

        for name, seconds in other.phase_seconds.items():
            self.add_phase(name, seconds)
        self.phase_events.extend(other.phase_events)
        for name, value in other.counters.items():
            self.add_counter(name, value)
        for name, value in other.gauges.items():
            self.observe_gauge(name, value)

    def take_phase_events(self) -> list[tuple[str, float, float]]:
        """Hand over the recorded phase intervals and forget them."""

        events, self.phase_events = self.phase_events, []
        return events

    def as_metrics_dict(self) -> dict[str, float]:
        """Flat metric view combining durations, counters and peak gauges."""

        return {**self.phase_seconds, **self.counters, **self.gauges}


def _synchronize_cuda() -> None:
    """Wait for the card's queued kernels, when there is a card."""

    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


_RECORDING: ContextVar[Stats | None] = ContextVar("vrl_stats", default=None)


@contextlib.contextmanager
def recording(stats: Stats | None = None) -> Iterator[Stats]:
    """Bind ``stats`` (a fresh one when omitted) as the accumulator writes land in.

    A nested ``recording`` opens its own scope: writes inside it reach the
    inner accumulator only, until the caller folds it in with ``absorb``.
    """

    bound = Stats() if stats is None else stats
    token = _RECORDING.set(bound)
    try:
        yield bound
    finally:
        _RECORDING.reset(token)


class timed:
    """Time a block, or every call of a function, as phase ``name``; repeats sum.

    ``with timed(name):`` times the block; ``@timed(name)`` times each call of
    a sync function, an async function (the awaited body, not the coroutine's
    creation) or a method. The accumulator is the one bound when the block
    starts; with none bound this does nothing. Recorded in a ``finally`` so a
    failing block still reports the time it took. ``profile`` is the trainer's
    GPU-phase mode: CUDA is synchronized on both ends so asynchronous kernels
    are counted, and the phase's wall-clock interval lands in ``phase_events``.
    """

    __slots__ = ("_start", "_stats", "_wall_start", "name", "profile")

    def __init__(self, name: str, *, profile: bool = False) -> None:
        self.name = name
        self.profile = profile
        self._stats: Stats | None = None
        self._start = 0.0
        self._wall_start = 0.0

    def __enter__(self) -> None:
        self._stats = _RECORDING.get()
        if self._stats is None:
            return
        if self.profile:
            _synchronize_cuda()
        self._start = time.perf_counter()
        self._wall_start = time.time()

    def __exit__(self, *exc_info: object) -> None:
        stats, self._stats = self._stats, None
        if stats is None:
            return
        if self.profile:
            _synchronize_cuda()
        stats.add_phase(self.name, time.perf_counter() - self._start)
        if self.profile:
            stats.phase_events.append((self.name, self._wall_start, time.time()))

    def __call__[**P, R](self, function: Callable[P, R]) -> Callable[P, R]:
        # Each call times itself with its own context, so concurrent calls of
        # one decorated function never share a start time.
        name, profile = self.name, self.profile
        if inspect.iscoroutinefunction(function):

            @functools.wraps(function)
            async def timed_coroutine(*args: Any, **kwargs: Any) -> Any:
                with timed(name, profile=profile):
                    return await function(*args, **kwargs)

            return timed_coroutine  # type: ignore[return-value]

        @functools.wraps(function)
        def timed_call(*args: P.args, **kwargs: P.kwargs) -> R:
            with timed(name, profile=profile):
                return function(*args, **kwargs)

        return timed_call


def elapsed(name: str, seconds: float) -> None:
    """Add an already measured duration to phase ``name`` of the bound accumulator."""

    stats = _RECORDING.get()
    if stats is not None:
        stats.add_phase(name, seconds)


def count(name: str, value: float = 1.0) -> None:
    """Add ``value`` to counter ``name`` of the bound accumulator."""

    stats = _RECORDING.get()
    if stats is not None:
        stats.add_counter(name, value)


def gauge(name: str, value: float) -> None:
    """Observe ``value`` for gauge ``name`` of the bound accumulator; the peak is kept."""

    stats = _RECORDING.get()
    if stats is not None:
        stats.observe_gauge(name, value)


def absorb(stats: Stats) -> None:
    """Fold an accumulator from another scope, thread or process into the bound one."""

    bound = _RECORDING.get()
    if bound is not None:
        bound.merge(stats)


@dataclass(frozen=True, slots=True)
class Recorded[T]:
    """A call's return value with what the call recorded, for a boundary the binding cannot cross."""

    value: T
    stats: Stats


def recorded[**P, T](function: Callable[P, T]) -> Callable[P, Recorded[T]]:
    """Run each call in its own scope and return ``Recorded(value, stats)``.

    For Ray actor methods (sync or async) and any other entry point whose
    caller sits in another process, thread or event loop; the caller hands the
    result to ``unwrap``. A raising call returns nothing, so its stats are
    dropped with it.
    """

    if inspect.iscoroutinefunction(function):

        @functools.wraps(function)
        async def recorded_coroutine(*args: Any, **kwargs: Any) -> Any:
            with recording() as stats:
                value = await function(*args, **kwargs)
            return Recorded(value, stats)

        return recorded_coroutine  # type: ignore[return-value]

    @functools.wraps(function)
    def recorded_call(*args: P.args, **kwargs: P.kwargs) -> Recorded[T]:
        with recording() as stats:
            value = function(*args, **kwargs)
        return Recorded(value, stats)

    return recorded_call


def unwrap[T](result: Recorded[T]) -> T:
    """Fold ``result.stats`` into the bound accumulator and return ``result.value``."""

    absorb(result.stats)
    return result.value


__all__ = [
    "Recorded",
    "Stats",
    "absorb",
    "count",
    "elapsed",
    "gauge",
    "recorded",
    "recording",
    "timed",
    "unwrap",
]
