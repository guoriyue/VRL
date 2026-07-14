"""Shared async polling and owner-observation helpers for continuous rollout tests."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from vrl.rollouts.orchestration.continuous.owner import (
    ContinuousRolloutOwner,
    _await_owner_future,
)
from vrl.rollouts.orchestration.continuous.types import ContinuousRolloutProducerState


async def _wait_until(condition: Callable[[], bool], timeout_s: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not condition():
        if loop.time() >= deadline:
            raise AssertionError("condition not reached before timeout")
        await asyncio.sleep(0.001)


@dataclass(frozen=True, slots=True)
class OwnerSnapshot:
    """Immutable copy of owner-loop state, observed only by tests."""

    producer_state: ContinuousRolloutProducerState | None
    queue_stats: dict[str, float]
    prompts: tuple[Any, ...]
    terminal_error: str | None


async def owner_snapshot(owner: ContinuousRolloutOwner) -> OwnerSnapshot:
    """Copy producer/queue state off the owner loop without racing it.

    The owner runs on its own thread and event loop; the copy must execute as
    a coroutine on that loop (the same loop as the producer) to stay race-free,
    so it is marshalled with run_coroutine_threadsafe exactly like the
    production owner commands. Production has no diagnostic read surface, so
    this observation seam lives with the tests that need it.
    """

    runtime, loop = owner._ensure_thread()

    async def _copy() -> OwnerSnapshot:
        producer = runtime.producer
        return OwnerSnapshot(
            producer_state=(None if producer is None else replace(producer.state)),
            queue_stats={} if runtime.queue is None else dict(runtime.queue.stats()),
            prompts=() if producer is None else tuple(producer.prompts),
            terminal_error=(
                None if runtime._terminal_error is None else repr(runtime._terminal_error)
            ),
        )

    future = asyncio.run_coroutine_threadsafe(_copy(), loop)
    return await _await_owner_future(future)
