"""One generation engine replica: the driver-side handle on its rank actor.

An engine is the data-parallel unit the driver talks to — the dispatch target,
weight-sync target, and health-verdict unit. Its rank is the per-GPU worker
actor behind it (``RayGenerationWorker``); one engine owns one rank, so an
engine call is that actor's call and its result is that rank's result.

Lifecycle stays rank-level on purpose: launching, killing, and liveness-probing
operate on rank actors (``RayActorGroup`` / the health monitor); the engine
carries the id the dispatcher, executor and session address it by, and vouches
for its rank's parking report.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vrl.generation.execution.types import WorkerMemoryParkingSnapshot
from vrl.ray.actor_group import RayActorHandle


class RayGenerationEngine:
    """Driver-side engine object over its rank actor handle."""

    __slots__ = ("engine_id", "rank")

    def __init__(self, engine_id: str, rank: RayActorHandle) -> None:
        if not engine_id:
            raise ValueError("engine_id must be non-empty")
        if not isinstance(rank, RayActorHandle):
            raise TypeError(
                f"engine {engine_id!r} requires one RayActorHandle, got {type(rank).__name__}",
            )
        self.engine_id = engine_id
        self.rank = rank

    def remote(self, method_name: str) -> Callable[..., Any]:
        """Submission surface for the dispatcher: the rank actor's own ``.remote``,
        so completion and cancellation timing are the rank's."""

        return getattr(self.rank.actor, method_name).remote

    async def sleep(self) -> WorkerMemoryParkingSnapshot:
        """Park the rank and return its validated physical-memory evidence."""

        value = await self.rank.actor.sleep.remote()
        if not isinstance(value, WorkerMemoryParkingSnapshot):
            raise TypeError(
                f"rank {self.rank.worker_id!r} returned invalid memory-parking "
                f"report {type(value).__name__}",
            )
        if value.worker_id != self.rank.worker_id:
            raise RuntimeError(
                "mismatched rank memory-parking report: "
                f"expected={self.rank.worker_id!r} actual={value.worker_id!r}",
            )
        value.validate()
        return value

    async def wake(self) -> None:
        """Restore the parked rank onto its assigned GPU."""

        await self.rank.actor.wake.remote()


__all__ = ["RayGenerationEngine"]
