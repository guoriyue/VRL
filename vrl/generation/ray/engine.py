"""One generation engine replica: driver-side composition of its rank actors.

An engine is the data-parallel unit the driver talks to — the dispatch target,
weight-sync target, and health-verdict unit. A rank is one per-GPU worker actor
inside it (``RayGenerationWorker``). Today every engine owns exactly one rank
(``ROLLOUT_GPUS_PER_ENGINE = 1``); the fan-out/aggregate semantics here are
generic over the rank count so a multi-GPU engine backend only changes the
rank program, never the driver chain.

Lifecycle stays rank-level on purpose: launching, killing, and liveness-probing
operate on rank actors (``RayActorGroup`` / the health monitor); the engine
only derives the verdict "any dead rank == dead engine".
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Any

from vrl.generation.execution.types import WorkerMemoryParkingSnapshot
from vrl.ray.actor_group import RayActorHandle
from vrl.ray.operation_deadline import cancel_ray_refs


class EngineCallRef:
    """Awaitable aggregate over one engine call's per-rank Ray refs.

    All ranks must succeed. The combined result is ``combine(results)``
    (default: rank 0's result). Any rank failure cancels the sibling refs and
    re-raises immediately — waiting only on rank 0 would turn a crashed
    non-zero rank into a hang once ranks run collectives, not into an error.

    ``child_refs`` is the duck-typed surface ``cancel_ray_refs`` flattens, so
    the generic pool can cancel an aggregate without importing this module.
    """

    __slots__ = ("_combine", "child_refs")

    def __init__(
        self,
        refs: Sequence[Any],
        *,
        combine: Callable[[list[Any]], Any] | None = None,
    ) -> None:
        if not refs:
            raise ValueError("EngineCallRef requires at least one rank ref")
        self.child_refs = tuple(refs)
        self._combine = combine

    def __await__(self) -> Any:
        return self._gather().__await__()

    async def _gather(self) -> Any:
        async def wait_one(ref: Any) -> Any:
            return await ref

        tasks = [asyncio.ensure_future(wait_one(ref)) for ref in self.child_refs]
        try:
            results = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            cancel_ray_refs(None, self.child_refs, root_error=None)
            raise
        return results[0] if self._combine is None else self._combine(results)


def uniform_rank_result(method_name: str) -> Callable[[list[Any]], Any]:
    """Combine that requires every rank to agree — e.g. the update_weights
    version echo: ranks that installed different policy versions mean the
    engine is internally inconsistent and must fail loud."""

    def combine(results: list[Any]) -> Any:
        first = results[0]
        if any(result != first for result in results[1:]):
            raise RuntimeError(
                f"engine ranks disagree on {method_name} result: {results!r}",
            )
        return first

    return combine


class RayGenerationEngine:
    """Driver-side engine object over its rank actor handles.

    The dispatcher, executor, weight sync, and session speak to this object;
    it fans control-plane calls out to every rank and aggregates. The
    single-rank case returns raw rank refs so today's behavior is unchanged
    byte-for-byte; the aggregate path is exercised by multi-rank unit tests
    until a multi-GPU engine backend lands.
    """

    __slots__ = ("engine_id", "ranks")

    def __init__(self, engine_id: str, ranks: Sequence[RayActorHandle]) -> None:
        if not engine_id:
            raise ValueError("engine_id must be non-empty")
        ranks = tuple(ranks)
        if not ranks:
            raise ValueError(f"engine {engine_id!r} requires at least one rank")
        rank_ids = tuple(rank.worker_id for rank in ranks)
        if len(set(rank_ids)) != len(rank_ids):
            raise ValueError(f"engine {engine_id!r} has duplicate rank ids: {rank_ids}")
        self.engine_id = engine_id
        self.ranks = ranks

    @property
    def primary(self) -> RayActorHandle:
        """Rank 0: the rank whose result represents the engine."""

        return self.ranks[0]

    def remote(
        self,
        method_name: str,
        *,
        combine: Callable[[list[Any]], Any] | None = None,
    ) -> Callable[..., Any]:
        """Submission surface for the dispatcher: the returned callable submits
        one call to every rank and returns a single awaitable ref.

        The single-rank case always returns the raw rank ref — a one-element
        aggregate carries no information (``combine`` over one result is that
        result), and the wrapper's extra async hop would change the pool's
        completion/cancellation timing for the common case."""

        if len(self.ranks) == 1:
            return getattr(self.ranks[0].actor, method_name).remote

        def submit(*args: Any, **kwargs: Any) -> EngineCallRef:
            refs = [
                getattr(rank.actor, method_name).remote(*args, **kwargs) for rank in self.ranks
            ]
            return EngineCallRef(refs, combine=combine)

        return submit

    async def sleep(self) -> tuple[WorkerMemoryParkingSnapshot, ...]:
        """Park every rank and return its validated physical-memory evidence."""

        refs = [rank.actor.sleep.remote() for rank in self.ranks]
        values = await asyncio.gather(*refs)
        snapshots: list[WorkerMemoryParkingSnapshot] = []
        for rank, value in zip(self.ranks, values, strict=True):
            if not isinstance(value, WorkerMemoryParkingSnapshot):
                raise TypeError(
                    f"rank {rank.worker_id!r} returned invalid memory-parking "
                    f"report {type(value).__name__}",
                )
            if value.worker_id != rank.worker_id:
                raise RuntimeError(
                    "mismatched rank memory-parking report: "
                    f"expected={rank.worker_id!r} actual={value.worker_id!r}",
                )
            value.validate()
            snapshots.append(value)
        return tuple(snapshots)

    async def wake(self) -> None:
        """Restore every parked rank onto its assigned GPU."""

        await asyncio.gather(*[rank.actor.wake.remote() for rank in self.ranks])


def rank_handles(engines: Sequence[RayGenerationEngine]) -> list[RayActorHandle]:
    """Flat rank view for the lifecycle machinery (kill, liveness, metadata)."""

    return [rank for engine in engines for rank in engine.ranks]


__all__ = [
    "EngineCallRef",
    "RayGenerationEngine",
    "rank_handles",
    "uniform_rank_result",
]
