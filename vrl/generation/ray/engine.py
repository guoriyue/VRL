"""One generation engine replica: driver-side composition of its rank actors.

An engine is the data-parallel unit the driver talks to — the dispatch target,
weight-sync target, and health-verdict unit. A rank is one per-GPU worker actor
inside it (``RayGenerationWorker``). Engines may own multiple ranks; calls
fan out to their rank actors and aggregate before returning to the dispatcher.

Lifecycle stays rank-level on purpose: launching, killing, and liveness-probing
operate on rank actors (``RayActorGroup`` / the health monitor); the engine
only derives the verdict "any dead rank == dead engine".
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

from vrl.generation.execution.types import GenerationBatchResult, WorkerMemoryParkingSnapshot
from vrl.ray.actor_group import RayActorHandle
from vrl.ray.operation_deadline import cancel_ray_refs
from vrl.runtime_errors import TerminalRuntimeError
from vrl.utils.cuda_memory import is_cuda_out_of_memory


class EngineCallRef:
    """Awaitable aggregate over one engine call's per-rank Ray refs.

    All rank awaitables must finish without raising. The combined result is
    ``combine(results)`` (default: rank 0's result). Error payloads returned as
    ordinary values need an explicit combiner; the default does not inspect them.
    Any raised rank failure cancels the sibling refs and
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
        tasks = [asyncio.ensure_future(ref) for ref in self.child_refs]
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
        if any(type(result) is not type(first) or result != first for result in results[1:]):
            raise RuntimeError(
                f"engine ranks disagree on {method_name} result: {results!r}",
            )
        return first

    return combine


class RayGenerationEngine:
    """Driver-side engine object over its rank actor handles.

    The dispatcher, executor, weight sync, and session speak to this object;
    it fans control-plane calls out to every rank and aggregates. The
    single-rank case returns raw rank refs; multi-rank calls wait for all rank
    refs before applying their result-combination policy.
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

        ``combine`` is called only for multi-rank engines. It must implement
        cross-rank aggregation; validation or transformation required for every
        result belongs in the caller after awaiting the ref.

        Single-rank engines return the raw rank ref and skip ``combine`` to
        preserve the pool's completion/cancellation timing for the common case.
        """

        if len(self.ranks) == 1:
            return getattr(self.ranks[0].actor, method_name).remote

        if combine is None and method_name == "execute_batch":
            combine = self._combine_batch_results

        def submit(*args: Any, **kwargs: Any) -> EngineCallRef:
            refs = self._submit_rank_calls(method_name, *args, **kwargs)
            return EngineCallRef(refs, combine=combine)

        return submit

    def _submit_rank_calls(self, method_name: str, *args: Any, **kwargs: Any) -> list[Any]:
        """Own rank refs until the complete fan-out can be handed to its waiter."""

        refs: list[Any] = []
        try:
            for rank in self.ranks:
                refs.append(getattr(rank.actor, method_name).remote(*args, **kwargs))
        except BaseException as cause:
            if not refs:
                raise
            error = TerminalRuntimeError(
                f"engine {self.engine_id!r} partially submitted {method_name!r}: "
                f"{len(refs)}/{len(self.ranks)} ranks received the call",
            )
            cancel_ray_refs(None, refs, root_error=error)
            raise error from cause
        return refs

    def _combine_batch_results(self, results: list[Any]) -> GenerationBatchResult:
        return combine_rank_batch_results(
            results,
            expected_worker_ids=[rank.worker_id for rank in self.ranks],
        )

    async def sleep(self) -> tuple[WorkerMemoryParkingSnapshot, ...]:
        """Park every rank and return its validated physical-memory evidence."""

        refs = self._submit_rank_calls("sleep")
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

        await asyncio.gather(*self._submit_rank_calls("wake"))


def rank_handles(engines: Sequence[RayGenerationEngine]) -> list[RayActorHandle]:
    """Flat rank view for the lifecycle machinery (kill, liveness, metadata)."""

    return [rank for engine in engines for rank in engine.ranks]


def combine_rank_batch_results(
    results: list[Any],
    *,
    expected_worker_ids: Sequence[str] | None = None,
) -> GenerationBatchResult:
    """Fold one generation batch's per-rank results into the result the driver acts on.

    Every rank must return a ``GenerationBatchResult`` for the same request and
    batch (and, when ``expected_worker_ids`` is given, its own worker id). A
    terminal failure on any rank wins over another rank's retryable OOM or
    graceful stale-slot discard, and keeps the reporting rank's identity, so
    the executor's stale-slot and OOM handling still sees it; otherwise the
    primary rank's payload carries every rank's metrics.
    """

    if not all(isinstance(result, GenerationBatchResult) for result in results):
        raise TypeError("generation engine ranks must return GenerationBatchResult")
    first = results[0]
    if expected_worker_ids is not None:
        for worker_id, result in zip(expected_worker_ids, results, strict=True):
            if result.worker_id != worker_id:
                raise RuntimeError(f"rank {worker_id!r} returned another worker's result")
    for result in results[1:]:
        if result.request_id != first.request_id or result.batch != first.batch:
            raise RuntimeError(
                "generation engine ranks returned different request/batch identities"
            )
    for result in results:
        if result.error and not result.stale_slot and not is_cuda_out_of_memory(result.error):
            return result
    for result in results:
        if result.stale_slot:
            return result
    for result in results:
        if result.error:
            return result
    if any(result.policy_version != first.policy_version for result in results[1:]):
        raise RuntimeError("generation engine ranks returned different policy versions")
    return replace(
        first,
        rank_metrics={result.worker_id: result.metrics for result in results},
    )


__all__ = [
    "EngineCallRef",
    "RayGenerationEngine",
    "combine_rank_batch_results",
    "rank_handles",
    "uniform_rank_result",
]
