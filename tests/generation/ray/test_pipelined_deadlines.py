"""Pipelined generation deadlines: admission first, then a budget per batch."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

import vrl.ray.actor_pool as actor_pool_module
from tests.generation.ray._helpers import GatedRef, ResolvedRef
from vrl.generation.execution.planner import EnginePlan
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.execution.types import (
    RequestBatchOutOfMemory,
)
from vrl.generation.ray.engine import RayGenerationEngine
from vrl.generation.ray.executor import RayGenerationExecutor
from vrl.ray.actor_group import RayActorHandle
from vrl.ray.actor_pool import RayActorDispatcher, RayActorJob


def _executor(*, timeout_s: float = 1.0) -> RayGenerationExecutor:
    return RayGenerationExecutor(
        engines=[
            RayGenerationEngine(
                "w0",
                [RayActorHandle(worker_id="w0", actor=object())],
            ),
        ],
        gatherer=object(),
        actor_dispatcher=RayActorDispatcher(("w0",)),
        generation_stall_timeout_s=timeout_s,
        pipelined=True,
        finalizers=[RayActorHandle(worker_id="finalize-0", actor=object())],
    )


def _batches(count: int) -> tuple[GenerationSampleBatch, ...]:
    return tuple(GenerationSampleBatch(0, index, 1) for index in range(count))


@pytest.mark.asyncio
async def test_pipelined_call_budget_is_the_stall_timeout_per_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One RPC produces every batch of the engine's share, so its budget covers each."""

    executor = _executor(timeout_s=2.0)
    budgets: list[tuple[str, float]] = []
    real_deadline = actor_pool_module.RayCallDeadline

    def recording_deadline(operation: str, timeout_s: float, **kwargs: Any) -> Any:
        budgets.append((operation, timeout_s))
        return real_deadline(operation, timeout_s, **kwargs)

    class _PipelineMethod:
        @staticmethod
        def remote(request: Any, **_kwargs: Any) -> ResolvedRef:
            return ResolvedRef(
                RequestBatchOutOfMemory(
                    request_id=request.request_id,
                    worker_id="w0",
                    error="expected fallback",
                ),
            )

    monkeypatch.setattr(actor_pool_module, "RayCallDeadline", recording_deadline)
    executor.engines[0] = RayGenerationEngine(
        "w0",
        [
            RayActorHandle(
                worker_id="w0",
                actor=SimpleNamespace(execute_request_batches=_PipelineMethod()),
            ),
        ],
    )

    result = await executor._execute_request_batches(
        SimpleNamespace(request_id="req-budget"),
        EnginePlan(sample_batches=_batches(3)),
        [],
    )

    assert isinstance(result, RequestBatchOutOfMemory)
    assert budgets == [("rollout.generation.pipelined", 6.0)]


@pytest.mark.asyncio
async def test_pipelined_submission_gets_deadline_only_after_fleet_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = _executor(timeout_s=0.01)
    gate = asyncio.Event()
    first_submitted = asyncio.Event()
    pipeline_calls: list[str] = []
    deadlines: list[str] = []
    real_deadline = actor_pool_module.RayCallDeadline

    def recording_deadline(operation: str, *args: Any, **kwargs: Any) -> Any:
        deadlines.append(operation)
        return real_deadline(operation, *args, **kwargs)

    def submit_first(_payload: Any) -> GatedRef:
        first_submitted.set()
        return GatedRef(gate, "first")

    class _PipelineMethod:
        @staticmethod
        def remote(request: Any, **_kwargs: Any) -> ResolvedRef:
            pipeline_calls.append(request.request_id)
            return ResolvedRef(
                RequestBatchOutOfMemory(
                    request_id=request.request_id,
                    worker_id="w0",
                    error="expected fallback",
                ),
            )

    monkeypatch.setattr(actor_pool_module, "RayCallDeadline", recording_deadline)
    executor.engines[0] = RayGenerationEngine(
        "w0",
        [
            RayActorHandle(
                worker_id="w0",
                actor=SimpleNamespace(execute_request_batches=_PipelineMethod()),
            ),
        ],
    )
    first = asyncio.create_task(
        executor.actor_dispatcher.run(
            [RayActorJob(0, "w0", submit_first, None)],
            operation="rollout.generation.batch",
            call_timeout_s=30.0,
        ),
    )
    await first_submitted.wait()

    pipelined = asyncio.create_task(
        executor._execute_request_batches(
            SimpleNamespace(request_id="req-admission"),
            EnginePlan(sample_batches=_batches(2)),
            [],
        ),
    )
    await asyncio.sleep(0.03)

    assert not pipelined.done()
    assert pipeline_calls == []
    assert deadlines == ["rollout.generation.batch"]

    gate.set()
    assert await first == [(0, "first")]
    result = await pipelined
    assert isinstance(result, RequestBatchOutOfMemory)
    assert pipeline_calls == ["req-admission"]
    assert deadlines == [
        "rollout.generation.batch",
        "rollout.generation.pipelined",
    ]


@pytest.mark.asyncio
async def test_finalize_gets_its_deadline_only_after_a_finalizer_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request whose merge queues behind a slow one does not burn its budget
    waiting: the finalizer has one slot, FIFO waiters, and the deadline starts
    at admission, exactly like the engines."""

    import time

    from vrl.generation.types import GenerationOutput
    from vrl.trajectory.types import TrajectoryBatch

    executor = _executor(timeout_s=5.0)
    first_gate = asyncio.Event()
    merges: list[str] = []
    deadline_starts: list[float] = []
    real_deadline = actor_pool_module.RayCallDeadline

    def recording_deadline(operation: str, *args: Any, **kwargs: Any) -> Any:
        assert operation == "rollout.generation.finalize"
        deadline_starts.append(time.perf_counter())
        return real_deadline(operation, *args, **kwargs)

    def _output(request: Any) -> GenerationOutput:
        return GenerationOutput(
            output=[],
            trajectory=TrajectoryBatch(
                request_id=request.request_id,
                family="test",
                task="t2i",
                sample_rows=[],
                axes={},
                segments={},
            ),
        )

    class _MergeMethod:
        @staticmethod
        def remote(request: Any, **_kwargs: Any) -> Any:
            merges.append(request.request_id)
            if request.request_id == "slow":
                return GatedRef(first_gate, _output(request))
            return ResolvedRef(_output(request))

    monkeypatch.setattr(actor_pool_module, "RayCallDeadline", recording_deadline)
    executor.finalizers = (
        RayActorHandle(
            worker_id="finalize-0", actor=SimpleNamespace(merge_request=_MergeMethod())
        ),
    )
    executor._finalizer_dispatcher = RayActorDispatcher(("finalize-0",))

    slow = asyncio.create_task(
        executor._finalize_request(SimpleNamespace(request_id="slow"), [], ["ref-a"]),
    )
    await asyncio.sleep(0)
    queued = asyncio.create_task(
        executor._finalize_request(SimpleNamespace(request_id="queued"), [], ["ref-b"]),
    )
    await asyncio.sleep(0.05)

    # The queued merge is neither submitted nor on the clock while the slow
    # one holds the finalizer's slot.
    assert merges == ["slow"]
    assert len(deadline_starts) == 1
    assert not queued.done()

    released_at = time.perf_counter()
    first_gate.set()
    assert (await slow).request_id == "slow"
    assert (await queued).request_id == "queued"
    assert merges == ["slow", "queued"]
    assert len(deadline_starts) == 2
    assert deadline_starts[1] >= released_at
