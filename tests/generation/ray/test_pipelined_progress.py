"""Pipelined generation progress and single-flight deadline tests."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from typing import Any

import pytest

import vrl.generation.ray.executor as executor_module
import vrl.ray.actor_pool as actor_pool_module
import vrl.ray.operation_deadline as deadline_module
from vrl.generation.execution.types import (
    DistributedWorkerHandle,
    PipelinedRequestOutOfMemory,
)
from vrl.generation.ray.executor import RayGenerationExecutor
from vrl.generation.ray.pipeline_protocol import (
    PipelinedProgressError,
    PipelinedRequestProgress,
)
from vrl.generation.ray.worker import RayGenerationWorker
from vrl.ray.actor_pool import RayActorDispatcher, RayActorJob
from vrl.ray.operation_deadline import (
    RayCallDeadline,
    RayOperationCancelled,
    RayOperationTimeout,
)


@pytest.fixture(autouse=True)
def _fast_progress_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        executor_module,
        "_PIPELINED_PROGRESS_POLL_INTERVAL_S",
        0.001,
    )


class _ResolvedRef:
    def __init__(self, value: Any) -> None:
        self.value = value

    def __await__(self):
        async def resolve() -> Any:
            return self.value

        return resolve().__await__()


class _GatedRef:
    def __init__(self, event: asyncio.Event, value: Any) -> None:
        self.event = event
        self.value = value

    def __await__(self):
        async def resolve() -> Any:
            await self.event.wait()
            return self.value

        return resolve().__await__()


def _executor(*, timeout_s: float = 1.0) -> RayGenerationExecutor:
    return RayGenerationExecutor(
        planner=object(),
        workers=[DistributedWorkerHandle(worker_id="w0", actor=object())],
        gatherer=object(),
        actor_dispatcher=RayActorDispatcher(("w0",)),
        generation_stall_timeout_s=timeout_s,
        pipelined=True,
    )


async def _await_pipelined_through_dispatcher(
    executor: RayGenerationExecutor,
    *,
    result_ref: Any,
    progress_remote: Any,
    request_id: str,
    total_chunks: int,
) -> Any:
    """Exercise the progress waiter under its production ObjectRef owner."""

    async def await_result(ref: Any, initial_deadline: RayCallDeadline) -> Any:
        return await executor._await_pipelined_result(
            result_ref=ref,
            progress_remote=progress_remote,
            request_id=request_id,
            total_chunks=total_chunks,
            initial_deadline=initial_deadline,
        )

    return await executor.actor_dispatcher.run_one(
        RayActorJob(
            job_index=0,
            worker_id="w0",
            remote_method=lambda _payload: result_ref,
            payload=None,
        ),
        operation="rollout.generation.pipelined",
        call_timeout_s=executor.generation_stall_timeout_s,
        await_result=await_result,
    )


def test_ray_worker_reports_only_the_active_pipelined_request() -> None:
    worker = object.__new__(RayGenerationWorker)
    worker._pipelined_progress_lock = threading.Lock()
    worker._pipelined_progress = None
    observed: list[PipelinedRequestProgress | None] = []

    class _Core:
        def execute_request_pipelined(
            self,
            request: Any,
            engine_plan: Any,
            sample_rows: Any,
            *,
            progress_callback: Any,
        ) -> str:
            del engine_plan, sample_rows
            observed.append(worker.pipelined_progress(request.request_id))
            progress_callback(1)
            observed.append(worker.pipelined_progress(request.request_id))
            progress_callback(2)
            return "complete"

    worker.core = _Core()
    request = SimpleNamespace(request_id="req-progress")
    plan = SimpleNamespace(chunks=("c0", "c1"))

    result = worker.execute_request_pipelined(request, plan, [])

    assert result == "complete"
    assert [snapshot.completed_chunks for snapshot in observed if snapshot is not None] == [0, 1]
    assert all(snapshot.total_chunks == 2 for snapshot in observed if snapshot is not None)
    assert worker.pipelined_progress("req-progress") is None
    assert worker.pipelined_progress("another-request") is None


@pytest.mark.asyncio
async def test_remote_pipelined_worker_requires_progress_endpoint() -> None:
    executor = _executor()

    class _RemoteMethod:
        @staticmethod
        def remote(*_args: Any) -> None:
            raise AssertionError("request must not start without progress endpoint")

    executor.workers[0] = DistributedWorkerHandle(
        worker_id="w0",
        actor=SimpleNamespace(execute_request_pipelined=_RemoteMethod()),
    )

    with pytest.raises(PipelinedProgressError, match="requires worker progress"):
        await executor._execute_request_pipelined(
            SimpleNamespace(request_id="req-missing-progress"),
            SimpleNamespace(chunks=("c0", "c1")),
            [],
        )


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

    def submit_first(_payload: Any) -> _GatedRef:
        first_submitted.set()
        return _GatedRef(gate, "first")

    class _PipelineMethod:
        @staticmethod
        def remote(request: Any, **_kwargs: Any) -> _ResolvedRef:
            pipeline_calls.append(request.request_id)
            return _ResolvedRef(
                PipelinedRequestOutOfMemory(
                    request_id=request.request_id,
                    worker_id="w0",
                    error="expected fallback",
                ),
            )

    class _ProgressMethod:
        @staticmethod
        def remote(_request_id: str) -> _ResolvedRef:
            raise AssertionError("an immediately resolved pipeline needs no progress RPC")

    monkeypatch.setattr(actor_pool_module, "RayCallDeadline", recording_deadline)
    executor.workers[0] = DistributedWorkerHandle(
        worker_id="w0",
        actor=SimpleNamespace(
            execute_request_pipelined=_PipelineMethod(),
            pipelined_progress=_ProgressMethod(),
        ),
    )
    first = asyncio.create_task(
        executor.actor_dispatcher.run(
            [RayActorJob(0, "w0", submit_first, None)],
            operation="rollout.generation.chunk",
            call_timeout_s=30.0,
        ),
    )
    await first_submitted.wait()

    pipelined = asyncio.create_task(
        executor._execute_request_pipelined(
            SimpleNamespace(request_id="req-admission"),
            SimpleNamespace(chunks=("c0", "c1")),
            [],
        ),
    )
    await asyncio.sleep(0.03)

    assert not pipelined.done()
    assert pipeline_calls == []
    assert deadlines == ["rollout.generation.chunk"]

    gate.set()
    assert await first == [(0, "first")]
    result = await pipelined
    assert isinstance(result, PipelinedRequestOutOfMemory)
    assert pipeline_calls == ["req-admission"]
    assert deadlines == [
        "rollout.generation.chunk",
        "rollout.generation.pipelined",
    ]


@pytest.mark.asyncio
async def test_pipelined_progress_resets_the_stall_deadline() -> None:
    executor = _executor(timeout_s=0.05)
    result_ready = asyncio.Event()
    result_ref = _GatedRef(result_ready, "complete")
    progress_calls = 0

    def progress_remote(request_id: str) -> _ResolvedRef:
        nonlocal progress_calls
        progress_calls += 1
        if progress_calls == 2:
            result_ready.set()
        return _ResolvedRef(
            PipelinedRequestProgress(
                request_id=request_id,
                completed_chunks=min(progress_calls, 2),
                total_chunks=2,
            ),
        )

    result = await _await_pipelined_through_dispatcher(
        executor,
        result_ref=result_ref,
        progress_remote=progress_remote,
        request_id="req-progress",
        total_chunks=2,
    )

    assert result == "complete"
    assert progress_calls == 2


@pytest.mark.asyncio
async def test_pipelined_progress_reset_preserves_initial_deadline_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = _executor(timeout_s=1.0)
    result_ref = _GatedRef(asyncio.Event(), "never")
    progress_calls = 0

    class _Ray:
        @staticmethod
        def cancel(_ref: Any, *, force: bool) -> None:
            assert force is False

    monkeypatch.setattr(deadline_module, "require_ray", lambda: _Ray)

    def progress_remote(request_id: str) -> _ResolvedRef:
        nonlocal progress_calls
        progress_calls += 1
        return _ResolvedRef(
            PipelinedRequestProgress(
                request_id=request_id,
                completed_chunks=1,
                total_chunks=2,
            ),
        )

    initial = RayCallDeadline(
        "custom.pipeline.operation",
        0.02,
        context="custom-deadline-context",
    )
    with pytest.raises(RayOperationTimeout) as caught:
        await executor._await_pipelined_result(
            result_ref=result_ref,
            progress_remote=progress_remote,
            request_id="req-identity",
            total_chunks=2,
            initial_deadline=initial,
        )

    assert progress_calls >= 2
    assert caught.value.operation == initial.operation
    assert caught.value.timeout_s == initial.timeout_s
    assert caught.value.context == initial.context


@pytest.mark.asyncio
async def test_pipelined_stall_cancels_the_result_ref(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = _executor(timeout_s=0.03)
    result_ref = _GatedRef(asyncio.Event(), "never")
    cancelled: list[Any] = []

    class _Ray:
        @staticmethod
        def cancel(ref: Any, *, force: bool) -> None:
            assert force is False
            cancelled.append(ref)

    monkeypatch.setattr(deadline_module, "require_ray", lambda: _Ray)

    def progress_remote(request_id: str) -> _ResolvedRef:
        return _ResolvedRef(
            PipelinedRequestProgress(
                request_id=request_id,
                completed_chunks=0,
                total_chunks=2,
            ),
        )

    with pytest.raises(RayOperationTimeout, match=r"rollout\.generation\.pipelined"):
        await _await_pipelined_through_dispatcher(
            executor,
            result_ref=result_ref,
            progress_remote=progress_remote,
            request_id="req-stalled",
            total_chunks=2,
        )

    assert result_ref in cancelled


@pytest.mark.asyncio
async def test_pipelined_stall_polling_does_not_accelerate_near_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        executor_module,
        "_PIPELINED_PROGRESS_POLL_INTERVAL_S",
        0.02,
    )
    executor = _executor(timeout_s=0.11)
    result_ref = _GatedRef(asyncio.Event(), "never")
    progress_calls = 0

    class _Ray:
        @staticmethod
        def cancel(_ref: Any, *, force: bool) -> None:
            assert force is False

    monkeypatch.setattr(deadline_module, "require_ray", lambda: _Ray)

    def progress_remote(request_id: str) -> _ResolvedRef:
        nonlocal progress_calls
        progress_calls += 1
        return _ResolvedRef(
            PipelinedRequestProgress(
                request_id=request_id,
                completed_chunks=0,
                total_chunks=2,
            ),
        )

    with pytest.raises(RayOperationTimeout):
        await _await_pipelined_through_dispatcher(
            executor,
            result_ref=result_ref,
            progress_remote=progress_remote,
            request_id="req-fixed-cadence",
            total_chunks=2,
        )

    assert 4 <= progress_calls <= 6


@pytest.mark.asyncio
async def test_pipelined_progress_rpc_stall_cancels_result_and_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = _executor(timeout_s=0.04)
    result_ref = _GatedRef(asyncio.Event(), "never")
    progress_ref = _GatedRef(asyncio.Event(), "never")
    cancelled: list[Any] = []

    class _Ray:
        @staticmethod
        def cancel(ref: Any, *, force: bool) -> None:
            assert force is False
            cancelled.append(ref)

    monkeypatch.setattr(deadline_module, "require_ray", lambda: _Ray)

    with pytest.raises(RayOperationTimeout, match=r"rollout\.generation\.pipelined"):
        await _await_pipelined_through_dispatcher(
            executor,
            result_ref=result_ref,
            progress_remote=lambda _request_id: progress_ref,
            request_id="req-progress-stalled",
            total_chunks=2,
        )

    assert result_ref in cancelled
    assert progress_ref in cancelled


@pytest.mark.asyncio
async def test_pipelined_caller_cancellation_cancels_active_refs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = _executor(timeout_s=0.1)
    result_ref = _GatedRef(asyncio.Event(), "never")
    progress_ref = _GatedRef(asyncio.Event(), "never")
    progress_called = asyncio.Event()
    cancelled: list[Any] = []

    class _Ray:
        @staticmethod
        def cancel(ref: Any, *, force: bool) -> None:
            assert force is False
            cancelled.append(ref)

    monkeypatch.setattr(deadline_module, "require_ray", lambda: _Ray)

    def progress_remote(_request_id: str) -> _GatedRef:
        progress_called.set()
        return progress_ref

    task = asyncio.create_task(
        _await_pipelined_through_dispatcher(
            executor,
            result_ref=result_ref,
            progress_remote=progress_remote,
            request_id="req-cancelled",
            total_chunks=2,
        ),
    )
    await progress_called.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError) as caught:
        await task

    assert isinstance(caught.value.__cause__, RayOperationCancelled)
    assert result_ref in cancelled
    assert progress_ref in cancelled


@pytest.mark.asyncio
async def test_pipelined_result_cancels_a_losing_progress_rpc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = _executor(timeout_s=0.1)
    result_ready = asyncio.Event()
    result_ref = _GatedRef(result_ready, "complete")
    progress_ref = _GatedRef(asyncio.Event(), "never")
    cancelled: list[Any] = []

    class _Ray:
        @staticmethod
        def cancel(ref: Any, *, force: bool) -> None:
            assert force is False
            cancelled.append(ref)

    monkeypatch.setattr(deadline_module, "require_ray", lambda: _Ray)

    def progress_remote(_request_id: str) -> _GatedRef:
        result_ready.set()
        return progress_ref

    result = await _await_pipelined_through_dispatcher(
        executor,
        result_ref=result_ref,
        progress_remote=progress_remote,
        request_id="req-result-wins",
        total_chunks=2,
    )

    assert result == "complete"
    assert cancelled == [progress_ref]


@pytest.mark.parametrize(
    ("snapshots", "message"),
    [
        ([object()], "invalid progress"),
        (
            [
                PipelinedRequestProgress(
                    request_id="wrong-request",
                    completed_chunks=0,
                    total_chunks=2,
                ),
            ],
            "request_id mismatch",
        ),
        (
            [
                PipelinedRequestProgress(
                    request_id="req-protocol",
                    completed_chunks=0,
                    total_chunks=3,
                ),
            ],
            "total_chunks mismatch",
        ),
        (
            [
                PipelinedRequestProgress(
                    request_id="req-protocol",
                    completed_chunks=1,
                    total_chunks=2,
                ),
                PipelinedRequestProgress(
                    request_id="req-protocol",
                    completed_chunks=0,
                    total_chunks=2,
                ),
            ],
            "progress regressed",
        ),
    ],
)
@pytest.mark.asyncio
async def test_pipelined_progress_protocol_failure_is_terminal_and_cancels_result(
    monkeypatch: pytest.MonkeyPatch,
    snapshots: list[Any],
    message: str,
) -> None:
    executor = _executor(timeout_s=0.1)
    result_ref = _GatedRef(asyncio.Event(), "never")
    remaining = list(snapshots)
    cancelled: list[Any] = []

    class _Ray:
        @staticmethod
        def cancel(ref: Any, *, force: bool) -> None:
            assert force is False
            cancelled.append(ref)

    monkeypatch.setattr(deadline_module, "require_ray", lambda: _Ray)

    def progress_remote(_request_id: str) -> _ResolvedRef:
        snapshot = remaining.pop(0) if remaining else snapshots[-1]
        return _ResolvedRef(snapshot)

    with pytest.raises(PipelinedProgressError, match=message):
        await _await_pipelined_through_dispatcher(
            executor,
            result_ref=result_ref,
            progress_remote=progress_remote,
            request_id="req-protocol",
            total_chunks=2,
        )

    assert result_ref in cancelled


@pytest.mark.asyncio
async def test_pipelined_executor_queues_requests_before_starting_deadline() -> None:
    executor = _executor()
    first_can_finish = asyncio.Event()
    first_entered = asyncio.Event()
    entered: list[str] = []

    async def execute_unlocked(request: Any) -> str:
        entered.append(request.request_id)
        if request.request_id == "first":
            first_entered.set()
            await first_can_finish.wait()
        return request.request_id

    executor._execute = execute_unlocked
    first = asyncio.create_task(executor.execute(SimpleNamespace(request_id="first")))
    await first_entered.wait()
    second = asyncio.create_task(executor.execute(SimpleNamespace(request_id="second")))
    await asyncio.sleep(0)

    assert entered == ["first"]
    first_can_finish.set()
    assert await asyncio.gather(first, second) == ["first", "second"]
    assert entered == ["first", "second"]


@pytest.mark.asyncio
async def test_cancelling_pipelined_lock_wait_does_not_mark_submitted_work() -> None:
    executor = _executor()
    first_can_finish = asyncio.Event()
    first_entered = asyncio.Event()
    entered: list[str] = []

    async def execute_unlocked(request: Any) -> str:
        entered.append(request.request_id)
        if request.request_id == "first":
            first_entered.set()
            await first_can_finish.wait()
        return request.request_id

    executor._execute = execute_unlocked
    first = asyncio.create_task(executor.execute(SimpleNamespace(request_id="first")))
    await first_entered.wait()
    waiting = asyncio.create_task(executor.execute(SimpleNamespace(request_id="waiting")))
    await asyncio.sleep(0)
    waiting.cancel()

    with pytest.raises(asyncio.CancelledError) as caught:
        await waiting
    assert caught.value.__cause__ is None
    assert entered == ["first"]

    first_can_finish.set()
    assert await first == "first"
    assert await executor.execute(SimpleNamespace(request_id="third")) == "third"
    assert entered == ["first", "third"]
