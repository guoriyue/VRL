"""Batch placement and pull-based actor dispatch on real Ray actors.

Every dispatched call is a real actor method on the package cluster
(``tests/ray/conftest.py``) returning a real ``ObjectRef``. Completion order is
controlled with real mechanisms: per-worker execution delays far apart, and a
real gate actor a worker method awaits until the test opens it. They pin the
dispatch contract: plan-time binding is kept bit-for-bit, pull dispatch never
changes gather order, deadlines start at real submission, and terminal
failures, cancellations and admission waits linearize as documented.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import pytest

import vrl.ray.actor_pool as actor_pool_module
from tests.rollouts.collector._helpers import Trace
from vrl.generation.execution.types import (
    GenerationBatchEnvelope,
    GenerationBatchResult,
)
from vrl.generation.ray.engine import RayGenerationEngine
from vrl.generation.ray.executor import RayGenerationExecutor
from vrl.generation.types import GenerationOutput, GenerationRequest
from vrl.ray.actor_group import RayActorHandle
from vrl.ray.actor_pool import (
    RayActorCallError,
    RayActorDispatcher,
    RayActorJob,
)
from vrl.ray.operation_deadline import RayOperationTimeout
from vrl.trajectory.types import TrajectoryBatch

# Far apart, so the fast worker's completions all land inside one slow call.
_FAST_S = 0.01
_SLOW_S = 1.0


class _Gate:
    """A real actor holding an event other actors await until the test opens it."""

    def __init__(self) -> None:
        self._event = asyncio.Event()

    async def wait(self) -> None:
        await self._event.wait()

    def open(self) -> None:
        self._event.set()


class _Worker:
    """A real async actor whose calls run, block on a gate, fail or never return."""

    def __init__(self, worker_id: str, delay_s: float = 0.0, gate: Any = None) -> None:
        self.worker_id = worker_id
        self.delay_s = delay_s
        self.gate = gate
        self._received: list[Any] = []

    def received(self) -> list[Any]:
        return list(self._received)

    async def run(self, payload: Any) -> tuple[str, Any]:
        self._received.append(payload)
        await asyncio.sleep(self.delay_s)
        return self.worker_id, payload

    async def echo(self, payload: Any) -> Any:
        self._received.append(payload)
        return payload

    async def hold(self, payload: Any) -> Any:
        self._received.append(payload)
        await self.gate.wait.remote()
        return payload

    async def hold_then_fail(self, payload: Any) -> Any:
        self._received.append(payload)
        await self.gate.wait.remote()
        raise ValueError(f"actor failed: {payload}")

    async def hang(self, payload: Any) -> Any:
        self._received.append(payload)
        await asyncio.Event().wait()

    async def execute_batch(self, envelope: GenerationBatchEnvelope) -> GenerationBatchResult:
        batch = envelope.batch
        self._received.append(batch.batch_key)
        await asyncio.sleep(self.delay_s)
        return GenerationBatchResult(
            request_id=envelope.request.request_id,
            worker_id=self.worker_id,
            batch=batch,
            output={"batch_key": batch.batch_key, "samples": batch.sample_count},
        )


@pytest.fixture
def spawn(local_ray) -> Iterator[Callable[..., Any]]:
    """Create real actors on the package cluster; every one is killed after the test."""

    created: list[Any] = []

    def make(cls: type, *args: Any) -> Any:
        actor = local_ray.remote(num_cpus=0)(cls).remote(*args)
        created.append(actor)
        # Started before use, so no deadline under test pays for process startup.
        local_ray.get(actor.__ray_ready__.remote(), timeout=60)
        return actor

    yield make
    for actor in created:
        local_ray.kill(actor, no_restart=True)


def _received(ray: Any, actor: Any) -> list[Any]:
    return ray.get(actor.received.remote(), timeout=30)


def _recording(remote: Callable[[Any], Any], log: list[Any], refs: list[Any]) -> Callable:
    """The real ``.remote`` submission, noting each payload and the ref it returned."""

    def submit(payload: Any) -> Any:
        log.append(payload)
        ref = remote(payload)
        refs.append(ref)
        return ref

    return submit


async def _until(condition: Callable[[], bool], timeout_s: float = 30.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not condition():
        assert loop.time() < deadline, "condition not reached"
        await asyncio.sleep(0.005)


def _observe_completion(ray: Any, refs: list[Any]) -> None:
    """Block the loop until the actor calls finished and Ray queued their callbacks."""

    ready, _ = ray.wait(refs, num_returns=len(refs), timeout=30)
    assert len(ready) == len(refs)
    time.sleep(0.05)


def _request(num_steps: int = 10, samples: int = 8, sbs: int = 2) -> GenerationRequest:
    return GenerationRequest(
        request_id="req-dispatch",
        family="sd3_5",
        task="t2i",
        inputs=["a test prompt"],
        samples_per_prompt=samples,
        sampling={"height": 64, "width": 64, "num_steps": num_steps},
        samples_per_generation_batch=sbs,
        runtime_debug=True,
    )


pytestmark = pytest.mark.slow_test


# ---------------------------------------------------------------- actor pool


def test_bound_jobs_keep_plan_time_binding_and_order(local_ray, spawn) -> None:
    """The static path is unchanged: binding and order preserved."""

    fast = spawn(_Worker, "w0", _FAST_S)
    slow = spawn(_Worker, "w1", _SLOW_S)
    jobs = [
        RayActorJob(
            job_index=i,
            worker_id=("w0" if i % 2 == 0 else "w1"),
            remote_method=(fast.run.remote if i % 2 == 0 else slow.run.remote),
            payload=f"batch-{i}",
        )
        for i in range(4)
    ]

    pairs = asyncio.run(
        RayActorDispatcher(("w0", "w1")).run(
            jobs,
            operation="test.actor_job",
            call_timeout_s=30.0,
        ),
    )

    assert [index for index, _ in pairs] == [0, 1, 2, 3]
    # Even though w1 is slow, its batches never migrate to w0.
    assert _received(local_ray, fast) == ["batch-0", "batch-2"]
    assert _received(local_ray, slow) == ["batch-1", "batch-3"]


def test_pull_dispatch_lets_fast_worker_take_more_chunks(local_ray, spawn) -> None:
    """Unbound jobs flow to whichever worker frees up first."""

    fast = spawn(_Worker, "w0", _FAST_S)
    slow = spawn(_Worker, "w1", _SLOW_S)
    jobs = [
        RayActorJob(job_index=i, worker_id=None, remote_method=None, payload=f"batch-{i}")
        for i in range(4)
    ]

    pairs = asyncio.run(
        RayActorDispatcher(("w0", "w1")).run(
            jobs,
            operation="test.actor_job",
            call_timeout_s=30.0,
            worker_methods={"w0": fast.run.remote, "w1": slow.run.remote},
        ),
    )

    assert [index for index, _ in pairs] == [0, 1, 2, 3]
    # w0 finishes every call inside w1's one, so it pulls every queued batch.
    assert _received(local_ray, fast) == ["batch-0", "batch-2", "batch-3"]
    assert _received(local_ray, slow) == ["batch-1"]


def test_unbound_jobs_without_worker_methods_fail_loudly() -> None:
    """Pull dispatch without worker handles is a hard error."""

    jobs = [RayActorJob(job_index=0, worker_id=None, remote_method=None, payload="x")]

    with pytest.raises(ValueError, match="worker_methods"):
        asyncio.run(
            RayActorDispatcher(("w0",)).run(
                jobs,
                operation="test.actor_job",
                call_timeout_s=30.0,
            ),
        )


@pytest.mark.asyncio
async def test_actor_pool_validates_every_worker_before_first_submission(spawn) -> None:
    worker = spawn(_Worker, "w0")
    submitted: list[Any] = []
    submit = _recording(worker.echo.remote, submitted, [])

    dispatcher = RayActorDispatcher(("w0",))
    with pytest.raises(ValueError, match="unknown Ray actor worker"):
        await dispatcher.run(
            [
                RayActorJob(0, "w0", submit, "would-leak"),
                RayActorJob(1, "unknown", submit, "invalid"),
            ],
            operation="test.prevalidate",
            call_timeout_s=30.0,
        )

    assert submitted == []
    assert await dispatcher.run(
        [RayActorJob(0, "w0", submit, "still-open")],
        operation="test.after_prevalidate",
        call_timeout_s=30.0,
    ) == [(0, "still-open")]


def test_schedule_telemetry_rows_are_emitted(spawn) -> None:
    """The dispatch loop emits one telemetry row per job."""

    worker = spawn(_Worker, "w0", _FAST_S)
    jobs = [
        RayActorJob(job_index=i, worker_id="w0", remote_method=worker.run.remote, payload=i)
        for i in range(2)
    ]
    schedule: list[dict[str, Any]] = []

    asyncio.run(
        RayActorDispatcher(("w0",)).run(
            jobs,
            operation="test.actor_job",
            call_timeout_s=30.0,
            schedule=schedule,
        ),
    )

    assert sorted(row["job_index"] for row in schedule) == [0, 1]
    for row in schedule:
        assert row["worker_id"] == "w0"
        assert row["queue_wait_s"] >= 0.0
        assert row["execution_s"] >= 0.0


@pytest.mark.asyncio
async def test_actor_pool_timeout_discards_completed_partial_result(
    local_ray, spawn, monkeypatch: pytest.MonkeyPatch
) -> None:
    fast = spawn(_Worker, "w0", _FAST_S)
    hung = spawn(_Worker, "w1")
    hung_refs: list[Any] = []
    cancels = Trace(monkeypatch)
    cancels.watch(local_ray, "cancel", "cancel")
    jobs = [
        RayActorJob(
            job_index=0,
            worker_id="w0",
            remote_method=fast.run.remote,
            payload="complete-first",
        ),
        RayActorJob(
            job_index=1,
            worker_id="w1",
            remote_method=_recording(hung.hang.remote, [], hung_refs),
            payload="never",
        ),
    ]

    with pytest.raises(RayOperationTimeout, match="worker_id=w1"):
        await RayActorDispatcher(("w0", "w1")).run(
            jobs,
            operation="test.actor_job",
            call_timeout_s=1.0,
        )

    assert _received(local_ray, fast) == ["complete-first"]
    # Only the call still in flight is cancelled, and never forcefully.
    assert [args for _, args in cancels.calls] == [(hung_refs[0],)]


@pytest.mark.asyncio
async def test_queued_job_gets_its_deadline_only_when_submitted(
    spawn, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_deadline = actor_pool_module.RayCallDeadline
    deadlines: list[Any] = []
    deadlines_seen_at_submit: list[int] = []

    def recording_deadline(*args: Any, **kwargs: Any) -> Any:
        deadline = real_deadline(*args, **kwargs)
        deadlines.append(deadline)
        return deadline

    worker = spawn(_Worker, "w0", _FAST_S)

    def submit(payload: str) -> Any:
        deadlines_seen_at_submit.append(len(deadlines))
        return worker.echo.remote(payload)

    monkeypatch.setattr(actor_pool_module, "RayCallDeadline", recording_deadline)
    jobs = [
        RayActorJob(job_index=index, worker_id="w0", remote_method=submit, payload=f"job-{index}")
        for index in range(2)
    ]

    results = await RayActorDispatcher(("w0",)).run(
        jobs,
        operation="test.actor_job",
        call_timeout_s=30.0,
    )

    assert results == [(0, "job-0"), (1, "job-1")]
    assert deadlines_seen_at_submit == [1, 2]
    assert len(deadlines) == 2


@pytest.mark.asyncio
async def test_partial_submission_failure_cancels_registered_refs_and_closes(
    local_ray, spawn, monkeypatch: pytest.MonkeyPatch
) -> None:
    held = spawn(_Worker, "w0")
    other = spawn(_Worker, "w1")
    held_refs: list[Any] = []
    cancels = Trace(monkeypatch)
    cancels.watch(local_ray, "cancel", "cancel")
    dispatcher = RayActorDispatcher(("w0", "w1"))
    jobs = [
        RayActorJob(0, "w0", _recording(held.hang.remote, [], held_refs), "active"),
        # A payload Ray cannot serialize fails the driver-side submission.
        RayActorJob(1, "w1", other.echo.remote, threading.Lock()),
    ]

    with pytest.raises(RayActorCallError) as caught:
        await dispatcher.run(
            jobs,
            operation="test.partial_submit",
            call_timeout_s=30.0,
        )

    assert isinstance(caught.value.__cause__, TypeError)
    assert [args for _, args in cancels.calls] == [(held_refs[0],)]
    with pytest.raises(RuntimeError) as closed:
        await dispatcher.run(
            [],
            operation="test.after_partial_submit",
            call_timeout_s=30.0,
        )
    assert closed.value.__cause__ is caught.value


@pytest.mark.asyncio
async def test_concurrent_runs_propagate_the_first_terminal_identity(local_ray, spawn) -> None:
    first_gate = spawn(_Gate)
    second_gate = spawn(_Gate)
    first_worker = spawn(_Worker, "w0", 0.0, first_gate)
    second_worker = spawn(_Worker, "w1", 0.0, second_gate)
    submitted: list[Any] = []
    dispatcher = RayActorDispatcher(("w0", "w1"))
    first_task = asyncio.create_task(
        dispatcher.run(
            [
                RayActorJob(
                    0, "w0", _recording(first_worker.hold_then_fail.remote, submitted, []), "first"
                )
            ],
            operation="test.first",
            call_timeout_s=30.0,
        ),
    )
    second_task = asyncio.create_task(
        dispatcher.run(
            [
                RayActorJob(
                    0,
                    "w1",
                    _recording(second_worker.hold_then_fail.remote, submitted, []),
                    "second",
                )
            ],
            operation="test.second",
            call_timeout_s=30.0,
        ),
    )
    await _until(lambda: len(submitted) == 2)

    local_ray.get(first_gate.open.remote(), timeout=30)
    with pytest.raises(RayActorCallError) as first:
        await first_task
    local_ray.get(second_gate.open.remote(), timeout=30)
    with pytest.raises(RuntimeError) as second:
        await second_task

    assert first.value.operation == "test.first"
    assert second.value.__cause__ is first.value


@pytest.mark.asyncio
async def test_cancellation_race_preserves_a_completed_actor_failure(local_ray, spawn) -> None:
    """The actor call failed and the loop observed it, but the caller's
    cancellation lands before the dispatcher handles the failure: the failure
    is the terminal identity, not a generic cancellation."""

    gate = spawn(_Gate)
    worker = spawn(_Worker, "w0", 0.0, gate)
    refs: list[Any] = []
    dispatcher = RayActorDispatcher(("w0",))
    task = asyncio.create_task(
        dispatcher.run(
            [RayActorJob(0, "w0", _recording(worker.hold_then_fail.remote, [], refs), "payload")],
            operation="test.cancel_race",
            call_timeout_s=30.0,
        ),
    )
    await _until(lambda: len(refs) == 1)

    local_ray.get(gate.open.remote(), timeout=30)
    _observe_completion(local_ray, refs)
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError) as caught:
        await task

    assert isinstance(caught.value.__cause__, RayActorCallError)
    actor_error = caught.value.__cause__.__cause__
    assert isinstance(actor_error, ValueError)
    assert "actor failed: payload" in str(actor_error)
    with pytest.raises(RuntimeError) as closed:
        await dispatcher.run(
            [],
            operation="test.after_cancel_race",
            call_timeout_s=30.0,
        )
    assert closed.value.__cause__ is caught.value.__cause__


@pytest.mark.asyncio
async def test_completed_actor_success_wins_cancellation_linearization(local_ray, spawn) -> None:
    gate = spawn(_Gate)
    worker = spawn(_Worker, "w0", 0.0, gate)
    refs: list[Any] = []
    dispatcher = RayActorDispatcher(("w0",))
    task = asyncio.create_task(
        dispatcher.run(
            [RayActorJob(0, "w0", _recording(worker.hold.remote, [], refs), "committed")],
            operation="test.cancel_after_success",
            call_timeout_s=30.0,
        ),
    )
    await _until(lambda: len(refs) == 1)

    local_ray.get(gate.open.remote(), timeout=30)
    _observe_completion(local_ray, refs)
    await asyncio.sleep(0)
    task.cancel()

    assert await task == [(0, "committed")]
    assert await dispatcher.run(
        [RayActorJob(0, "w0", worker.echo.remote, "next")],
        operation="test.after_committed_cancel",
        call_timeout_s=30.0,
    ) == [(0, "next")]


@pytest.mark.asyncio
async def test_actor_completion_is_the_linearization_point_for_cancellation(
    local_ray, spawn
) -> None:
    """The actor finished, but the driver loop has not yet run Ray's completion
    callback when the caller cancels: the committed call still wins, because
    what the actor did -- possibly a weight install -- already happened."""

    gate = spawn(_Gate)
    worker = spawn(_Worker, "w0", 0.0, gate)
    refs: list[Any] = []
    dispatcher = RayActorDispatcher(("w0",))
    task = asyncio.create_task(
        dispatcher.run(
            [RayActorJob(0, "w0", _recording(worker.hold.remote, [], refs), "committed")],
            operation="test.cancel_before_observation",
            call_timeout_s=30.0,
        ),
    )
    await _until(lambda: len(refs) == 1)

    local_ray.get(gate.open.remote(), timeout=30)
    # Block the loop until the call finished: the completion callback is
    # queued, not yet run, when the cancellation is delivered.
    _observe_completion(local_ray, refs)
    task.cancel()

    assert await task == [(0, "committed")]


@pytest.mark.asyncio
async def test_unobserved_actor_failure_wins_over_cancellation(local_ray, spawn) -> None:
    gate = spawn(_Gate)
    worker = spawn(_Worker, "w0", 0.0, gate)
    refs: list[Any] = []
    dispatcher = RayActorDispatcher(("w0",))
    task = asyncio.create_task(
        dispatcher.run(
            [RayActorJob(0, "w0", _recording(worker.hold_then_fail.remote, [], refs), "payload")],
            operation="test.fail_before_observation",
            call_timeout_s=30.0,
        ),
    )
    await _until(lambda: len(refs) == 1)

    local_ray.get(gate.open.remote(), timeout=30)
    _observe_completion(local_ray, refs)
    task.cancel()
    with pytest.raises(asyncio.CancelledError) as caught:
        await task

    assert isinstance(caught.value.__cause__, RayActorCallError)
    assert "actor failed: payload" in str(caught.value.__cause__.__cause__)


def _admission_worker(spawn: Callable[..., Any], held_payload: str) -> tuple[Any, Any, Callable]:
    """One real worker whose ``held_payload`` call blocks on a gate; others echo."""

    gate = spawn(_Gate)
    worker = spawn(_Worker, "w0", 0.0, gate)
    received: list[str] = []

    def remote(payload: str) -> Any:
        received.append(payload)
        if payload == held_payload:
            return worker.hold.remote(payload)
        return worker.echo.remote(payload)

    remote.received = received  # type: ignore[attr-defined]
    return gate, worker, remote


@pytest.mark.asyncio
async def test_concurrent_requests_start_deadline_after_shared_worker_admission(
    local_ray, spawn, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_deadline = actor_pool_module.RayCallDeadline
    deadlines: list[Any] = []

    def recording_deadline(*args: Any, **kwargs: Any) -> Any:
        deadline = real_deadline(*args, **kwargs)
        deadlines.append(deadline)
        return deadline

    gate, _worker, remote = _admission_worker(spawn, "first")
    monkeypatch.setattr(actor_pool_module, "RayCallDeadline", recording_deadline)
    dispatcher = RayActorDispatcher(("w0",))

    async def dispatch(payload: str) -> list[tuple[int, Any]]:
        return await dispatcher.run(
            [RayActorJob(job_index=0, worker_id="w0", remote_method=remote, payload=payload)],
            operation="test.actor_job",
            call_timeout_s=30.0,
        )

    first = asyncio.create_task(dispatch("first"))
    await _until(lambda: remote.received == ["first"])
    second = asyncio.create_task(dispatch("second"))
    await asyncio.sleep(0.1)

    # The second call waits for the worker locally, with no deadline running.
    assert remote.received == ["first"]
    assert len(deadlines) == 1

    local_ray.get(gate.open.remote(), timeout=30)
    assert await first == [(0, "first")]
    assert await second == [(0, "second")]
    assert remote.received == ["first", "second"]
    assert len(deadlines) == 2


@pytest.mark.asyncio
async def test_cancelling_local_admission_wait_keeps_dispatcher_open(local_ray, spawn) -> None:
    gate, _worker, remote = _admission_worker(spawn, "first")
    dispatcher = RayActorDispatcher(("w0",))

    async def dispatch(payload: str) -> list[tuple[int, Any]]:
        return await dispatcher.run(
            [RayActorJob(job_index=0, worker_id="w0", remote_method=remote, payload=payload)],
            operation="test.actor_job",
            call_timeout_s=30.0,
        )

    first = asyncio.create_task(dispatch("first"))
    await _until(lambda: remote.received == ["first"])
    waiting = asyncio.create_task(dispatch("cancel-before-submit"))
    await asyncio.sleep(0.05)
    waiting.cancel()

    with pytest.raises(asyncio.CancelledError) as caught:
        await waiting
    assert caught.value.__cause__ is None
    assert remote.received == ["first"]

    local_ray.get(gate.open.remote(), timeout=30)
    assert await first == [(0, "first")]
    assert await dispatch("third") == [(0, "third")]
    assert remote.received == ["first", "third"]


@pytest.mark.asyncio
async def test_cancelling_middle_admission_waiter_preserves_identity_fifo(
    local_ray, spawn
) -> None:
    gate, _worker, remote = _admission_worker(spawn, "active")
    dispatcher = RayActorDispatcher(("w0",))

    async def dispatch(payload: str) -> list[tuple[int, Any]]:
        return await dispatcher.run(
            [RayActorJob(0, "w0", remote, payload)],
            operation="test.identity_fifo",
            call_timeout_s=30.0,
        )

    active = asyncio.create_task(dispatch("active"))
    await _until(lambda: remote.received == ["active"])
    head = asyncio.create_task(dispatch("head"))
    await asyncio.sleep(0)
    middle = asyncio.create_task(dispatch("middle"))
    await asyncio.sleep(0)
    tail = asyncio.create_task(dispatch("tail"))
    await _until(lambda: len(dispatcher._admission_queues["w0"]) == 3)

    middle.cancel()
    with pytest.raises(asyncio.CancelledError):
        await middle
    assert len(dispatcher._admission_queues["w0"]) == 2

    local_ray.get(gate.open.remote(), timeout=30)
    assert await asyncio.wait_for(active, timeout=30) == [(0, "active")]
    assert await asyncio.wait_for(head, timeout=30) == [(0, "head")]
    assert await asyncio.wait_for(tail, timeout=30) == [(0, "tail")]
    assert await dispatch("after") == [(0, "after")]
    assert remote.received == ["active", "head", "tail", "after"]


# ----------------------------------------------------- executor end to end


class _ListGatherer:
    """Gathers the workers' model-free batch payloads in batch order.

    The batch payload shape is owned by the binding that produced it; these
    workers produce coordinates, so this is the binding's gatherer.
    """

    def merge_generation_batches(
        self,
        request: GenerationRequest,
        batches: list[Any],
    ) -> GenerationOutput:
        return GenerationOutput(
            output=list(batches),
            trajectory=TrajectoryBatch(
                request_id=request.request_id,
                family=request.family,
                task=request.task,
                sample_rows=request.sample_rows(),
                axes={},
                segments={},
            ),
        )


def _executor(actors: list[tuple[str, Any]]) -> RayGenerationExecutor:
    engines = [
        RayGenerationEngine(worker_id, [RayActorHandle(worker_id=worker_id, actor=actor)])
        for worker_id, actor in actors
    ]
    return RayGenerationExecutor(
        engines,
        _ListGatherer(),
        actor_dispatcher=RayActorDispatcher(tuple(engine.engine_id for engine in engines)),
        generation_stall_timeout_s=30.0,
    )


@pytest.mark.asyncio
async def test_executor_round_robin_dispatches_per_plan_binding(local_ray, spawn) -> None:
    """Round-robin plan binding reaches the real dispatch: a slow engine keeps
    its own batches."""

    fast = spawn(_Worker, "w0", _FAST_S)
    slow = spawn(_Worker, "w1", _SLOW_S / 2)
    executor = _executor([("w0", fast), ("w1", slow)])

    output = await executor.execute(_request(num_steps=10, samples=8, sbs=2))

    assert _received(local_ray, fast) == ["prompt:0:samples:0:2", "prompt:0:samples:4:6"]
    assert _received(local_ray, slow) == ["prompt:0:samples:2:4", "prompt:0:samples:6:8"]
    assert output.runtime_debug is not None
    schedule = output.runtime_debug["chunk_schedule"]
    assert [row["assigned_worker"] for row in schedule] == ["w0", "w1", "w0", "w1"]
    for row in schedule:
        assert row["sample_count"] == 2


@pytest.mark.asyncio
async def test_executor_runtime_debug_exposes_chunk_schedule(spawn) -> None:
    """runtime_debug surfaces per-batch placement telemetry."""

    executor = _executor([("w0", spawn(_Worker, "w0")), ("w1", spawn(_Worker, "w1"))])
    request = _request(num_steps=10, samples=8, sbs=2)
    request.runtime_debug = True

    output = await executor.execute(request)

    assert output.runtime_debug is not None
    debug_schedule = output.runtime_debug["chunk_schedule"]
    assert len(debug_schedule) == 4
    for row in debug_schedule:
        assert {
            "batch_key",
            "sample_count",
            "assigned_worker",
            "queue_wait_s",
            "execution_s",
        } <= set(row)


@pytest.mark.parametrize("worker_ids", ["ab", b"ab", [""], [1], [["w0"]]])
def test_dispatcher_rejects_invalid_worker_identity(worker_ids):
    with pytest.raises(ValueError, match="worker ids must be"):
        RayActorDispatcher(worker_ids)
