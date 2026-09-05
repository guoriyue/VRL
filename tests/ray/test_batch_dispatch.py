"""Deterministic tests for batch placement and pull-based actor dispatch.

These use awaitable fake refs so completion order is fully controlled — no Ray
runtime, no slow markers. They pin the Track A contract:
round_robin keeps plan-time binding bit-for-bit; dynamic binds at dispatch
time (pull + LPT) and never changes gather order.

The fakes are a controlled clock, not a Ray protocol fake, and the ``real_cover``
labels below name what covers each half for real: the dispatch loop's ObjectRef
handling by ``test_ray_actor_pool.py``, and the executor's whole
envelope-over-the-wire-to-result crossing by the real-cluster twins in
``tests/ray/test_real_batch_execution.py``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from typing import Any

import pytest

import vrl.ray.actor_pool as actor_pool_module
import vrl.ray.operation_deadline as deadline_module
from vrl.generation.execution.batch_placement import DistributedExecutionPlanner
from vrl.generation.execution.types import (
    BatchSizeProbeResult,
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
from vrl.trajectory import TrajectoryBatch

# Carried by the tests that actually drive `_FakeRef`/`_FakeWorker`; the planner
# and argument-validation tests below use no double, so a module-level pytestmark
# would over-claim on their behalf.
_CONTROLLED_CLOCK = pytest.mark.real_cover(
    "tests/ray/test_ray_actor_pool.py::test_actor_dispatcher_awaits_real_object_refs",
    why=(
        "the fake refs control event-loop completion ORDER, which a real Ray cluster cannot "
        "make deterministic; the protocol assumption they encode — a real ObjectRef awaits "
        "directly and resolves to the task result — is pinned against a live cluster there"
    ),
)


class _FakeRef:
    """One in-flight fake actor call; completion_rank orders completion.

    ``RayActorDispatcher`` awaits refs directly (like real Ray ObjectRefs), so
    the fake controls order by suspending ``completion_rank`` event-loop steps
    before resolving: a lower-rank ref finishes first, one per ``asyncio.wait``
    iteration, with no wall-clock sleeps.
    """

    def __init__(self, result: Any, completion_rank: int) -> None:
        self.result = result
        self.completion_rank = completion_rank

    def __await__(self) -> Generator[Any, None, Any]:
        for _ in range(self.completion_rank):
            yield
        return self.result


class _NeverRef:
    def __await__(self) -> Generator[Any, None, Any]:
        while True:
            yield


class _GatedRef:
    def __init__(self, gate: asyncio.Event, result: Any) -> None:
        self.gate = gate
        self.result = result

    def __await__(self):
        async def wait() -> Any:
            await self.gate.wait()
            return self.result

        return wait().__await__()


class _GatedErrorRef:
    def __init__(self, gate: asyncio.Event, error: BaseException) -> None:
        self.gate = gate
        self.error = error

    def __await__(self):
        async def wait() -> Any:
            await self.gate.wait()
            raise self.error

        return wait().__await__()


class _FakeWorker:
    """Remote method returning refs whose completion rank encodes speed."""

    def __init__(self, worker_id: str, speed_rank_base: int) -> None:
        self.worker_id = worker_id
        self._rank = speed_rank_base
        self.received: list[Any] = []

    def remote(self, payload: Any) -> _FakeRef:
        self.received.append(payload)
        self._rank += 1
        return _FakeRef(result=(self.worker_id, payload), completion_rank=self._rank)


def _request(num_steps: int = 10, samples: int = 8, sbs: int = 2) -> GenerationRequest:
    return GenerationRequest(
        request_id="req-dispatch",
        family="sd3_5",
        task="t2i",
        inputs=["a test prompt"],
        samples_per_prompt=samples,
        sampling={
            "height": 64,
            "width": 64,
            "num_steps": num_steps,
            "samples_per_generation_batch": sbs,
        },
        runtime_debug=True,
    )


def _worker_ids(count: int) -> list[str]:
    return [f"w{idx}" for idx in range(count)]


# ---------------------------------------------------------------- actor pool


@_CONTROLLED_CLOCK
def test_bound_jobs_keep_plan_time_binding_and_order() -> None:
    """Checks the static path is unchanged: binding and order preserved."""
    fast = _FakeWorker("w0", speed_rank_base=0)
    slow = _FakeWorker("w1", speed_rank_base=100)
    jobs = [
        RayActorJob(
            job_index=i,
            worker_id=("w0" if i % 2 == 0 else "w1"),
            remote_method=(fast.remote if i % 2 == 0 else slow.remote),
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
    assert fast.received == ["batch-0", "batch-2"]
    assert slow.received == ["batch-1", "batch-3"]


@_CONTROLLED_CLOCK
def test_pull_dispatch_lets_fast_worker_take_more_chunks() -> None:
    """Checks unbound jobs flow to whichever worker frees up first."""
    fast = _FakeWorker("w0", speed_rank_base=0)
    slow = _FakeWorker("w1", speed_rank_base=100)
    jobs = [
        RayActorJob(job_index=i, worker_id=None, remote_method=None, payload=f"batch-{i}")
        for i in range(4)
    ]

    pairs = asyncio.run(
        RayActorDispatcher(("w0", "w1")).run(
            jobs,
            operation="test.actor_job",
            call_timeout_s=30.0,
            worker_methods={"w0": fast.remote, "w1": slow.remote},
        ),
    )

    assert [index for index, _ in pairs] == [0, 1, 2, 3]
    # w0 completes first every time, so it pulls every queued batch.
    assert fast.received == ["batch-0", "batch-2", "batch-3"]
    assert slow.received == ["batch-1"]


@_CONTROLLED_CLOCK
def test_lpt_priority_orders_submission() -> None:
    """Checks higher-priority (more expensive) batches are submitted first."""
    worker = _FakeWorker("w0", speed_rank_base=0)
    jobs = [
        RayActorJob(
            job_index=0, worker_id=None, remote_method=None, payload="small", priority=1.0
        ),
        RayActorJob(
            job_index=1, worker_id=None, remote_method=None, payload="large", priority=10.0
        ),
        RayActorJob(
            job_index=2, worker_id=None, remote_method=None, payload="medium", priority=5.0
        ),
    ]

    pairs = asyncio.run(
        RayActorDispatcher(("w0",)).run(
            jobs,
            operation="test.actor_job",
            call_timeout_s=30.0,
            worker_methods={"w0": worker.remote},
        ),
    )

    assert worker.received == ["large", "medium", "small"]
    # Gather order stays by job_index regardless of submission order.
    assert [index for index, _ in pairs] == [0, 1, 2]


def test_unbound_jobs_without_worker_methods_fail_loudly() -> None:
    """Checks pull dispatch without worker handles is a hard error."""
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
async def test_actor_pool_validates_every_worker_before_first_submission() -> None:
    submitted: list[str] = []

    def submit(payload: str) -> _FakeRef:
        submitted.append(payload)
        return _FakeRef(payload, completion_rank=1)

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


@_CONTROLLED_CLOCK
def test_schedule_telemetry_rows_are_emitted() -> None:
    """Checks the dispatch loop emits one telemetry row per job."""
    worker = _FakeWorker("w0", speed_rank_base=0)
    jobs = [
        RayActorJob(job_index=i, worker_id="w0", remote_method=worker.remote, payload=i)
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Deadline:
        def __init__(self, operation: str, timeout_s: float, context: str | None = None) -> None:
            del operation, timeout_s
            self.context = context
            self.expires_at = 0.0 if "worker_id=w1" in str(context) else 1.0
            self._remaining_calls = 0

        def remaining_s(self) -> float:
            self._remaining_calls += 1
            if "worker_id=w1" in str(self.context) and self._remaining_calls > 1:
                return 0.0
            return 30.0

        def timeout_error(self) -> RayOperationTimeout:
            return RayOperationTimeout(
                "rollout.generation.batch",
                30.0,
                context=self.context,
            )

    cancelled: list[Any] = []

    class _Ray:
        @staticmethod
        def cancel(ref: Any, *, force: bool) -> None:
            assert force is False
            cancelled.append(ref)

    fast = _FakeWorker("w0", speed_rank_base=0)
    never_ref = _NeverRef()
    monkeypatch.setattr(actor_pool_module, "RayCallDeadline", _Deadline)
    monkeypatch.setattr(deadline_module, "require_ray", lambda: _Ray)
    jobs = [
        RayActorJob(
            job_index=0,
            worker_id="w0",
            remote_method=fast.remote,
            payload="complete-first",
        ),
        RayActorJob(
            job_index=1,
            worker_id="w1",
            remote_method=lambda _payload: never_ref,
            payload="never",
        ),
    ]

    with pytest.raises(RayOperationTimeout, match="worker_id=w1"):
        await RayActorDispatcher(("w0", "w1")).run(
            jobs,
            operation="test.actor_job",
            call_timeout_s=30.0,
        )

    assert fast.received == ["complete-first"]
    assert cancelled == [never_ref]


@pytest.mark.asyncio
async def test_queued_job_gets_its_deadline_only_when_submitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_deadline = actor_pool_module.RayCallDeadline
    deadlines: list[Any] = []
    deadlines_seen_at_submit: list[int] = []

    def recording_deadline(*args: Any, **kwargs: Any) -> Any:
        deadline = real_deadline(*args, **kwargs)
        deadlines.append(deadline)
        return deadline

    class _Worker:
        def remote(self, payload: str) -> _FakeRef:
            deadlines_seen_at_submit.append(len(deadlines))
            return _FakeRef(payload, completion_rank=1)

    monkeypatch.setattr(actor_pool_module, "RayCallDeadline", recording_deadline)
    worker = _Worker()
    jobs = [
        RayActorJob(
            job_index=index,
            worker_id="w0",
            remote_method=worker.remote,
            payload=f"job-{index}",
        )
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


@_CONTROLLED_CLOCK
@pytest.mark.asyncio
async def test_partial_submission_failure_cancels_registered_refs_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active_ref = _NeverRef()
    cancelled: list[Any] = []

    class _Ray:
        @staticmethod
        def cancel(ref: Any, *, force: bool) -> None:
            assert force is False
            cancelled.append(ref)

    def reject(_payload: Any) -> Any:
        raise ValueError("driver submission failed")

    monkeypatch.setattr(deadline_module, "require_ray", lambda: _Ray)
    dispatcher = RayActorDispatcher(("w0", "w1"))
    jobs = [
        RayActorJob(0, "w0", lambda _payload: active_ref, "active"),
        RayActorJob(1, "w1", reject, "rejected"),
    ]

    with pytest.raises(RayActorCallError) as caught:
        await dispatcher.run(
            jobs,
            operation="test.partial_submit",
            call_timeout_s=30.0,
        )

    assert isinstance(caught.value.__cause__, ValueError)
    assert cancelled == [active_ref]
    with pytest.raises(RuntimeError) as closed:
        await dispatcher.run(
            [],
            operation="test.after_partial_submit",
            call_timeout_s=30.0,
        )
    assert closed.value.__cause__ is caught.value


@_CONTROLLED_CLOCK
@pytest.mark.asyncio
async def test_concurrent_runs_propagate_the_first_terminal_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_gate = asyncio.Event()
    second_gate = asyncio.Event()
    first_ref = _GatedErrorRef(first_gate, ValueError("first actor failure"))
    second_ref = _GatedErrorRef(second_gate, ValueError("second actor failure"))

    class _Ray:
        @staticmethod
        def cancel(_ref: Any, *, force: bool) -> None:
            assert force is False

    monkeypatch.setattr(deadline_module, "require_ray", lambda: _Ray)
    dispatcher = RayActorDispatcher(("w0", "w1"))
    first_task = asyncio.create_task(
        dispatcher.run(
            [RayActorJob(0, "w0", lambda _payload: first_ref, "first")],
            operation="test.first",
            call_timeout_s=30.0,
        ),
    )
    second_task = asyncio.create_task(
        dispatcher.run(
            [RayActorJob(0, "w1", lambda _payload: second_ref, "second")],
            operation="test.second",
            call_timeout_s=30.0,
        ),
    )
    await asyncio.sleep(0)

    first_gate.set()
    with pytest.raises(RayActorCallError) as first:
        await first_task
    second_gate.set()
    with pytest.raises(RuntimeError) as second:
        await second_task

    assert first.value.operation == "test.first"
    assert second.value.__cause__ is first.value


@_CONTROLLED_CLOCK
@pytest.mark.asyncio
async def test_cancellation_race_preserves_a_completed_actor_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = asyncio.Event()
    actor_error = ValueError("actor failed before caller cancellation")
    ref = _GatedErrorRef(gate, actor_error)

    class _Ray:
        @staticmethod
        def cancel(_ref: Any, *, force: bool) -> None:
            assert force is False

    monkeypatch.setattr(deadline_module, "require_ray", lambda: _Ray)
    dispatcher = RayActorDispatcher(("w0",))
    task = asyncio.create_task(
        dispatcher.run(
            [RayActorJob(0, "w0", lambda _payload: ref, "payload")],
            operation="test.cancel_race",
            call_timeout_s=30.0,
        ),
    )
    await asyncio.sleep(0)

    gate.set()
    task.cancel()
    with pytest.raises(asyncio.CancelledError) as caught:
        await task

    assert isinstance(caught.value.__cause__, RayActorCallError)
    assert caught.value.__cause__.__cause__ is actor_error
    with pytest.raises(RuntimeError) as closed:
        await dispatcher.run(
            [],
            operation="test.after_cancel_race",
            call_timeout_s=30.0,
        )
    assert closed.value.__cause__ is caught.value.__cause__


@_CONTROLLED_CLOCK
@pytest.mark.asyncio
async def test_completed_actor_success_wins_cancellation_linearization() -> None:
    gate = asyncio.Event()
    dispatcher = RayActorDispatcher(("w0",))
    task = asyncio.create_task(
        dispatcher.run(
            [RayActorJob(0, "w0", lambda _payload: _GatedRef(gate, "committed"), None)],
            operation="test.cancel_after_success",
            call_timeout_s=30.0,
        ),
    )
    await asyncio.sleep(0)

    gate.set()
    task.cancel()

    assert await task == [(0, "committed")]
    assert await dispatcher.run(
        [RayActorJob(0, "w0", lambda _payload: _FakeRef("next", 1), None)],
        operation="test.after_committed_cancel",
        call_timeout_s=30.0,
    ) == [(0, "next")]


@pytest.mark.asyncio
async def test_concurrent_requests_start_deadline_after_shared_worker_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_deadline = actor_pool_module.RayCallDeadline
    deadlines: list[Any] = []
    first_gate = asyncio.Event()
    received: list[str] = []

    def recording_deadline(*args: Any, **kwargs: Any) -> Any:
        deadline = real_deadline(*args, **kwargs)
        deadlines.append(deadline)
        return deadline

    def remote(payload: str) -> Any:
        received.append(payload)
        if payload == "first":
            return _GatedRef(first_gate, payload)
        return _FakeRef(payload, completion_rank=1)

    monkeypatch.setattr(actor_pool_module, "RayCallDeadline", recording_deadline)
    dispatcher = RayActorDispatcher(("w0",))

    async def dispatch(payload: str) -> list[tuple[int, Any]]:
        return await dispatcher.run(
            [
                RayActorJob(
                    job_index=0,
                    worker_id="w0",
                    remote_method=remote,
                    payload=payload,
                ),
            ],
            operation="test.actor_job",
            call_timeout_s=30.0,
        )

    first = asyncio.create_task(dispatch("first"))
    await asyncio.sleep(0)
    second = asyncio.create_task(dispatch("second"))
    await asyncio.sleep(0)

    assert received == ["first"]
    assert len(deadlines) == 1

    first_gate.set()
    assert await first == [(0, "first")]
    assert await second == [(0, "second")]
    assert received == ["first", "second"]
    assert len(deadlines) == 2


@pytest.mark.asyncio
async def test_cancelling_local_admission_wait_keeps_dispatcher_open() -> None:
    first_gate = asyncio.Event()
    received: list[str] = []

    def remote(payload: str) -> Any:
        received.append(payload)
        if payload == "first":
            return _GatedRef(first_gate, payload)
        return _FakeRef(payload, completion_rank=1)

    dispatcher = RayActorDispatcher(("w0",))

    async def dispatch(payload: str) -> list[tuple[int, Any]]:
        return await dispatcher.run(
            [
                RayActorJob(
                    job_index=0,
                    worker_id="w0",
                    remote_method=remote,
                    payload=payload,
                ),
            ],
            operation="test.actor_job",
            call_timeout_s=30.0,
        )

    first = asyncio.create_task(dispatch("first"))
    await asyncio.sleep(0)
    waiting = asyncio.create_task(dispatch("cancel-before-submit"))
    await asyncio.sleep(0)
    waiting.cancel()

    with pytest.raises(asyncio.CancelledError) as caught:
        await waiting
    assert caught.value.__cause__ is None
    assert received == ["first"]

    first_gate.set()
    assert await first == [(0, "first")]
    assert await dispatch("third") == [(0, "third")]
    assert received == ["first", "third"]


@pytest.mark.asyncio
async def test_cancelling_middle_admission_waiter_preserves_identity_fifo() -> None:
    active_gate = asyncio.Event()
    received: list[str] = []

    def remote(payload: str) -> Any:
        received.append(payload)
        if payload == "active":
            return _GatedRef(active_gate, payload)
        return _FakeRef(payload, completion_rank=1)

    dispatcher = RayActorDispatcher(("w0",))

    async def dispatch(payload: str) -> list[tuple[int, Any]]:
        return await dispatcher.run(
            [RayActorJob(0, "w0", remote, payload)],
            operation="test.identity_fifo",
            call_timeout_s=30.0,
        )

    active = asyncio.create_task(dispatch("active"))
    await asyncio.sleep(0)
    head = asyncio.create_task(dispatch("head"))
    await asyncio.sleep(0)
    middle = asyncio.create_task(dispatch("middle"))
    await asyncio.sleep(0)
    tail = asyncio.create_task(dispatch("tail"))

    for _ in range(20):
        if len(dispatcher._admission_queues["w0"]) == 3:
            break
        await asyncio.sleep(0)
    assert len(dispatcher._admission_queues["w0"]) == 3

    middle.cancel()
    with pytest.raises(asyncio.CancelledError):
        await middle
    assert len(dispatcher._admission_queues["w0"]) == 2

    active_gate.set()
    assert await asyncio.wait_for(active, timeout=1) == [(0, "active")]
    assert await asyncio.wait_for(head, timeout=1) == [(0, "head")]
    assert await asyncio.wait_for(tail, timeout=1) == [(0, "tail")]
    assert await dispatch("after") == [(0, "after")]
    assert received == ["active", "head", "tail", "after"]


def test_round_robin_planner_binds_workers_at_plan_time() -> None:
    """Checks the default strategy keeps the historical binding."""
    planner = DistributedExecutionPlanner()
    plan = planner.plan_with_engine(_request(), _worker_ids(2))

    worker_ids = [assignment.engine_id for assignment in plan.assignments]
    assert worker_ids == ["w0", "w1", "w0", "w1"]
    assert all(a.estimated_cost > 0 for a in plan.assignments)


def test_dynamic_planner_leaves_chunks_unbound_with_costs() -> None:
    """Checks dynamic strategy defers binding and carries cost estimates."""
    planner = DistributedExecutionPlanner(strategy="dynamic")
    request = _request(num_steps=10, samples=8, sbs=2)
    plan = planner.plan_with_engine(request, _worker_ids(2))

    assert all(a.engine_id is None for a in plan.assignments)
    assert len(plan.assignments) == 4
    assert all(a.estimated_cost > 0 for a in plan.assignments)
    assert all(a.batch is a.envelope.batch for a in plan.assignments)
    # Batch identity and order (the gather contract) are untouched.
    assert [a.batch.sample_start for a in plan.assignments] == [0, 2, 4, 6]


def test_planner_cost_uses_steps_axis() -> None:
    """Checks the cost hint scales with samples x steps."""
    request = _request(num_steps=35, samples=4, sbs=4)
    plan = DistributedExecutionPlanner().plan_with_engine(request, _worker_ids(1))

    assignment = plan.assignments[0]
    assert assignment.estimated_cost == assignment.batch.sample_count * 35


def test_planner_rejects_unknown_strategy() -> None:
    """Checks the strategy vocabulary is closed."""
    with pytest.raises(ValueError, match="round_robin"):
        DistributedExecutionPlanner(strategy="work_stealing")  # type: ignore[arg-type]


# ----------------------------------------------------- executor end to end


# Carried by the three `execute` tests. Their fake actor is called in-process, so
# no envelope is ever pickled: a field that became unserializable (a lambda, an
# open handle, a torch device reference) would pass here and break on production's
# first batch. That crossing is what the twins named here run for real; the
# completion ORDER these tests pin is what a real cluster cannot give.
_CONTROLLED_CLOCK_OVER_A_REAL_WIRE = pytest.mark.real_cover(
    "tests/ray/test_real_batch_execution.py",
    why=(
        "a real cluster cannot make batch completion order deterministic, which is the whole "
        "point of the fake refs; the envelope -> pickle -> actor -> GenerationBatchResult crossing "
        "they therefore skip is pinned against a live cluster by both twins in the named file"
    ),
)


class _FakeActor:
    """Fake Ray actor: execute_batch.remote returns a real batch result."""

    def __init__(self, worker_id: str, speed_rank_base: int) -> None:
        self.worker_id = worker_id
        self._rank = speed_rank_base
        self.executed: list[str] = []

        class _ExecuteChunk:
            @staticmethod
            def remote(envelope: GenerationBatchEnvelope) -> _FakeRef:
                return self._execute(envelope)

        self.execute_batch = _ExecuteChunk()

    def _execute(self, envelope: GenerationBatchEnvelope) -> _FakeRef:
        batch = envelope.batch
        self.executed.append(batch.batch_key)
        self._rank += 1
        result = GenerationBatchResult(
            request_id=envelope.request.request_id,
            worker_id=self.worker_id,
            batch=batch,
            output={"batch_key": batch.batch_key, "samples": batch.sample_count},
        )
        return _FakeRef(result=result, completion_rank=self._rank)


class _ListGatherer:
    def gather_batches(
        self,
        request: GenerationRequest,
        sample_rows: Any,
        batches: list[Any],
    ) -> GenerationOutput:
        return GenerationOutput(
            output=list(batches),
            trajectory=TrajectoryBatch(
                request_id=request.request_id,
                family=request.family,
                task=request.task,
                sample_rows=list(sample_rows),
                axes={},
                segments={},
            ),
        )


def _executor(strategy: str, actors: list[_FakeActor]) -> RayGenerationExecutor:
    engines = [
        RayGenerationEngine(
            actor.worker_id,
            [RayActorHandle(worker_id=actor.worker_id, actor=actor)],
        )
        for actor in actors
    ]
    return RayGenerationExecutor(
        DistributedExecutionPlanner(strategy=strategy),  # type: ignore[arg-type]
        engines,
        _ListGatherer(),
        actor_dispatcher=RayActorDispatcher(
            tuple(engine.engine_id for engine in engines),
        ),
        generation_stall_timeout_s=30.0,
    )


@pytest.mark.real_cover(
    "tests/generation/ray/test_runtime_config.py"
    "::test_real_ray_probe_fan_out_resolves_auto_once_across_the_fleet",
    why=(
        "the gated ref makes probe/batch submission order deterministic; the named "
        "test sends the real request and keyword arguments through live Ray actors"
    ),
)
@pytest.mark.asyncio
async def test_batch_size_probe_shares_actor_admission_with_explicit_generation() -> None:
    gate = asyncio.Event()
    probe_requests: list[str] = []
    actor = _FakeActor("w0", 0)

    class _Probe:
        @staticmethod
        def remote(
            request: GenerationRequest,
            *,
            max_samples: int,
        ) -> _GatedRef:
            assert max_samples == 8
            probe_requests.append(request.request_id)
            return _GatedRef(
                gate,
                BatchSizeProbeResult(
                    samples_per_generation_batch=2,
                    budget_bytes=1,
                    trials=(),
                ),
            )

    actor.probe_batch_size = _Probe()
    executor = _executor("round_robin", [actor])
    request = _request(samples=2, sbs=1)
    probe = asyncio.create_task(
        executor.probe_batch_sizes(request, max_samples=8),
    )
    await asyncio.sleep(0)
    generation = asyncio.create_task(executor.execute(request))
    await asyncio.sleep(0)

    assert probe_requests == [request.request_id]
    assert actor.executed == []

    gate.set()
    assert await probe == [
        BatchSizeProbeResult(
            samples_per_generation_batch=2,
            budget_bytes=1,
            trials=(),
        ),
    ]
    output = await generation
    assert len(output.output) == 2
    assert actor.executed == [
        "prompt:0:samples:0:1",
        "prompt:0:samples:1:2",
    ]


@_CONTROLLED_CLOCK_OVER_A_REAL_WIRE
@pytest.mark.asyncio
async def test_executor_round_robin_dispatches_per_plan_binding() -> None:
    """Checks config strategy round_robin reaches the actual dispatch."""
    actors = [_FakeActor("w0", 0), _FakeActor("w1", 100)]
    executor = _executor("round_robin", actors)

    request = _request(num_steps=10, samples=8, sbs=2)
    output = await executor.execute(request)

    # 4 batches alternate w0/w1 even though w1 is much slower: plan-time binding.
    assert actors[0].executed == ["prompt:0:samples:0:2", "prompt:0:samples:4:6"]
    assert actors[1].executed == ["prompt:0:samples:2:4", "prompt:0:samples:6:8"]
    assert output.runtime_debug is not None
    schedule = output.runtime_debug["chunk_schedule"]
    assert [row["assigned_worker"] for row in schedule] == ["w0", "w1", "w0", "w1"]
    for row in schedule:
        assert row["assignment_strategy"] == "round_robin"
        assert row["sample_count"] == 2
        # Cost follows the source formula (samples x num_steps), not a literal.
        assert row["estimated_cost"] == row["sample_count"] * request.sampling["num_steps"]


@_CONTROLLED_CLOCK_OVER_A_REAL_WIRE
@pytest.mark.asyncio
async def test_executor_dynamic_dispatches_by_pull() -> None:
    """Checks config strategy dynamic actually changes worker placement."""
    actors = [_FakeActor("w0", 0), _FakeActor("w1", 100)]
    executor = _executor("dynamic", actors)

    output = await executor.execute(_request(num_steps=10, samples=8, sbs=2))

    # The fast worker pulls every queued batch once the slow one is busy.
    assert len(actors[0].executed) == 3
    assert len(actors[1].executed) == 1
    assert output.runtime_debug is not None
    schedule = output.runtime_debug["chunk_schedule"]
    assert all(row["assignment_strategy"] == "dynamic" for row in schedule)
    by_worker = {
        worker: sum(1 for row in schedule if row["assigned_worker"] == worker)
        for worker in ("w0", "w1")
    }
    assert by_worker == {"w0": 3, "w1": 1}
    # Gather order is untouched by dynamic placement.
    assert [entry["batch_key"] for entry in output.output] == [
        "prompt:0:samples:0:2",
        "prompt:0:samples:2:4",
        "prompt:0:samples:4:6",
        "prompt:0:samples:6:8",
    ]


@_CONTROLLED_CLOCK_OVER_A_REAL_WIRE
@pytest.mark.asyncio
async def test_executor_runtime_debug_exposes_chunk_schedule() -> None:
    """Checks runtime_debug surfaces per-batch placement telemetry."""
    actors = [_FakeActor("w0", 0), _FakeActor("w1", 100)]
    executor = _executor("round_robin", actors)
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
            "assignment_strategy",
            "estimated_cost",
            "assigned_worker",
            "queue_wait_s",
            "execution_s",
        } <= set(row)
