"""Deterministic tests for chunk placement and pull-based actor dispatch.

These use awaitable fake refs so completion order is fully controlled — no Ray
runtime, no slow markers. They pin the Track A contract:
round_robin keeps plan-time binding bit-for-bit; dynamic binds at dispatch
time (pull + LPT) and never changes gather order.

The fakes are a controlled clock, not a Ray protocol fake, and the ``real_cover``
labels below name what covers each half for real: the dispatch loop's ObjectRef
handling by ``test_ray_actor_pool.py``, and the executor's whole
envelope-over-the-wire-to-result crossing by the real-cluster twins in
``tests/ray/test_real_chunk_execution.py``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from typing import Any

import pytest

import vrl.ray.actor_pool as actor_pool_module
import vrl.ray.operation_deadline as deadline_module
from vrl.generation.execution.chunk_placement import (
    ChunkPlacementPolicy,
    DistributedExecutionPlanner,
)
from vrl.generation.execution.types import (
    ChunkExecutionEnvelope,
    ChunkExecutionResult,
    DistributedWorkerHandle,
)
from vrl.generation.ray.executor import RayGenerationExecutor
from vrl.generation.types import GenerationOutput, GenerationRequest
from vrl.ray.actor_pool import (
    RayActorCallError,
    RayActorDispatcher,
    RayActorJob,
    run_actor_jobs,
)
from vrl.ray.operation_deadline import RayOperationCancelled, RayOperationTimeout

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
        sampling={"height": 64, "width": 64, "num_steps": num_steps, "samples_per_chunk": sbs},
        metadata={"dataset": "unit", "_runtime_debug": True},
    )


def _workers(count: int) -> list[DistributedWorkerHandle]:
    return [
        DistributedWorkerHandle(
            worker_id=f"w{idx}",
            actor=None,
        )
        for idx in range(count)
    ]


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
            payload=f"chunk-{i}",
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
    # Even though w1 is slow, its chunks never migrate to w0.
    assert fast.received == ["chunk-0", "chunk-2"]
    assert slow.received == ["chunk-1", "chunk-3"]


@_CONTROLLED_CLOCK
def test_pull_dispatch_lets_fast_worker_take_more_chunks() -> None:
    """Checks unbound jobs flow to whichever worker frees up first."""
    fast = _FakeWorker("w0", speed_rank_base=0)
    slow = _FakeWorker("w1", speed_rank_base=100)
    jobs = [
        RayActorJob(job_index=i, worker_id=None, remote_method=None, payload=f"chunk-{i}")
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
    # w0 completes first every time, so it pulls every queued chunk.
    assert fast.received == ["chunk-0", "chunk-2", "chunk-3"]
    assert slow.received == ["chunk-1"]


@_CONTROLLED_CLOCK
def test_lpt_priority_orders_submission() -> None:
    """Checks higher-priority (more expensive) chunks are submitted first."""
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
                "rollout.generation.chunk",
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


@_CONTROLLED_CLOCK
def test_run_actor_jobs_compatibility_facade_uses_dispatcher() -> None:
    worker = _FakeWorker("w0", speed_rank_base=0)
    jobs = [
        RayActorJob(
            job_index=index,
            worker_id="w0",
            remote_method=worker.remote,
            payload=f"chunk-{index}",
        )
        for index in range(2)
    ]

    with pytest.warns(DeprecationWarning, match="RayActorDispatcher"):
        pairs = asyncio.run(
            run_actor_jobs(
                jobs,
                generation_stall_timeout_s=30.0,
            ),
        )

    assert pairs == [
        (0, ("w0", "chunk-0")),
        (1, ("w0", "chunk-1")),
    ]


@pytest.mark.asyncio
async def test_admission_wait_tracks_new_worker_without_losing_per_worker_fifo() -> None:
    w1_gate = asyncio.Event()
    a0_gate = asyncio.Event()
    b_gate = asyncio.Event()
    w1_submitted = asyncio.Event()
    a0_submitted = asyncio.Event()
    a1_submitted = asyncio.Event()
    a2_submitted = asyncio.Event()
    b_submitted = asyncio.Event()
    w0_order: list[str] = []

    def hold_w1(payload: str) -> _GatedRef:
        w1_submitted.set()
        return _GatedRef(w1_gate, payload)

    def run_a_w0(payload: str) -> Any:
        w0_order.append(payload)
        if payload == "a0":
            a0_submitted.set()
            return _GatedRef(a0_gate, payload)
        a1_submitted.set()
        return _FakeRef(payload, completion_rank=1)

    def run_a_w1(payload: str) -> _FakeRef:
        a2_submitted.set()
        return _FakeRef(payload, completion_rank=1)

    def run_b(payload: str) -> _GatedRef:
        w0_order.append(payload)
        b_submitted.set()
        return _GatedRef(b_gate, payload)

    dispatcher = RayActorDispatcher(("w0", "w1"))
    holding_w1 = asyncio.create_task(
        dispatcher.run(
            [RayActorJob(0, "w1", hold_w1, "hold-w1")],
            operation="test.hold_w1",
            call_timeout_s=30.0,
        ),
    )
    await w1_submitted.wait()
    run_a = asyncio.create_task(
        dispatcher.run(
            [
                RayActorJob(0, "w0", run_a_w0, "a0"),
                RayActorJob(1, "w0", run_a_w0, "a1"),
                RayActorJob(2, "w1", run_a_w1, "a2"),
            ],
            operation="test.dynamic_candidates",
            call_timeout_s=30.0,
        ),
    )
    await a0_submitted.wait()
    run_b_task = asyncio.create_task(
        dispatcher.run(
            [RayActorJob(0, "w0", run_b, "b")],
            operation="test.wait_w0_first",
            call_timeout_s=30.0,
        ),
    )

    try:
        for _ in range(20):
            if dispatcher._admission_queues["w0"]:
                break
            await asyncio.sleep(0)
        assert dispatcher._admission_queues["w0"]

        a0_gate.set()
        await asyncio.wait_for(b_submitted.wait(), timeout=1)
        assert w0_order == ["a0", "b"]
        assert not a1_submitted.is_set()

        b_gate.set()
        await asyncio.wait_for(a1_submitted.wait(), timeout=1)
        assert w0_order == ["a0", "b", "a1"]
        assert not a2_submitted.is_set()

        w1_gate.set()
        await asyncio.wait_for(a2_submitted.wait(), timeout=1)
        assert await holding_w1 == [(0, "hold-w1")]
        assert await run_b_task == [(0, "b")]
        assert await run_a == [(0, "a0"), (1, "a1"), (2, "a2")]
    finally:
        a0_gate.set()
        b_gate.set()
        w1_gate.set()
        await asyncio.gather(
            holding_w1,
            run_a,
            run_b_task,
            return_exceptions=True,
        )


@pytest.mark.asyncio
async def test_submit_ready_does_not_retry_blocked_jobs_before_slot_change() -> None:
    w0_gate = asyncio.Event()
    w1_gate = asyncio.Event()
    w0_submitted = asyncio.Event()
    w1_submitted = asyncio.Event()
    attempts: list[tuple[str, ...]] = []

    def w0_remote(payload: str) -> Any:
        if payload == "w0-active":
            w0_submitted.set()
            return _GatedRef(w0_gate, payload)
        return _FakeRef(payload, completion_rank=1)

    def w1_remote(payload: str) -> Any:
        if payload == "w1-active":
            w1_submitted.set()
            return _GatedRef(w1_gate, payload)
        return _FakeRef(payload, completion_rank=1)

    dispatcher = RayActorDispatcher(("w0", "w1"))
    original_try_acquire = dispatcher._try_acquire

    def recording_try_acquire(
        candidates: tuple[str, ...],
        *,
        waiter: Any,
    ) -> str | None:
        attempts.append(candidates)
        return original_try_acquire(candidates, waiter=waiter)

    dispatcher._try_acquire = recording_try_acquire
    task = asyncio.create_task(
        dispatcher.run(
            [
                RayActorJob(0, "w0", w0_remote, "w0-active"),
                RayActorJob(1, "w0", w0_remote, "w0-pending-1"),
                RayActorJob(2, "w1", w1_remote, "w1-active"),
                RayActorJob(3, "w0", w0_remote, "w0-pending-2"),
            ],
            operation="test.bounded_submit_scan",
            call_timeout_s=30.0,
        ),
    )
    await asyncio.gather(w0_submitted.wait(), w1_submitted.wait())

    assert attempts == [("w0",), ("w0",), ("w1",)]

    w0_gate.set()
    w1_gate.set()
    assert await task == [
        (0, "w0-active"),
        (1, "w0-pending-1"),
        (2, "w1-active"),
        (3, "w0-pending-2"),
    ]


@pytest.mark.asyncio
async def test_busy_worker_does_not_block_an_independent_worker() -> None:
    w0_gate = asyncio.Event()
    received: list[tuple[str, str]] = []

    def w0_remote(payload: str) -> _GatedRef:
        received.append(("w0", payload))
        return _GatedRef(w0_gate, payload)

    def w1_remote(payload: str) -> _FakeRef:
        received.append(("w1", payload))
        return _FakeRef(payload, completion_rank=1)

    dispatcher = RayActorDispatcher(("w0", "w1"))
    first = asyncio.create_task(
        dispatcher.run(
            [RayActorJob(0, "w0", w0_remote, "slow")],
            operation="test.actor_job",
            call_timeout_s=30.0,
        ),
    )
    await asyncio.sleep(0)

    assert await dispatcher.run(
        [RayActorJob(0, "w1", w1_remote, "fast")],
        operation="test.actor_job",
        call_timeout_s=30.0,
    ) == [(0, "fast")]
    assert received == [("w0", "slow"), ("w1", "fast")]

    w0_gate.set()
    assert await first == [(0, "slow")]


@pytest.mark.asyncio
async def test_actor_pool_caller_cancellation_cancels_submitted_refs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    never_ref = _NeverRef()
    cancelled: list[Any] = []

    class _Ray:
        @staticmethod
        def cancel(ref: Any, *, force: bool) -> None:
            assert force is False
            cancelled.append(ref)

    monkeypatch.setattr(deadline_module, "require_ray", lambda: _Ray)
    dispatcher = RayActorDispatcher(("w0",))
    task = asyncio.create_task(
        dispatcher.run(
            [
                RayActorJob(
                    job_index=0,
                    worker_id="w0",
                    remote_method=lambda _payload: never_ref,
                    payload="never",
                ),
            ],
            operation="test.actor_job",
            call_timeout_s=30.0,
        ),
    )
    await asyncio.sleep(0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError) as caught:
        await task

    assert isinstance(caught.value.__cause__, RayOperationCancelled)
    assert cancelled == [never_ref]


@_CONTROLLED_CLOCK
@pytest.mark.parametrize("failure_mode", ["raise", "non-awaitable"])
@pytest.mark.asyncio
async def test_run_one_waiter_factory_failure_cancels_ref_and_closes_dispatcher(
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    ref = _NeverRef()
    cancelled: list[Any] = []

    class _Ray:
        @staticmethod
        def cancel(value: Any, *, force: bool) -> None:
            assert force is False
            cancelled.append(value)

    def invalid_waiter(_ref: Any, _deadline: Any) -> Any:
        if failure_mode == "raise":
            raise ValueError("waiter construction failed")
        return object()

    monkeypatch.setattr(deadline_module, "require_ray", lambda: _Ray)
    dispatcher = RayActorDispatcher(("w0",))

    with pytest.raises(RayActorCallError) as caught:
        await dispatcher.run_one(
            RayActorJob(0, "w0", lambda _payload: ref, None),
            operation="test.custom_wait",
            call_timeout_s=30.0,
            await_result=invalid_waiter,
        )

    expected_cause = ValueError if failure_mode == "raise" else TypeError
    assert isinstance(caught.value.__cause__, expected_cause)
    assert cancelled == [ref]
    with pytest.raises(RuntimeError) as closed:
        await dispatcher.run(
            [],
            operation="test.after_custom_wait_failure",
            call_timeout_s=30.0,
        )
    assert closed.value.__cause__ is caught.value


# ------------------------------------------------------------------ planner


def test_round_robin_planner_binds_workers_at_plan_time() -> None:
    """Checks the default strategy keeps the historical binding."""
    planner = DistributedExecutionPlanner()
    plan = planner.plan_with_engine(_request(), _workers(2))

    worker_ids = [assignment.worker_id for assignment in plan.assignments]
    assert worker_ids == ["w0", "w1", "w0", "w1"]
    assert all(a.estimated_cost > 0 for a in plan.assignments)


def test_dynamic_planner_leaves_chunks_unbound_with_costs() -> None:
    """Checks dynamic strategy defers binding and carries cost estimates."""
    planner = DistributedExecutionPlanner(
        policy=ChunkPlacementPolicy(strategy="dynamic"),
    )
    request = _request(num_steps=10, samples=8, sbs=2)
    plan = planner.plan_with_engine(request, _workers(2))

    assert all(a.worker_id is None for a in plan.assignments)
    assert len(plan.assignments) == 4
    assert all(a.estimated_cost > 0 for a in plan.assignments)
    assert all(a.chunk is a.envelope.chunk for a in plan.assignments)
    assert all(not hasattr(a, "node_id") for a in plan.assignments)
    assert all(not hasattr(a, "gpu_ids") for a in plan.assignments)
    # Chunk identity and order (the gather contract) are untouched.
    assert [a.chunk.sample_start for a in plan.assignments] == [0, 2, 4, 6]


def test_planner_cost_uses_steps_axis() -> None:
    """Checks the cost hint scales with samples x steps."""
    request = _request(num_steps=35, samples=4, sbs=4)
    plan = DistributedExecutionPlanner().plan_with_engine(request, _workers(1))

    assignment = plan.assignments[0]
    assert assignment.estimated_cost == assignment.chunk.sample_count * 35


def test_placement_policy_rejects_unknown_strategy() -> None:
    """Checks the strategy vocabulary is closed."""
    with pytest.raises(ValueError, match="round_robin"):
        ChunkPlacementPolicy(strategy="work_stealing")


# ----------------------------------------------------- executor end to end


# Carried by the three `execute` tests. Their fake actor is called in-process, so
# no envelope is ever pickled: a field that became unserializable (a lambda, an
# open handle, a torch device reference) would pass here and break on production's
# first chunk. That crossing is what the twins named here run for real; the
# completion ORDER these tests pin is what a real cluster cannot give.
_CONTROLLED_CLOCK_OVER_A_REAL_WIRE = pytest.mark.real_cover(
    "tests/ray/test_real_chunk_execution.py",
    why=(
        "a real cluster cannot make chunk completion order deterministic, which is the whole "
        "point of the fake refs; the envelope -> pickle -> actor -> ChunkExecutionResult crossing "
        "they therefore skip is pinned against a live cluster by both twins in the named file"
    ),
)


class _FakeActor:
    """Fake Ray actor: execute_chunk.remote returns a real chunk result."""

    def __init__(self, worker_id: str, speed_rank_base: int) -> None:
        self.worker_id = worker_id
        self._rank = speed_rank_base
        self.executed: list[str] = []

        class _ExecuteChunk:
            @staticmethod
            def remote(envelope: ChunkExecutionEnvelope) -> _FakeRef:
                return self._execute(envelope)

        self.execute_chunk = _ExecuteChunk()

    def _execute(self, envelope: ChunkExecutionEnvelope) -> _FakeRef:
        chunk = envelope.chunk
        self.executed.append(chunk.chunk_key)
        self._rank += 1
        result = ChunkExecutionResult(
            request_id=envelope.request.request_id,
            worker_id=self.worker_id,
            chunk=chunk,
            output={"chunk_key": chunk.chunk_key, "samples": chunk.sample_count},
        )
        return _FakeRef(result=result, completion_rank=self._rank)


class _ListGatherer:
    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_rows: Any,
        chunks: list[Any],
    ) -> GenerationOutput:
        return GenerationOutput(
            request_id=request.request_id,
            family=request.family,
            task=request.task,
            sample_rows=list(sample_rows),
            output=list(chunks),
        )


def _executor(strategy: str, actors: list[_FakeActor]) -> RayGenerationExecutor:
    workers = [
        DistributedWorkerHandle(
            worker_id=actor.worker_id,
            actor=actor,
        )
        for actor in actors
    ]
    return RayGenerationExecutor(
        DistributedExecutionPlanner(
            policy=ChunkPlacementPolicy(strategy=strategy),
        ),
        workers,
        _ListGatherer(),
        actor_dispatcher=RayActorDispatcher(
            tuple(worker.worker_id for worker in workers),
        ),
        generation_stall_timeout_s=30.0,
    )


@pytest.mark.real_cover(
    "tests/generation/ray/test_runtime_config.py"
    "::test_real_ray_probe_fan_out_resolves_auto_once_across_the_fleet",
    why=(
        "the gated ref makes probe/chunk submission order deterministic; the named "
        "test sends the real request and keyword arguments through live Ray actors"
    ),
)
@pytest.mark.asyncio
async def test_chunk_size_probe_shares_actor_admission_with_explicit_generation() -> None:
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
                {
                    "samples_per_chunk": 2,
                    "budget_bytes": 1,
                    "trials": [],
                },
            )

    actor.probe_chunk_size = _Probe()
    executor = _executor("round_robin", [actor])
    request = _request(samples=2, sbs=1)
    probe = asyncio.create_task(
        executor.probe_chunk_sizes(request, max_samples=8),
    )
    await asyncio.sleep(0)
    generation = asyncio.create_task(executor.execute(request))
    await asyncio.sleep(0)

    assert probe_requests == [request.request_id]
    assert actor.executed == []

    gate.set()
    assert await probe == [
        {
            "samples_per_chunk": 2,
            "budget_bytes": 1,
            "trials": [],
        },
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

    # 4 chunks alternate w0/w1 even though w1 is much slower: plan-time binding.
    assert actors[0].executed == ["prompt:0:samples:0:2", "prompt:0:samples:4:6"]
    assert actors[1].executed == ["prompt:0:samples:2:4", "prompt:0:samples:6:8"]
    schedule = output.extra["runtime_debug"]["chunk_schedule"]
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

    # The fast worker pulls every queued chunk once the slow one is busy.
    assert len(actors[0].executed) == 3
    assert len(actors[1].executed) == 1
    schedule = output.extra["runtime_debug"]["chunk_schedule"]
    assert all(row["assignment_strategy"] == "dynamic" for row in schedule)
    by_worker = {
        worker: sum(1 for row in schedule if row["assigned_worker"] == worker)
        for worker in ("w0", "w1")
    }
    assert by_worker == {"w0": 3, "w1": 1}
    # Gather order is untouched by dynamic placement.
    assert [entry["chunk_key"] for entry in output.output] == [
        "prompt:0:samples:0:2",
        "prompt:0:samples:2:4",
        "prompt:0:samples:4:6",
        "prompt:0:samples:6:8",
    ]


@_CONTROLLED_CLOCK_OVER_A_REAL_WIRE
@pytest.mark.asyncio
async def test_executor_runtime_debug_exposes_chunk_schedule() -> None:
    """Checks runtime_debug surfaces per-chunk placement telemetry."""
    actors = [_FakeActor("w0", 0), _FakeActor("w1", 100)]
    executor = _executor("round_robin", actors)
    request = _request(num_steps=10, samples=8, sbs=2)
    request.metadata["_runtime_debug"] = True

    output = await executor.execute(request)

    debug_schedule = output.extra["runtime_debug"]["chunk_schedule"]
    assert len(debug_schedule) == 4
    for row in debug_schedule:
        assert {
            "chunk_key",
            "sample_count",
            "assignment_strategy",
            "estimated_cost",
            "assigned_worker",
            "queue_wait_s",
            "execution_s",
        } <= set(row)
