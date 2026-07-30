"""Deterministic tests for chunk placement and pull-based actor dispatch.

These use awaitable fake refs so completion order is fully controlled — no Ray
runtime, no slow markers. They pin the Track A contract:
round_robin keeps plan-time binding bit-for-bit; dynamic binds at dispatch
time (pull + LPT) and never changes gather order.

The fakes are a controlled clock, not a Ray protocol fake; the ``real_cover``
labels below name the live-cluster twin that pins the protocol assumption.
"""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from typing import Any

import pytest

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
from vrl.ray.actor_pool import RayActorJob, run_actor_jobs

# Carried by the tests that actually drive `_FakeRef`/`_FakeWorker`; the planner
# and argument-validation tests below use no double, so a module-level pytestmark
# would over-claim on their behalf.
_CONTROLLED_CLOCK = pytest.mark.real_cover(
    "tests/ray/test_ray_actor_pool.py::test_run_actor_jobs_awaits_real_object_refs",
    why=(
        "the fake refs control event-loop completion ORDER, which a real Ray cluster cannot "
        "make deterministic; the protocol assumption they encode — a real ObjectRef awaits "
        "directly and resolves to the task result — is pinned against a live cluster there"
    ),
)


class _FakeRef:
    """One in-flight fake actor call; completion_rank orders completion.

    ``run_actor_jobs`` now awaits refs directly (like real Ray ObjectRefs), so
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
        run_actor_jobs(
            jobs,
            max_inflight_per_actor=1,
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
        run_actor_jobs(
            jobs,
            max_inflight_per_actor=1,
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
        run_actor_jobs(
            jobs,
            max_inflight_per_actor=1,
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
        asyncio.run(run_actor_jobs(jobs))


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
        run_actor_jobs(
            jobs,
            max_inflight_per_actor=1,
            schedule=schedule,
        ),
    )

    assert sorted(row["job_index"] for row in schedule) == [0, 1]
    for row in schedule:
        assert row["worker_id"] == "w0"
        assert row["queue_wait_s"] >= 0.0
        assert row["execution_s"] >= 0.0


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
    )


@_CONTROLLED_CLOCK
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


@_CONTROLLED_CLOCK
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


@_CONTROLLED_CLOCK
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
