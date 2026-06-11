"""Deterministic tests for chunk placement and pull-based actor dispatch.

These use a fake ray module so completion order is fully controlled — no Ray
runtime, no slow markers. They pin the Track A contract:
round_robin keeps plan-time binding bit-for-bit; dynamic binds at dispatch
time (pull + LPT) and never changes gather order.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import vrl.ray.actor_pool as actor_pool_module
from vrl.generation.execution.chunk_placement import (
    ChunkPlacementPolicy,
    DistributedExecutionPlanner,
    estimate_chunk_cost,
)
from vrl.generation.execution.types import DistributedWorkerHandle
from vrl.generation.types import GenerationRequest
from vrl.models.diffusion.capabilities import diffusion_family_capability
from vrl.ray.actor_pool import RayActorJob, run_actor_jobs


class _FakeRef:
    """One in-flight fake actor call; completion_rank orders ray.wait."""

    def __init__(self, result: Any, completion_rank: int) -> None:
        self.result = result
        self.completion_rank = completion_rank


class _FakeRay:
    @staticmethod
    def wait(refs: list[_FakeRef], num_returns: int = 1) -> tuple[list[_FakeRef], list[_FakeRef]]:
        ready = min(refs, key=lambda ref: ref.completion_rank)
        return [ready], [ref for ref in refs if ref is not ready]

    @staticmethod
    def get(ref: _FakeRef) -> Any:
        return ref.result


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


@pytest.fixture(autouse=True)
def _fake_ray(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(actor_pool_module, "require_ray", lambda: _FakeRay())


def _request(num_steps: int = 10, samples: int = 8, sbs: int = 2) -> GenerationRequest:
    return GenerationRequest(
        request_id="req-dispatch",
        family="sd3_5",
        task="t2i",
        prompts=["a test prompt"],
        samples_per_prompt=samples,
        sampling={"height": 64, "width": 64, "num_steps": num_steps, "sample_batch_size": sbs},
        return_artifacts={"output"},
        metadata={"dataset": "unit"},
    )


def _workers(count: int) -> list[DistributedWorkerHandle]:
    return [
        DistributedWorkerHandle(
            worker_id=f"w{idx}",
            node_id="node-0",
            gpu_ids=(idx,),
            actor=None,
        )
        for idx in range(count)
    ]


# ---------------------------------------------------------------- actor pool


def test_bound_jobs_keep_plan_time_binding_and_order() -> None:
    """Checks the static path is unchanged: binding and order preserved."""
    fast = _FakeWorker("w0", speed_rank_base=0)
    slow = _FakeWorker("w1", speed_rank_base=100)
    jobs = [
        RayActorJob(job_index=i, worker_id=("w0" if i % 2 == 0 else "w1"),
                    remote_method=(fast.remote if i % 2 == 0 else slow.remote),
                    payload=f"chunk-{i}")
        for i in range(4)
    ]

    pairs = asyncio.run(run_actor_jobs(jobs, max_inflight_per_actor=1))

    assert [index for index, _ in pairs] == [0, 1, 2, 3]
    # Even though w1 is slow, its chunks never migrate to w0.
    assert fast.received == ["chunk-0", "chunk-2"]
    assert slow.received == ["chunk-1", "chunk-3"]


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


def test_lpt_priority_orders_submission() -> None:
    """Checks higher-priority (more expensive) chunks are submitted first."""
    worker = _FakeWorker("w0", speed_rank_base=0)
    jobs = [
        RayActorJob(job_index=0, worker_id=None, remote_method=None, payload="small", priority=1.0),
        RayActorJob(job_index=1, worker_id=None, remote_method=None, payload="large", priority=10.0),
        RayActorJob(job_index=2, worker_id=None, remote_method=None, payload="medium", priority=5.0),
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


def test_schedule_telemetry_rows_are_emitted() -> None:
    """Checks the dispatch loop emits one telemetry row per job."""
    worker = _FakeWorker("w0", speed_rank_base=0)
    jobs = [
        RayActorJob(job_index=i, worker_id="w0", remote_method=worker.remote, payload=i)
        for i in range(2)
    ]
    schedule: list[dict[str, Any]] = []

    asyncio.run(run_actor_jobs(jobs, max_inflight_per_actor=1, schedule=schedule))

    assert sorted(row["job_index"] for row in schedule) == [0, 1]
    for row in schedule:
        assert row["worker_id"] == "w0"
        assert row["queue_wait_s"] >= 0.0
        assert row["execution_s"] >= 0.0


# ------------------------------------------------------------------ planner


def test_round_robin_planner_binds_workers_at_plan_time() -> None:
    """Checks the default strategy keeps the historical binding."""
    planner = DistributedExecutionPlanner(
        diffusion_family_capability("sd3_5", "t2i"),
    )
    plan = planner.plan_with_engine(_request(), _workers(2))

    worker_ids = [assignment.worker_id for assignment in plan.assignments]
    assert worker_ids == ["w0", "w1", "w0", "w1"]
    assert all(a.estimated_cost > 0 for a in plan.assignments)


def test_dynamic_planner_leaves_chunks_unbound_with_costs() -> None:
    """Checks dynamic strategy defers binding and carries cost estimates."""
    planner = DistributedExecutionPlanner(
        diffusion_family_capability("sd3_5", "t2i"),
        policy=ChunkPlacementPolicy(strategy="dynamic"),
    )
    request = _request(num_steps=10, samples=8, sbs=2)
    plan = planner.plan_with_engine(request, _workers(2))

    assert all(a.worker_id is None for a in plan.assignments)
    # 8 samples / sbs 2 = 4 chunks of 2 samples x 10 steps = cost 20 each.
    assert [a.estimated_cost for a in plan.assignments] == [20.0] * 4
    # Chunk identity and order (the gather contract) are untouched.
    assert [a.chunk.sample_start for a in plan.assignments] == [0, 2, 4, 6]


def test_estimate_chunk_cost_uses_steps_axis() -> None:
    """Checks the cost hint scales with samples x steps."""
    request = _request(num_steps=35, samples=4, sbs=4)
    plan = DistributedExecutionPlanner(
        diffusion_family_capability("sd3_5", "t2i"),
    ).plan_with_engine(request, _workers(1))

    chunk = plan.assignments[0].chunk
    assert estimate_chunk_cost(request, chunk) == chunk.sample_count * 35


def test_placement_policy_rejects_unknown_strategy() -> None:
    """Checks the strategy vocabulary is closed."""
    with pytest.raises(ValueError, match="round_robin"):
        ChunkPlacementPolicy(strategy="work_stealing")
