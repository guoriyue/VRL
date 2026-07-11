"""Tests for shared Ray actor substrate."""

from __future__ import annotations

try:
    import pytest
except ModuleNotFoundError:  # Ray workers import this module for test actors.
    pytest = None

import asyncio

from vrl.ray.actor_group import RayActorGroup
from vrl.ray.actor_pool import RayActorJob, run_actor_jobs
from vrl.ray.placement import validate_actor_gpu_ids

# Every test here spins up Ray (~seconds each) — slow by nature, run nightly not per-PR.
pytestmark = pytest.mark.slow_test if pytest is not None else ()


class _EchoWorker:
    def __init__(self, worker_id: str, config: dict) -> None:
        self.worker_id = worker_id
        self.config = dict(config)

    def startup(self) -> None:
        self.started = True

    def worker_metadata(self) -> dict:
        return {"worker_id": self.worker_id, "node_ip": "test-node", "gpu_ids": []}

    def echo(self, payload: int) -> tuple[str, int]:
        return self.worker_id, payload + int(self.config["offset"])


def test_ray_actor_group_launch_lifecycle() -> None:
    """Checks Ray actor group launch wiring, worker configs, metadata, and shutdown."""
    ray = pytest.importorskip("ray")
    ray.shutdown()
    group = None
    try:
        ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=2, log_to_driver=False)
        group = RayActorGroup.launch(
            worker_cls=_EchoWorker,
            worker_configs=[{"offset": 10}, {"offset": 20}],
            worker_ids=["w0", "w1"],
            num_cpus=0.5,
            num_gpus=0.0,
            startup_method="startup",
        )

        assert [handle.worker_id for handle in group.handles] == ["w0", "w1"]
        assert all(handle.node_ip == "test-node" for handle in group.handles)
        results = ray.get(
            [
                group.handles[0].actor.echo.remote(1),
                group.handles[1].actor.echo.remote(2),
            ],
        )
        assert results == [("w0", 11), ("w1", 22)]

        group.shutdown()
        assert group.handles == []
    finally:
        if group is not None:
            group.shutdown()
        ray.shutdown()


class _PayloadWorker:
    def __init__(self, worker_id: str) -> None:
        self.worker_id = worker_id

    def execute(self, payload: int) -> tuple[str, int]:
        return self.worker_id, payload * 2


def test_run_actor_jobs_awaits_real_object_refs() -> None:
    """Real-Ray twin of tests/ray/test_chunk_dispatch.py: the deterministic
    fake refs there encode the assumption that real ObjectRefs are directly
    awaitable inside the dispatch loop and resolve to the task result. Pin it
    against a live cluster for both plan-time-bound and pull-dispatched jobs
    (placement distribution is scheduling-dependent, so only totals and gather
    order are asserted here; the distribution contract stays deterministic in
    the fake-ref tests)."""
    ray = pytest.importorskip("ray")
    ray.shutdown()
    try:
        ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=2, log_to_driver=False)
        actor_cls = ray.remote(num_cpus=0)(_PayloadWorker)
        w0 = actor_cls.remote("w0")
        w1 = actor_cls.remote("w1")

        bound = [
            RayActorJob(
                job_index=i,
                worker_id=("w0" if i % 2 == 0 else "w1"),
                remote_method=(w0.execute.remote if i % 2 == 0 else w1.execute.remote),
                payload=i,
            )
            for i in range(4)
        ]
        pairs = asyncio.run(run_actor_jobs(bound, max_inflight_per_actor=1))
        assert [index for index, _ in pairs] == [0, 1, 2, 3]
        assert [result for _, result in pairs] == [
            ("w0", 0), ("w1", 2), ("w0", 4), ("w1", 6),
        ]

        unbound = [
            RayActorJob(job_index=i, worker_id=None, remote_method=None, payload=i)
            for i in range(4)
        ]
        pairs = asyncio.run(
            run_actor_jobs(
                unbound,
                max_inflight_per_actor=1,
                worker_methods={"w0": w0.execute.remote, "w1": w1.execute.remote},
            ),
        )
        assert [index for index, _ in pairs] == [0, 1, 2, 3]
        results = [result for _, result in pairs]
        assert sorted(payload for _, payload in results) == [0, 2, 4, 6]
        assert {worker for worker, _ in results} <= {"w0", "w1"}
    finally:
        ray.shutdown()


def test_validate_actor_gpu_ids_rejects_unexpected_assignment() -> None:
    """Checks validate actor GPU IDs rejects unexpected assignment."""
    with pytest.raises(RuntimeError, match="outside resolved reward devices"):
        validate_actor_gpu_ids(
            [{"worker_id": "reward-0", "gpu_ids": [2]}],
            expected_gpu_ids=(1,),
            role="reward",
        )


def test_validate_actor_gpu_ids_cross_node_accepts_remote_local_zero() -> None:
    """Checks validate actor GPU IDs cross-node accepts remote local zero."""
    result = validate_actor_gpu_ids(
        [
            {"worker_id": "generation-0", "node_ip": "10.0.0.2", "gpu_ids": [0]},
            {"worker_id": "generation-1", "node_ip": "10.0.0.3", "gpu_ids": [0]},
        ],
        expected_gpu_ids=(1, 2),
        role="generation",
        cross_node=True,
        driver_node_ip="10.0.0.1",
    )

    assert result == (0, 0)


def test_validate_actor_gpu_ids_cross_node_rejects_driver_node() -> None:
    """Checks validate actor GPU IDs cross-node rejects driver node."""
    with pytest.raises(RuntimeError, match="driver/head node"):
        validate_actor_gpu_ids(
            [{"worker_id": "generation-0", "node_ip": "10.0.0.1", "gpu_ids": [0]}],
            expected_gpu_ids=(1,),
            role="generation",
            cross_node=True,
            driver_node_ip="10.0.0.1",
        )


def test_validate_actor_gpu_ids_cross_node_rejects_shared_gpu() -> None:
    """Checks validate actor GPU IDs cross-node rejects shared GPU."""
    with pytest.raises(RuntimeError, match="share GPU"):
        validate_actor_gpu_ids(
            [
                {"worker_id": "generation-0", "node_ip": "10.0.0.2", "gpu_ids": [0]},
                {"worker_id": "generation-1", "node_ip": "10.0.0.2", "gpu_ids": [0]},
            ],
            expected_gpu_ids=(1, 2),
            role="generation",
            cross_node=True,
            driver_node_ip="10.0.0.1",
        )


def test_validate_actor_gpu_ids_cross_node_requires_gpu() -> None:
    """Checks validate actor GPU IDs cross-node requires GPU."""
    with pytest.raises(RuntimeError, match="no assigned GPU ids"):
        validate_actor_gpu_ids(
            [{"worker_id": "generation-0", "node_ip": "10.0.0.2", "gpu_ids": []}],
            expected_gpu_ids=(1,),
            role="generation",
            cross_node=True,
            driver_node_ip="10.0.0.1",
        )
