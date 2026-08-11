"""Tests for shared Ray actor substrate."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from vrl.ray.actor_group import RayActorGroup
from vrl.ray.actor_pool import RayActorDispatcher, RayActorJob
from vrl.ray.operation_deadline import RayOperationTimeout
from vrl.ray.placement import validate_actor_gpu_ids

# The two real-Ray tests here share the package cluster (tests/ray/conftest.py);
# the five validate_actor_gpu_ids tests below are pure, but stay in the nightly
# lane with them so this module has one lane instead of two.
pytestmark = pytest.mark.slow_test


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


def test_ray_actor_group_launch_lifecycle(local_ray) -> None:
    """Launch really places two actors, hands back their own metadata, routes
    per-worker config into them, and shutdown drops every handle."""
    ray = local_ray
    group = None
    try:
        group = RayActorGroup.launch(
            worker_cls=_EchoWorker,
            worker_configs=[{"offset": 10}, {"offset": 20}],
            worker_ids=["w0", "w1"],
            num_cpus=0.5,
            num_gpus=0.0,
            rpc_timeout_s=30.0,
            operation_prefix="test.echo",
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
        # The cluster is shared: this test owns its actors, not the cluster.
        if group is not None:
            group.shutdown()


class _PayloadWorker:
    def __init__(self, worker_id: str) -> None:
        self.worker_id = worker_id

    def execute(self, payload: int) -> tuple[str, int]:
        return self.worker_id, payload * 2


def test_actor_dispatcher_awaits_real_object_refs(local_ray) -> None:
    """Real-Ray twin of tests/ray/test_batch_dispatch.py: the deterministic
    fake refs there encode the assumption that real ObjectRefs are directly
    awaitable inside the dispatch loop and resolve to the task result. Pin it
    against a live cluster for both plan-time-bound and pull-dispatched jobs
    (placement distribution is scheduling-dependent, so only totals and gather
    order are asserted here; the distribution contract stays deterministic in
    the fake-ref tests)."""
    ray = local_ray
    actor_cls = ray.remote(num_cpus=0)(_PayloadWorker)
    w0 = actor_cls.remote("w0")
    w1 = actor_cls.remote("w1")
    try:
        bound = [
            RayActorJob(
                job_index=i,
                worker_id=("w0" if i % 2 == 0 else "w1"),
                remote_method=(w0.execute.remote if i % 2 == 0 else w1.execute.remote),
                payload=i,
            )
            for i in range(4)
        ]
        pairs = asyncio.run(
            RayActorDispatcher(("w0", "w1")).run(
                bound,
                operation="test.actor_job",
                call_timeout_s=30.0,
            ),
        )
        assert [index for index, _ in pairs] == [0, 1, 2, 3]
        assert [result for _, result in pairs] == [
            ("w0", 0),
            ("w1", 2),
            ("w0", 4),
            ("w1", 6),
        ]

        unbound = [
            RayActorJob(job_index=i, worker_id=None, remote_method=None, payload=i)
            for i in range(4)
        ]
        pairs = asyncio.run(
            RayActorDispatcher(("w0", "w1")).run(
                unbound,
                operation="test.actor_job",
                call_timeout_s=30.0,
                worker_methods={"w0": w0.execute.remote, "w1": w1.execute.remote},
            ),
        )
        assert [index for index, _ in pairs] == [0, 1, 2, 3]
        results = [result for _, result in pairs]
        assert sorted(payload for _, payload in results) == [0, 2, 4, 6]
        assert {worker for worker, _ in results} <= {"w0", "w1"}
    finally:
        # The cluster is shared: leaving these two alive would have later tests
        # scheduling against a fleet they did not create.
        for actor in (w0, w1):
            ray.kill(actor, no_restart=True)


@pytest.mark.asyncio
async def test_hung_business_call_times_out_while_health_group_responds(local_ray) -> None:
    """A live health thread cannot disguise a stalled default-group call."""

    ray = local_ray
    health_group = "test_health"

    class _HungWorker:
        def __init__(self) -> None:
            self._business_started = threading.Event()

        def block(self, _payload: None) -> None:
            self._business_started.set()
            threading.Event().wait()

        @ray.method(concurrency_group=health_group)
        def health(self) -> bool:
            return self._business_started.is_set()

    actor_cls = ray.remote(
        num_cpus=0,
        concurrency_groups={health_group: 1},
    )(_HungWorker)
    actor = actor_cls.remote()
    dispatcher = RayActorDispatcher(("w0",))
    started_at = time.monotonic()
    task = asyncio.create_task(
        dispatcher.run(
            [
                RayActorJob(
                    job_index=0,
                    worker_id="w0",
                    remote_method=actor.block.remote,
                    payload=None,
                ),
            ],
            operation="rollout.generation.batch",
            call_timeout_s=1.0,
        ),
    )
    try:
        business_started = False
        while not business_started:
            business_started = await asyncio.to_thread(
                ray.get,
                actor.health.remote(),
                timeout=1.0,
            )
            if not business_started:
                await asyncio.sleep(0.01)

        with pytest.raises(RayOperationTimeout, match=r"rollout\.generation\.batch"):
            await task

        assert time.monotonic() - started_at < 3.0
        assert (
            await asyncio.to_thread(
                ray.get,
                actor.health.remote(),
                timeout=1.0,
            )
            is True
        )
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        ray.kill(actor, no_restart=True)


@pytest.mark.asyncio
async def test_concurrent_calls_wait_locally_before_starting_real_ray_deadline(local_ray) -> None:
    """Two healthy sync calls each get a full budget after real actor admission."""

    ray = local_ray

    class _SlowWorker:
        @staticmethod
        def execute(payload: str) -> str:
            if payload != "warm":
                time.sleep(0.6)
            return payload

    actor = ray.remote(num_cpus=0)(_SlowWorker).remote()
    dispatcher = RayActorDispatcher(("w0",))
    assert ray.get(actor.execute.remote("warm")) == "warm"

    async def dispatch(payload: str) -> list[tuple[int, str]]:
        return await dispatcher.run(
            [
                RayActorJob(
                    job_index=0,
                    worker_id="w0",
                    remote_method=actor.execute.remote,
                    payload=payload,
                ),
            ],
            operation="test.sync_actor",
            # Each warmed call finishes comfortably inside this budget, while
            # two mailbox-queued calls would take about 1.2 seconds.
            call_timeout_s=1.0,
        )

    try:
        first, second = await asyncio.gather(
            dispatch("first"),
            dispatch("second"),
        )
        assert first == [(0, "first")]
        assert second == [(0, "second")]
    finally:
        ray.kill(actor, no_restart=True)


def test_validate_actor_gpu_ids_rejects_unexpected_assignment() -> None:
    """Single-node: a worker holding a GPU outside the resolved device set is a
    hard error -- that GPU belongs to another role."""
    with pytest.raises(RuntimeError, match="outside resolved reward devices"):
        validate_actor_gpu_ids(
            [{"worker_id": "reward-0", "gpu_ids": [2]}],
            expected_gpu_ids=(1,),
            role="reward",
        )


def test_validate_actor_gpu_ids_cross_node_accepts_remote_local_zero() -> None:
    """Cross-node: every remote node has its own ordinal space, so two workers
    both reporting local GPU 0 on different nodes is correct, not a collision."""
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
    """Cross-node drops the placement-group trainer reservation, so a rollout
    worker that landed on the head node would sit on the trainer's GPU."""
    with pytest.raises(RuntimeError, match="driver/head node"):
        validate_actor_gpu_ids(
            [{"worker_id": "generation-0", "node_ip": "10.0.0.1", "gpu_ids": [0]}],
            expected_gpu_ids=(1,),
            role="generation",
            cross_node=True,
            driver_node_ip="10.0.0.1",
        )


def test_validate_actor_gpu_ids_cross_node_rejects_shared_gpu() -> None:
    """Cross-node uniqueness is per ``(node_ip, gpu_id)``: the same local ordinal
    twice on ONE node means two workers time-share a physical GPU."""
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
    """A rollout worker with no GPU at all fails closed instead of silently
    running the model on CPU."""
    with pytest.raises(RuntimeError, match="no assigned GPU ids"):
        validate_actor_gpu_ids(
            [{"worker_id": "generation-0", "node_ip": "10.0.0.2", "gpu_ids": []}],
            expected_gpu_ids=(1,),
            role="generation",
            cross_node=True,
            driver_node_ip="10.0.0.1",
        )
