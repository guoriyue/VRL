"""Tests for shared Ray actor substrate."""

from __future__ import annotations

import time

import pytest

from vrl.ray.actor_group import RayActorGroup
from vrl.ray.placement import require_actor_gpu_ids

# The real-Ray tests here share the package cluster (tests/ray/conftest.py);
# the five require_actor_gpu_ids tests below are pure, but stay in the nightly
# lane with them so this module has one lane instead of two.
pytestmark = pytest.mark.slow_test


class _EchoWorker:
    def __init__(self, worker_id: str, config: dict) -> None:
        self.worker_id = worker_id
        self.config = dict(config)

    def startup(self) -> None:
        self.started = True

    def worker_metadata(self) -> dict:
        # What a production worker reports: its own Ray node and GPU assignment.
        from vrl.ray.dependencies import current_gpu_ids, current_node_ip

        return {
            "worker_id": self.worker_id,
            "node_ip": current_node_ip(),
            "gpu_ids": current_gpu_ids(),
        }

    def echo(self, payload: int) -> tuple[str, int]:
        return self.worker_id, payload + int(self.config["offset"])


def test_failed_startup_actor_is_reclaimed_by_placement_removal(local_ray, monkeypatch) -> None:
    from ray.util.placement_group import placement_group, remove_placement_group

    class FailingStartupWorker(_EchoWorker):
        def startup(self):
            raise RuntimeError("startup failed")

    ray = local_ray
    pg = placement_group([{"CPU": 1}])
    retained = []

    # The kill RPC is the fault: report it failed for every actor, so only
    # placement-group removal can reclaim the candidate.
    def fail_kill(_ray, actors):
        retained.extend(actors)
        return [(actor, RuntimeError("kill unavailable")) for actor in actors]

    monkeypatch.setattr("vrl.ray.actor_group.kill_actors", fail_kill)
    try:
        ray.get(pg.ready(), timeout=10)
        with pytest.raises(RuntimeError, match="startup failed"):
            RayActorGroup.launch(
                worker_cls=FailingStartupWorker,
                worker_configs=[{"offset": 1}],
                worker_ids=["candidate"],
                num_cpus=0.1,
                num_gpus=0,
                rpc_timeout_s=10,
                operation_prefix="test.startup",
                placement_group=pg,
                bundle_indices=[0],
                startup_method="startup",
            )
        assert len(retained) == 1
        actor = retained[0]
        assert ray.get(actor.echo.remote(1), timeout=5) == ("candidate", 2)
        remove_placement_group(pg)
        deadline = time.monotonic() + 10
        while True:
            try:
                ray.get(actor.echo.remote(1), timeout=1)
            except ray.exceptions.ActorDiedError:
                break
            assert time.monotonic() < deadline, "placement removal left its actor alive"
            time.sleep(0.01)
    finally:
        for actor in retained:
            ray.kill(actor, no_restart=True)
        remove_placement_group(pg)


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
        # Single-node cluster: both actors report the driver's own Ray node.
        driver_node = ray.util.get_node_ip_address()
        assert all(handle.node_ip == driver_node for handle in group.handles)
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


def test_validate_actor_gpu_ids_rejects_unexpected_assignment() -> None:
    """Single-node: a worker holding a GPU outside the resolved device set is a
    hard error -- that GPU belongs to another role."""
    with pytest.raises(RuntimeError, match="outside resolved reward devices"):
        require_actor_gpu_ids(
            [{"worker_id": "reward-0", "gpu_ids": [2]}],
            expected_gpu_ids=(1,),
            role="reward",
        )


def test_validate_actor_gpu_ids_cross_node_accepts_remote_local_zero() -> None:
    """Cross-node: every remote node has its own ordinal space, so two workers
    both reporting local GPU 0 on different nodes is correct, not a collision."""
    result = require_actor_gpu_ids(
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
    with pytest.raises(RuntimeError, match="driver node"):
        require_actor_gpu_ids(
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
        require_actor_gpu_ids(
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
        require_actor_gpu_ids(
            [{"worker_id": "generation-0", "node_ip": "10.0.0.2", "gpu_ids": []}],
            expected_gpu_ids=(1,),
            role="generation",
            cross_node=True,
            driver_node_ip="10.0.0.1",
        )
