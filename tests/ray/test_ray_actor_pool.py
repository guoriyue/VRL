"""Tests for shared Ray actor substrate."""

from __future__ import annotations

import asyncio

import pytest

from vrl.ray.actor_group import RayActorGroup
from vrl.ray.placement import validate_actor_gpu_ids


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


def test_ray_actor_group_maps_payloads_in_order() -> None:
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

        results = asyncio.run(group.map_method("echo", [1, 2, 3]))

        assert results == [("w0", 11), ("w1", 22), ("w0", 13)]
    finally:
        if group is not None:
            group.shutdown()
        ray.shutdown()


def test_validate_actor_gpu_ids_rejects_unexpected_assignment() -> None:
    with pytest.raises(RuntimeError, match="outside resolved reward devices"):
        validate_actor_gpu_ids(
            [{"worker_id": "reward-0", "gpu_ids": [2]}],
            expected_gpu_ids=(1,),
            role="reward",
        )
