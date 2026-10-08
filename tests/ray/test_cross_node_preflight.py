"""The cross-node rollout preflight (``vrl.ray.placement``) on a real two-node cluster.

Each case starts an isolated two-node Ray cluster in a subprocess: the head
(where the driver attaches) on the host address and a second node on the
loopback address, so ``ray.nodes()`` reports two distinct node IPs and the
preflight classifies driver and non-driver GPU capacity from the real cluster.
The resources argument is the real ``ResolvedDistributedResources`` output, so
the whole cross_node config -> resolution -> preflight chain is exercised.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from vrl.config.schema import parse_config
from vrl.ray import placement
from vrl.ray.resources import ResolvedDistributedResources

pytestmark = pytest.mark.slow_test


def _resources() -> ResolvedDistributedResources:
    return ResolvedDistributedResources.from_root(
        parse_config(
            OmegaConf.create(
                {
                    "distributed": {
                        "resources": {
                            "visible_devices": "auto",
                            "cross_node": True,
                            "trainer": {"num_gpus": 1},
                            "rollout": {"num_gpus": 1, "num_engines": 1},
                        },
                    },
                },
            )
        ),
    )


def _preflight_on_two_nodes(driver_node_gpus: int) -> None:
    """Subprocess entry point: run the preflight against a real two-node cluster.

    Prints ``ACCEPTED`` when the preflight returns, or ``REJECTED: <message>``.
    """

    import os

    import ray
    from ray.cluster_utils import Cluster

    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    cluster = Cluster()
    try:
        cluster.add_node(
            num_cpus=1,
            num_gpus=driver_node_gpus,
            include_dashboard=False,
            object_store_memory=80 * 1024 * 1024,
        )
        cluster.add_node(
            num_cpus=1,
            num_gpus=1,
            node_ip_address="127.0.0.1",
            object_store_memory=80 * 1024 * 1024,
        )
        ray.init(address=cluster.address, _skip_env_hook=True)
        ips = {node["NodeManagerAddress"] for node in ray.nodes() if node["Alive"]}
        assert len(ips) == 2, ips
        try:
            result = placement.cross_node_preflight(ray, _resources())
        except RuntimeError as error:
            print(f"REJECTED: {error}")
        else:
            assert result is None
            print("ACCEPTED")
    finally:
        ray.shutdown()
        cluster.shutdown()


def _run_on_two_nodes(driver_node_gpus: int) -> str:
    import subprocess
    import sys

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from tests.conftest import ray_uv_hook_disabled\n"
                "from tests.ray.test_cross_node_preflight import _preflight_on_two_nodes\n"
                # A subprocess still has the outer `uv run` ancestor; the
                # parent's hook guard does not cross processes.
                "with ray_uv_hook_disabled():\n"
                f"    _preflight_on_two_nodes({int(driver_node_gpus)})\n"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=150,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    (verdict,) = [
        line for line in completed.stdout.splitlines() if line.startswith(("ACCEPTED", "REJECTED"))
    ]
    return verdict


def test_preflight_rejects_a_driver_node_with_ray_gpus() -> None:
    """Plain cross-node: any Ray GPU on the driver node fails fast (--num-gpus=0 required)."""

    verdict = _run_on_two_nodes(driver_node_gpus=1)

    assert verdict.startswith("REJECTED:")
    assert "num-gpus=0" in verdict


def test_preflight_accepts_a_gpu_less_driver_node_with_remote_gpus() -> None:
    """Plain cross-node: a driver node with --num-gpus=0 and enough remote GPUs passes."""

    assert _run_on_two_nodes(driver_node_gpus=0) == "ACCEPTED"
