"""Tests for Ray cluster-topology inspection (vrl.ray.dependencies).

Topology runs against real Ray clusters: the package's single-node cluster, and
an isolated two-node one whose second node sits on the loopback address, so
``ray.nodes()`` reports two distinct node IPs and the driver-versus-non-driver
split is read from a live cluster.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.rollouts.collector._helpers import Trace
from vrl.ray import dependencies


@pytest.mark.slow_test
def test_inspect_cluster_single_node_has_no_non_driver_gpus(local_ray):
    """A single-node cluster offers zero rollout GPUs, however many it has: they
    all belong to the driver, which is what the cross-node preflight rejects."""

    topo = dependencies.ClusterTopology.discover(local_ray)

    assert topo.driver_gpus == float(local_ray.cluster_resources()["GPU"])
    assert topo.driver_gpus > 0
    assert topo.non_driver_gpus == 0.0


_TWO_NODE_PROBE = """
import json, os, time
os.environ.pop("CUDA_VISIBLE_DEVICES", None)
import ray
from ray.cluster_utils import Cluster
from vrl.ray import dependencies

cluster = Cluster(initialize_head=True, head_node_args={"num_cpus": 1, "num_gpus": 1})
worker = cluster.add_node(num_cpus=1, num_gpus=2, node_ip_address="127.0.0.1")
cluster.wait_for_nodes()
ray.init(address=cluster.address)
try:
    ips = {entry["NodeManagerAddress"] for entry in ray.nodes() if entry["Alive"]}
    assert len(ips) == 2, ips
    live = dependencies.ClusterTopology.discover(ray)
    cluster.remove_node(worker)
    deadline = time.monotonic() + 30
    while sum(1 for entry in ray.nodes() if entry["Alive"]) != 1:
        assert time.monotonic() < deadline, "removed node never left the cluster"
        time.sleep(0.2)
    dead = dependencies.ClusterTopology.discover(ray)
    print(json.dumps({
        "live": [live.driver_gpus, live.non_driver_gpus],
        "dead": [dead.driver_gpus, dead.non_driver_gpus],
        "dead_entries": sum(1 for entry in ray.nodes() if not entry["Alive"]),
    }))
finally:
    ray.shutdown()
    cluster.shutdown()
"""


@pytest.mark.slow_test
def test_inspect_cluster_splits_driver_node_and_skips_dead_nodes():
    """The driver's own node (where this process attached) holds the driver GPUs;
    every other alive node is summed into the rollout pool. A dead node's GPUs
    are not schedulable, so they stop counting once it is removed, while Ray
    still lists it as a dead entry.

    A real two-node cluster in its own process, so the package cluster is
    untouched.
    """

    repo = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "-c", _TWO_NODE_PROBE],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )
    report = json.loads(completed.stdout.strip().splitlines()[-1])

    assert report["live"] == [1.0, 2.0]
    assert report["dead_entries"] == 1
    assert report["dead"] == [1.0, 0.0]


@pytest.mark.slow_test
def test_current_gpu_ids_reports_the_actor_assignment(local_ray):
    """Inside a real actor holding two logical GPUs, the ids are Ray's own
    assignment as integer ordinals, none dropped."""

    @local_ray.remote(num_cpus=0, num_gpus=2)
    class _GpuHolder:
        def ids(self) -> tuple[list[int], list[object]]:
            import ray

            from vrl.ray import dependencies as actor_dependencies

            return actor_dependencies.current_gpu_ids(), list(ray.get_gpu_ids())

    holder = _GpuHolder.remote()
    try:
        parsed, raw = local_ray.get(holder.ids.remote(), timeout=60)
    finally:
        local_ray.kill(holder, no_restart=True)

    assert len(parsed) == 2
    assert parsed == [int(value) for value in raw]


# Ray's ``get_gpu_ids`` answer is the one boundary these two read; on a real
# cluster it is whatever the raylet's device list holds, so the value forms a
# raylet can report (ints, decimal strings, device UUIDs) are written here.


@pytest.mark.parametrize(
    "values, expected", [([], []), ([0, 2], [0, 2]), (["0", "2"], [0, 2]), ([0, "2"], [0, 2])]
)
def test_current_gpu_ids_preserves_all_assigned_ordinals(monkeypatch, values, expected):
    import ray

    monkeypatch.setattr(ray, "get_gpu_ids", lambda: values)
    assert dependencies.current_gpu_ids() == expected


@pytest.mark.parametrize("invalid", ["GPU-uuid", 1.5, True, None, -1])
def test_current_gpu_ids_rejects_instead_of_dropping_or_truncating(monkeypatch, invalid):
    import ray

    monkeypatch.setattr(ray, "get_gpu_ids", lambda: [0, invalid])
    with pytest.raises(ValueError, match=r"Ray GPU ID\[1\]"):
        dependencies.current_gpu_ids()


@pytest.mark.slow_test
def test_cluster_topology_preserves_driver_node_lookup_failure(local_ray, monkeypatch):
    failure = RuntimeError("driver node lookup failed")
    lookup = Trace(monkeypatch)
    lookup.watch(dependencies, "current_node_ip", "node_ip")
    lookup.fail("node_ip", failure)

    with pytest.raises(RuntimeError, match="driver node lookup failed") as caught:
        dependencies.ClusterTopology.discover(local_ray)

    assert caught.value is failure
    assert lookup.events == ["node_ip"]
