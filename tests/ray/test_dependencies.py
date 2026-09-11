"""Tests for Ray cluster-topology inspection (vrl.ray.dependencies)."""

from __future__ import annotations

import pytest

from tests.ray._helpers import fake_ray, node
from vrl.ray import dependencies

_MULTI_NODE_TOPOLOGY = pytest.mark.real_cover(
    None,
    why=(
        "a live 3-node / 2-GPU Ray cluster cannot be created inside a unit test, and "
        "ClusterTopology.from_ray only reaches the cluster through the injected ray module, so there is "
        "nowhere for a real counterpart to live; tests/e2e/test_real_checkpoint_rl.py is "
        "single-node only"
    ),
    tracked_in="docs/sprints/done/SPRINT_one-real-ray-cluster.md",
)

pytestmark = _MULTI_NODE_TOPOLOGY


def test_inspect_cluster_splits_driver_vs_non_driver(monkeypatch):
    """The driver's own node is excluded from the rollout GPU pool; the other
    alive nodes are summed into it."""
    monkeypatch.setattr(dependencies, "current_node_ip", lambda: "10.0.0.1")
    topo = dependencies.ClusterTopology.from_ray(
        fake_ray([node("10.0.0.1", 0.0), node("10.0.0.2", 1.0), node("10.0.0.3", 2.0)]),
    )
    assert topo.driver_gpus == 0.0
    assert topo.non_driver_gpus == 3.0


def test_inspect_cluster_single_node_has_no_non_driver_gpus(monkeypatch):
    """A single-node cluster offers zero rollout GPUs, however many it has: they
    all belong to the driver, which is what the cross-node preflight rejects."""
    monkeypatch.setattr(dependencies, "current_node_ip", lambda: "10.0.0.1")
    topo = dependencies.ClusterTopology.from_ray(fake_ray([node("10.0.0.1", 1.0)]))
    assert topo.driver_gpus == 1.0
    assert topo.non_driver_gpus == 0.0


def test_inspect_cluster_skips_dead_nodes(monkeypatch):
    """A dead node's GPUs are not schedulable, so they must not count toward the
    rollout budget the preflight checks."""
    monkeypatch.setattr(dependencies, "current_node_ip", lambda: "10.0.0.1")
    topo = dependencies.ClusterTopology.from_ray(
        fake_ray([node("10.0.0.1", 0.0), node("10.0.0.2", 1.0, alive=False)]),
    )
    assert topo.non_driver_gpus == 0.0


def test_inspect_cluster_counts_the_current_node_as_driver(monkeypatch):
    """The driver is whichever node this process runs on; no second injection
    seam exists (tests steer it exactly like production would experience it)."""
    monkeypatch.setattr(dependencies, "current_node_ip", lambda: "10.0.0.2")
    topo = dependencies.ClusterTopology.from_ray(
        fake_ray([node("10.0.0.1", 1.0), node("10.0.0.2", 1.0)]),
    )
    assert topo.driver_gpus == 1.0
    assert topo.non_driver_gpus == 1.0


@pytest.mark.parametrize(
    "values, expected", [([], []), ([0, 2], [0, 2]), (["0", "2"], [0, 2]), ([0, "2"], [0, 2])]
)
def test_current_gpu_ids_preserves_all_assigned_ordinals(monkeypatch, values, expected):
    from types import SimpleNamespace

    monkeypatch.setattr(
        dependencies, "require_ray", lambda: SimpleNamespace(get_gpu_ids=lambda: values)
    )
    assert dependencies.current_gpu_ids() == expected


@pytest.mark.parametrize("invalid", ["GPU-uuid", "bad", "", "1.5", "-1", 1.5, True, None, -1])
def test_current_gpu_ids_rejects_instead_of_dropping_or_truncating(monkeypatch, invalid):
    from types import SimpleNamespace

    monkeypatch.setattr(
        dependencies, "require_ray", lambda: SimpleNamespace(get_gpu_ids=lambda: [0, invalid])
    )
    with pytest.raises(ValueError, match=r"Ray GPU ID\[1\]"):
        dependencies.current_gpu_ids()


def test_cluster_topology_preserves_driver_node_lookup_failure(monkeypatch):
    failure = RuntimeError("driver node lookup failed")

    def fail():
        raise failure

    monkeypatch.setattr(dependencies, "current_node_ip", fail)
    with pytest.raises(RuntimeError, match="driver node lookup failed") as caught:
        dependencies.ClusterTopology.from_ray(fake_ray([node("10.0.0.1", 2.0)]))
    assert caught.value is failure
