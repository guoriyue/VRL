"""Tests for Ray cluster-topology inspection (vrl.ray.dependencies)."""

from __future__ import annotations

import pytest

from tests.ray._helpers import fake_ray, node
from vrl.ray import dependencies

_MULTI_NODE_TOPOLOGY = pytest.mark.real_cover(
    None,
    why=(
        "a live 3-node / 2-GPU Ray cluster cannot be created inside a unit test, and "
        "inspect_cluster only reaches the cluster through the injected ray module, so there is "
        "nowhere for a real counterpart to live; tests/e2e/test_real_checkpoint_rl.py is "
        "single-node only"
    ),
    tracked_in="docs/sprints/planned/SPRINT_one-real-ray-cluster.md",
)

pytestmark = _MULTI_NODE_TOPOLOGY


def test_inspect_cluster_splits_driver_vs_non_driver(monkeypatch):
    """The driver's own node is excluded from the rollout GPU pool; the other
    alive nodes are summed into it."""
    monkeypatch.setattr(dependencies, "current_node_ip", lambda: "10.0.0.1")
    topo = dependencies.inspect_cluster(
        fake_ray([node("10.0.0.1", 0.0), node("10.0.0.2", 1.0), node("10.0.0.3", 2.0)]),
    )
    assert topo.driver_gpus == 0.0
    assert topo.non_driver_gpus == 3.0


def test_inspect_cluster_single_node_has_no_non_driver_gpus(monkeypatch):
    """A single-node cluster offers zero rollout GPUs, however many it has: they
    all belong to the driver, which is what the cross-node preflight rejects."""
    monkeypatch.setattr(dependencies, "current_node_ip", lambda: "10.0.0.1")
    topo = dependencies.inspect_cluster(fake_ray([node("10.0.0.1", 1.0)]))
    assert topo.driver_gpus == 1.0
    assert topo.non_driver_gpus == 0.0


def test_inspect_cluster_skips_dead_nodes(monkeypatch):
    """A dead node's GPUs are not schedulable, so they must not count toward the
    rollout budget the preflight checks."""
    monkeypatch.setattr(dependencies, "current_node_ip", lambda: "10.0.0.1")
    topo = dependencies.inspect_cluster(
        fake_ray([node("10.0.0.1", 0.0), node("10.0.0.2", 1.0, alive=False)]),
    )
    assert topo.non_driver_gpus == 0.0


def test_inspect_cluster_accepts_explicit_driver_ip():
    """An explicit driver ip overrides autodetection, so a caller that already
    knows the head node does not depend on this process being on it."""
    topo = dependencies.inspect_cluster(
        fake_ray([node("10.0.0.1", 1.0), node("10.0.0.2", 1.0)]),
        driver_node_ip="10.0.0.2",
    )
    assert topo.driver_gpus == 1.0
    assert topo.non_driver_gpus == 1.0
