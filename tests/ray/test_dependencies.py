"""Tests for Ray cluster-topology inspection (vrl.ray.dependencies)."""

from __future__ import annotations

from types import SimpleNamespace

from vrl.ray import dependencies


def _ray(nodes):
    return SimpleNamespace(nodes=lambda: nodes)


def _node(ip, gpu, *, alive=True):
    return {"Alive": alive, "NodeManagerAddress": ip, "Resources": {"GPU": gpu}}


def test_inspect_cluster_splits_driver_vs_non_driver(monkeypatch):
    monkeypatch.setattr(dependencies, "current_node_ip", lambda: "10.0.0.1")
    topo = dependencies.inspect_cluster(
        _ray([_node("10.0.0.1", 0.0), _node("10.0.0.2", 1.0), _node("10.0.0.3", 2.0)]),
    )
    assert topo.driver_gpus == 0.0
    assert topo.non_driver_gpus == 3.0
    assert topo.has_non_driver_gpus is True


def test_inspect_cluster_single_node_has_no_non_driver_gpus(monkeypatch):
    monkeypatch.setattr(dependencies, "current_node_ip", lambda: "10.0.0.1")
    topo = dependencies.inspect_cluster(_ray([_node("10.0.0.1", 1.0)]))
    assert topo.driver_gpus == 1.0
    assert topo.non_driver_gpus == 0.0
    assert topo.has_non_driver_gpus is False


def test_inspect_cluster_skips_dead_nodes(monkeypatch):
    monkeypatch.setattr(dependencies, "current_node_ip", lambda: "10.0.0.1")
    topo = dependencies.inspect_cluster(
        _ray([_node("10.0.0.1", 0.0), _node("10.0.0.2", 1.0, alive=False)]),
    )
    assert topo.non_driver_gpus == 0.0
    assert topo.has_non_driver_gpus is False


def test_inspect_cluster_accepts_explicit_driver_ip():
    topo = dependencies.inspect_cluster(
        _ray([_node("10.0.0.1", 1.0), _node("10.0.0.2", 1.0)]),
        driver_node_ip="10.0.0.2",
    )
    assert topo.driver_gpus == 1.0
    assert topo.non_driver_gpus == 1.0
