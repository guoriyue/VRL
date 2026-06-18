"""Tests for multi-node cross_node auto-detection in the online recipe.

These exercise the helpers with a stub Ray object so no real cluster is needed.
The single-node path (no external cluster) must leave cross_node untouched; a
multi-node cluster (GPUs on a non-driver node) must auto-enable it; an explicit
cross_node is always respected.
"""

from __future__ import annotations

from types import SimpleNamespace

from omegaconf import OmegaConf

from vrl.scripts.common import online


def _ray(nodes, *, initialized=False, attach_error=False):
    state = {"initialized": initialized}

    def _init(address=None):
        if attach_error:
            raise ConnectionError("no running Ray instance")
        state["initialized"] = True

    return SimpleNamespace(
        is_initialized=lambda: state["initialized"],
        init=_init,
        nodes=lambda: nodes,
    )


def _node(ip, gpu, *, alive=True):
    return {"Alive": alive, "NodeManagerAddress": ip, "Resources": {"GPU": gpu}}


def test_cluster_has_non_driver_gpus_true_for_multinode(monkeypatch):
    monkeypatch.setattr(online, "current_node_ip", lambda: "10.0.0.1")
    ray = _ray([_node("10.0.0.1", 0.0), _node("10.0.0.2", 1.0)])
    assert online._cluster_has_non_driver_gpus(ray) is True


def test_cluster_has_non_driver_gpus_false_for_single_node(monkeypatch):
    monkeypatch.setattr(online, "current_node_ip", lambda: "10.0.0.1")
    ray = _ray([_node("10.0.0.1", 1.0)])
    assert online._cluster_has_non_driver_gpus(ray) is False


def test_autodetect_respects_explicit_cross_node():
    cfg = OmegaConf.create(
        {"distributed": {"resources": {"cross_node": False, "trainer": {"num_gpus": 1}}}},
    )
    # attach_error=True would raise if ray.init were called; an explicit setting
    # must short-circuit before any attach.
    ray = _ray([_node("10.0.0.1", 0.0), _node("10.0.0.2", 1.0)], attach_error=True)
    online._maybe_autodetect_cross_node(cfg, ray)
    assert cfg.distributed.resources.cross_node is False


def test_autodetect_noop_when_no_external_cluster(monkeypatch):
    monkeypatch.setattr(online, "current_node_ip", lambda: "10.0.0.1")
    cfg = OmegaConf.create({"distributed": {"resources": {"trainer": {"num_gpus": 1}}}})
    ray = _ray([], attach_error=True)  # ray.init(address="auto") raises ConnectionError
    online._maybe_autodetect_cross_node(cfg, ray)
    assert OmegaConf.select(cfg, "distributed.resources.cross_node") is None


def test_autodetect_enables_cross_node_on_multinode(monkeypatch):
    monkeypatch.setattr(online, "current_node_ip", lambda: "10.0.0.1")
    cfg = OmegaConf.create({"distributed": {"resources": {"trainer": {"num_gpus": 1}}}})
    ray = _ray([_node("10.0.0.1", 0.0), _node("10.0.0.2", 1.0)])
    online._maybe_autodetect_cross_node(cfg, ray)
    assert cfg.distributed.resources.cross_node is True


def test_autodetect_single_node_attached_does_not_enable(monkeypatch):
    monkeypatch.setattr(online, "current_node_ip", lambda: "10.0.0.1")
    cfg = OmegaConf.create({"distributed": {"resources": {"trainer": {"num_gpus": 1}}}})
    ray = _ray([_node("10.0.0.1", 1.0)])  # attach succeeds, single node only
    online._maybe_autodetect_cross_node(cfg, ray)
    assert OmegaConf.select(cfg, "distributed.resources.cross_node") is None
