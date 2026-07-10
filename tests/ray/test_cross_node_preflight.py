"""Tests for the cross-node rollout preflight (vrl.ray.placement).

Pure (no Ray spin-up): a fake ``ray`` exposes ``nodes()`` and ``current_node_ip``
is monkeypatched, so these run in the fast PR subset unlike the slow_test launcher
integration tests.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vrl.ray import dependencies, placement


def _ray(nodes):
    return SimpleNamespace(nodes=lambda: nodes)


def _node(ip, gpu, *, alive=True):
    return {"Alive": alive, "NodeManagerAddress": ip, "Resources": {"GPU": gpu}}


def _resources(*, rollout_num_gpus):
    return SimpleNamespace(rollout_num_gpus=rollout_num_gpus)


def test_preflight_non_hybrid_rejects_driver_ray_gpu(monkeypatch):
    """Plain cross-node: any Ray GPU on the head fails fast (--num-gpus=0 required)."""
    monkeypatch.setattr(dependencies, "current_node_ip", lambda: "10.0.0.1")
    ray = _ray([_node("10.0.0.1", 1.0), _node("10.0.0.2", 1.0)])
    with pytest.raises(RuntimeError, match="num-gpus=0"):
        placement.cross_node_preflight(ray, _resources(rollout_num_gpus=1))


def test_preflight_non_hybrid_accepts_head_with_zero_gpus(monkeypatch):
    """Plain cross-node: head with --num-gpus=0 + enough remote GPUs passes."""
    monkeypatch.setattr(dependencies, "current_node_ip", lambda: "10.0.0.1")
    ray = _ray([_node("10.0.0.1", 0.0), _node("10.0.0.2", 1.0)])
    placement.cross_node_preflight(ray, _resources(rollout_num_gpus=1))
