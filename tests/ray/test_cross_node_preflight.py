"""Tests for the cross-node rollout preflight (vrl.ray.placement).

Pure (no Ray spin-up): the cluster topology is a stand-in (see
``tests/ray/_helpers.py``) and ``current_node_ip`` is monkeypatched, so these run
in the fast PR subset unlike the slow_test launcher integration tests. The
resources argument is the real ``resolve_distributed_resources`` output, so the
whole cross_node config -> resolution -> preflight chain is exercised.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from tests.ray._helpers import fake_ray, node
from vrl.ray import dependencies, placement
from vrl.ray.resources import resolve_distributed_resources

_MULTI_NODE_TOPOLOGY = pytest.mark.real_cover(
    None,
    why=(
        "a live 2-node Ray cluster cannot be created inside a unit test, and inspect_cluster only "
        "reaches the cluster through the injected ray module, so there is nowhere for a real "
        "counterpart to live; tests/e2e/test_real_checkpoint_rl.py is single-node only"
    ),
    tracked_in="docs/sprints/done/SPRINT_one-real-ray-cluster.md",
)


def _resources():
    """The real resolver, not a namespace with one attribute on it.

    ``cross_node_preflight``'s second argument is always
    ``resolve_distributed_resources(cfg)`` output in production. Standing in a
    ``SimpleNamespace(rollout_num_gpus=...)`` skipped the entire cross_node
    config -> resolution chain, so a resolution change that produced the wrong
    rollout GPU budget could not fail this test. Costs ~0.13ms resolved.
    """

    return resolve_distributed_resources(
        OmegaConf.create(
            {
                "distributed": {
                    "resources": {
                        "visible_devices": "auto",
                        "cross_node": True,
                        "trainer": {"num_gpus": 1},
                        "rollout": {"num_gpus": 1, "gpus_per_worker": 1, "num_workers": 1},
                    },
                },
            },
        ),
    )


@_MULTI_NODE_TOPOLOGY
def test_preflight_non_hybrid_rejects_driver_ray_gpu(monkeypatch):
    """Plain cross-node: any Ray GPU on the head fails fast (--num-gpus=0 required)."""
    monkeypatch.setattr(dependencies, "current_node_ip", lambda: "10.0.0.1")
    ray = fake_ray([node("10.0.0.1", 1.0), node("10.0.0.2", 1.0)])
    with pytest.raises(RuntimeError, match="num-gpus=0"):
        placement.cross_node_preflight(ray, _resources())


@_MULTI_NODE_TOPOLOGY
def test_preflight_non_hybrid_accepts_head_with_zero_gpus(monkeypatch):
    """Plain cross-node: head with --num-gpus=0 + enough remote GPUs passes."""
    monkeypatch.setattr(dependencies, "current_node_ip", lambda: "10.0.0.1")
    ray = fake_ray([node("10.0.0.1", 0.0), node("10.0.0.2", 1.0)])
    # State the pass condition instead of leaving it to "nothing was raised":
    # preflight signals acceptance by returning None, and a future version that
    # returns a verdict object has to come back through this line.
    assert placement.cross_node_preflight(ray, _resources()) is None
