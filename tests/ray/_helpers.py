"""Shared Ray test helpers: a live-cluster topology stand-in.

``inspect_cluster(ray, ...)`` and ``cross_node_preflight(ray, ...)`` take the
``ray`` module as a parameter, which is the seam these two build against: a
3-node / 2-GPU cluster cannot be created inside a unit test, but the code that
reads one can still be driven for real. What the tests assert is the topology
production computes from ``nodes()``, never a value written here.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def fake_ray(nodes: list[dict[str, Any]]) -> Any:
    """A ``ray`` stand-in exposing only the one call cluster inspection makes."""

    return SimpleNamespace(nodes=lambda: nodes)


def node(ip: str, gpu: float, *, alive: bool = True) -> dict[str, Any]:
    """One entry shaped like ``ray.nodes()`` returns."""

    return {"Alive": alive, "NodeManagerAddress": ip, "Resources": {"GPU": gpu}}
