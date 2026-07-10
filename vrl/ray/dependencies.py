"""Lazy Ray dependency and actor metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def require_ray() -> Any:
    """Import Ray lazily so base package imports do not require Ray."""

    try:
        import ray
    except ImportError as exc:  # pragma: no cover - exercised only without Ray
        raise ImportError("Ray runtime requires `ray`. Install Ray or disable Ray usage.") from exc
    return ray


def current_node_ip() -> str:
    """Return the Ray node IP for the current actor process."""

    ray = require_ray()
    return str(ray.util.get_node_ip_address())


def current_gpu_ids() -> list[int]:
    """Return integer GPU IDs assigned to the current Ray actor."""

    ray = require_ray()
    out: list[int] = []
    for gpu_id in ray.get_gpu_ids():
        try:
            out.append(int(gpu_id))
        except (TypeError, ValueError):
            continue
    return out


@dataclass(frozen=True, slots=True)
class ClusterTopology:
    """Live Ray-cluster GPU layout: GPUs on the driver/head node vs on the other
    (worker) nodes.

    The single basis for the single-node-vs-multi-node decision: cross_node
    auto-detect (``run_online_recipe``) and the cross-node preflight
    (``vrl.ray.placement.cross_node_preflight``) both read it instead of each
    re-walking ``ray.nodes()``.
    """

    driver_gpus: float
    non_driver_gpus: float

    @property
    def has_non_driver_gpus(self) -> bool:
        """GPUs exist off the driver node -- i.e. a multi-node rollout topology."""

        return self.non_driver_gpus > 0


def inspect_cluster(ray: Any, *, driver_node_ip: str | None = None) -> ClusterTopology:
    """Sum alive-node GPUs split by driver vs non-driver node.

    Requires an initialized/attached Ray cluster. ``driver_node_ip`` defaults to
    the current process's node ip; nodes matching it count as the driver/head.
    """

    if driver_node_ip is None:
        try:
            driver_node_ip = current_node_ip()
        except Exception:
            driver_node_ip = None
    driver_gpus = 0.0
    non_driver_gpus = 0.0
    for node in ray.nodes():
        if not node.get("Alive"):
            continue
        node_gpus = float(node.get("Resources", {}).get("GPU", 0.0))
        node_ip = node.get("NodeManagerAddress")
        if driver_node_ip is not None and node_ip == driver_node_ip:
            driver_gpus += node_gpus
        else:
            non_driver_gpus += node_gpus
    return ClusterTopology(driver_gpus=driver_gpus, non_driver_gpus=non_driver_gpus)


__all__ = [
    "ClusterTopology",
    "current_gpu_ids",
    "current_node_ip",
    "inspect_cluster",
    "require_ray",
]
