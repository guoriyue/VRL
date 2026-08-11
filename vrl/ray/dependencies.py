"""Lazy Ray dependency and actor metadata helpers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


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

    The single basis for the single-node-vs-multi-node decision: the cross-node
    preflight (``vrl.ray.placement.cross_node_preflight``) reads it instead of
    re-walking ``ray.nodes()``.
    """

    driver_gpus: float
    non_driver_gpus: float


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


def kill_actors(ray: Any, actors: list[Any]) -> list[tuple[Any, Exception]]:
    """Best-effort kill actors and return failures to the resource owner.

    Cleanup never raises mid-sweep: failures come back so the owner (actor
    group, placement owner, generation session, health monitor) can retain
    the failed handles and refuse to report cleanup as complete.
    """

    failures: list[tuple[Any, Exception]] = []
    for actor in actors:
        try:
            ray.kill(actor, no_restart=True)
        except Exception as error:
            failures.append((actor, error))
            logger.warning("Failed to kill owned Ray actor %r", actor, exc_info=True)
    return failures


def kill_and_retain[OwnedItem](
    ray: Any,
    items: list[OwnedItem],
    get_actor: Callable[[OwnedItem], Any],
) -> tuple[list[OwnedItem], list[tuple[Any, Exception]]]:
    """Kill each item's actor and retain only the items whose kill FAILED.

    The two Ray resource owners (``RayActorGroup.shutdown`` and the generation
    session) both own cleanup truth: they drop handles for actors that
    died and keep the ones they could not kill so cleanup is not falsely reported
    complete. ``get_actor`` maps one owned item to its Ray actor handle (or
    ``None`` when already released). Returns ``(surviving, failures)`` where
    ``surviving`` are the still-owned items whose actor kill raised and
    ``failures`` are the ``(actor, error)`` pairs from :func:`kill_actors`.
    """

    actors = [actor for item in items if (actor := get_actor(item)) is not None]
    failures = kill_actors(ray, actors)
    failed_actor_ids = {id(actor) for actor, _ in failures}
    surviving = [
        item
        for item in items
        if (actor := get_actor(item)) is not None and id(actor) in failed_actor_ids
    ]
    return surviving, failures


__all__ = [
    "ClusterTopology",
    "current_gpu_ids",
    "current_node_ip",
    "inspect_cluster",
    "kill_actors",
    "kill_and_retain",
    "require_ray",
]
