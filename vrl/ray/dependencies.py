"""Lazy Ray dependency and actor metadata helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from vrl.utils.validation import require_int

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
    for index, gpu_id in enumerate(ray.get_gpu_ids()):
        path = f"Ray GPU ID[{index}]"
        if isinstance(gpu_id, str):
            if not gpu_id.isascii() or not gpu_id.isdecimal():
                raise ValueError(f"{path} must be a non-negative integer ordinal, got {gpu_id!r}")
            gpu_id = int(gpu_id)
        out.append(require_int(gpu_id, path=path, minimum=0))
    return out


@dataclass(frozen=True, slots=True)
class ClusterTopology:
    """Ray-advertised logical GPU capacity on driver and non-driver nodes.

    This is total configured scheduling capacity on alive nodes, not physical
    GPU inventory, currently unallocated capacity, free VRAM, or device health.
    The driver node is where this process runs; it need not be the Ray head.
    """

    driver_gpus: float
    non_driver_gpus: float

    @classmethod
    def discover(cls, ray: Any) -> ClusterTopology:
        """Read logical GPU capacity from an already attached Ray cluster.

        Classify nodes by the current process's IP. This queries Ray's resource
        declarations; it neither initializes Ray nor probes physical hardware.
        """

        driver_node_ip = current_node_ip()
        driver_gpus = 0.0
        non_driver_gpus = 0.0
        for node in ray.nodes():
            if not node.get("Alive"):
                continue
            node_gpus = float(node.get("Resources", {}).get("GPU", 0.0))
            node_ip = node.get("NodeManagerAddress")
            if node_ip == driver_node_ip:
                driver_gpus += node_gpus
            else:
                non_driver_gpus += node_gpus
        return cls(driver_gpus=driver_gpus, non_driver_gpus=non_driver_gpus)


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


def raise_if_kill_failures(failures: list[tuple[Any, Exception]], *, what: str) -> None:
    """Refuse to report ``what``'s cleanup as complete when any kill failed."""

    if failures:
        raise RuntimeError(
            f"{what} cleanup incomplete: {len(failures)} actor kill(s) failed",
        ) from failures[0][1]


def note_kill_failures(
    error: BaseException,
    failures: list[tuple[Any, Exception]],
    *,
    what: str,
) -> None:
    """Attach kill failures to an error already propagating out of ``what``."""

    if failures:
        error.add_note(f"{what} cleanup incomplete: {len(failures)} actor kill(s) failed")


__all__ = [
    "ClusterTopology",
    "current_gpu_ids",
    "current_node_ip",
    "kill_actors",
    "note_kill_failures",
    "raise_if_kill_failures",
    "require_ray",
]
