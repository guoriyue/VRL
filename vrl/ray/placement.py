"""Shared Ray placement helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from vrl.ray.dependencies import require_ray
from vrl.ray.lifecycle import remove_placement_group


def create_placement_group(
    bundles: Sequence[Mapping[str, float]],
    *,
    strategy: str,
) -> Any:
    """Create a Ray placement group and wait until it is ready."""

    if not bundles:
        raise ValueError("Ray placement group requires at least one bundle")
    ray = require_ray()
    from ray.util.placement_group import placement_group

    pg = placement_group([dict(bundle) for bundle in bundles], strategy=str(strategy))
    ray.get(pg.ready())
    return pg


def actor_scheduling_strategy(
    placement_group: Any,
    *,
    bundle_index: int | None = None,
    capture_child_tasks: bool = True,
) -> Any:
    """Build a placement-group scheduling strategy for one actor."""

    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    return PlacementGroupSchedulingStrategy(
        placement_group=placement_group,
        placement_group_bundle_index=bundle_index,
        placement_group_capture_child_tasks=capture_child_tasks,
    )


def shutdown_placement_group(placement_group: Any | None) -> None:
    """Remove a placement group if one exists."""

    remove_placement_group(placement_group)


def validate_actor_gpu_ids(
    metadata: Sequence[Mapping[str, Any]],
    *,
    expected_gpu_ids: Sequence[int],
    role: str,
) -> tuple[int, ...]:
    """Validate that Ray actors received only the expected GPU IDs."""

    expected = {int(gpu_id) for gpu_id in expected_gpu_ids}
    if not expected:
        return ()

    actual: set[int] = set()
    for meta in metadata:
        worker_id = str(meta.get("worker_id", "unknown"))
        worker_gpu_ids = tuple(int(gpu_id) for gpu_id in meta.get("gpu_ids", ()))
        if not worker_gpu_ids:
            raise RuntimeError(f"Ray {role} worker {worker_id} has no assigned GPU ids")
        outside = set(worker_gpu_ids) - expected
        if outside:
            raise RuntimeError(
                f"Ray {role} worker {worker_id} assigned GPU ids "
                f"{sorted(worker_gpu_ids)}, outside resolved {role} devices "
                f"{sorted(expected)}",
            )
        actual.update(worker_gpu_ids)

    if actual != expected:
        raise RuntimeError(
            f"Ray {role} placement did not cover the resolved {role} devices: "
            f"actual={sorted(actual)} expected={sorted(expected)}",
        )
    return tuple(sorted(actual))


__all__ = [
    "actor_scheduling_strategy",
    "create_placement_group",
    "shutdown_placement_group",
    "validate_actor_gpu_ids",
]
