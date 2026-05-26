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
    cross_node: bool = False,
    driver_node_ip: str | None = None,
) -> tuple[int, ...]:
    """Validate that Ray actors received only the expected GPU IDs.

    Single-node mode asserts every actor's node-local GPU id falls inside the
    globally resolved ordinal set. Cross-node mode cannot use that assumption
    (each node has its own ordinal space and Ray remaps ``CUDA_VISIBLE_DEVICES``
    per actor), so it instead validates that every worker (a) got a GPU, (b) does
    not run on the driver/head node, and (c) holds a unique ``(node_ip, gpu_id)``
    pair so no two workers share a physical GPU.
    """

    if cross_node:
        return _validate_cross_node_actor_gpu_ids(
            metadata,
            role=role,
            driver_node_ip=driver_node_ip,
        )

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


def _validate_cross_node_actor_gpu_ids(
    metadata: Sequence[Mapping[str, Any]],
    *,
    role: str,
    driver_node_ip: str | None,
) -> tuple[int, ...]:
    """Node-aware GPU validation for cross-node rollout actors."""

    seen_pairs: set[tuple[str, int]] = set()
    gpu_ids: list[int] = []
    for meta in metadata:
        worker_id = str(meta.get("worker_id", "unknown"))
        node_ip = str(meta.get("node_ip", ""))
        worker_gpu_ids = tuple(int(gpu_id) for gpu_id in meta.get("gpu_ids", ()))
        if not worker_gpu_ids:
            raise RuntimeError(f"Ray {role} worker {worker_id} has no assigned GPU ids")
        if driver_node_ip is not None and node_ip == str(driver_node_ip):
            raise RuntimeError(
                f"Ray {role} worker {worker_id} landed on the driver/head node "
                f"{node_ip}; cross-node rollout must run off the trainer node. "
                "Start the head with `ray start --head --num-gpus=0` so the trainer "
                "GPU stays out of Ray's scheduling pool.",
            )
        for gpu_id in worker_gpu_ids:
            pair = (node_ip, gpu_id)
            if pair in seen_pairs:
                raise RuntimeError(
                    f"Ray {role} workers share GPU {gpu_id} on node {node_ip}; "
                    "each rollout worker must own a distinct physical GPU.",
                )
            seen_pairs.add(pair)
            gpu_ids.append(gpu_id)
    return tuple(sorted(gpu_ids))


__all__ = [
    "actor_scheduling_strategy",
    "create_placement_group",
    "shutdown_placement_group",
    "validate_actor_gpu_ids",
]
