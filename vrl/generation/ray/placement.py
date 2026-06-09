"""Generation placement planning over shared Ray placement helpers."""

from __future__ import annotations

import logging
import socket
from dataclasses import dataclass
from typing import Any

from vrl.generation.ray.config import RayGenerationConfig
from vrl.ray.dependencies import current_gpu_ids, current_node_ip, require_ray
from vrl.ray.placement import actor_scheduling_strategy, create_placement_group
from vrl.ray.resources import (
    ResolvedDistributedResources,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RayPlacement:
    """Placement group and stable bundle order for generation workers."""

    placement_group: Any
    ordered_bundle_indices: list[int]
    trainer_bundle_indices: list[int]
    trainer_reservation_actors: list[Any]
    trainer_gpu_ids: tuple[int, ...]
    rollout_gpu_ids: tuple[int, ...]


class _InfoActor:
    def get_ip_and_gpu_ids(self) -> tuple[str, tuple[int, ...]]:
        return current_node_ip(), tuple(current_gpu_ids())


def create_generation_placement_group(config: RayGenerationConfig) -> RayPlacement:
    """Create a placement group for generation workers and stable-sort bundles."""

    ray = require_ray()

    resources = config.resources
    bundles: list[dict[str, float]] = []
    trainer_bundle_indices: list[int] = []
    if resources is not None and resources.requires_trainer_reservation:
        for _ in resources.trainer_devices:
            trainer_bundle_indices.append(len(bundles))
            bundles.append(_trainer_reservation_bundle())

    rollout_bundle_indices: list[int] = []
    for _ in range(config.num_workers):
        rollout_bundle_indices.append(len(bundles))
        bundles.append(_rollout_bundle(config))

    pg = create_placement_group(bundles, strategy=config.placement_strategy)

    trainer_actors, trainer_gpu_ids = _start_trainer_reservations(
        ray,
        pg,
        trainer_bundle_indices,
    )
    ordered, rollout_gpu_ids = _probe_rollout_bundles(
        ray,
        pg,
        config,
        rollout_bundle_indices,
    )

    _log_placement(
        ordered,
        rollout_bundle_indices,
        rollout_gpu_ids,
        trainer_gpu_ids,
        resources,
    )
    return RayPlacement(
        placement_group=pg,
        ordered_bundle_indices=ordered,
        trainer_bundle_indices=trainer_bundle_indices,
        trainer_reservation_actors=trainer_actors,
        trainer_gpu_ids=trainer_gpu_ids,
        rollout_gpu_ids=rollout_gpu_ids,
    )


def _rollout_bundle(config: RayGenerationConfig) -> dict[str, float]:
    bundle = {"CPU": float(config.cpus_per_worker)}
    if config.gpus_per_worker > 0:
        bundle["GPU"] = float(config.gpus_per_worker)
    return bundle


def _trainer_reservation_bundle() -> dict[str, float]:
    return {"CPU": 0.001, "GPU": 1.0}


def _start_trainer_reservations(
    ray: Any,
    pg: Any,
    trainer_bundle_indices: list[int],
) -> tuple[list[Any], tuple[int, ...]]:
    if not trainer_bundle_indices:
        return [], ()

    RemoteReservationActor = ray.remote(num_cpus=0.001, num_gpus=1.0)(_InfoActor)
    actors = [
        RemoteReservationActor.options(
            scheduling_strategy=actor_scheduling_strategy(
                placement_group=pg,
                bundle_index=bundle_idx,
                capture_child_tasks=True,
            ),
        ).remote()
        for bundle_idx in trainer_bundle_indices
    ]
    placement = ray.get([actor.get_ip_and_gpu_ids.remote() for actor in actors])
    gpu_ids: list[int] = []
    for _, ids in placement:
        gpu_ids.extend(ids)
    return actors, tuple(gpu_ids)


def _probe_rollout_bundles(
    ray: Any,
    pg: Any,
    config: RayGenerationConfig,
    rollout_bundle_indices: list[int],
) -> tuple[list[int], tuple[int, ...]]:
    RemoteInfoActor = ray.remote(
        num_cpus=config.cpus_per_worker,
        num_gpus=config.gpus_per_worker,
    )(_InfoActor)

    info_actors = [
        RemoteInfoActor.options(
            scheduling_strategy=actor_scheduling_strategy(
                placement_group=pg,
                bundle_index=bundle_idx,
                capture_child_tasks=True,
            ),
        ).remote()
        for bundle_idx in rollout_bundle_indices
    ]

    try:
        ip_gpu_pairs = ray.get([actor.get_ip_and_gpu_ids.remote() for actor in info_actors])
    finally:
        for actor in info_actors:
            ray.kill(actor, no_restart=True)

    bundle_infos = []
    rollout_gpu_ids: list[int] = []
    for probe_idx, (node_ip, gpu_ids) in enumerate(ip_gpu_pairs):
        gpu_id = int(gpu_ids[0]) if gpu_ids else -1
        bundle_idx = rollout_bundle_indices[probe_idx]
        bundle_infos.append((bundle_idx, node_ip, gpu_id))
        rollout_gpu_ids.extend(gpu_ids)

    ordered = [idx for idx, _, _ in sorted(bundle_infos, key=_sort_node_gpu_key)]
    return ordered, tuple(rollout_gpu_ids)


def _sort_node_gpu_key(item: tuple[int, str, int]) -> tuple[list[int], int, int]:
    """Stable sort key for generation placement bundles."""

    index, node_identifier, gpu_id = item
    try:
        node_ip_parts = [int(part) for part in node_identifier.split(".")]
    except ValueError:
        try:
            resolved = socket.gethostbyname(node_identifier)
            node_ip_parts = [int(part) for part in resolved.split(".")]
        except (socket.gaierror, TypeError):
            node_ip_parts = [ord(char) for char in node_identifier]
    return node_ip_parts, gpu_id, index


def _log_placement(
    ordered: list[int],
    rollout_bundle_indices: list[int],
    rollout_gpu_ids: tuple[int, ...],
    trainer_gpu_ids: tuple[int, ...],
    resources: ResolvedDistributedResources | None,
) -> None:
    if trainer_gpu_ids:
        logger.info("Ray trainer reservation actual GPU ids: %s", list(trainer_gpu_ids))

    for logical_idx, bundle_idx in enumerate(ordered):
        try:
            probe_idx = rollout_bundle_indices.index(bundle_idx)
        except ValueError:
            probe_idx = logical_idx
        gpu_id = rollout_gpu_ids[probe_idx] if probe_idx < len(rollout_gpu_ids) else -1
        logger.info(
            "Ray generation bundle %d -> actual bundle %d gpu=%s",
            logical_idx,
            bundle_idx,
            gpu_id,
        )
    if resources is not None:
        logger.info(
            "Ray resolved GPU plan trainer=%s rollout=%s actual_rollout=%s",
            list(resources.trainer_devices),
            list(resources.rollout_devices),
            list(rollout_gpu_ids),
        )

__all__ = [
    "RayPlacement",
    "create_generation_placement_group",
]
