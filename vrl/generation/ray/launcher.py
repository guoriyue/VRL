"""Launch Ray generation workers and assemble the collector-facing runtime."""

from __future__ import annotations

import contextlib
import logging
import socket
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.generation.protocols import ChunkGatherer
from vrl.generation.ray.dependencies import require_ray
from vrl.generation.ray.executor import DistributedGenerationExecutor
from vrl.generation.ray.planner import DistributedExecutionPlanner
from vrl.generation.ray.runtime import RayGenerationRuntime
from vrl.generation.ray.types import RayWorkerHandle
from vrl.generation.ray.weight_sync import RayGenerationWeightSync
from vrl.generation.ray.worker import RayGenerationWorker
from vrl.generation.resources import (
    ResolvedDistributedResources,
    format_distributed_resource_plan,
)
from vrl.generation.runtime.config import GenerationRuntimeConfig

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

    @property
    def ordered_rollout_bundle_indices(self) -> list[int]:
        return self.ordered_bundle_indices


@dataclass(slots=True)
class RayGenerationLauncher:
    """Create Ray generation actors and return a ``RayGenerationRuntime``."""

    init_ray: bool = True
    ray_init_kwargs: dict[str, Any] = field(default_factory=dict)

    def launch(
        self,
        config: GenerationRuntimeConfig | Mapping[str, Any],
        launch_contract: GenerationRuntimeLaunchContract | Mapping[str, Any],
        gatherer: ChunkGatherer,
    ) -> RayGenerationRuntime:
        rollout_config = GenerationRuntimeConfig.from_cfg(config)
        if rollout_config.backend != "ray":
            raise ValueError(
                "RayGenerationLauncher requires distributed backend='ray', "
                f"got {rollout_config.backend!r}",
            )

        contract = GenerationRuntimeLaunchContract.from_value(launch_contract)
        if not contract.family:
            raise ValueError("GenerationRuntimeLaunchContract.family is required")
        chunk_gatherer = _require_chunk_gatherer(gatherer)

        ray = require_ray()
        if self.init_ray and not ray.is_initialized():
            ray.init(**self.ray_init_kwargs)

        placement = create_generation_placement_group(rollout_config)
        if rollout_config.resources is not None:
            logger.info(
                format_distributed_resource_plan(
                    rollout_config.resources,
                    actual_placement={
                        "trainer_gpu_ids": list(placement.trainer_gpu_ids),
                        "rollout_gpu_ids": list(placement.rollout_gpu_ids),
                    },
                ),
            )

        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        RemoteGenerationWorker = ray.remote(
            num_cpus=rollout_config.cpus_per_worker,
            num_gpus=rollout_config.gpus_per_worker,
        )(RayGenerationWorker)

        actors: list[Any] = []
        worker_ids: list[str] = []
        for logical_idx, bundle_idx in enumerate(placement.ordered_bundle_indices):
            worker_id = f"rollout-{logical_idx}"
            actor = RemoteGenerationWorker.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=placement.placement_group,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=bundle_idx,
                ),
            ).remote(worker_id, contract)
            actors.append(actor)
            worker_ids.append(worker_id)

        try:
            ray.get([actor.load_policy.remote() for actor in actors])
            metadata = ray.get([actor.worker_metadata.remote() for actor in actors])
            _validate_worker_gpu_ids(rollout_config, metadata)
        except Exception:
            _kill_actors(ray, actors)
            _kill_actors(ray, placement.trainer_reservation_actors)
            _remove_placement_group(ray, placement.placement_group)
            raise

        workers = [
            RayWorkerHandle(
                worker_id=worker_id,
                node_id=str(meta.get("node_ip", "unknown")),
                gpu_ids=tuple(int(gpu_id) for gpu_id in meta.get("gpu_ids", ())),
                actor=actor,
            )
            for worker_id, actor, meta in zip(worker_ids, actors, metadata, strict=True)
        ]

        executor = DistributedGenerationExecutor(
            DistributedExecutionPlanner(contract.extra.get("family_capability")),
            workers,
            chunk_gatherer,
            max_inflight_chunks_per_worker=rollout_config.max_inflight_chunks_per_worker,
        )
        weight_sync = (
            RayGenerationWeightSync(workers)
            if rollout_config.sync_trainable_state != "disabled"
            else None
        )
        runtime = RayGenerationRuntime(
            executor,
            weight_sync=weight_sync,
            owned_workers=workers,
            owned_actors=placement.trainer_reservation_actors,
            placement_group=placement.placement_group,
        )
        if contract.policy_version is not None:
            runtime.current_policy_version = contract.policy_version
        return runtime


class _InfoActor:
    def get_ip_and_gpu_ids(self) -> tuple[str, tuple[int, ...]]:
        ray = require_ray()
        gpu_ids: list[int] = []
        for gpu_id in ray.get_gpu_ids():
            try:
                gpu_ids.append(int(gpu_id))
            except (TypeError, ValueError):
                continue
        return str(ray.util.get_node_ip_address()), tuple(gpu_ids)


def create_generation_placement_group(config: GenerationRuntimeConfig) -> RayPlacement:
    """Create a placement group for generation workers and stable-sort bundles."""

    ray = require_ray()
    from ray.util.placement_group import placement_group

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

    pg = placement_group(bundles, strategy=config.placement_strategy)
    ray.get(pg.ready())

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


def _kill_actors(ray: Any, actors: list[Any]) -> None:
    for actor in actors:
        with contextlib.suppress(Exception):
            ray.kill(actor, no_restart=True)


def _require_chunk_gatherer(gatherer: Any) -> ChunkGatherer:
    gather_chunks = getattr(gatherer, "gather_chunks", None)
    if not callable(gather_chunks):
        raise TypeError(
            f"{type(gatherer).__name__} does not implement gather_chunks(...)",
        )
    return gatherer


def _validate_worker_gpu_ids(
    config: GenerationRuntimeConfig,
    metadata: list[Mapping[str, Any]],
) -> None:
    resources = config.resources
    if resources is None or resources.rollout_gpus_per_worker <= 0:
        return

    expected = set(resources.rollout_devices)
    actual: set[int] = set()
    for meta in metadata:
        worker_id = str(meta.get("worker_id", "unknown"))
        worker_gpu_ids = tuple(int(gpu_id) for gpu_id in meta.get("gpu_ids", ()))
        if not worker_gpu_ids:
            raise RuntimeError(f"Ray generation worker {worker_id} has no assigned GPU ids")
        outside = set(worker_gpu_ids) - expected
        if outside:
            raise RuntimeError(
                f"Ray generation worker {worker_id} assigned GPU ids "
                f"{sorted(worker_gpu_ids)}, outside resolved rollout devices "
                f"{sorted(expected)}",
            )
        actual.update(worker_gpu_ids)

    if actual != expected:
        raise RuntimeError(
            "Ray generation placement did not cover the resolved rollout devices: "
            f"actual={sorted(actual)} expected={sorted(expected)}",
        )


def _remove_placement_group(ray: Any, placement_group: Any) -> None:
    with contextlib.suppress(Exception):
        from ray.util import remove_placement_group

        remove_placement_group(placement_group)


def _rollout_bundle(config: GenerationRuntimeConfig) -> dict[str, float]:
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

    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    RemoteReservationActor = ray.remote(num_cpus=0.001, num_gpus=1.0)(_InfoActor)
    actors = [
        RemoteReservationActor.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=bundle_idx,
                placement_group_capture_child_tasks=True,
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
    config: GenerationRuntimeConfig,
    rollout_bundle_indices: list[int],
) -> tuple[list[int], tuple[int, ...]]:
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    RemoteInfoActor = ray.remote(
        num_cpus=config.cpus_per_worker,
        num_gpus=config.gpus_per_worker,
    )(_InfoActor)

    info_actors = [
        RemoteInfoActor.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=bundle_idx,
                placement_group_capture_child_tasks=True,
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


RayRolloutLauncher = RayGenerationLauncher
create_rollout_placement_group = create_generation_placement_group


__all__ = [
    "RayGenerationLauncher",
    "RayPlacement",
    "RayRolloutLauncher",
    "create_generation_placement_group",
    "create_rollout_placement_group",
]
