"""Launch Ray generation workers and assemble the collector-facing runtime."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from vrl.generation.execution.distributed import (
    DistributedExecutionPlanner,
    DistributedWorkerHandle,
)
from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.generation.protocols import ChunkGatherer
from vrl.generation.ray.executor import RayGenerationExecutor
from vrl.generation.ray.placement import (
    RayPlacement,
    create_generation_placement_group,
)
from vrl.generation.ray.runtime import RayGenerationRuntime
from vrl.generation.ray.weight_sync import RayGenerationWeightSync
from vrl.generation.ray.worker import RayGenerationWorker
from vrl.generation.resources import format_distributed_resource_plan
from vrl.generation.runtime.config import GenerationRuntimeConfig
from vrl.ray.actor_group import RayActorGroup
from vrl.ray.dependencies import require_ray
from vrl.ray.lifecycle import kill_actors, remove_placement_group
from vrl.ray.placement import validate_actor_gpu_ids

logger = logging.getLogger(__name__)


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

        worker_ids: list[str] = []
        worker_configs: list[GenerationRuntimeLaunchContract] = []
        for logical_idx, _bundle_idx in enumerate(placement.ordered_bundle_indices):
            worker_id = f"rollout-{logical_idx}"
            worker_ids.append(worker_id)
            worker_configs.append(contract)

        try:
            actor_group = RayActorGroup.launch(
                worker_cls=RayGenerationWorker,
                worker_configs=worker_configs,
                worker_ids=worker_ids,
                num_cpus=rollout_config.cpus_per_worker,
                num_gpus=rollout_config.gpus_per_worker,
                placement_group=placement.placement_group,
                bundle_indices=placement.ordered_bundle_indices,
                startup_method="load_policy",
            )
            metadata = [
                {
                    "worker_id": handle.worker_id,
                    "node_ip": handle.node_id,
                    "gpu_ids": handle.gpu_ids,
                }
                for handle in actor_group.handles
            ]
            _validate_worker_gpu_ids(rollout_config, metadata)
        except Exception:
            if "actor_group" in locals():
                actor_group.shutdown()
            kill_actors(ray, placement.trainer_reservation_actors)
            remove_placement_group(placement.placement_group)
            raise

        workers = [
            DistributedWorkerHandle(
                worker_id=handle.worker_id,
                node_id=handle.node_id,
                gpu_ids=handle.gpu_ids,
                actor=handle.actor,
            )
            for handle in actor_group.handles
        ]

        executor = RayGenerationExecutor(
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

    validate_actor_gpu_ids(
        metadata,
        expected_gpu_ids=resources.rollout_devices,
        role="generation",
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
