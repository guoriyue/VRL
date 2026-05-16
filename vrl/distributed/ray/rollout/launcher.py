"""Launch Ray rollout workers and assemble the collector-facing runtime."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from vrl.distributed.ray.dependencies import require_ray
from vrl.distributed.ray.rollout.executor import DistributedRolloutExecutor
from vrl.distributed.ray.rollout.placement import create_rollout_placement_group
from vrl.distributed.ray.rollout.planner import DistributedExecutionPlanner
from vrl.distributed.ray.rollout.runtime import RayDistributedRuntime
from vrl.distributed.ray.rollout.types import RayWorkerHandle
from vrl.distributed.ray.rollout.weight_sync import RayRolloutWeightSync
from vrl.distributed.ray.rollout.worker import RayRolloutWorker
from vrl.distributed.resources import format_distributed_resource_plan
from vrl.engine.core.launch_contract import GenerationRuntimeLaunchContract
from vrl.engine.execution.gather import ChunkGatherer, require_chunk_gatherer
from vrl.rollouts.runtime.config import RolloutBackendConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RayRolloutLauncher:
    """Create Ray rollout actors and return a ``RayDistributedRuntime``."""

    init_ray: bool = True
    ray_init_kwargs: dict[str, Any] = field(default_factory=dict)

    def launch(
        self,
        config: RolloutBackendConfig | Mapping[str, Any],
        launch_contract: GenerationRuntimeLaunchContract | Mapping[str, Any],
        gatherer: ChunkGatherer,
    ) -> RayDistributedRuntime:
        rollout_config = RolloutBackendConfig.from_cfg(config)
        if rollout_config.backend != "ray":
            raise ValueError(
                "RayRolloutLauncher requires distributed rollout backend='ray', "
                f"got {rollout_config.backend!r}",
            )

        contract = GenerationRuntimeLaunchContract.from_value(launch_contract)
        if not contract.family:
            raise ValueError("GenerationRuntimeLaunchContract.family is required")
        chunk_gatherer = require_chunk_gatherer(gatherer)

        ray = require_ray()
        if self.init_ray and not ray.is_initialized():
            ray.init(**self.ray_init_kwargs)

        placement = create_rollout_placement_group(rollout_config)
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

        RemoteRolloutWorker = ray.remote(
            num_cpus=rollout_config.cpus_per_worker,
            num_gpus=rollout_config.gpus_per_worker,
        )(RayRolloutWorker)

        actors: list[Any] = []
        worker_ids: list[str] = []
        for logical_idx, bundle_idx in enumerate(placement.ordered_bundle_indices):
            worker_id = f"rollout-{logical_idx}"
            actor = RemoteRolloutWorker.options(
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

        executor = DistributedRolloutExecutor(
            DistributedExecutionPlanner(contract.extra.get("family_capability")),
            workers,
            chunk_gatherer,
            max_inflight_chunks_per_worker=rollout_config.max_inflight_chunks_per_worker,
        )
        weight_sync = (
            RayRolloutWeightSync(workers)
            if rollout_config.sync_trainable_state != "disabled"
            else None
        )
        runtime = RayDistributedRuntime(
            executor,
            weight_sync=weight_sync,
            owned_workers=workers,
            owned_actors=placement.trainer_reservation_actors,
            placement_group=placement.placement_group,
        )
        if contract.policy_version is not None:
            runtime.current_policy_version = contract.policy_version
        return runtime


def _kill_actors(ray: Any, actors: list[Any]) -> None:
    for actor in actors:
        with contextlib.suppress(Exception):
            ray.kill(actor, no_restart=True)


def _validate_worker_gpu_ids(
    config: RolloutBackendConfig,
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
            raise RuntimeError(f"Ray rollout worker {worker_id} has no assigned GPU ids")
        outside = set(worker_gpu_ids) - expected
        if outside:
            raise RuntimeError(
                f"Ray rollout worker {worker_id} assigned GPU ids "
                f"{sorted(worker_gpu_ids)}, outside resolved rollout devices "
                f"{sorted(expected)}",
            )
        actual.update(worker_gpu_ids)

    if actual != expected:
        raise RuntimeError(
            "Ray rollout placement did not cover the resolved rollout devices: "
            f"actual={sorted(actual)} expected={sorted(expected)}",
        )


def _remove_placement_group(ray: Any, placement_group: Any) -> None:
    with contextlib.suppress(Exception):
        from ray.util import remove_placement_group

        remove_placement_group(placement_group)


__all__ = [
    "RayRolloutLauncher",
]
