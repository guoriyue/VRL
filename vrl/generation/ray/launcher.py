"""Launch Ray generation workers and assemble the collector-facing runtime."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from vrl.generation.execution import (
    ChunkPlacementPolicy,
    DistributedExecutionPlanner,
    DistributedWorkerHandle,
)
from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.generation.protocols import ChunkGatherer, GenerationRuntime
from vrl.generation.ray.config import RayGenerationConfig
from vrl.generation.ray.executor import RayGenerationExecutor
from vrl.generation.ray.runtime import RayGenerationRuntime
from vrl.generation.ray.weight_sync import RayGenerationWeightSync
from vrl.generation.ray.worker import RayGenerationWorker
from vrl.models.dtypes import dtype_to_config_string
from vrl.models.interfaces.runtime import torch_compile_model_config
from vrl.ray.actor_group import RayActorGroup
from vrl.ray.dependencies import current_node_ip, import_from_path, require_ray
from vrl.ray.placement import RolePlacement, validate_actor_gpu_ids
from vrl.utils.config import cfg_path, to_builtin_deep

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RayGenerationLaunchInputs:
    """Serializable worker launch contract plus driver-side pure gatherer."""

    launch_contract: GenerationRuntimeLaunchContract
    gatherer: ChunkGatherer


@dataclass(slots=True)
class RayGenerationLauncher:
    """Create Ray generation actors and return a ``RayGenerationRuntime``."""

    init_ray: bool = True
    ray_init_kwargs: dict[str, Any] = field(default_factory=dict)

    def launch(
        self,
        config: RayGenerationConfig | Mapping[str, Any],
        launch_contract: GenerationRuntimeLaunchContract | Mapping[str, Any],
        gatherer: ChunkGatherer,
        *,
        placement: RolePlacement,
    ) -> RayGenerationRuntime:
        """Launch rollout workers and return a collector-facing runtime.

        Workers are scheduled into a run-level placement group owned by a
        ``GlobalRayPlacementOwner``. The launcher uses the rollout role's bundles,
        validates the workers against the role's expected GPUs, and never removes
        the group; the owner does that once at run shutdown.
        """

        rollout_config = RayGenerationConfig.from_cfg(config)

        contract = GenerationRuntimeLaunchContract.from_value(launch_contract)
        if not contract.family:
            raise ValueError("GenerationRuntimeLaunchContract.family is required")
        chunk_gatherer = _require_chunk_gatherer(gatherer)

        ray = require_ray()
        if self.init_ray and not ray.is_initialized():
            ray.init(**self.ray_init_kwargs)

        placement_group = placement.placement_group
        bundle_indices = list(placement.bundle_indices)
        owned_placement_group = None  # owned by GlobalRayPlacementOwner
        expected_gpu_ids = placement.expected_gpu_ids

        worker_ids = [f"rollout-{logical_idx}" for logical_idx in range(len(bundle_indices))]
        worker_configs = [contract for _ in bundle_indices]

        try:
            actor_group = RayActorGroup.launch(
                worker_cls=RayGenerationWorker,
                worker_configs=worker_configs,
                worker_ids=worker_ids,
                num_cpus=rollout_config.cpus_per_worker,
                num_gpus=rollout_config.gpus_per_worker,
                placement_group=placement_group,
                bundle_indices=bundle_indices,
                startup_method="load_policy",
            )
            metadata = [
                {
                    "worker_id": handle.worker_id,
                    "node_ip": handle.node_ip,
                    "gpu_ids": handle.gpu_ids,
                }
                for handle in actor_group.handles
            ]
            _validate_worker_gpu_ids(
                rollout_config,
                metadata,
                expected_gpu_ids=expected_gpu_ids,
            )
        except Exception:
            if "actor_group" in locals():
                actor_group.shutdown()
            # The launcher only created the workers; the placement group belongs
            # to the GlobalRayPlacementOwner.
            raise

        workers = [
            DistributedWorkerHandle(
                worker_id=handle.worker_id,
                node_id=handle.node_ip,
                gpu_ids=handle.gpu_ids,
                actor=handle.actor,
            )
            for handle in actor_group.handles
        ]

        executor = RayGenerationExecutor(
            DistributedExecutionPlanner(
                contract.extra.get("family_capability"),
                policy=ChunkPlacementPolicy(
                    strategy=rollout_config.chunk_placement_strategy,
                ),
            ),
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
            owned_actors=[],
            placement_group=owned_placement_group,
        )
        if contract.policy_version is not None:
            runtime.current_policy_version = contract.policy_version
        return runtime

    def launch_from_cfg(
        self,
        cfg: Any,
        *,
        driver_bundle: Any | None = None,
        driver_policy: Any | None = None,
        trainable_modules: Mapping[str, Any] | Iterable[Any] | None = None,
        launch_contract: Any | None = None,
        gatherer: Any | None = None,
        placement: RolePlacement | None = None,
    ) -> GenerationRuntime:
        """Build the Ray generation runtime from training config and launch inputs."""

        config = RayGenerationConfig.from_cfg(cfg).validate_driver_state(
            driver_bundle=driver_bundle,
            driver_policy=driver_policy,
            trainable_modules=trainable_modules,
        )
        resolved_contract = (
            launch_contract
            if launch_contract is not None
            else cfg_path(
                cfg,
                "distributed.rollout.launch_contract",
                None,
            )
        )
        if resolved_contract is not None and gatherer is not None:
            if placement is None:
                raise ValueError(
                    "RayGenerationLauncher.launch_from_cfg requires placement: "
                    "rollout workers schedule into the run-level placement group "
                    "built by a GlobalRayPlacementOwner.",
                )
            if config.release_after_collect:
                from vrl.generation.ray.runtime import ReleasableRayGenerationRuntime

                return ReleasableRayGenerationRuntime(
                    config,
                    resolved_contract,
                    gatherer,
                    placement=placement,
                )
            return self.launch(
                config,
                resolved_contract,
                gatherer,
                placement=placement,
            )

        raise ValueError(
            "Ray generation launch requires launch_contract plus gatherer so "
            "RayGenerationLauncher can construct workers through the "
            "runtime_builder+executor_cls path.",
        )

    @staticmethod
    def build_inputs(
        cfg: Any,
        entry: Any,
        *,
        weight_dtype: Any,
        executor_kwargs: Mapping[str, Any] | None = None,
        policy_version: int = 0,
    ) -> RayGenerationLaunchInputs:
        """Build Ray generation launch inputs from a resolved family entry."""

        ray_config = RayGenerationConfig.from_cfg(cfg)

        runtime_device = "cuda" if ray_config.gpus_per_worker > 0 else "cpu"
        runtime_build = _call_runtime_build_extractor(
            entry,
            cfg,
            runtime_device,
            dtype_to_config_string(weight_dtype),
        )
        resolved_executor_kwargs = _build_executor_kwargs(entry, cfg)
        resolved_executor_kwargs.update(dict(executor_kwargs or {}))
        runtime_extra = _runtime_extra(cfg)
        runtime_extra["family_capability"] = entry.capability.to_dict()
        runtime_build_payload = _runtime_build_payload(runtime_build)
        _apply_rollout_compile_override(runtime_build_payload, cfg, entry)

        return RayGenerationLaunchInputs(
            launch_contract=GenerationRuntimeLaunchContract(
                family=entry.family,
                task=entry.task,
                model_build=runtime_build_payload,
                executor_kwargs=resolved_executor_kwargs,
                policy_version=policy_version,
                runtime_builder=entry.runtime_builder,
                executor_cls=entry.executor_cls,
                extra=runtime_extra,
            ),
            gatherer=_build_gatherer(entry),
        )


def _require_chunk_gatherer(gatherer: Any) -> ChunkGatherer:
    gather_chunks = getattr(gatherer, "gather_chunks", None)
    if not callable(gather_chunks):
        raise TypeError(
            f"{type(gatherer).__name__} does not implement gather_chunks(...)",
        )
    return gatherer


def _validate_worker_gpu_ids(
    config: RayGenerationConfig,
    metadata: list[Mapping[str, Any]],
    *,
    expected_gpu_ids: tuple[int, ...] | None = None,
) -> None:
    resources = config.resources
    if resources is None or resources.rollout_gpus_per_worker <= 0:
        return

    driver_node_ip: str | None = None
    if resources.cross_node:
        try:
            driver_node_ip = current_node_ip()
        except Exception:
            driver_node_ip = None

    # The placement owner supplies the role's expected GPUs (empty under
    # cross-node, where the node-aware check applies instead).
    expected = resources.rollout_devices if expected_gpu_ids is None else expected_gpu_ids
    validate_actor_gpu_ids(
        metadata,
        expected_gpu_ids=expected,
        role="generation",
        cross_node=resources.cross_node,
        driver_node_ip=driver_node_ip,
    )


def _cross_node_preflight(ray: Any, resources: Any) -> None:
    """Fail fast when the live Ray cluster cannot host cross-node rollout.

    Runs after ``ray.init()`` (resolution earlier in the pipeline cannot see the
    cluster). Verifies that non-driver nodes expose enough GPUs for rollout, and
    that the driver/head node does not expose Ray GPUs — otherwise rollout actors
    could be scheduled onto the trainer GPU, since cross-node mode intentionally
    drops the placement-group trainer reservation.
    """

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

    needed = resources.rollout_num_gpus
    if non_driver_gpus < needed:
        raise RuntimeError(
            f"cross_node rollout needs {needed} GPU(s) on non-driver Ray nodes, but "
            f"only {non_driver_gpus:g} are available. Join more rollout workers, e.g. "
            "`ray start --address=<head>:6379 --num-gpus=<n>`.",
        )
    if driver_gpus > 0:
        raise RuntimeError(
            f"cross_node rollout: the driver/head node exposes {driver_gpus:g} Ray "
            "GPU(s), so rollout could be scheduled onto the trainer GPU. Start the "
            "head with `ray start --head --num-gpus=0` so the trainer GPU stays out "
            "of Ray's scheduling pool.",
        )


def _call_runtime_build_extractor(
    entry: Any,
    cfg: Any,
    device: str,
    weight_dtype: str,
) -> Any:
    extractor = import_from_path(entry.runtime_spec_extractor)
    return extractor(cfg, device, weight_dtype)


def _build_gatherer(entry: Any) -> ChunkGatherer:
    gatherer_cls = import_from_path(entry.gatherer.import_path)
    return gatherer_cls(**entry.gatherer.kwargs)


def _build_executor_kwargs(entry: Any, cfg: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    metadata = entry.executor_kwargs
    if metadata.include_sample_batch_size:
        sample_batch_size = cfg_path(cfg, "rollout.sample_batch_size", None)
        if sample_batch_size is not None:
            kwargs["sample_batch_size"] = int(sample_batch_size)
    if metadata.include_reference_image:
        reference_image = cfg_path(cfg, "model.reference_image", None)
        if reference_image:
            kwargs["reference_image"] = str(reference_image)
    return kwargs


def _runtime_build_payload(runtime_build: Any) -> dict[str, Any]:
    payload = asdict(runtime_build)
    payload["device"] = str(payload["device"])
    payload["dtype"] = dtype_to_config_string(payload["dtype"])
    if payload.get("frozen_dtype") is not None:
        payload["frozen_dtype"] = dtype_to_config_string(payload["frozen_dtype"])
    return payload


# Launcher-local input path; the written compile block is owned by the runtime
# spec via ``torch_compile_model_config`` (single source of key + shape).
_ROLLOUT_COMPILE_CFG_PATH = "rollout.denoise_compile"


def _apply_rollout_compile_override(payload: dict[str, Any], cfg: Any, entry: Any) -> None:
    """Map rollout-only compile config onto the worker runtime model config.

    A rollout family opts in through ``capability.supports_torch_compile``.
    """

    compile_cfg = cfg_path(cfg, _ROLLOUT_COMPILE_CFG_PATH, None)
    if compile_cfg is None:
        return
    compile_cfg = to_builtin_deep(compile_cfg)
    if not isinstance(compile_cfg, Mapping):
        raise TypeError(f"{_ROLLOUT_COMPILE_CFG_PATH} must be a mapping")
    if not compile_cfg.get("enable", False):
        return

    if not entry.capability.supports_torch_compile:
        raise ValueError(
            f"{entry.family} does not support torch compile but "
            f"{_ROLLOUT_COMPILE_CFG_PATH}.enable is set",
        )

    raw_model_config = payload.get("model_config") or {}
    if not isinstance(raw_model_config, Mapping):
        raise TypeError("runtime model_config must be a mapping")
    model_config = dict(raw_model_config)
    model_config.update(
        torch_compile_model_config(
            enable=True,
            mode=str(compile_cfg.get("mode", "default")),
        )
    )
    payload["model_config"] = model_config


def _runtime_extra(cfg: Any) -> dict[str, Any]:
    profiler_cfg = cfg_path(cfg, "rollout.torch_profiler", None)
    if profiler_cfg is None:
        return {}
    profiler = to_builtin_deep(profiler_cfg)
    if not isinstance(profiler, dict):
        return {}
    return {
        "torch_profiler": profiler,
        # trainer.output_dir is required (??? in base yaml, enforced at load
        # time) — no silent fallback that would shadow the contract.
        "profiler_output_dir": str(cfg_path(cfg, "trainer.output_dir")),
    }


__all__ = [
    "RayGenerationLaunchInputs",
    "RayGenerationLauncher",
]
