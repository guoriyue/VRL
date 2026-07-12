"""Launch Ray generation workers and assemble the collector-facing runtime."""

from __future__ import annotations

import asyncio
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
from vrl.generation.ray.launch_inputs import RayGenerationLaunchInputs
from vrl.generation.ray.runtime import RayGenerationRuntime
from vrl.generation.ray.utils import (
    all_workers_support_versioned_slots,
    require_chunk_gatherer,
    validate_worker_gpu_ids,
)
from vrl.generation.ray.weight_sync import RayGenerationWeightSync
from vrl.generation.ray.worker import RayGenerationWorker
from vrl.models.dtypes import dtype_to_config_string
from vrl.ray.actor_group import RayActorGroup
from vrl.ray.dependencies import require_ray
from vrl.ray.placement import RolePlacement
from vrl.utils.config import cfg_path, import_from_path, to_builtin_deep

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RayGenerationLauncher:
    """Create Ray generation actors and return a ``RayGenerationRuntime``."""

    init_ray: bool = True
    # Standalone launcher use must be ownership-safe too. Online recipes already
    # initialize explicitly; callers that intend to attach can override address.
    ray_init_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"address": "local"},
    )

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
        chunk_gatherer = require_chunk_gatherer(gatherer)

        ray = require_ray()
        if self.init_ray and not ray.is_initialized():
            ray.init(**self.ray_init_kwargs)

        placement_group = placement.placement_group
        bundle_indices = list(placement.bundle_indices)
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
            validate_worker_gpu_ids(
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
                actor=handle.actor,
            )
            for handle in actor_group.handles
        ]

        executor = RayGenerationExecutor(
            DistributedExecutionPlanner(
                policy=ChunkPlacementPolicy(
                    strategy=rollout_config.chunk_placement_strategy,
                ),
            ),
            workers,
            chunk_gatherer,
            max_inflight_chunks_per_worker=rollout_config.max_inflight_chunks_per_worker,
            pipelined=rollout_config.pipelined,
        )
        weight_sync = (
            RayGenerationWeightSync(workers) if rollout_config.sync_trainable_state else None
        )
        runtime = RayGenerationRuntime(
            executor,
            weight_sync=weight_sync,
            owned_workers=workers,
            colocated=rollout_config.allow_driver_gpu_overlap,
        )
        if contract.policy_version is not None:
            runtime.current_policy_version = contract.policy_version
        # Non-draining weight sync is safe only when EVERY worker retains versioned
        # trainable-state slots (a chunk stamped v1 may land on any worker). Workers
        # already loaded their model (startup_method="load_policy"), so query the
        # AND once here; absence/error keeps the safe draining barrier.
        runtime.supports_non_draining_weight_sync = all_workers_support_versioned_slots(
            ray,
            workers,
            weight_sync=weight_sync,
        )
        return runtime

    async def launch_async(
        self,
        config: RayGenerationConfig | Mapping[str, Any],
        launch_contract: GenerationRuntimeLaunchContract | Mapping[str, Any],
        gatherer: ChunkGatherer,
        *,
        placement: RolePlacement,
    ) -> RayGenerationRuntime:
        """Launch without blocking the runtime's lifecycle event loop.

        Ray initialization stays on the caller thread because it owns process
        signal setup. Once connected, actor startup and policy load can run in
        a worker thread; the runtime's shielded activation task remains the
        ownership boundary if the external waiter is cancelled.
        """

        ray = require_ray()
        if self.init_ray and not ray.is_initialized():
            ray.init(**self.ray_init_kwargs)
        return await asyncio.to_thread(
            self.launch,
            config,
            launch_contract,
            gatherer,
            placement=placement,
        )

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
            # On-demand vs resident comes from the topology-derived lifecycle
            # plan (resources.lifecycle), the single source of truth. Without a
            # resolved plan (hand-built configs in tests) default to resident.
            resources = config.resources
            rollout_on_demand = (
                resources is not None and resources.lifecycle.rollout.mode == "on_demand"
            )
            if rollout_on_demand:
                return RayGenerationRuntime.with_on_demand_activation(
                    config,
                    RayGenerationLaunchInputs(
                        launch_contract=resolved_contract,
                        gatherer=gatherer,
                    ),
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
        resources = ray_config.resources
        if resources is not None and resources.rollout_gpu_memory_fraction is not None:
            # Worker-side allocator cap for colocated rollout (applied in load_policy).
            runtime_extra["gpu_memory_fraction"] = resources.rollout_gpu_memory_fraction
        _validate_model_compile_supported(cfg, entry)
        runtime_build_payload = _runtime_build_payload(runtime_build)

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
    from vrl.generation.diffusion.executor import GENERIC_DIFFUSION_EXECUTOR

    kwargs: dict[str, Any] = {}
    # Which executor kwargs to thread is DERIVED from the family capability,
    # not declared on the entry: diffusion executors take a chunk batch size
    # (AR ones don't), and reference-conditioned executors take a reference
    # image. One source (entry.capability) drives both.
    if entry.capability.trajectory_kind == "diffusion":
        samples_per_chunk = cfg_path(cfg, "rollout.samples_per_chunk", None)
        # ``auto`` belongs to the request and is resolved by RayGenerationRuntime
        # before dispatch. Do not feed it to the executor constructor, whose
        # fixed fallback accepts only an integer.
        if samples_per_chunk is not None and samples_per_chunk != "auto":
            kwargs["samples_per_chunk"] = int(samples_per_chunk)
    if entry.capability.supports_reference_conditioning:
        reference_image = cfg_path(cfg, "model.reference_image", None)
        if reference_image:
            kwargs["reference_image"] = str(reference_image)
    # Families on the shared executor read their whole executor config block
    # from yaml in ONE pass (config is homogeneous — no per-field extraction);
    # family/task identity comes from the registry entry. Families with their
    # own executor hardcode these as class attrs and skip this.
    if entry.executor_cls == GENERIC_DIFFUSION_EXECUTOR:
        kwargs.update(dict(cfg_path(cfg, "model.executor", {}) or {}))
        kwargs["family"] = entry.family
        kwargs["task"] = entry.task
    return kwargs


def _runtime_build_payload(runtime_build: Any) -> dict[str, Any]:
    payload = asdict(runtime_build)
    payload["device"] = str(payload["device"])
    payload["dtype"] = dtype_to_config_string(payload["dtype"])
    if payload.get("frozen_dtype") is not None:
        payload["frozen_dtype"] = dtype_to_config_string(payload["frozen_dtype"])
    return payload


def _validate_model_compile_supported(cfg: Any, entry: Any) -> None:
    """Fail fast when the single public compile knob is set for unsupported families."""

    if not bool(cfg_path(cfg, "model.torch_compile.enable", False)):
        return
    if not entry.capability.supports_torch_compile:
        raise ValueError(
            f"{entry.family} does not support torch compile but model.torch_compile.enable is set",
        )


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
