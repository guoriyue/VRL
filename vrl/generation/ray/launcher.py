"""Launch Ray generation workers and assemble the collector-facing runtime."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from vrl.generation.execution import (
    ChunkPlacementPolicy,
    DistributedExecutionPlanner,
    DistributedWorkerHandle,
)
from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.generation.protocols import GenerationRuntime
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
from vrl.generation.ray.worker import HEALTH_CONCURRENCY_GROUP, RayGenerationWorker
from vrl.models.dtypes import dtype_to_wire_name
from vrl.ray.actor_group import RayActorGroup
from vrl.ray.dependencies import require_ray
from vrl.ray.placement import RolePlacement
from vrl.utils.config import cfg_path, to_builtin_deep

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
        config: RayGenerationConfig,
        launch_inputs: RayGenerationLaunchInputs,
        *,
        placement: RolePlacement,
    ) -> RayGenerationRuntime:
        """Launch rollout workers and return a collector-facing runtime.

        Workers are scheduled into a run-level placement group owned by a
        ``GlobalRayPlacementOwner``. The launcher uses the rollout role's bundles,
        validates the workers against the role's expected GPUs, and never removes
        the group; the owner does that once at run shutdown.
        """

        if not isinstance(config, RayGenerationConfig):
            raise TypeError(
                f"config must be a RayGenerationConfig, got {type(config).__name__}",
            )
        if not isinstance(launch_inputs, RayGenerationLaunchInputs):
            raise TypeError(
                "launch_inputs must be RayGenerationLaunchInputs, "
                f"got {type(launch_inputs).__name__}",
            )
        rollout_config = config
        contract = launch_inputs.launch_contract
        chunk_gatherer = require_chunk_gatherer(launch_inputs.gatherer)

        bundle_indices = list(placement.bundle_indices)
        if rollout_config.pipelined and len(bundle_indices) != 1:
            raise ValueError(
                "pipelined Ray generation requires exactly one rollout placement "
                f"bundle; received {len(bundle_indices)}. Per-worker request "
                "pipelining is not implemented.",
            )
        ray = require_ray()
        if self.init_ray and not ray.is_initialized():
            ray.init(**self.ray_init_kwargs)

        placement_group = placement.placement_group
        expected_gpu_ids = placement.expected_gpu_ids

        worker_ids = [f"rollout-{logical_idx}" for logical_idx in range(len(bundle_indices))]
        worker_configs = [contract for _ in bundle_indices]
        try:
            actor_group = RayActorGroup.launch(
                worker_cls=RayGenerationWorker,
                worker_configs=worker_configs,
                worker_ids=worker_ids,
                num_cpus=rollout_config.cpus_per_worker,
                num_gpus=rollout_config.resources.rollout_gpus_per_worker,
                placement_group=placement_group,
                bundle_indices=bundle_indices,
                startup_method="load_policy",
                # One dedicated thread so liveness probes never queue behind
                # generation; the default group keeps its serialization.
                concurrency_groups={HEALTH_CONCURRENCY_GROUP: 1},
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
                max_inflight_chunks_per_worker=(rollout_config.max_inflight_chunks_per_worker),
                pipelined=rollout_config.pipelined,
            )
            weight_sync = (
                RayGenerationWeightSync(workers) if rollout_config.sync_trainable_state else None
            )
            runtime = RayGenerationRuntime(
                executor,
                weight_sync=weight_sync,
                owned_workers=workers,
                colocated=rollout_config.resources.colocated,
                health_check_interval_s=rollout_config.health_check_interval_s,
                health_check_timeout_s=rollout_config.health_check_timeout_s,
                health_check_first_wait_s=rollout_config.health_check_first_wait_s,
            )
            if contract.policy_version is not None:
                runtime.current_policy_version = contract.policy_version
            # Non-draining weight sync is safe only when EVERY worker retains
            # versioned trainable-state slots (a chunk stamped v1 may land on any
            # worker). Query the AND before publishing the runtime candidate.
            runtime.supports_non_draining_weight_sync = all_workers_support_versioned_slots(
                ray,
                workers,
                weight_sync=weight_sync,
            )
            runtime.start_health_monitoring()
            return runtime
        except BaseException as error:
            if "actor_group" in locals():
                try:
                    actor_group.shutdown()
                except BaseException as cleanup_error:
                    error.add_note(
                        f"rollout startup actor cleanup also failed: {cleanup_error!r}",
                    )
            # The launcher only created the workers; the placement group belongs
            # to the GlobalRayPlacementOwner.
            raise

    async def launch_async(
        self,
        config: RayGenerationConfig,
        launch_inputs: RayGenerationLaunchInputs,
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
            launch_inputs,
            placement=placement,
        )

    def launch_from_cfg(
        self,
        cfg: Any,
        *,
        resources: Any,
        entry: Any,
        driver_bundle: Any,
        placement: RolePlacement,
    ) -> GenerationRuntime:
        """Build the Ray generation runtime from training config and launch inputs."""

        config = RayGenerationConfig.from_cfg(
            cfg,
            resources=resources,
        ).validate_driver_state(
            driver_bundle=driver_bundle,
        )
        runtime_device = "cuda" if config.resources.rollout_gpus_per_worker > 0 else "cpu"
        build = entry.resolve_model_build(cfg, runtime_device)
        # The lifecycle resolver, not the model config, owns whether rollout
        # workers will receive trainable-state updates. Thread that resolved fact
        # into the one build option that needs it before the Ray payload is frozen.
        if build.rollout is not None:
            build.rollout = replace(
                build.rollout,
                # Full-finetune sync replaces base parameters; LoRA sync only
                # sends adapters. Resolve that lifecycle fact once here so the
                # quantizer does not have to reinterpret model configuration.
                base_weight_sync=(config.sync_trainable_state and not build.use_lora),
            )
        if bool(cfg_path(cfg, "model.torch_compile.enable", False)) and not (
            entry.runtime_capabilities.supports_torch_compile
        ):
            raise ValueError(
                f"{entry.family} does not support torch compile but "
                "model.torch_compile.enable is set",
            )
        schedule_mode = str(
            cfg_path(
                cfg,
                "trainer.rollout_orchestration.schedule_mode",
                "strict_on_policy",
            ),
        )
        launch_inputs = RayGenerationLaunchInputs(
            launch_contract=GenerationRuntimeLaunchContract(
                family=entry.family,
                model_build=_model_build_payload(build),
                executor_kwargs=build_executor_kwargs(entry, cfg),
                policy_version=0,
                torch_profiler=_runtime_profiler(cfg),
                # The schedule is the source of truth for whether a worker may
                # serve an older request after a weight sync. Full-parameter
                # payloads fail closed to the draining path until a byte-budget
                # gate proves their retained slots fit in host RAM.
                versioned_weight_sync=(
                    config.sync_trainable_state
                    and schedule_mode == "continuous"
                    and build.use_lora
                ),
            ),
            gatherer=entry.new_gatherer(),
        )
        # On-demand vs resident comes from the topology-derived lifecycle plan
        # (resources.lifecycle), the single source of truth. Hand-built configs
        # without a resolved plan default to resident.
        resources = config.resources
        rollout_on_demand = resources.lifecycle.rollout.mode == "on_demand"
        if rollout_on_demand:
            return RayGenerationRuntime.with_on_demand_activation(
                config,
                launch_inputs,
                placement=placement,
            )
        return self.launch(
            config,
            launch_inputs,
            placement=placement,
        )


def build_executor_kwargs(entry: Any, cfg: Any) -> dict[str, Any]:
    """Project one experiment config into registered executor constructor kwargs.

    The Ray launcher and test-owned rollout preview share this boundary so
    family executor settings are interpreted in exactly one place.
    """
    from vrl.families.registry import GENERIC_FULL_SEQUENCE_DENOISE_EXECUTOR

    kwargs: dict[str, Any] = {}
    # Only executors that publish this constructor capability receive the
    # request-chunk size; generation regime does not determine their API.
    if entry.runtime_capabilities.accepts_samples_per_chunk:
        samples_per_chunk = cfg_path(cfg, "rollout.samples_per_chunk", None)
        # ``auto`` belongs to the request and is resolved by RayGenerationRuntime
        # before dispatch. Do not feed it to the executor constructor, whose
        # fixed fallback accepts only an integer.
        if samples_per_chunk is not None and samples_per_chunk != "auto":
            kwargs["samples_per_chunk"] = int(samples_per_chunk)
    # Families on the shared executor read their whole executor config block
    # from yaml in ONE pass (config is homogeneous — no per-field extraction);
    # family/task identity comes from the registry entry. Families with their
    # own executor hardcode these as class attrs and skip this.
    if entry.executor_cls == GENERIC_FULL_SEQUENCE_DENOISE_EXECUTOR:
        kwargs.update(dict(cfg_path(cfg, "model.executor", {}) or {}))
    return kwargs


def _model_build_payload(build: Any) -> dict[str, Any]:
    payload = asdict(build)
    # Family is the launch contract identity and is restored worker-side. Do not
    # serialize it again inside the nested model-build payload.
    payload.pop("family", None)
    payload["device"] = str(payload["device"])
    payload["parameter_dtype"] = dtype_to_wire_name(payload["parameter_dtype"])
    rollout = payload.get("rollout")
    if rollout is not None:
        rollout["prompt_encoder_dtype"] = dtype_to_wire_name(
            rollout["prompt_encoder_dtype"],
        )
    return payload


def _runtime_profiler(cfg: Any) -> dict[str, Any]:
    profiler_cfg = cfg_path(cfg, "rollout.torch_profiler", None)
    if profiler_cfg is None:
        return {}
    profiler = to_builtin_deep(profiler_cfg)
    if not isinstance(profiler, dict):
        return {}
    # trainer.output_dir is required (??? in base yaml, enforced at load time),
    # so profiler output has no independent fallback or duplicate wire key.
    profiler["output_dir"] = str(cfg_path(cfg, "trainer.output_dir"))
    return profiler


__all__ = [
    "RayGenerationLaunchInputs",
    "RayGenerationLauncher",
    "build_executor_kwargs",
]
