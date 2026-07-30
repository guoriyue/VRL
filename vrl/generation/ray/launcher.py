"""Launch Ray generation workers and assemble the collector-facing runtime."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Any

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
from vrl.generation.ray.weight_sync import RayGenerationWeightSync
from vrl.generation.ray.worker import HEALTH_CONCURRENCY_GROUP, RayGenerationWorker
from vrl.models.dtypes import dtype_to_wire_name
from vrl.ray.actor_group import RayActorGroup
from vrl.ray.dependencies import current_node_ip, require_ray
from vrl.ray.operation_deadline import RayOperationTimeout, get_ray_refs
from vrl.ray.placement import RolePlacement, validate_actor_gpu_ids
from vrl.utils.config import cfg_path, plain_mapping, to_builtin_deep

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vrl.config.precision import PrecisionPolicy
    from vrl.config.schema import RootConfig


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
        worker = rollout_config.worker
        contract = launch_inputs.launch_contract
        chunk_gatherer = _require_chunk_gatherer(launch_inputs.gatherer)

        bundle_indices = list(placement.bundle_indices)
        if worker.pipelined and len(bundle_indices) != 1:
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
        # The same registry-owned gatherer serves the driver executor and the
        # single-worker pipelined path. Ray serializes it to each actor; workers
        # never re-resolve family identity from the neutral execution layer.
        worker_configs = [launch_inputs for _ in bundle_indices]
        try:
            actor_group = RayActorGroup.launch(
                worker_cls=RayGenerationWorker,
                worker_configs=worker_configs,
                worker_ids=worker_ids,
                num_cpus=worker.cpus_per_worker,
                num_gpus=rollout_config.resources.rollout_gpus_per_worker,
                worker_rpc_timeout_s=worker.worker_rpc_timeout_s,
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
            _validate_worker_gpu_ids(
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
                        strategy=worker.chunk_placement_strategy,
                    ),
                ),
                workers,
                chunk_gatherer,
                max_inflight_chunks_per_worker=worker.max_inflight_chunks_per_worker,
                generation_stall_timeout_s=worker.generation_stall_timeout_s,
                pipelined=worker.pipelined,
            )
            weight_sync = (
                RayGenerationWeightSync(
                    workers,
                    worker_rpc_timeout_s=worker.worker_rpc_timeout_s,
                )
                if worker.sync_trainable_state
                else None
            )
            runtime = RayGenerationRuntime(
                executor,
                weight_sync=weight_sync,
                owned_workers=workers,
                colocated=rollout_config.resources.colocated,
                health_check_interval_s=worker.health_check_interval_s,
                health_check_timeout_s=worker.health_check_timeout_s,
                health_check_first_wait_s=worker.health_check_first_wait_s,
            )
            if contract.policy_version is not None:
                runtime.current_policy_version = contract.policy_version
            # Non-draining weight sync is safe only when EVERY worker retains
            # versioned trainable-state slots (a chunk stamped v1 may land on any
            # worker). Query the AND before publishing the runtime candidate.
            runtime.supports_non_draining_weight_sync = _all_workers_support_versioned_slots(
                ray,
                workers,
                weight_sync=weight_sync,
                worker_rpc_timeout_s=worker.worker_rpc_timeout_s,
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
        root: RootConfig,
        *,
        precision: PrecisionPolicy,
        config: RayGenerationConfig,
        entry: Any,
        driver_bundle: Any,
        expected_model_identity: dict[str, Any],
        placement: RolePlacement,
    ) -> GenerationRuntime:
        """Build launch inputs from config and launch one resolved Ray runtime."""

        config.validate_driver_state(
            driver_bundle=driver_bundle,
        )
        runtime_device = "cuda" if config.resources.rollout_gpus_per_worker > 0 else "cpu"
        build = entry.resolve_model_build(
            root,
            runtime_device,
            precision=precision,
        )
        # The lifecycle resolver, not the model config, owns whether rollout
        # workers will receive trainable-state updates. Thread that resolved fact
        # into the one build option that needs it before the Ray payload is frozen.
        if build.rollout is not None:
            build.rollout = replace(
                build.rollout,
                # Full-finetune sync replaces base parameters; LoRA sync only
                # sends adapters. Resolve that lifecycle fact once here so the
                # quantizer does not have to reinterpret model configuration.
                base_weight_sync=(config.worker.sync_trainable_state and not build.use_lora),
            )
        if bool(cfg_path(root, "model.torch_compile.enable", False)) and not (
            entry.runtime_capabilities.supports_torch_compile
        ):
            raise ValueError(
                f"{entry.family} does not support torch compile but "
                "model.torch_compile.enable is set",
            )
        from vrl.models.checkpoint_identity import resolve_checkpoint_model_identity

        rollout_model_identity = resolve_checkpoint_model_identity(build)
        if rollout_model_identity != expected_model_identity:
            raise ValueError(
                "rollout model identity does not match the driver replay model "
                "identity before Ray worker launch: "
                f"replay={expected_model_identity!r}, rollout={rollout_model_identity!r}",
            )
        schedule_mode = str(
            cfg_path(
                root,
                "trainer.rollout_orchestration.schedule_mode",
                "strict_on_policy",
            ),
        )
        launch_inputs = RayGenerationLaunchInputs(
            launch_contract=GenerationRuntimeLaunchContract(
                family=entry.family,
                model_build=_model_build_payload(build),
                expected_model_identity=expected_model_identity,
                executor_kwargs=build_executor_kwargs(entry, root),
                policy_version=0,
                torch_profiler=_runtime_profiler(root),
                # The schedule is the source of truth for whether a worker may
                # serve an older request after a weight sync. Full-parameter
                # payloads fail closed to the draining path until a byte-budget
                # gate proves their retained slots fit in host RAM.
                versioned_weight_sync=(
                    config.worker.sync_trainable_state
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

    executor_config = cfg_path(cfg, "model.executor", None)
    entry.validate_model_runtime_sections(
        executor_config=executor_config,
        memory_config=cfg_path(cfg, "model.memory", None),
    )

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
    if (
        entry.executor_cls == GENERIC_FULL_SEQUENCE_DENOISE_EXECUTOR
        and executor_config is not None
    ):
        kwargs.update(
            plain_mapping(
                executor_config,
                field_name="model.executor",
            ),
        )
    return kwargs


def _require_chunk_gatherer(gatherer: Any) -> ChunkGatherer:
    """Require the collector-facing chunk-gatherer protocol.

    ``gatherer`` is ``Any`` because the registry builds it from an unvalidated
    dotted-string import; this is the boundary that turns it into a protocol.
    """

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
    """Validate launched workers against the resolved rollout placement."""

    resources = config.resources
    if resources.rollout_gpus_per_worker <= 0:
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


def _all_workers_support_versioned_slots(
    ray: Any,
    workers: list[DistributedWorkerHandle],
    *,
    weight_sync: Any | None,
    worker_rpc_timeout_s: float,
) -> bool:
    """Return whether every worker supports versioned trainable-state slots.

    Non-draining weight sync needs slots on all workers because a chunk stamped
    with an older policy version can be placed on any worker. A missing weight
    syncer, an empty worker set, or any failed capability query keeps the safe
    draining barrier.
    """

    if weight_sync is None:
        return False
    actors = [worker.actor for worker in workers if worker.actor is not None]
    if not actors or len(actors) != len(workers):
        return False
    try:
        results = get_ray_refs(
            ray,
            [actor.supports_versioned_trainable_state.remote() for actor in actors],
            operation="rollout.startup.versioned_slots",
            timeout_s=worker_rpc_timeout_s,
            context=f"workers={len(actors)}",
        )
    except RayOperationTimeout:
        raise
    except Exception:
        return False
    return bool(results) and all(bool(result) for result in results)


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
        # Absence is the wire representation of the universal no-offload
        # default. Only a selected pipeline residency mode needs to cross Ray.
        if rollout.get("pipeline_offload_mode") == "none":
            rollout.pop("pipeline_offload_mode")
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
