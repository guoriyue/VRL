"""Launch Ray generation workers and assemble the collector-facing runtime."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import replace
from functools import partial
from typing import Any

from vrl.generation.execution.batch_placement import DistributedExecutionPlanner
from vrl.generation.ray.config import RayGenerationConfig
from vrl.generation.ray.engine import RayGenerationEngine, rank_handles
from vrl.generation.ray.executor import RayGenerationExecutor
from vrl.generation.ray.launch_inputs import RayGenerationLaunchInputs
from vrl.generation.ray.runtime import RayGenerationRuntime
from vrl.generation.ray.session import RayGenerationSession
from vrl.generation.ray.weight_sync import RayGenerationWeightSync
from vrl.generation.ray.worker import HEALTH_CONCURRENCY_GROUP, RayGenerationWorker
from vrl.ray.actor_group import RayActorGroup, RayActorHandle
from vrl.ray.actor_pool import RayActorDispatcher
from vrl.ray.dependencies import current_node_ip, require_ray
from vrl.ray.operation_deadline import get_ray_refs
from vrl.ray.placement import RolePlacement, validate_actor_gpu_ids
from vrl.ray.resources import ROLLOUT_GPUS_PER_ENGINE

logger = logging.getLogger(__name__)


class RayGenerationLauncher:
    """Create Ray generation actors and return a ``RayGenerationRuntime``."""

    def __init__(
        self,
        *,
        init_ray: bool = True,
        ray_init_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.init_ray = bool(init_ray)
        # Standalone launcher use must be ownership-safe too. Online recipes
        # initialize explicitly; callers that intend to attach can override it.
        self.ray_init_kwargs = (
            {"address": "local"} if ray_init_kwargs is None else dict(ray_init_kwargs)
        )

    def _launch_session(
        self,
        config: RayGenerationConfig,
        launch_inputs: RayGenerationLaunchInputs,
        *,
        placement: RolePlacement,
    ) -> RayGenerationSession:
        """Launch one actor fleet after its GPU owner has yielded.

        Workers are scheduled into a run-level placement group owned by a
        ``GlobalRayPlacementOwner``. The launcher uses the rollout role's bundles,
        validates the workers against the role's expected GPUs, and never removes
        the group; the owner does that once at run shutdown.
        """

        worker = config.worker

        bundle_indices = list(placement.bundle_indices)
        # One engine per ROLLOUT_GPUS_PER_ENGINE bundles; the resolver's
        # num_engines arithmetic guarantees divisibility (one GPU per engine
        # today, so groups have size 1 and rank ids equal engine ids).
        if len(bundle_indices) % ROLLOUT_GPUS_PER_ENGINE:
            raise ValueError(
                f"rollout placement bundles ({len(bundle_indices)}) are not "
                f"divisible into engines of {ROLLOUT_GPUS_PER_ENGINE} rank(s)",
            )
        engine_count = len(bundle_indices) // ROLLOUT_GPUS_PER_ENGINE
        if worker.pipelined and engine_count != 1:
            raise ValueError(
                "pipelined Ray generation requires exactly one rollout engine; "
                f"received {engine_count}. Per-engine request "
                "pipelining is not implemented.",
            )
        ray = require_ray()
        if self.init_ray and not ray.is_initialized():
            ray.init(**self.ray_init_kwargs)

        placement_group = placement.placement_group
        expected_gpu_ids = placement.expected_gpu_ids

        engine_ids = [f"rollout-{engine_idx}" for engine_idx in range(engine_count)]
        rank_ids = [
            engine_id if ROLLOUT_GPUS_PER_ENGINE == 1 else f"{engine_id}.r{rank_idx}"
            for engine_id in engine_ids
            for rank_idx in range(ROLLOUT_GPUS_PER_ENGINE)
        ]
        # The same registry-owned gatherer serves the driver executor and the
        # single-engine pipelined path. Ray serializes it to each rank actor;
        # ranks never re-resolve family identity from the neutral execution layer.
        rank_configs = [launch_inputs for _ in rank_ids]
        try:
            actor_group = RayActorGroup.launch(
                worker_cls=RayGenerationWorker,
                worker_configs=rank_configs,
                worker_ids=rank_ids,
                num_cpus=worker.cpus_per_worker,
                # Each GPU rank owns one whole GPU; a CPU fleet grants none.
                num_gpus=1 if config.resources.rollout_devices else 0,
                rpc_timeout_s=worker.worker_rpc_timeout_s,
                operation_prefix="rollout",
                placement_group=placement_group,
                bundle_indices=bundle_indices,
                startup_method="load_policy",
                # One dedicated thread so liveness probes never queue behind
                # generation; the default group keeps its serialization.
                concurrency_groups={HEALTH_CONCURRENCY_GROUP: 1},
            )
            _validate_rank_gpu_ids(
                config,
                actor_group.handles,
                expected_gpu_ids=expected_gpu_ids,
            )
            engines = [
                RayGenerationEngine(
                    engine_id,
                    actor_group.handles[
                        engine_idx * ROLLOUT_GPUS_PER_ENGINE : (engine_idx + 1)
                        * ROLLOUT_GPUS_PER_ENGINE
                    ],
                )
                for engine_idx, engine_id in enumerate(engine_ids)
            ]
            actor_dispatcher = RayActorDispatcher(
                tuple(engine.engine_id for engine in engines),
            )

            executor = RayGenerationExecutor(
                DistributedExecutionPlanner(
                    strategy=worker.batch_placement_strategy,
                ),
                engines,
                launch_inputs.gatherer,
                actor_dispatcher=actor_dispatcher,
                generation_stall_timeout_s=worker.generation_stall_timeout_s,
                pipelined=worker.pipelined,
            )
            weight_sync = (
                RayGenerationWeightSync(
                    engines,
                    actor_dispatcher=actor_dispatcher,
                    worker_rpc_timeout_s=worker.worker_rpc_timeout_s,
                )
                if worker.sync_trainable_state
                else None
            )
            supports_non_draining_weight_sync = _all_ranks_support_versioned_slots(
                ray,
                rank_handles(engines),
                weight_sync=weight_sync,
                worker_rpc_timeout_s=worker.worker_rpc_timeout_s,
            )
            return RayGenerationSession(
                executor,
                weight_sync=weight_sync,
                owned_engines=engines,
                supports_non_draining_weight_sync=supports_non_draining_weight_sync,
            )
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

    async def _launch_session_async(
        self,
        config: RayGenerationConfig,
        launch_inputs: RayGenerationLaunchInputs,
        *,
        placement: RolePlacement,
    ) -> RayGenerationSession:
        """Launch a session without blocking the runtime's lifecycle event loop.

        Ray initialization stays on the caller thread because it owns process
        signal setup. Once connected, actor startup and policy load can run in
        a worker thread; the runtime's shielded activation task remains the
        ownership boundary if the external waiter is cancelled.
        """

        ray = require_ray()
        if self.init_ray and not ray.is_initialized():
            ray.init(**self.ray_init_kwargs)
        return await asyncio.to_thread(
            self._launch_session,
            config,
            launch_inputs,
            placement=placement,
        )

    def create_runtime(
        self,
        config: RayGenerationConfig,
        launch_inputs: RayGenerationLaunchInputs,
        *,
        placement: RolePlacement | None,
    ) -> RayGenerationRuntime:
        """Create the sole Ray runtime with eager or deferred session ownership."""

        if placement is None:
            # ``GlobalRayPlacementOwner.rollout_placement`` yields None when the
            # resolved topology assigned no rollout bundles; fail with the config
            # knob instead of an AttributeError deep inside the session launch.
            raise ValueError(
                "Ray generation launch requires a rollout placement, but the "
                "run-level placement owner resolved no rollout bundles. Check "
                "distributed.resources.rollout.* (num_gpus/devices) and that the "
                "placement group was created before launch.",
            )
        if not isinstance(config, RayGenerationConfig):
            raise TypeError(
                f"config must be a RayGenerationConfig, got {type(config).__name__}",
            )
        if not isinstance(launch_inputs, RayGenerationLaunchInputs):
            raise TypeError(
                "launch_inputs must be RayGenerationLaunchInputs, "
                f"got {type(launch_inputs).__name__}",
            )
        resources = config.resources
        worker = config.worker
        deferred = resources.lifecycle.rollout_mode == "on_demand"
        if deferred and resources.rollout_devices:
            launch_inputs = replace(
                launch_inputs,
                launch_contract=replace(
                    launch_inputs.launch_contract,
                    sleep_offload=True,
                ),
            )

        session = None
        session_factory = None
        if deferred:
            session_factory = partial(
                self._launch_session_async,
                config,
                launch_inputs,
                placement=placement,
            )
        else:
            session = self._launch_session(
                config,
                launch_inputs,
                placement=placement,
            )

        try:
            runtime = RayGenerationRuntime(
                session=session,
                session_factory=session_factory,
                initial_policy_version=launch_inputs.launch_contract.policy_version,
                supports_weight_sync=worker.sync_trainable_state,
                colocated=resources.colocated,
                health_check_interval_s=worker.health_check_interval_s,
                health_check_timeout_s=worker.health_check_timeout_s,
                health_check_first_wait_s=worker.health_check_first_wait_s,
            )
            runtime.start_health_monitoring()
            return runtime
        except BaseException as error:
            if session is not None:
                session.force_close()
                try:
                    session.kill_workers()
                except BaseException as cleanup_error:
                    error.add_note(
                        f"resident rollout startup cleanup also failed: {cleanup_error!r}",
                    )
            raise


def _validate_rank_gpu_ids(
    config: RayGenerationConfig,
    metadata: Sequence[RayActorHandle],
    *,
    expected_gpu_ids: tuple[int, ...] | None = None,
) -> None:
    """Validate launched rank actors against the resolved rollout placement."""

    resources = config.resources
    if not resources.rollout_devices:
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


def _all_ranks_support_versioned_slots(
    ray: Any,
    ranks: Sequence[RayActorHandle],
    *,
    weight_sync: Any | None,
    worker_rpc_timeout_s: float,
) -> bool:
    """Return whether every rank supports versioned trainable-state slots.

    Non-draining weight sync needs slots on all ranks because a batch stamped
    with an older policy version can be placed on any engine. A missing weight
    syncer or an empty fleet keeps the safe draining barrier. A query
    failure means the candidate fleet is broken, not merely unsupported, and
    therefore propagates to launcher-owned actor cleanup.
    """

    if weight_sync is None:
        return False
    actors = [rank.actor for rank in ranks]
    if not actors:
        return False
    results = get_ray_refs(
        ray,
        [actor.supports_versioned_trainable_state.remote() for actor in actors],
        operation="rollout.startup.versioned_slots",
        timeout_s=worker_rpc_timeout_s,
        context=f"ranks={len(actors)}",
    )
    return bool(results) and all(bool(result) for result in results)


__all__ = [
    "RayGenerationLauncher",
]
