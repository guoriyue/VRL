"""Launch Ray generation workers and assemble the collector-facing runtime."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from functools import partial

from vrl.generation.ray.config import RolloutWorkerConfig
from vrl.generation.ray.engine import RayGenerationEngine
from vrl.generation.ray.executor import RayGenerationExecutor
from vrl.generation.ray.finalizer import RayGenerationFinalizer
from vrl.generation.ray.launch_inputs import RayGenerationLaunchInputs
from vrl.generation.ray.runtime import RayGenerationRuntime
from vrl.generation.ray.session import RayGenerationSession
from vrl.generation.ray.worker import RayGenerationWorker
from vrl.ray.actor_group import RayActorGroup, RayActorHandle
from vrl.ray.actor_pool import RayActorDispatcher
from vrl.ray.dependencies import current_node_ip
from vrl.ray.placement import RolePlacement, require_actor_gpu_ids
from vrl.ray.resources import ResourcePlan

logger = logging.getLogger(__name__)


class RayGenerationLauncher:
    """Create Ray generation actors and return a ``RayGenerationRuntime``.

    Callers connect Ray before launching: the rollout placement group this
    launcher schedules into already requires a live cluster.
    """

    @staticmethod
    def _validate_rank_gpu_ids(
        resources: ResourcePlan,
        metadata: Sequence[RayActorHandle],
    ) -> None:
        """Validate launched rank actors against the resolved rollout placement."""

        if not resources.rollout_devices:
            return

        driver_node_ip: str | None = None
        if resources.cross_node:
            driver_node_ip = current_node_ip()

        # Under cross-node the node-aware check applies and the ids are ignored.
        require_actor_gpu_ids(
            metadata,
            expected_gpu_ids=resources.rollout_devices,
            role="generation",
            cross_node=resources.cross_node,
            driver_node_ip=driver_node_ip,
        )

    def _launch_session(
        self,
        resources: ResourcePlan,
        worker: RolloutWorkerConfig,
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

        bundle_indices = list(placement.bundle_indices)
        # One engine per rollout bundle: the engine's single rank owns the
        # bundle's GPU, so engine ids and worker ids coincide.
        placement_group = placement.placement_group
        engine_ids = [f"rollout-{engine_idx}" for engine_idx in range(len(bundle_indices))]
        # The same registry-owned gatherer serves the driver executor and the
        # single-engine pipelined path. Ray serializes it to each rank actor;
        # ranks never re-resolve family identity from the neutral execution layer.
        actor_group: RayActorGroup | None = None
        finalizer_group: RayActorGroup | None = None
        try:
            actor_group = RayActorGroup.launch(
                worker_cls=RayGenerationWorker,
                worker_configs=[launch_inputs for _ in engine_ids],
                worker_ids=engine_ids,
                num_cpus=worker.cpus_per_worker,
                # Each GPU rank owns one whole GPU; a CPU fleet grants none.
                num_gpus=1 if resources.rollout_devices else 0,
                rpc_timeout_s=worker.worker_rpc_timeout_s,
                operation_prefix="rollout",
                placement_group=placement_group,
                bundle_indices=bundle_indices,
                startup_method="load_policy",
            )
            self._validate_rank_gpu_ids(resources, actor_group.handles)
            engines = [
                RayGenerationEngine(engine_id, [handle])
                for engine_id, handle in zip(engine_ids, actor_group.handles, strict=True)
            ]
            actor_dispatcher = RayActorDispatcher(
                tuple(engine.engine_id for engine in engines),
            )
            finalizer_handles: list[RayActorHandle] = []
            if worker.pipelined:
                # One CPU finalizer per engine, pinned to the engine's bundle.
                # That keeps the merge local for a single-engine fleet; with
                # several engines a request's payloads come from every engine,
                # so part of the input crosses nodes either way. It reserves no
                # CPU: bundles are sized for the rank.
                finalizer_group = RayActorGroup.launch(
                    worker_cls=RayGenerationFinalizer,
                    worker_configs=[launch_inputs.gatherer for _ in engine_ids],
                    worker_ids=[f"{engine_id}.finalize" for engine_id in engine_ids],
                    num_cpus=0,
                    num_gpus=0,
                    rpc_timeout_s=worker.worker_rpc_timeout_s,
                    operation_prefix="rollout.finalize",
                    placement_group=placement_group,
                    bundle_indices=bundle_indices,
                )
                finalizer_handles = list(finalizer_group.handles)

            executor = RayGenerationExecutor(
                engines,
                launch_inputs.gatherer,
                actor_dispatcher=actor_dispatcher,
                generation_stall_timeout_s=worker.generation_stall_timeout_s,
                finalizers=finalizer_handles,
            )
            return RayGenerationSession(executor, worker_rpc_timeout_s=worker.worker_rpc_timeout_s)
        except BaseException as error:
            for group in (actor_group, finalizer_group):
                if group is None:
                    continue
                try:
                    group.shutdown()
                except BaseException as cleanup_error:
                    error.add_note(
                        f"rollout startup actor cleanup also failed: {cleanup_error!r}",
                    )
            # The launcher only created the workers; the placement group belongs
            # to the GlobalRayPlacementOwner.
            raise

    async def _launch_session_async(
        self,
        resources: ResourcePlan,
        worker: RolloutWorkerConfig,
        launch_inputs: RayGenerationLaunchInputs,
        *,
        placement: RolePlacement,
    ) -> RayGenerationSession:
        """Launch a session without blocking the runtime's lifecycle event loop.

        Actor startup and policy load run in a worker thread that cancellation
        cannot stop; a cancelled activation waits for the thread's fleet and
        closes it.
        """

        return await asyncio.to_thread(
            self._launch_session,
            resources,
            worker,
            launch_inputs,
            placement=placement,
        )

    def create_runtime(
        self,
        resources: ResourcePlan,
        worker: RolloutWorkerConfig,
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
        deferred = resources.lifecycle.rollout_mode == "on_demand"
        session = None
        session_factory = None
        if deferred:
            session_factory = partial(
                self._launch_session_async,
                resources,
                worker,
                launch_inputs,
                placement=placement,
            )
        else:
            session = self._launch_session(
                resources,
                worker,
                launch_inputs,
                placement=placement,
            )

        try:
            runtime = RayGenerationRuntime(
                session=session,
                session_factory=session_factory,
                initial_policy_version=launch_inputs.launch_config.policy_version,
            )
            return runtime
        except BaseException as error:
            if session is not None:
                session.force_close()
                try:
                    session.kill_engines()
                except BaseException as cleanup_error:
                    error.add_note(
                        f"resident rollout startup cleanup also failed: {cleanup_error!r}",
                    )
            raise


__all__ = [
    "RayGenerationLauncher",
]
