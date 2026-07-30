"""Synchronous weight-version coordination for Ray generation workers."""

from __future__ import annotations

from typing import Any, Protocol

from vrl.generation.execution.types import DistributedWorkerHandle
from vrl.ray.actor_pool import RayActorDispatcher, RayActorJob
from vrl.ray.dependencies import require_ray
from vrl.ray.operation_deadline import validate_ray_timeout


class GenerationWeightSync(Protocol):
    """Push trainable generation state to workers with a known policy version."""

    async def push_to_rollout_workers(
        self,
        state_ref: Any,
        policy_version: int,
    ) -> None: ...


class RayGenerationWeightSync(GenerationWeightSync):
    """Call ``update_weights`` on every Ray generation worker."""

    def __init__(
        self,
        workers: list[DistributedWorkerHandle],
        *,
        actor_dispatcher: RayActorDispatcher,
        worker_rpc_timeout_s: float,
    ) -> None:
        self.workers = list(workers)
        expected_worker_ids = tuple(worker.worker_id for worker in self.workers)
        if actor_dispatcher.worker_ids != expected_worker_ids:
            raise ValueError(
                "RayGenerationWeightSync actor dispatcher does not own its worker fleet: "
                f"{actor_dispatcher.worker_ids} != {expected_worker_ids}",
            )
        self.actor_dispatcher = actor_dispatcher
        self.worker_rpc_timeout_s = validate_ray_timeout(
            worker_rpc_timeout_s,
            name="worker_rpc_timeout_s",
        )

    async def push_to_rollout_workers(
        self,
        state_ref: Any,
        policy_version: int,
    ) -> None:
        remote_workers: list[tuple[DistributedWorkerHandle, Any]] = []
        for worker in self.workers:
            actor = worker.actor
            if actor is None:
                raise RuntimeError(f"worker {worker.worker_id!r} has no actor")
            update_weights = actor.update_weights
            remote = getattr(update_weights, "remote", None)
            if callable(remote):
                remote_workers.append((worker, remote))
            else:
                installed = update_weights(state_ref, policy_version)
                worker.require_installed_policy_version(installed, policy_version)

        if not remote_workers:
            return

        ray = require_ray()
        # Serialize the (potentially large) state dict once into the object
        # store and hand every worker the same ObjectRef. Passing the dict
        # straight to each actor.update_weights.remote(...) makes Ray
        # re-serialize and store one copy per worker, so weight-sync cost grew
        # linearly in worker count for identical data. Ray auto-dereferences
        # the ref into the real dict before the worker method runs.
        shared_state = ray.put(state_ref)
        remote_jobs = [
            RayActorJob(
                job_index=job_index,
                worker_id=worker.worker_id,
                remote_method=remote,
                payload=shared_state,
                keyword_args={"policy_version": policy_version},
            )
            for job_index, (worker, remote) in enumerate(remote_workers)
        ]
        installed_pairs = await self.actor_dispatcher.run(
            remote_jobs,
            operation="rollout.weight_sync",
            call_timeout_s=self.worker_rpc_timeout_s,
        )
        for (worker, _remote), (_job_index, installed) in zip(
            remote_workers,
            installed_pairs,
            strict=True,
        ):
            worker.require_installed_policy_version(installed, policy_version)


__all__ = [
    "GenerationWeightSync",
    "RayGenerationWeightSync",
]
