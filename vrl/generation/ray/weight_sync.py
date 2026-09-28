"""Awaited weight-version coordination for Ray generation engines."""

from __future__ import annotations

from typing import Any

from vrl.generation.ray.engine import RayGenerationEngine
from vrl.ray.actor_pool import RayActorDispatcher, RayActorJob
from vrl.ray.dependencies import require_ray
from vrl.utils.deadline import require_timeout
from vrl.utils.validation import require_int


class RayGenerationWeightSync:
    """Broadcast ``update_weights`` to every rank of every generation engine.

    The engine call fans out to all its ranks and completes when every rank has
    installed the payload; a rank failure fails the whole sync.
    """

    def __init__(
        self,
        engines: list[RayGenerationEngine],
        *,
        actor_dispatcher: RayActorDispatcher,
        worker_rpc_timeout_s: float,
    ) -> None:
        self.engines = list(engines)
        self.actor_dispatcher = actor_dispatcher
        self.worker_rpc_timeout_s = require_timeout(
            worker_rpc_timeout_s,
            name="worker_rpc_timeout_s",
        )

    async def push_to_rollout_engines(
        self,
        trainable_state: Any,
        policy_version: int,
    ) -> None:
        require_int(policy_version, path="policy_version", minimum=0)
        if not self.engines:
            return

        ray = require_ray()
        # Serialize the (potentially large) state dict once into the object
        # store and hand every rank the same ObjectRef. Passing the dict
        # straight to each actor.update_weights.remote(...) makes Ray
        # re-serialize and store one copy per rank, so weight-sync cost grew
        # linearly in fleet size for identical data. Ray auto-dereferences
        # the ref into the real dict before the rank method runs.
        shared_state_ref = ray.put(trainable_state)
        remote_jobs = [
            RayActorJob(
                job_index=job_index,
                worker_id=engine.engine_id,
                remote_method=engine.remote("update_weights"),
                payload=shared_state_ref,
                keyword_args={"policy_version": policy_version},
            )
            for job_index, engine in enumerate(self.engines)
        ]
        await self.actor_dispatcher.run(
            remote_jobs,
            operation="rollout.weight_sync",
            call_timeout_s=self.worker_rpc_timeout_s,
        )


__all__ = [
    "RayGenerationWeightSync",
]
