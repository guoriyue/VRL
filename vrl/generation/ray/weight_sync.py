"""Awaited weight-version coordination for Ray generation engines."""

from __future__ import annotations

from typing import Any, Protocol

from vrl.generation.ray.engine import RayGenerationEngine
from vrl.ray.actor_pool import RayActorDispatcher, RayActorJob
from vrl.ray.dependencies import require_ray
from vrl.utils.deadline import require_timeout
from vrl.utils.validation import require_int


class GenerationWeightSync(Protocol):
    """Push a trainable-state payload with a known policy version.

    The sender supplies the payload itself; transport references belong to the
    implementation. None retains the worker's existing bootstrap semantics.
    """

    async def push_to_rollout_engines(
        self,
        trainable_state: Any,
        policy_version: int,
    ) -> None: ...


class RayGenerationWeightSync:
    """Broadcast ``update_weights`` to every rank of every generation engine.

    The engine call fans out to all its ranks and requires their version
    echoes to agree (``RayGenerationEngine.remote_uniform``); this layer then
    validates the agreed echo against the expected version per engine.
    """

    def __init__(
        self,
        engines: list[RayGenerationEngine],
        *,
        actor_dispatcher: RayActorDispatcher,
        worker_rpc_timeout_s: float,
    ) -> None:
        self.engines = list(engines)
        expected_engine_ids = tuple(engine.engine_id for engine in self.engines)
        if actor_dispatcher.worker_ids != expected_engine_ids:
            raise ValueError(
                "RayGenerationWeightSync actor dispatcher does not own its engine fleet: "
                f"{actor_dispatcher.worker_ids} != {expected_engine_ids}",
            )
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
                remote_method=engine.remote_uniform("update_weights"),
                payload=shared_state_ref,
                keyword_args={"policy_version": policy_version},
            )
            for job_index, engine in enumerate(self.engines)
        ]
        policy_version_acks = await self.actor_dispatcher.run(
            remote_jobs,
            operation="rollout.weight_sync",
            call_timeout_s=self.worker_rpc_timeout_s,
        )
        for engine, (_job_index, acknowledged_policy_version) in zip(
            self.engines,
            policy_version_acks,
            strict=True,
        ):
            self._validate_policy_version_match(
                engine, acknowledged_policy_version, policy_version
            )

    @staticmethod
    def _validate_policy_version_match(
        engine: RayGenerationEngine,
        acknowledged_policy_version: Any,
        expected_policy_version: int,
    ) -> None:
        """Validate one untyped engine ACK at the Ray weight-sync boundary."""

        if (
            isinstance(acknowledged_policy_version, bool)
            or not isinstance(acknowledged_policy_version, int)
            or acknowledged_policy_version < 0
        ):
            raise RuntimeError(
                f"engine {engine.engine_id!r} returned invalid policy version acknowledgment {acknowledged_policy_version!r}",
            )
        if acknowledged_policy_version != expected_policy_version:
            raise RuntimeError(
                f"engine {engine.engine_id!r} acknowledged policy version {acknowledged_policy_version}, "
                f"expected {expected_policy_version}",
            )


__all__ = [
    "GenerationWeightSync",
    "RayGenerationWeightSync",
]
