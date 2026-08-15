"""Synchronous weight-version coordination for Ray generation engines."""

from __future__ import annotations

from typing import Any, Protocol

from vrl.generation.ray.engine import RayGenerationEngine, uniform_rank_result
from vrl.ray.actor_pool import RayActorDispatcher, RayActorJob
from vrl.ray.dependencies import require_ray
from vrl.utils.deadline import validate_timeout


class GenerationWeightSync(Protocol):
    """Push trainable generation state to engines with a known policy version."""

    async def push_to_rollout_engines(
        self,
        state_ref: Any,
        policy_version: int,
    ) -> None: ...


class RayGenerationWeightSync:
    """Broadcast ``update_weights`` to every rank of every generation engine.

    The engine call fans out to all its ranks and requires their version
    echoes to agree (``uniform_rank_result``); this layer then validates the
    agreed echo against the expected version per engine.
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
        self.worker_rpc_timeout_s = validate_timeout(
            worker_rpc_timeout_s,
            name="worker_rpc_timeout_s",
        )

    async def push_to_rollout_engines(
        self,
        state_ref: Any,
        policy_version: int,
    ) -> None:
        remote_engines: list[tuple[RayGenerationEngine, Any]] = []
        for engine in self.engines:
            update_weights = engine.primary.actor.update_weights
            if callable(getattr(update_weights, "remote", None)):
                remote_engines.append(
                    (
                        engine,
                        engine.remote(
                            "update_weights",
                            combine=uniform_rank_result("update_weights"),
                        ),
                    ),
                )
            else:
                # Local test double: call the single rank directly.
                installed = update_weights(state_ref, policy_version)
                _require_installed_policy_version(engine, installed, policy_version)

        if not remote_engines:
            return

        ray = require_ray()
        # Serialize the (potentially large) state dict once into the object
        # store and hand every rank the same ObjectRef. Passing the dict
        # straight to each actor.update_weights.remote(...) makes Ray
        # re-serialize and store one copy per rank, so weight-sync cost grew
        # linearly in fleet size for identical data. Ray auto-dereferences
        # the ref into the real dict before the rank method runs.
        shared_state = ray.put(state_ref)
        remote_jobs = [
            RayActorJob(
                job_index=job_index,
                worker_id=engine.engine_id,
                remote_method=remote,
                payload=shared_state,
                keyword_args={"policy_version": policy_version},
            )
            for job_index, (engine, remote) in enumerate(remote_engines)
        ]
        installed_pairs = await self.actor_dispatcher.run(
            remote_jobs,
            operation="rollout.weight_sync",
            call_timeout_s=self.worker_rpc_timeout_s,
        )
        for (engine, _remote), (_job_index, installed) in zip(
            remote_engines,
            installed_pairs,
            strict=True,
        ):
            _require_installed_policy_version(engine, installed, policy_version)


def _require_installed_policy_version(
    engine: RayGenerationEngine,
    installed: Any,
    expected: int,
) -> None:
    """Validate one untyped engine ACK at the Ray weight-sync boundary."""

    try:
        installed_version = int(installed)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"engine {engine.engine_id!r} returned invalid installed policy version {installed!r}",
        ) from exc
    if installed_version != int(expected):
        raise RuntimeError(
            f"engine {engine.engine_id!r} installed policy version {installed_version}, "
            f"expected {int(expected)}",
        )


__all__ = [
    "GenerationWeightSync",
    "RayGenerationWeightSync",
]
