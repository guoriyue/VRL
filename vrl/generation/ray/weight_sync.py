"""Synchronous weight-version coordination for Ray generation engines."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Protocol

from vrl.generation.ray.engine import RayGenerationEngine, uniform_rank_result
from vrl.ray.actor_pool import RayActorDispatcher, RayActorJob
from vrl.ray.dependencies import require_ray
from vrl.utils.config import require_exact_int
from vrl.utils.deadline import require_timeout


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
    agreed echo against the expected version per engine. ``verify_content`` is
    an opt-in acceptance probe: every rank must read back the installed parameters
    before returning that echo. Normal sync does not pay for device readback.
    """

    def __init__(
        self,
        engines: list[RayGenerationEngine],
        *,
        actor_dispatcher: RayActorDispatcher,
        worker_rpc_timeout_s: float,
        verify_content: bool = False,
        bucket_bytes: int | None = None,
    ) -> None:
        if bucket_bytes is not None and (type(bucket_bytes) is not int or bucket_bytes < 1):
            raise ValueError("bucket_bytes must be a positive integer")
        self.bucket_bytes = bucket_bytes
        self.verify_content = bool(verify_content)
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
        state_ref: Any,
        policy_version: int,
    ) -> None:
        require_exact_int(policy_version, path="policy_version", minimum=0)
        if self.bucket_bytes is not None and state_ref is not None:
            await self._push_bucketed(state_ref, policy_version)
            return
        verification = {"verify_content": True} if self.verify_content else {}
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
                installed = update_weights(state_ref, policy_version, **verification)
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
                keyword_args={"policy_version": policy_version, **verification},
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

    async def _push_bucketed(self, state: Any, policy_version: int) -> None:
        from vrl.generation.weight_transfer import iter_weight_buckets, weight_manifest

        manifest = weight_manifest(state)
        transfer_id = uuid.uuid4().hex
        ray = require_ray()
        assert self.bucket_bytes is not None

        async def broadcast(method: str, payload: Any, **kwargs: Any) -> None:
            shared = ray.put(payload)
            jobs = [
                RayActorJob(
                    job_index=index,
                    worker_id=engine.engine_id,
                    remote_method=engine.remote(method, combine=uniform_rank_result(method)),
                    payload=shared,
                    keyword_args=kwargs,
                )
                for index, engine in enumerate(self.engines)
            ]
            installed = await self.actor_dispatcher.run(
                jobs,
                operation=f"rollout.weight_sync.{method}",
                call_timeout_s=self.worker_rpc_timeout_s,
            )
            for engine, (_, version) in zip(self.engines, installed, strict=True):
                _require_installed_policy_version(engine, version, policy_version)

        try:
            await broadcast(
                "begin_weight_transfer",
                manifest,
                transfer_id=transfer_id,
                policy_version=policy_version,
            )
            # Await every receiver before putting the next independently owned
            # slice. Receiver copies ensure completed buckets can be reclaimed.
            for chunk in iter_weight_buckets(state, self.bucket_bytes):
                await broadcast("receive_weight_bucket", chunk, transfer_id=transfer_id)
            await broadcast(
                "commit_weight_transfer", transfer_id, verify_content=self.verify_content
            )
        except BaseException as error:
            # The dispatcher may already reject admissions after a rank failure.
            # Abort is cleanup, so bypass its normal admission path with a bound.
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(
                        *[
                            engine.remote("abort_weight_transfer")(transfer_id)
                            for engine in self.engines
                        ],
                        return_exceptions=True,
                    ),
                    timeout=self.worker_rpc_timeout_s,
                )
                for result in results:
                    if isinstance(result, BaseException):
                        error.add_note(f"weight transfer abort failed: {result}")
            except BaseException as cleanup_error:
                error.add_note(f"weight transfer abort failed: {cleanup_error}")
            raise


def _require_installed_policy_version(
    engine: RayGenerationEngine,
    installed: Any,
    expected: int,
) -> None:
    """Validate one untyped engine ACK at the Ray weight-sync boundary."""

    if isinstance(installed, bool) or not isinstance(installed, int) or installed < 0:
        raise RuntimeError(
            f"engine {engine.engine_id!r} returned invalid installed policy version {installed!r}",
        )
    if installed != expected:
        raise RuntimeError(
            f"engine {engine.engine_id!r} installed policy version {installed}, "
            f"expected {expected}",
        )


__all__ = [
    "GenerationWeightSync",
    "RayGenerationWeightSync",
]
