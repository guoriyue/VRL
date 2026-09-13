"""Resources owned by one launched Ray generation engine fleet."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from vrl.generation.execution.types import WorkerMemoryParkingSnapshot
from vrl.generation.ray.engine import RayGenerationEngine, rank_handles
from vrl.generation.ray.executor import RayGenerationExecutor
from vrl.generation.ray.weight_sync import GenerationWeightSync
from vrl.ray.dependencies import kill_actors, require_ray
from vrl.utils.deadline import OperationDeadline

logger = logging.getLogger(__name__)

# Parking moves whole model states between GPU and pinned host memory; the
# bound catches a wedged CUDA runtime without tripping on a slow multi-GiB
# offload. A timeout raises the shared terminal OperationTimeout so the runtime
# treats the fleet as failed instead of surfacing a bare asyncio error.
_RANK_PARK_TIMEOUT_S = 120.0
# Graceful policy release before actor kill; a slow release is not worth
# delaying cleanup longer than this — the kill that follows reclaims the GPU.
_POLICY_RELEASE_TIMEOUT_S = 60.0


class RayGenerationSession:
    """Own one launched engine fleet without owning its public lifecycle.

    ``RayGenerationRuntime`` decides when operations are admitted, which failure
    is terminal, and whether cleanup is graceful or forceful. This object retains
    only the concrete resources needed to execute and close one launched fleet.
    Engines are the operation unit (dispatch, weight sync, parking); their rank
    actors are the lifecycle unit (kill, liveness, graceful release).
    """

    def __init__(
        self,
        executor: RayGenerationExecutor,
        weight_sync: GenerationWeightSync | None,
        owned_engines: list[RayGenerationEngine],
        *,
        supports_non_draining_weight_sync: bool = False,
    ) -> None:
        if executor is None:
            raise ValueError("Ray generation session requires an executor")
        self.executor = executor
        self.weight_sync = weight_sync
        executor_dispatcher = getattr(executor, "actor_dispatcher", None)
        weight_sync_dispatcher = getattr(weight_sync, "actor_dispatcher", None)
        if (
            executor_dispatcher is not None
            and weight_sync_dispatcher is not None
            and executor_dispatcher is not weight_sync_dispatcher
        ):
            raise ValueError(
                "Ray generation and weight sync must share one actor dispatcher",
            )
        self.engines = list(owned_engines)
        self.rank_handles = rank_handles(self.engines)
        engine_ids = tuple(engine.engine_id for engine in self.engines)
        if len(set(engine_ids)) != len(engine_ids):
            raise RuntimeError(f"duplicate generation engine ids: {engine_ids}")
        rank_ids = tuple(rank.worker_id for rank in self.rank_handles)
        if len(set(rank_ids)) != len(rank_ids):
            raise RuntimeError(f"duplicate generation rank ids: {rank_ids}")
        self.supports_non_draining_weight_sync = bool(
            supports_non_draining_weight_sync,
        )
        self._release_wait_task: asyncio.Task[Any] | None = None
        self._force_close = False

    async def update_weights(self, trainable_state: Any, policy_version: int) -> None:
        weight_sync = self.weight_sync
        if weight_sync is None:
            raise RuntimeError("RayGenerationSession has no GenerationWeightSync")
        await weight_sync.push_to_rollout_engines(
            trainable_state,
            policy_version,
        )

    async def sleep_engines(self) -> tuple[WorkerMemoryParkingSnapshot, ...]:
        """Park every engine and return validated per-rank physical evidence."""

        if not self.engines:
            return ()
        deadline = OperationDeadline(
            "generation.engine_sleep",
            _RANK_PARK_TIMEOUT_S,
            context=f"engines={len(self.engines)}",
        )
        try:
            per_engine = await asyncio.wait_for(
                asyncio.gather(*[engine.sleep() for engine in self.engines]),
                timeout=deadline.remaining_s(),
            )
        except TimeoutError as cause:
            # A rank may itself raise TimeoutError before the budget runs
            # out; only an exhausted deadline is this barrier's own expiry.
            if deadline.remaining_s() > 0:
                raise
            raise deadline.timeout_error() from cause
        return tuple(snapshot for snapshots in per_engine for snapshot in snapshots)

    async def wake_engines(self) -> None:
        """Restore every parked engine onto its assigned GPUs."""

        if not self.engines:
            return
        deadline = OperationDeadline(
            "generation.engine_wake",
            _RANK_PARK_TIMEOUT_S,
            context=f"engines={len(self.engines)}",
        )
        try:
            await asyncio.wait_for(
                asyncio.gather(*[engine.wake() for engine in self.engines]),
                timeout=deadline.remaining_s(),
            )
        except TimeoutError as cause:
            if deadline.remaining_s() > 0:
                raise
            raise deadline.timeout_error() from cause

    async def close(self, *, force: bool) -> None:
        """Release rank policies, kill actors, and retain failed handles."""

        if not self.rank_handles:
            return
        if force:
            self.force_close()
        ray = require_ray()
        release_refs: list[Any] = []
        if not self._force_close:
            for rank in self.rank_handles:
                try:
                    release_refs.append(rank.actor.release_policy.remote())
                except Exception:
                    logger.warning(
                        "generation rank %s policy release submission failed; "
                        "forcing actor cleanup",
                        rank.worker_id,
                        exc_info=True,
                    )

        try:
            if release_refs:
                release_wait_task = asyncio.create_task(
                    asyncio.to_thread(
                        ray.get,
                        release_refs,
                        timeout=_POLICY_RELEASE_TIMEOUT_S,
                    ),
                )
                self._release_wait_task = release_wait_task
                try:
                    await release_wait_task
                except asyncio.CancelledError:
                    if not self._force_close:
                        raise
                except Exception:
                    logger.warning(
                        "generation policy release wait failed; forcing actor cleanup",
                        exc_info=True,
                    )
                finally:
                    if self._release_wait_task is release_wait_task:
                        self._release_wait_task = None
        finally:
            self.kill_engines()

    def kill_engines(self) -> None:
        """Synchronously kill every rank actor and retain handles that fail."""

        if not self.rank_handles:
            return
        ray = require_ray()
        failures = kill_actors(ray, [rank.actor for rank in self.rank_handles])
        failed_actor_ids = {id(actor) for actor, _ in failures}
        self.rank_handles[:] = [
            rank for rank in self.rank_handles if id(rank.actor) in failed_actor_ids
        ]
        # Once shutdown starts, engines are no longer executable. Retry cleanup
        # through the retained rank handles without rebuilding partial engines.
        self.engines.clear()
        if failures:
            raise RuntimeError(
                "Ray generation session cleanup incomplete: "
                f"{len(failures)} rank actor kill(s) failed",
            ) from failures[0][1]

    def force_close(self) -> None:
        """Upgrade current or future cleanup to skip graceful rank release."""

        self._force_close = True
        release_wait_task = self._release_wait_task
        if release_wait_task is not None and not release_wait_task.done():
            release_wait_task.cancel()


__all__ = ["RayGenerationSession"]
