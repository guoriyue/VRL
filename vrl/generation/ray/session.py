"""Resources owned by one launched Ray generation worker session."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from vrl.generation.execution.types import WorkerMemoryParkingSnapshot
from vrl.generation.ray.executor import RayGenerationExecutor
from vrl.generation.ray.weight_sync import GenerationWeightSync
from vrl.ray.actor_group import RayActorHandle
from vrl.ray.dependencies import require_ray
from vrl.ray.resource_cleanup import kill_and_retain
from vrl.utils.deadline import OperationDeadline

logger = logging.getLogger(__name__)

# Parking moves whole model states between GPU and pinned host memory; the
# bound catches a wedged CUDA runtime without tripping on a slow multi-GiB
# offload. A timeout raises the shared terminal OperationTimeout so the runtime
# treats the fleet as failed instead of surfacing a bare asyncio error.
_WORKER_PARK_TIMEOUT_S = 120.0
# Graceful policy release before actor kill; a slow release is not worth
# delaying cleanup longer than this — the kill that follows reclaims the GPU.
_POLICY_RELEASE_TIMEOUT_S = 60.0


class RayGenerationSession:
    """Own one launched worker fleet without owning its public lifecycle.

    ``RayGenerationRuntime`` decides when operations are admitted, which failure
    is terminal, and whether cleanup is graceful or forceful. This object retains
    only the concrete resources needed to execute and close one launched fleet.
    """

    def __init__(
        self,
        executor: RayGenerationExecutor,
        weight_sync: GenerationWeightSync | None,
        owned_workers: list[RayActorHandle],
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
        self.workers = list(owned_workers)
        worker_ids = tuple(worker.worker_id for worker in self.workers)
        if len(set(worker_ids)) != len(worker_ids):
            raise RuntimeError(f"duplicate generation worker ids: {worker_ids}")
        self.supports_non_draining_weight_sync = bool(
            supports_non_draining_weight_sync,
        )
        self._release_wait_task: asyncio.Task[Any] | None = None
        self._force_close = False

    async def update_weights(self, state_ref: Any, policy_version: int) -> None:
        weight_sync = self.weight_sync
        if weight_sync is None:
            raise RuntimeError("RayGenerationSession has no GenerationWeightSync")
        await weight_sync.push_to_rollout_workers(
            state_ref,
            int(policy_version),
        )

    async def sleep_workers(self) -> tuple[WorkerMemoryParkingSnapshot, ...]:
        """Park every worker and return validated physical-memory evidence."""

        if not self.workers:
            return ()
        deadline = OperationDeadline(
            "generation.worker_sleep",
            _WORKER_PARK_TIMEOUT_S,
            context=f"workers={len(self.workers)}",
        )
        refs = [worker.actor.sleep.remote() for worker in self.workers]
        try:
            values = await asyncio.wait_for(
                asyncio.gather(*refs),
                timeout=deadline.remaining_s(),
            )
        except TimeoutError as cause:
            # A worker may itself raise TimeoutError before the budget runs
            # out; only an exhausted deadline is this barrier's own expiry.
            if deadline.remaining_s() > 0:
                raise
            raise deadline.timeout_error() from cause
        snapshots: list[WorkerMemoryParkingSnapshot] = []
        for worker, value in zip(self.workers, values, strict=True):
            if not isinstance(value, WorkerMemoryParkingSnapshot):
                raise TypeError(
                    f"worker {worker.worker_id!r} returned invalid memory-parking "
                    f"report {type(value).__name__}",
                )
            if value.worker_id != worker.worker_id:
                raise RuntimeError(
                    "mismatched worker memory-parking report: "
                    f"expected={worker.worker_id!r} actual={value.worker_id!r}",
                )
            value.validate()
            snapshots.append(value)
        return tuple(snapshots)

    async def wake_workers(self) -> None:
        """Restore every parked worker onto its assigned GPU."""

        if not self.workers:
            return
        deadline = OperationDeadline(
            "generation.worker_wake",
            _WORKER_PARK_TIMEOUT_S,
            context=f"workers={len(self.workers)}",
        )
        refs = [worker.actor.wake.remote() for worker in self.workers]
        try:
            await asyncio.wait_for(
                asyncio.gather(*refs),
                timeout=deadline.remaining_s(),
            )
        except TimeoutError as cause:
            if deadline.remaining_s() > 0:
                raise
            raise deadline.timeout_error() from cause

    async def close(self, *, force: bool) -> None:
        """Release worker policies, kill actors, and retain failed handles."""

        if not self.workers:
            return
        if force:
            self.force_close()
        ray = require_ray()
        release_refs: list[Any] = []
        if not self._force_close:
            for worker in self.workers:
                actor = worker.actor
                remote = getattr(getattr(actor, "release_policy", None), "remote", None)
                if not callable(remote):
                    continue
                try:
                    release_refs.append(remote())
                except Exception:
                    logger.warning(
                        "generation worker %s policy release submission failed; "
                        "forcing actor cleanup",
                        worker.worker_id,
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
            self.kill_workers()

    def kill_workers(self) -> None:
        """Synchronously kill the fleet and retain handles that fail to die."""

        if not self.workers:
            return
        ray = require_ray()
        surviving, failures = kill_and_retain(
            ray,
            self.workers,
            lambda worker: worker.actor,
        )
        self.workers[:] = surviving
        if failures:
            raise RuntimeError(
                "Ray generation session cleanup incomplete: "
                f"{len(failures)} worker actor kill(s) failed",
            ) from failures[0][1]

    def force_close(self) -> None:
        """Upgrade current or future cleanup to skip graceful worker release."""

        self._force_close = True
        release_wait_task = self._release_wait_task
        if release_wait_task is not None and not release_wait_task.done():
            release_wait_task.cancel()


__all__ = ["RayGenerationSession"]
