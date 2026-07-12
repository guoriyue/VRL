"""Synchronous weight-version coordination for Ray generation workers."""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

from vrl.generation.execution.types import DistributedWorkerHandle
from vrl.generation.ray.utils import require_installed_policy_version
from vrl.ray.dependencies import require_ray


class GenerationWeightSync(Protocol):
    """Push trainable generation state to workers with a known policy version."""

    async def push_to_rollout_workers(
        self,
        state_ref: Any,
        policy_version: int,
    ) -> None: ...


class RayGenerationWeightSync(GenerationWeightSync):
    """Call ``update_weights`` on every Ray generation worker."""

    def __init__(self, workers: list[DistributedWorkerHandle]) -> None:
        self.workers = list(workers)

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
                require_installed_policy_version(
                    worker_id=worker.worker_id,
                    installed=installed,
                    expected=policy_version,
                )

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
        update_refs = [
            update_remote(shared_state, policy_version)
            for _worker, update_remote in remote_workers
        ]
        # Ray ObjectRefs are asyncio-awaitable. Waiting on them directly keeps
        # the ACK barrier owned by this event loop; cancelling owner shutdown
        # cannot leave a detached ``to_thread(ray.get)`` blocked indefinitely.
        installed_versions = await asyncio.gather(*update_refs)
        for (worker, _update), installed in zip(
            remote_workers,
            installed_versions,
            strict=True,
        ):
            require_installed_policy_version(
                worker_id=worker.worker_id,
                installed=installed,
                expected=policy_version,
            )


__all__ = [
    "GenerationWeightSync",
    "RayGenerationWeightSync",
]
