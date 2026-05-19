"""Bounded Ray actor submit / wait / gather helpers."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Any

from vrl.ray.dependencies import require_ray


@dataclass(frozen=True, slots=True)
class RayActorJob:
    """One actor method call scheduled through the shared Ray pool."""

    job_index: int
    worker_id: str
    remote_method: Any
    payload: Any


async def run_actor_jobs(
    jobs: list[RayActorJob],
    *,
    max_inflight_per_actor: int = 1,
) -> list[tuple[int, Any]]:
    """Run actor jobs with per-actor backpressure and return ordered pairs."""

    if max_inflight_per_actor < 1:
        raise ValueError("max_inflight_per_actor must be >= 1")
    if not jobs:
        return []

    ray = require_ray()
    pending = deque(jobs)
    inflight_by_worker = {job.worker_id: 0 for job in jobs}
    ref_to_job: dict[Any, tuple[int, str]] = {}
    result_pairs: list[tuple[int, Any]] = []

    def _submit_ready() -> None:
        made_progress = True
        while pending and made_progress:
            made_progress = False
            for _ in range(len(pending)):
                job = pending.popleft()
                if inflight_by_worker[job.worker_id] >= max_inflight_per_actor:
                    pending.append(job)
                    continue
                ref = job.remote_method(job.payload)
                ref_to_job[ref] = (job.job_index, job.worker_id)
                inflight_by_worker[job.worker_id] += 1
                made_progress = True

    _submit_ready()
    while ref_to_job:
        ready, _ = await asyncio.to_thread(
            ray.wait,
            list(ref_to_job),
            num_returns=1,
        )
        job_index, worker_id = ref_to_job.pop(ready[0])
        inflight_by_worker[worker_id] -= 1
        result = await asyncio.to_thread(ray.get, ready[0])
        result_pairs.append((job_index, result))
        _submit_ready()

    return sorted(result_pairs, key=lambda pair: pair[0])


__all__ = ["RayActorJob", "run_actor_jobs"]
