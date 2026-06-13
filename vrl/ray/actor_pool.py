"""Bounded Ray actor submit / wait / gather helpers."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RayActorJob:
    """One actor method call scheduled through the shared Ray pool.

    ``worker_id``/``remote_method`` may be left ``None`` for pull-based
    dispatch: the pool then binds the job to whichever worker from
    ``worker_methods`` has a free slot (least inflight first). ``priority``
    orders submission (higher first, stable for ties) — with pull dispatch
    this implements LPT: submit expensive chunks first so no worker is left
    finishing one large job at the tail.
    """

    job_index: int
    worker_id: str | None
    remote_method: Any
    payload: Any
    priority: float = 0.0


async def run_actor_jobs(
    jobs: list[RayActorJob],
    *,
    max_inflight_per_actor: int = 1,
    worker_methods: Mapping[str, Any] | None = None,
    schedule: list[dict[str, Any]] | None = None,
) -> list[tuple[int, Any]]:
    """Run actor jobs with per-actor backpressure and return ordered pairs.

    Jobs with a ``worker_id`` keep their plan-time binding (static placement,
    the round-robin regression baseline). Jobs without one require
    ``worker_methods`` and are bound at dispatch time. ``schedule``, when
    given, receives one telemetry row per job: queue wait and execution time
    are only observable inside this loop, so the instrumentation lives here.
    """

    if max_inflight_per_actor < 1:
        raise ValueError("max_inflight_per_actor must be >= 1")
    if not jobs:
        return []
    unbound = [job for job in jobs if job.worker_id is None]
    if unbound and not worker_methods:
        raise ValueError(
            f"{len(unbound)} job(s) have no worker binding; pull-based "
            "dispatch requires worker_methods",
        )

    # Stable sort: equal priorities (the static path always submits 0.0)
    # preserve caller order bit-for-bit.
    pending = deque(sorted(jobs, key=lambda job: -job.priority))
    inflight_by_worker = {
        job.worker_id: 0 for job in jobs if job.worker_id is not None
    }
    for worker_id in worker_methods or ():
        inflight_by_worker.setdefault(worker_id, 0)
    ref_to_job: dict[Any, tuple[int, str]] = {}
    result_pairs: list[tuple[int, Any]] = []
    pool_started = time.perf_counter()
    submit_time_by_ref: dict[Any, float] = {}

    def _dispatch_worker_for(job: RayActorJob) -> tuple[str, Any] | None:
        """Resolve (worker_id, method); None when no slot is free."""
        if job.worker_id is not None:
            if inflight_by_worker[job.worker_id] >= max_inflight_per_actor:
                return None
            return job.worker_id, job.remote_method
        free = [
            (count, worker_id)
            for worker_id, count in inflight_by_worker.items()
            if count < max_inflight_per_actor and worker_id in (worker_methods or {})
        ]
        if not free:
            return None
        _, worker_id = min(free)
        return worker_id, worker_methods[worker_id]  # type: ignore[index]

    def _submit_ready() -> list[Any]:
        """Submit every job whose worker has a free slot; return the new refs."""
        new_refs: list[Any] = []
        made_progress = True
        while pending and made_progress:
            made_progress = False
            for _ in range(len(pending)):
                job = pending.popleft()
                dispatch = _dispatch_worker_for(job)
                if dispatch is None:
                    pending.append(job)
                    continue
                worker_id, method = dispatch
                ref = method(job.payload)
                ref_to_job[ref] = (job.job_index, worker_id)
                inflight_by_worker[worker_id] += 1
                submit_time_by_ref[ref] = time.perf_counter()
                new_refs.append(ref)
                made_progress = True
        return new_refs

    async def _await_ref(ref: Any) -> tuple[Any, Any]:
        # Ray ObjectRefs are awaitable; awaiting resolves to the value or
        # re-raises the worker exception — same fail-fast as the old
        # ray.get(ready[0]), but without bouncing through a thread pool.
        return ref, await ref

    # task -> ref, so a completed wrapper task maps back to its job/telemetry.
    waiters: dict[Any, Any] = {}

    def _spawn(refs: list[Any]) -> None:
        for ref in refs:
            waiters[asyncio.ensure_future(_await_ref(ref))] = ref

    # Fail-fast on worker error: a raising waiter propagates immediately and
    # in-flight refs on other actors are abandoned (not cancelled) — their
    # results are dropped when the caller tears the group down. The finally
    # block cancels the leftover waiters so no "Task was destroyed but it is
    # pending" warning leaks; it does not cancel the underlying Ray work.
    _spawn(_submit_ready())
    try:
        while waiters:
            done, _ = await asyncio.wait(
                waiters,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                waiters.pop(task, None)
                ref, result = task.result()
                job_index, worker_id = ref_to_job.pop(ref)
                inflight_by_worker[worker_id] -= 1
                result_pairs.append((job_index, result))
                if schedule is not None:
                    submitted = submit_time_by_ref.pop(ref)
                    finished = time.perf_counter()
                    schedule.append(
                        {
                            "job_index": job_index,
                            "worker_id": worker_id,
                            "queue_wait_s": submitted - pool_started,
                            "execution_s": finished - submitted,
                        },
                    )
            _spawn(_submit_ready())
    finally:
        for task in waiters:
            task.cancel()
        if waiters:
            await asyncio.gather(*waiters, return_exceptions=True)

    return sorted(result_pairs, key=lambda pair: pair[0])


__all__ = ["RayActorJob", "run_actor_jobs"]
