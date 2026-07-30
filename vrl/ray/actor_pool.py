"""Bounded Ray actor submit / wait / gather helpers."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from vrl.ray.operation_deadline import (
    RayCallDeadline,
    RayOperationCancelled,
    cancel_ray_refs,
    validate_ray_timeout,
)
from vrl.runtime_errors import TerminalRuntimeError


@dataclass(frozen=True, slots=True)
class RayActorJob:
    """One actor method call scheduled through the shared Ray pool.

    ``worker_id``/``remote_method`` may be left ``None`` for pull-based
    dispatch: the pool then binds the job to whichever worker from
    ``worker_methods`` has a free slot (least inflight first). ``priority``
    orders submission (higher first, stable for ties) — with pull dispatch
    this implements LPT: submit expensive chunks first so no worker is left
    finishing one large job at the tail. ``keyword_args`` carries the small
    number of actor methods whose wire contract is not one positional payload.
    """

    job_index: int
    worker_id: str | None
    remote_method: Any
    payload: Any
    priority: float = 0.0
    keyword_args: Mapping[str, Any] = field(default_factory=dict)


class RayActorCallError(TerminalRuntimeError):
    """An actor call failed outside the worker's typed result protocol."""

    def __init__(self, operation: str, *, worker_id: str, job_index: int) -> None:
        self.operation = operation
        self.worker_id = worker_id
        self.job_index = int(job_index)
        super().__init__(
            f"Ray actor operation {operation!r} failed "
            f"(worker_id={worker_id}, job_index={job_index})",
        )


class RayActorDispatcher:
    """Map concurrent driver jobs onto one real slot per synchronous actor.

    The dispatcher belongs to the executor, not to one request. Keeping its
    admission state across ``run`` calls prevents concurrent requests from
    pre-queueing calls in a synchronous actor mailbox before their own
    operation deadline starts.
    """

    def __init__(self, worker_ids: list[str] | tuple[str, ...]) -> None:
        worker_ids = tuple(worker_ids)
        if not worker_ids:
            raise ValueError("RayActorDispatcher requires at least one worker")
        if len(set(worker_ids)) != len(worker_ids):
            raise ValueError(f"duplicate Ray actor worker ids: {worker_ids}")
        self._worker_ids = worker_ids
        self._available = set(worker_ids)
        self._changed = asyncio.Event()
        self._terminal_error: TerminalRuntimeError | None = None
        self._active_refs: dict[Any, str] = {}

    async def run(
        self,
        jobs: list[RayActorJob],
        *,
        operation: str,
        call_timeout_s: float,
        worker_methods: Mapping[str, Any] | None = None,
        schedule: list[dict[str, Any]] | None = None,
    ) -> list[tuple[int, Any]]:
        """Run jobs with cross-request actor admission and ordered results."""

        if not operation:
            raise ValueError("Ray actor operation must not be empty")
        call_timeout_s = validate_ray_timeout(
            call_timeout_s,
            name="call_timeout_s",
        )
        self._require_open()
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
        ref_to_job: dict[Any, tuple[int, str]] = {}
        result_pairs: list[tuple[int, Any]] = []
        pool_started = time.perf_counter()
        submit_time_by_ref: dict[Any, float] = {}
        deadline_by_ref: dict[Any, RayCallDeadline] = {}

        def dispatch_worker_for(job: RayActorJob) -> tuple[str, Any] | None:
            if job.worker_id is not None:
                worker_id = self._try_acquire((job.worker_id,))
                if worker_id is None:
                    return None
                return worker_id, job.remote_method
            candidates = tuple((worker_methods or {}).keys())
            worker_id = self._try_acquire(candidates)
            if worker_id is None:
                return None
            return worker_id, worker_methods[worker_id]  # type: ignore[index]

        def submit_ready() -> list[Any]:
            """Submit every job with a real free worker slot."""

            new_refs: list[Any] = []
            made_progress = True
            while pending and made_progress:
                made_progress = False
                for _ in range(len(pending)):
                    job = pending.popleft()
                    dispatch = dispatch_worker_for(job)
                    if dispatch is None:
                        pending.append(job)
                        continue
                    worker_id, method = dispatch
                    deadline = RayCallDeadline(
                        operation,
                        call_timeout_s,
                        context=f"worker_id={worker_id}, job_index={job.job_index}",
                    )
                    try:
                        ref = method(job.payload, **job.keyword_args)
                    except BaseException as cause:
                        self._release(worker_id)
                        if not ref_to_job:
                            raise
                        error = RayActorCallError(
                            operation,
                            worker_id=worker_id,
                            job_index=job.job_index,
                        )
                        terminal, doomed = self._close(error)
                        cancel_ray_refs(None, doomed, root_error=terminal)
                        if terminal is not error:
                            self._require_open()
                        raise error from cause
                    try:
                        self._register(ref, worker_id)
                    except BaseException as cause:
                        error = RayActorCallError(
                            operation,
                            worker_id=worker_id,
                            job_index=job.job_index,
                        )
                        terminal, doomed = self._close(error)
                        if ref not in doomed:
                            doomed = (*doomed, ref)
                        cancel_ray_refs(None, doomed, root_error=terminal)
                        if terminal is not error:
                            self._require_open()
                        raise error from cause
                    ref_to_job[ref] = (job.job_index, worker_id)
                    if schedule is not None:
                        submit_time_by_ref[ref] = time.perf_counter()
                    deadline_by_ref[ref] = deadline
                    new_refs.append(ref)
                    made_progress = True
            return new_refs

        async def await_ref(ref: Any) -> tuple[Any, Any]:
            return ref, await ref

        # task -> ref, so a completed wrapper maps back to its job/telemetry.
        waiters: dict[asyncio.Future[Any], Any] = {}

        def spawn(refs: list[Any]) -> None:
            for ref in refs:
                waiters[asyncio.ensure_future(await_ref(ref))] = ref

        admission_waiter: asyncio.Task[None] | None = None
        try:
            spawn(submit_ready())
            while waiters or pending:
                spawn(submit_ready())
                if pending and admission_waiter is None:
                    admission_waiter = asyncio.create_task(
                        self._wait_for_available(self._pending_workers(pending, worker_methods)),
                    )

                waiting: set[asyncio.Future[Any] | asyncio.Task[None]] = set(waiters)
                if admission_waiter is not None:
                    waiting.add(admission_waiter)
                wait_timeout_s = (
                    min(deadline_by_ref[ref].remaining_s() for ref in waiters.values())
                    if waiters
                    else None
                )
                done, _ = await asyncio.wait(
                    waiting,
                    timeout=wait_timeout_s,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    expired_ref = min(
                        waiters.values(),
                        key=lambda ref: deadline_by_ref[ref].expires_at,
                    )
                    error = deadline_by_ref[expired_ref].timeout_error()
                    terminal, doomed = self._close(error)
                    cancel_ray_refs(None, doomed, root_error=terminal)
                    if terminal is not error:
                        self._require_open()
                    raise error

                if admission_waiter is not None and admission_waiter in done:
                    done.remove(admission_waiter)
                    admission_waiter.result()
                    admission_waiter = None

                for task in done:
                    ref = waiters.pop(task)
                    job_index, worker_id = ref_to_job.pop(ref)
                    deadline_by_ref.pop(ref)
                    try:
                        _, result = task.result()
                    except BaseException as cause:
                        self._finish(ref, worker_id)
                        if self._terminal_error is not None:
                            self._require_open()
                        error = RayActorCallError(
                            operation,
                            worker_id=worker_id,
                            job_index=job_index,
                        )
                        terminal, doomed = self._close(error)
                        cancel_ray_refs(None, doomed, root_error=terminal)
                        raise error from cause
                    self._finish(ref, worker_id)
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
            self._require_open()
        except asyncio.CancelledError as cancellation:
            unresolved = {task: ref for task, ref in waiters.items() if not task.done()}
            first_failure: tuple[int, str, BaseException] | None = None
            for task, ref in waiters.items():
                if not task.done():
                    continue
                job_index, worker_id = ref_to_job.pop(ref)
                deadline_by_ref.pop(ref)
                self._finish(ref, worker_id)
                try:
                    failure = task.exception()
                except asyncio.CancelledError as error:
                    failure = error
                if failure is not None and first_failure is None:
                    first_failure = (job_index, worker_id, failure)

            terminal = self._terminal_error
            if terminal is None and first_failure is not None:
                job_index, worker_id, failure = first_failure
                proposed = RayActorCallError(
                    operation,
                    worker_id=worker_id,
                    job_index=job_index,
                )
                proposed.__cause__ = failure
                terminal, doomed = self._close(proposed)
                cancel_ray_refs(None, doomed, root_error=terminal)
            elif terminal is None and unresolved:
                proposed = RayOperationCancelled(
                    operation,
                    context=f"submitted_refs={len(unresolved)}",
                )
                terminal, doomed = self._close(proposed)
                cancel_ray_refs(None, doomed, root_error=terminal)
            if terminal is not None:
                raise cancellation from terminal
            raise
        finally:
            if admission_waiter is not None:
                admission_waiter.cancel()
            for task in waiters:
                task.cancel()
            local_tasks = [*waiters]
            if admission_waiter is not None:
                local_tasks.append(admission_waiter)
            if local_tasks:
                await asyncio.gather(*local_tasks, return_exceptions=True)

        return sorted(result_pairs, key=lambda pair: pair[0])

    def _try_acquire(self, candidates: tuple[str, ...]) -> str | None:
        self._require_open()
        self._validate_workers(candidates)
        for worker_id in candidates:
            if worker_id in self._available:
                self._available.remove(worker_id)
                return worker_id
        return None

    def _register(self, ref: Any, worker_id: str) -> None:
        self._require_open()
        if ref in self._active_refs:
            raise RuntimeError("Ray actor dispatcher received a duplicate ObjectRef")
        self._active_refs[ref] = worker_id

    def _finish(self, ref: Any, worker_id: str) -> None:
        self._active_refs.pop(ref, None)
        if self._terminal_error is None:
            self._release(worker_id)

    def _release(self, worker_id: str) -> None:
        if self._terminal_error is not None:
            return
        if worker_id in self._available:
            raise RuntimeError(f"Ray actor worker {worker_id!r} slot released twice")
        self._available.add(worker_id)
        self._changed.set()

    def _close(
        self,
        error: TerminalRuntimeError,
    ) -> tuple[TerminalRuntimeError, tuple[Any, ...]]:
        if self._terminal_error is None:
            self._terminal_error = error
            self._available.clear()
            doomed = tuple(self._active_refs)
            self._active_refs.clear()
            self._changed.set()
            return error, doomed
        return self._terminal_error, ()

    def _require_open(self) -> None:
        if self._terminal_error is not None:
            raise RuntimeError(
                "Ray actor dispatcher is terminally closed"
            ) from self._terminal_error

    async def _wait_for_available(self, candidates: tuple[str, ...]) -> None:
        self._validate_workers(candidates)
        while True:
            self._require_open()
            if any(worker_id in self._available for worker_id in candidates):
                return
            self._changed.clear()
            self._require_open()
            if any(worker_id in self._available for worker_id in candidates):
                continue
            await self._changed.wait()

    def _pending_workers(
        self,
        pending: deque[RayActorJob],
        worker_methods: Mapping[str, Any] | None,
    ) -> tuple[str, ...]:
        bound = {job.worker_id for job in pending if job.worker_id is not None}
        dynamic = any(job.worker_id is None for job in pending)
        return tuple(
            worker_id
            for worker_id in self._worker_ids
            if worker_id in bound or (dynamic and worker_id in (worker_methods or {}))
        )

    def _validate_workers(self, worker_ids: tuple[str, ...]) -> None:
        unknown = [worker_id for worker_id in worker_ids if worker_id not in self._worker_ids]
        if unknown:
            raise ValueError(f"unknown Ray actor worker ids: {unknown}")


__all__ = ["RayActorCallError", "RayActorDispatcher", "RayActorJob"]
