"""Bounded Ray actor submit / wait / gather helpers."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
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


@dataclass(slots=True)
class _AdmissionWaiter:
    """One caller's independent FIFO position on each compatible worker."""

    candidates: tuple[str, ...] = ()


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

    The dispatcher belongs to the actor fleet, not to one request or operation.
    Keeping its admission state across generation and weight-sync calls prevents
    either caller from pre-queueing work in a synchronous actor mailbox before
    its own operation deadline starts.
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
        self._admission_queues = {worker_id: deque[_AdmissionWaiter]() for worker_id in worker_ids}

    @property
    def worker_ids(self) -> tuple[str, ...]:
        """Return the fleet identity whose default-group slots are owned here."""

        return self._worker_ids

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
        bound_worker_ids = tuple(job.worker_id for job in jobs if job.worker_id is not None)
        self._validate_workers(bound_worker_ids)
        invalid_bound_methods = [
            job.job_index
            for job in jobs
            if job.worker_id is not None and not callable(job.remote_method)
        ]
        if invalid_bound_methods:
            raise ValueError(
                f"bound Ray actor jobs require callable remote methods: {invalid_bound_methods}",
            )
        if worker_methods is not None:
            self._validate_workers(tuple(worker_methods))
            invalid_worker_methods = [
                worker_id for worker_id, method in worker_methods.items() if not callable(method)
            ]
            if invalid_worker_methods:
                raise ValueError(
                    "pull-based Ray actor dispatch requires callable worker methods: "
                    f"{invalid_worker_methods}",
                )

        # Stable sort: equal priorities (the static path always submits 0.0)
        # preserve caller order bit-for-bit.
        pending = deque(sorted(jobs, key=lambda job: -job.priority))
        ref_to_job: dict[Any, tuple[int, str]] = {}
        result_pairs: list[tuple[int, Any]] = []
        pool_started = time.perf_counter()
        submit_time_by_ref: dict[Any, float] = {}
        deadline_by_ref: dict[Any, RayCallDeadline] = {}
        admission = _AdmissionWaiter()
        submitted_count = 0

        def dispatch_worker_for(job: RayActorJob) -> tuple[str, Any] | None:
            if job.worker_id is not None:
                worker_id = self._try_acquire(
                    (job.worker_id,),
                    waiter=admission,
                )
                if worker_id is None:
                    return None
                return worker_id, job.remote_method
            candidates = tuple((worker_methods or {}).keys())
            worker_id = self._try_acquire(candidates, waiter=admission)
            if worker_id is None:
                return None
            return worker_id, worker_methods[worker_id]  # type: ignore[index]

        def submit_ready() -> list[Any]:
            """Submit compatible jobs until no real worker slot remains."""

            nonlocal submitted_count
            new_refs: list[Any] = []
            failed_before_submission = 0
            for _ in range(len(pending)):
                job = pending.popleft()
                dispatch = dispatch_worker_for(job)
                if dispatch is None:
                    pending.append(job)
                    failed_before_submission += 1
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
                    if submitted_count == 0:
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
                submitted_count += 1
                if schedule is not None:
                    submit_time_by_ref[ref] = time.perf_counter()
                deadline_by_ref[ref] = deadline
                new_refs.append(ref)
                if not self._available:
                    # No later job in this synchronous pass can acquire a slot.
                    # Restore failed jobs ahead of the untouched suffix.
                    pending.rotate(failed_before_submission)
                    break
            return new_refs

        async def await_ref(ref: Any) -> tuple[Any, Any]:
            return ref, await ref

        # task -> ref, so a completed wrapper maps back to its job/telemetry.
        waiters: dict[asyncio.Future[Any], Any] = {}

        def spawn(refs: list[Any]) -> None:
            for ref in refs:
                waiters[asyncio.ensure_future(await_ref(ref))] = ref

        def finish_success(
            ref: Any,
            *,
            job_index: int,
            worker_id: str,
            result: Any,
        ) -> None:
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

        admission_task: asyncio.Task[None] | None = None
        try:
            while waiters or pending:
                spawn(submit_ready())
                candidates: tuple[str, ...] = ()
                if pending:
                    candidates = self._pending_workers(pending, worker_methods)
                    active_workers = {worker_id for _job_index, worker_id in ref_to_job.values()}
                    candidates = tuple(
                        worker_id for worker_id in candidates if worker_id not in active_workers
                    )
                self._sync_admission_waiter(candidates, waiter=admission)
                if candidates and admission_task is None:
                    admission_task = asyncio.create_task(
                        self._wait_for_available(admission),
                    )

                waiting: set[asyncio.Future[Any] | asyncio.Task[None]] = set(waiters)
                if admission_task is not None:
                    waiting.add(admission_task)
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

                if admission_task is not None and admission_task in done:
                    done.remove(admission_task)
                    admission_task.result()
                    admission_task = None

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
                    finish_success(
                        ref,
                        job_index=job_index,
                        worker_id=worker_id,
                        result=result,
                    )
            self._require_open()
        except asyncio.CancelledError as cancellation:
            first_failure: tuple[int, str, BaseException] | None = None
            for task, ref in waiters.items():
                if not task.done():
                    continue
                job_index, worker_id = ref_to_job.pop(ref)
                deadline_by_ref.pop(ref)
                try:
                    _, result = task.result()
                except asyncio.CancelledError as error:
                    failure = error
                except BaseException as error:
                    failure = error
                else:
                    finish_success(
                        ref,
                        job_index=job_index,
                        worker_id=worker_id,
                        result=result,
                    )
                    continue
                self._finish(ref, worker_id)
                if first_failure is None:
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
            elif terminal is None and submitted_count and (ref_to_job or pending):
                proposed = RayOperationCancelled(
                    operation,
                    context=(f"submitted_jobs={submitted_count}, pending_jobs={len(pending)}"),
                )
                terminal, doomed = self._close(proposed)
                cancel_ray_refs(None, doomed, root_error=terminal)
            if terminal is not None:
                raise cancellation from terminal
            if not (
                not pending
                and not ref_to_job
                and len(result_pairs) == submitted_count == len(jobs)
            ):
                raise
            # Every actor call completed successfully before cancellation won
            # the driver race. Complete the transport transaction so callers
            # still run ACK validation and publish any resulting state.
            self._require_open()
        finally:
            self._remove_admission_waiter(admission)
            if admission_task is not None:
                admission_task.cancel()
            for task in waiters:
                task.cancel()
            local_tasks = [*waiters]
            if admission_task is not None:
                local_tasks.append(admission_task)
            if local_tasks:
                await asyncio.gather(*local_tasks, return_exceptions=True)

        return sorted(result_pairs, key=lambda pair: pair[0])

    async def run_one(
        self,
        job: RayActorJob,
        *,
        operation: str,
        call_timeout_s: float,
        await_result: Callable[[Any, RayCallDeadline], Awaitable[Any]],
    ) -> Any:
        """Run one bound job whose waiter owns a progress-aware deadline.

        ``run`` owns ordinary fixed-deadline calls. This seam keeps the same
        fleet admission and terminal state while allowing a protocol-specific
        waiter to replace its deadline only after verified progress.
        """

        if not operation:
            raise ValueError("Ray actor operation must not be empty")
        call_timeout_s = validate_ray_timeout(
            call_timeout_s,
            name="call_timeout_s",
        )
        if job.worker_id is None or job.remote_method is None:
            raise ValueError("run_one requires a bound worker and remote method")
        if not callable(job.remote_method):
            raise ValueError("run_one requires a callable remote method")
        if not callable(await_result):
            raise ValueError("run_one requires a callable result waiter")
        self._require_open()
        worker_id = job.worker_id
        admission = _AdmissionWaiter()
        while self._try_acquire((worker_id,), waiter=admission) is None:
            self._sync_admission_waiter(
                (worker_id,),
                waiter=admission,
            )
            await self._wait_for_available(admission)

        # Start the initial budget immediately before submission, after local
        # admission. It therefore includes serialization, transport, and actor
        # execution, but not legitimate waiting behind an earlier fleet call.
        deadline = RayCallDeadline(
            operation,
            call_timeout_s,
            context=f"worker_id={worker_id}, job_index={job.job_index}",
        )
        try:
            ref = job.remote_method(job.payload, **job.keyword_args)
        except BaseException:
            self._release(worker_id)
            raise
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

        async def wait_for_result() -> Any:
            return await await_result(ref, deadline)

        waiter = asyncio.create_task(wait_for_result())
        try:
            result = await asyncio.shield(waiter)
        except asyncio.CancelledError as cancellation:
            if waiter.done():
                try:
                    waiter.result()
                except asyncio.CancelledError as cause:
                    proposed = RayOperationCancelled(
                        operation,
                        context=f"worker_id={worker_id}, job_index={job.job_index}",
                    )
                    nested = cause.__cause__
                    if isinstance(nested, TerminalRuntimeError):
                        proposed = nested
                    terminal, doomed = self._close(proposed)
                    cancel_ray_refs(None, doomed, root_error=terminal)
                    raise cancellation from terminal
                except BaseException as cause:
                    proposed = (
                        cause
                        if isinstance(cause, TerminalRuntimeError)
                        else RayActorCallError(
                            operation,
                            worker_id=worker_id,
                            job_index=job.job_index,
                        )
                    )
                    if proposed is not cause:
                        proposed.__cause__ = cause
                    terminal, doomed = self._close(proposed)
                    cancel_ray_refs(None, doomed, root_error=terminal)
                    raise cancellation from terminal
                else:
                    # The actor completed before cancellation won the driver
                    # race. Its state is known, so release the physical slot and
                    # propagate an ordinary caller cancellation.
                    self._finish(ref, worker_id)
                    terminal = self._terminal_error
                    if terminal is not None:
                        raise cancellation from terminal
                    raise

            proposed = RayOperationCancelled(
                operation,
                context=f"worker_id={worker_id}, job_index={job.job_index}",
            )
            terminal, doomed = self._close(proposed)
            cancel_ray_refs(None, doomed, root_error=terminal)
            raise cancellation from terminal
        except BaseException as cause:
            error = (
                cause
                if isinstance(cause, TerminalRuntimeError)
                else RayActorCallError(
                    operation,
                    worker_id=worker_id,
                    job_index=job.job_index,
                )
            )
            terminal, doomed = self._close(error)
            cancel_ray_refs(None, doomed, root_error=terminal)
            if terminal is not error:
                self._require_open()
            if error is cause:
                raise
            raise error from cause
        else:
            self._finish(ref, worker_id)
            self._require_open()
            return result
        finally:
            if not waiter.done():
                waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)

    def _try_acquire(
        self,
        candidates: tuple[str, ...],
        *,
        waiter: _AdmissionWaiter,
    ) -> str | None:
        self._require_open()
        self._validate_workers(candidates)
        for worker_id in candidates:
            if worker_id not in self._available:
                continue
            first_waiter = self._first_waiter_for(worker_id)
            if first_waiter is not None and first_waiter is not waiter:
                continue
            self._remove_admission_waiter(waiter)
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
            for queue in self._admission_queues.values():
                queue.clear()
            self._changed.set()
            return error, doomed
        return self._terminal_error, ()

    def _require_open(self) -> None:
        if self._terminal_error is not None:
            raise RuntimeError(
                "Ray actor dispatcher is terminally closed"
            ) from self._terminal_error

    async def _wait_for_available(
        self,
        waiter: _AdmissionWaiter,
    ) -> None:
        try:
            while True:
                self._require_open()
                candidates = waiter.candidates
                if not candidates:
                    return
                if self._can_acquire(candidates, waiter=waiter):
                    return
                self._changed.clear()
                self._require_open()
                candidates = waiter.candidates
                if not candidates:
                    return
                if self._can_acquire(candidates, waiter=waiter):
                    continue
                await self._changed.wait()
        except BaseException:
            self._remove_admission_waiter(waiter)
            raise

    def _sync_admission_waiter(
        self,
        candidates: tuple[str, ...],
        *,
        waiter: _AdmissionWaiter,
    ) -> None:
        self._validate_workers(candidates)
        if candidates == waiter.candidates:
            return
        previous = set(waiter.candidates)
        desired = set(candidates)
        for worker_id in previous - desired:
            self._admission_queues[worker_id].remove(waiter)
        for worker_id in candidates:
            if worker_id not in previous:
                self._admission_queues[worker_id].append(waiter)
        waiter.candidates = candidates
        self._changed.set()

    def _can_acquire(
        self,
        candidates: tuple[str, ...],
        *,
        waiter: _AdmissionWaiter,
    ) -> bool:
        for worker_id in candidates:
            if worker_id not in self._available:
                continue
            first_waiter = self._first_waiter_for(worker_id)
            if first_waiter is None or first_waiter is waiter:
                return True
        return False

    def _first_waiter_for(self, worker_id: str) -> _AdmissionWaiter | None:
        queue = self._admission_queues[worker_id]
        return queue[0] if queue else None

    def _remove_admission_waiter(self, waiter: _AdmissionWaiter) -> None:
        if not waiter.candidates:
            return
        for worker_id in waiter.candidates:
            queue = self._admission_queues[worker_id]
            if waiter in queue:
                queue.remove(waiter)
        waiter.candidates = ()
        self._changed.set()

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
