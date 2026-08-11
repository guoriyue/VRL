"""Ray-backed generation executor that gathers chunk results.

The per-request slice of the Ray adapter: split one ``EnginePlan`` across the
worker fleet, await the chunk RPCs (with stall deadlines and pipelined
progress probing), and reassemble outputs through the model-free gatherer.
It deliberately owns no lifecycle — admission, terminal failure, and shutdown
belong to ``RayGenerationRuntime``, while the live actors and this executor
are held together by ``RayGenerationSession``.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from typing import Any

from vrl.generation.execution.chunk_memory import build_chunk_memory_shadow
from vrl.generation.execution.chunk_placement import DistributedExecutionPlanner
from vrl.generation.execution.ids import build_sample_rows
from vrl.generation.execution.planner import EnginePlan
from vrl.generation.execution.types import (
    ChunkExecutionEnvelope,
    ChunkExecutionResult,
    ChunkSizeProbeResult,
    PipelinedRequestOutOfMemory,
    StaleSlotDiscard,
)
from vrl.generation.protocols import ChunkGatherer, ChunkResult
from vrl.generation.ray.pipeline_protocol import (
    PipelinedProgressError,
    PipelinedRequestProgress,
)
from vrl.generation.types import GenerationOutput, GenerationRequest, GenerationSampleRow
from vrl.ray.actor_group import RayActorHandle
from vrl.ray.actor_pool import RayActorDispatcher, RayActorJob
from vrl.ray.operation_deadline import (
    RayCallDeadline,
    cancel_ray_refs,
)
from vrl.utils.deadline import validate_timeout

logger = logging.getLogger(__name__)

# Bounds progress-probe traffic on the shared health concurrency group. This is
# a wire cadence, not a user-facing generation SLA.
_PIPELINED_PROGRESS_POLL_INTERVAL_S = 1.0


class RayGenerationExecutor:
    """Execute one GenerationRequest across generation workers."""

    def __init__(
        self,
        planner: DistributedExecutionPlanner,
        workers: list[RayActorHandle],
        gatherer: ChunkGatherer,
        *,
        actor_dispatcher: RayActorDispatcher,
        generation_stall_timeout_s: float,
        pipelined: bool = False,
    ) -> None:
        if not workers:
            raise ValueError("RayGenerationExecutor requires at least one worker")
        if pipelined and len(workers) != 1:
            raise ValueError(
                "pipelined Ray generation requires exactly one rollout worker; "
                f"received {len(workers)}. Per-worker request pipelining is not "
                "implemented.",
            )
        self.planner = planner
        self.workers = list(workers)
        self.gatherer = gatherer
        expected_worker_ids = tuple(worker.worker_id for worker in self.workers)
        if actor_dispatcher.worker_ids != expected_worker_ids:
            raise ValueError(
                "RayGenerationExecutor actor dispatcher does not own its worker fleet: "
                f"{actor_dispatcher.worker_ids} != {expected_worker_ids}",
            )
        self.actor_dispatcher = actor_dispatcher
        self.generation_stall_timeout_s = validate_timeout(
            generation_stall_timeout_s,
            name="generation_stall_timeout_s",
        )
        # Config validation rejects multi-worker use; this constructor repeats the
        # guard for callers that construct executors directly.
        self.pipelined = bool(pipelined)
        # A synchronous single-worker actor cannot execute two pipelined requests
        # concurrently. Queue them on the driver so a later request does not spend
        # its stall budget waiting behind an earlier, legitimately long request.
        self._pipelined_request_lock = asyncio.Lock() if self.pipelined else None

    async def execute(self, request: GenerationRequest) -> GenerationOutput:
        """Execute with single-flight admission for the pipelined worker."""

        lock = self._pipelined_request_lock
        if lock is None:
            return await self._execute(request)
        async with lock:
            return await self._execute(request)

    async def probe_chunk_sizes(
        self,
        request: GenerationRequest,
        *,
        max_samples: int,
    ) -> list[ChunkSizeProbeResult]:
        """Probe every worker through the same actor admission as generation."""

        lock = self._pipelined_request_lock
        if lock is None:
            return await self._probe_chunk_sizes(request, max_samples=max_samples)
        async with lock:
            return await self._probe_chunk_sizes(request, max_samples=max_samples)

    async def _probe_chunk_sizes(
        self,
        request: GenerationRequest,
        *,
        max_samples: int,
    ) -> list[ChunkSizeProbeResult]:
        result_pairs: list[tuple[int, Any]] = []
        remote_jobs: list[RayActorJob] = []
        for job_index, worker in enumerate(self.workers):
            actor = worker.actor
            probe = getattr(actor, "probe_chunk_size", None)
            if probe is None:
                raise RuntimeError(
                    f"worker {worker.worker_id!r} does not support the "
                    "chunk-size probe required by samples_per_chunk: auto",
                )
            remote = getattr(probe, "remote", None)
            if callable(remote):
                remote_jobs.append(
                    RayActorJob(
                        job_index=job_index,
                        worker_id=worker.worker_id,
                        remote_method=remote,
                        payload=request,
                        keyword_args={"max_samples": max_samples},
                    ),
                )
            else:
                result_pairs.append(
                    (job_index, probe(request, max_samples=max_samples)),
                )
        if remote_jobs:
            result_pairs.extend(
                await self.actor_dispatcher.run(
                    remote_jobs,
                    operation="rollout.generation.chunk_size_probe",
                    call_timeout_s=self.generation_stall_timeout_s,
                ),
            )
        results = [result for _, result in sorted(result_pairs, key=lambda pair: pair[0])]
        for worker, result in zip(self.workers, results, strict=True):
            if not isinstance(result, ChunkSizeProbeResult):
                raise TypeError(
                    f"worker {worker.worker_id!r} returned invalid chunk-size probe "
                    f"result {type(result).__name__}",
                )
        return results

    async def _execute(self, request: GenerationRequest) -> GenerationOutput:
        import time

        from vrl.utils.profiling import profile_range

        _gen_start = time.perf_counter()
        sample_rows = build_sample_rows(request)
        with profile_range("engine.plan"):
            generation_plan = self.planner.plan_with_engine(
                request,
                tuple(worker.worker_id for worker in self.workers),
            )
        assignments = list(generation_plan.assignments)
        engine_plan = generation_plan.engine_plan
        pipelined_oom: PipelinedRequestOutOfMemory | None = None
        if self.pipelined and len(engine_plan.chunks) >= 2:
            pipelined_result = await self._execute_request_pipelined(
                request,
                engine_plan,
                sample_rows,
            )
            if isinstance(pipelined_result, GenerationOutput):
                logger.info(
                    "generation wall: path=per_request_pipelined chunks=%d wall_s=%.3f",
                    len(engine_plan.chunks),
                    time.perf_counter() - _gen_start,
                )
                return pipelined_result
            pipelined_oom = pipelined_result
            logger.warning(
                "pipelined generation request %s OOMed on worker %s; retrying "
                "through per-chunk split admission: %s",
                request.request_id,
                pipelined_oom.worker_id,
                pipelined_oom.error,
            )
        worker_by_id = {worker.worker_id: worker for worker in self.workers}
        strategy = self.planner.strategy
        runtime_debug_on = request.runtime_debug
        remote_jobs: list[RayActorJob] = []
        result_pairs: list[tuple[int, ChunkExecutionResult]] = []
        schedule_rows: list[dict[str, Any]] = []

        for job_index, assignment in enumerate(assignments):
            if assignment.worker_id is None:
                # Dynamic placement: binding happens in the actor pool. The
                # estimated cost becomes the submission priority (LPT).
                remote_jobs.append(
                    RayActorJob(
                        job_index=job_index,
                        worker_id=None,
                        remote_method=None,
                        payload=assignment.envelope,
                        priority=assignment.estimated_cost,
                    ),
                )
                continue
            worker = worker_by_id[assignment.worker_id]
            actor = worker.actor
            execute_chunk = actor.execute_chunk
            remote = getattr(execute_chunk, "remote", None)
            if callable(remote):
                remote_jobs.append(
                    RayActorJob(
                        job_index=job_index,
                        worker_id=worker.worker_id,
                        remote_method=remote,
                        payload=assignment.envelope,
                    ),
                )
            else:
                result_pairs.append(
                    (job_index, execute_chunk(assignment.envelope)),
                )

        if remote_jobs:
            worker_methods = None
            if any(job.worker_id is None for job in remote_jobs):
                worker_methods = self._remote_worker_methods()
            result_pairs.extend(
                await self.actor_dispatcher.run(
                    remote_jobs,
                    operation="rollout.generation.chunk",
                    call_timeout_s=self.generation_stall_timeout_s,
                    worker_methods=worker_methods,
                    schedule=schedule_rows if runtime_debug_on else None,
                ),
            )

        results = [result for _, result in sorted(result_pairs, key=lambda pair: pair[0])]

        if len(results) != len(assignments):
            raise RuntimeError(
                "distributed rollout returned wrong number of chunks: "
                f"{len(results)} != {len(assignments)}",
            )

        envelope_by_chunk_key = {
            assignment.envelope.chunk_key: assignment.envelope for assignment in assignments
        }
        for result in results:
            _require_correlated_result(result, envelope_by_chunk_key)

        # A stale-slot result is a typed graceful discard, not a failure: the
        # request's policy version was evicted from its worker's slot window under
        # a non-draining weight sync. Route it BEFORE OOM-degrade (which hard-raises
        # on any non-OOM error) and before the version assert (a stale slot stamps
        # request.policy_version, so it would pass that check silently). Raising a
        # distinct StaleSlotDiscard lets the producer count it as a stale discard
        # instead of a collect error. One evicted chunk poisons the whole request,
        # so the group is discarded — partial mixed-version output is never built.
        stale = [result for result in results if result.stale_slot]
        if stale:
            evicted = stale[0]
            raise StaleSlotDiscard(
                "distributed rollout discarded a stale trainable-state slot "
                f"(worker_id={evicted.worker_id}, "
                f"policy_version={evicted.policy_version}, "
                f"chunks={len(stale)}/{len(results)}): {evicted.error}",
            )

        results, oom_splits = await self._degrade_oom_chunks(
            results,
            envelope_by_chunk_key=envelope_by_chunk_key,
            worker_by_id=worker_by_id,
        )

        for result in results:
            if (
                request.policy_version is not None
                and result.policy_version != request.policy_version
            ):
                raise RuntimeError(
                    "distributed rollout policy_version mismatch "
                    f"(worker_id={result.worker_id}, "
                    f"expected={request.policy_version}, "
                    f"actual={result.policy_version})",
                )

        chunk_outputs: list[ChunkResult] = []
        for result in results:
            if result.output is None:
                raise RuntimeError(
                    f"distributed rollout chunk returned no output: {result}",
                )
            chunk_outputs.append(result.output)

        output = self.gatherer.gather_chunks(request, sample_rows, chunk_outputs)
        # Raw per-chunk memory readings (drift monitor for the startup
        # chunk-size probe). Log-only provenance; nothing here changes chunk sizing.
        memory_shadow = build_chunk_memory_shadow(
            results,
        )
        if memory_shadow:
            for row in memory_shadow:
                logger.info(
                    "chunk memory: chunk=%s n=%d peak=%.0fMB "
                    "(denoise=%.0fMB decode=%.0fMB baseline=%.0fMB) "
                    "budget=%.0fMB non_torch=%.0fMB",
                    row["chunk_key"],
                    row["sample_count"],
                    row["peak_bytes"] / 2**20,
                    row["denoise_peak_bytes"] / 2**20,
                    row["decode_peak_bytes"] / 2**20,
                    row["baseline_allocated_bytes"] / 2**20,
                    row["budget_bytes"] / 2**20,
                    row["non_torch_bytes"] / 2**20,
                )
        schedule_summary: list[dict[str, Any]] = []
        if schedule_rows:
            by_index = {row["job_index"]: row for row in schedule_rows}
            schedule_summary = [
                {
                    "chunk_key": assignment.chunk.chunk_key,
                    "sample_count": assignment.chunk.sample_count,
                    "assignment_strategy": strategy,
                    "estimated_cost": assignment.estimated_cost,
                    "assigned_worker": by_index[job_index]["worker_id"],
                    "queue_wait_s": by_index[job_index]["queue_wait_s"],
                    "execution_s": by_index[job_index]["execution_s"],
                }
                for job_index, assignment in enumerate(assignments)
                if job_index in by_index
            ]
        if runtime_debug_on:
            for row in schedule_summary:
                logger.info(
                    "ray chunk schedule [%s]: chunk=%s worker=%s samples=%d "
                    "cost=%.0f queue_wait=%.3fs exec=%.3fs",
                    strategy,
                    row["chunk_key"],
                    row["assigned_worker"],
                    row["sample_count"],
                    row["estimated_cost"],
                    row["queue_wait_s"],
                    row["execution_s"],
                )
        worker_debug_rows: list[dict[str, Any]] = []
        if runtime_debug_on:
            for result in results:
                worker = worker_by_id[result.worker_id]
                worker_debug_rows.append(
                    {
                        "worker_id": result.worker_id,
                        "node_ip": worker.node_ip,
                        "gpu_ids": list(worker.gpu_ids),
                        "policy_version": result.policy_version,
                        "chunk_key": result.chunk.chunk_key,
                        **result.metrics,
                    },
                )
        debug_payload: dict[str, Any] = {}
        if worker_debug_rows:
            debug_payload["ray_chunks"] = worker_debug_rows
        if runtime_debug_on and schedule_summary:
            debug_payload["chunk_schedule"] = schedule_summary
        if runtime_debug_on and oom_splits:
            debug_payload["chunk_oom_splits"] = oom_splits
        if debug_payload:
            output.runtime_debug = debug_payload
        logger.info(
            "generation wall: path=%s chunks=%d wall_s=%.3f",
            (
                "per_chunk_dispatch_after_pipelined_oom"
                if pipelined_oom is not None
                else "per_chunk_dispatch"
            ),
            len(assignments),
            time.perf_counter() - _gen_start,
        )
        return output

    async def _execute_request_pipelined(
        self,
        request: GenerationRequest,
        engine_plan: EnginePlan,
        sample_rows: list[GenerationSampleRow],
    ) -> GenerationOutput | PipelinedRequestOutOfMemory:
        """Single-worker stage-overlap path (opt-in, ``pipelined=True``):
        the whole request's chunks run software-pipelined on one worker
        (``forward_plan_pipelined``), returning the already-gathered
        GenerationOutput. The pipeline keeps depth 1 (about two chunks resident),
        so a typed OOM response falls back to the normal per-chunk dispatch and
        split admission path. Version safety is enforced in the worker (slot
        activation / StaleSlotDiscard); a stale request raises and is counted as a
        graceful discard upstream, never trained off-policy."""

        worker = self.workers[0]
        actor = worker.actor
        call = actor.execute_request_pipelined
        remote = getattr(call, "remote", None)
        if callable(remote):
            progress = getattr(actor, "pipelined_progress", None)
            progress_remote = getattr(progress, "remote", None)
            if not callable(progress_remote):
                raise PipelinedProgressError(
                    "pipelined Ray generation requires worker progress reporting",
                )

            async def await_pipelined_result(
                result_ref: Any,
                initial_deadline: RayCallDeadline,
            ) -> Any:
                try:
                    return await self._await_pipelined_result(
                        result_ref=result_ref,
                        progress_remote=progress_remote,
                        request_id=request.request_id,
                        total_chunks=len(engine_plan.chunks),
                        initial_deadline=initial_deadline,
                    )
                except StaleSlotDiscard as error:
                    # A stale version is a known, graceful business outcome. Let
                    # the fleet dispatcher release the actor slot before the
                    # public executor re-raises the typed discard.
                    return error

            result = await self.actor_dispatcher.run_one(
                RayActorJob(
                    job_index=0,
                    worker_id=worker.worker_id,
                    remote_method=remote,
                    payload=request,
                    keyword_args={
                        "engine_plan": engine_plan,
                        "sample_rows": sample_rows,
                    },
                ),
                operation="rollout.generation.pipelined",
                call_timeout_s=self.generation_stall_timeout_s,
                await_result=await_pipelined_result,
            )
        else:
            result = call(request, engine_plan, sample_rows)
        if isinstance(result, StaleSlotDiscard):
            raise result
        if not isinstance(result, (GenerationOutput, PipelinedRequestOutOfMemory)):
            raise TypeError(
                f"pipelined generation worker returned unsupported result {type(result).__name__}",
            )
        if result.request_id != request.request_id:
            raise RuntimeError(
                "pipelined generation request_id mismatch: "
                f"{result.request_id!r} != {request.request_id!r}",
            )
        if (
            isinstance(result, PipelinedRequestOutOfMemory)
            and result.worker_id != worker.worker_id
        ):
            raise RuntimeError(
                "pipelined generation worker_id mismatch: "
                f"{result.worker_id!r} != {worker.worker_id!r}",
            )
        return result

    async def _await_pipelined_result(
        self,
        *,
        result_ref: Any,
        progress_remote: Any,
        request_id: str,
        total_chunks: int,
        initial_deadline: RayCallDeadline,
    ) -> Any:
        """Reset one stall deadline only when the worker completes a new chunk."""

        deadline = initial_deadline
        # Reserve at least half of even a very short stall budget for the first
        # progress RPC. The one-second constant remains the steady-state traffic
        # ceiling; this derived cadence is fixed for the request, so polling does
        # not accelerate into a busy loop as a deadline approaches.
        progress_poll_interval_s = min(
            _PIPELINED_PROGRESS_POLL_INTERVAL_S,
            initial_deadline.timeout_s / 2,
        )
        result_task = asyncio.ensure_future(result_ref)
        progress_task: asyncio.Future[Any] | None = None
        progress_ref: Any | None = None
        completed_chunks = 0
        try:
            while True:
                remaining_s = deadline.remaining_s()
                done, _ = await asyncio.wait(
                    {result_task},
                    timeout=min(progress_poll_interval_s, remaining_s),
                )
                if done:
                    return result_task.result()
                if deadline.remaining_s() <= 0:
                    raise deadline.timeout_error()

                progress_ref = progress_remote(request_id)
                progress_task = asyncio.ensure_future(progress_ref)
                done, _ = await asyncio.wait(
                    {result_task, progress_task},
                    timeout=deadline.remaining_s(),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    raise deadline.timeout_error()
                if result_task in done:
                    result = result_task.result()
                    if not progress_task.done():
                        failures = cancel_ray_refs(
                            None,
                            [progress_ref],
                            root_error=None,
                        )
                        if failures:
                            logger.warning(
                                "pipelined progress cancellation failed after "
                                "result completion: %r",
                                failures[0],
                            )
                    return result

                snapshot = progress_task.result()
                progress_task = None
                progress_ref = None
                if snapshot is None:
                    continue
                if not isinstance(snapshot, PipelinedRequestProgress):
                    raise PipelinedProgressError(
                        f"pipelined worker returned invalid progress {type(snapshot).__name__}",
                    )
                if snapshot.request_id != request_id:
                    raise PipelinedProgressError(
                        "pipelined progress request_id mismatch: "
                        f"{snapshot.request_id!r} != {request_id!r}",
                    )
                if snapshot.total_chunks != total_chunks:
                    raise PipelinedProgressError(
                        "pipelined progress total_chunks mismatch: "
                        f"{snapshot.total_chunks} != {total_chunks}",
                    )
                if snapshot.completed_chunks < completed_chunks:
                    raise PipelinedProgressError(
                        "pipelined progress regressed: "
                        f"{snapshot.completed_chunks} < {completed_chunks}",
                    )
                if snapshot.completed_chunks > completed_chunks:
                    completed_chunks = snapshot.completed_chunks
                    deadline = replace(initial_deadline)
        except asyncio.CancelledError as cancellation:
            if progress_ref is not None:
                cancel_ray_refs(None, [progress_ref], root_error=cancellation)
            raise
        except BaseException as error:
            if progress_ref is not None:
                cancel_ray_refs(None, [progress_ref], root_error=error)
            raise
        finally:
            if progress_task is not None and not progress_task.done():
                progress_task.cancel()
            if not result_task.done():
                result_task.cancel()
            await asyncio.gather(
                *(task for task in (result_task, progress_task) if task is not None),
                return_exceptions=True,
            )

    async def _degrade_oom_chunks(
        self,
        results: list[ChunkExecutionResult],
        *,
        envelope_by_chunk_key: dict[str, Any],
        worker_by_id: dict[str, RayActorHandle],
    ) -> tuple[list[ChunkExecutionResult], list[dict[str, Any]]]:
        """Split OOM chunks in half and re-run until success or single sample.

        The retry lives on the driver so vrl/ray stays chunk-agnostic, and the
        gatherer reassembles by (prompt_index, sample_start) metadata, so the
        extra child results need no positional bookkeeping. Children rebind to
        the worker that OOMed: the fleet-owned dispatcher exposes one real
        slot per synchronous actor, so the two halves run sequentially instead
        of landing concurrently on the GPU that just proved too full.
        """

        final: list[ChunkExecutionResult] = []
        pending = list(results)
        splits: list[dict[str, Any]] = []
        while pending:
            retry_jobs: list[RayActorJob] = []
            local_calls: list[tuple[Any, Any]] = []
            for result in pending:
                parent_envelope = envelope_by_chunk_key[result.chunk.chunk_key]
                if not result.error:
                    final.append(result)
                    continue
                chunk = result.chunk
                if not _is_oom_error(result.error) or chunk.sample_count <= 1:
                    raise RuntimeError(
                        "distributed rollout chunk failed "
                        f"(worker_id={result.worker_id}, chunk={chunk}): "
                        f"{result.error}",
                    )
                worker = worker_by_id.get(result.worker_id)
                if worker is None:
                    raise RuntimeError(
                        "distributed rollout chunk OOMed on unknown worker "
                        f"{result.worker_id!r}: {result.error}",
                    )
                children = chunk.split()
                logger.warning(
                    "ray chunk %s OOMed on worker %s; splitting %d samples into %s",
                    chunk.chunk_key,
                    result.worker_id,
                    chunk.sample_count,
                    [child.chunk_key for child in children],
                )
                splits.append(
                    {
                        "chunk_key": chunk.chunk_key,
                        "worker_id": result.worker_id,
                        "sample_count": chunk.sample_count,
                        "children": [child.chunk_key for child in children],
                    },
                )
                execute_chunk = worker.actor.execute_chunk
                remote = getattr(execute_chunk, "remote", None)
                for child in children:
                    child_envelope = replace(parent_envelope, chunk=child)
                    envelope_by_chunk_key[child_envelope.chunk_key] = child_envelope
                    if callable(remote):
                        retry_jobs.append(
                            RayActorJob(
                                job_index=len(retry_jobs),
                                worker_id=result.worker_id,
                                remote_method=remote,
                                payload=child_envelope,
                            ),
                        )
                    else:
                        local_calls.append((execute_chunk, child_envelope))
            pending = []
            if retry_jobs:
                pairs = await self.actor_dispatcher.run(
                    retry_jobs,
                    operation="rollout.generation.chunk",
                    call_timeout_s=self.generation_stall_timeout_s,
                )
                pending.extend(result for _, result in pairs)
            pending.extend(call(envelope) for call, envelope in local_calls)
            for result in pending:
                _require_correlated_result(result, envelope_by_chunk_key)
        return final, splits

    def _remote_worker_methods(self) -> dict[str, Any]:
        """Collect remote execute_chunk handles for pull-based dispatch."""

        methods: dict[str, Any] = {}
        for worker in self.workers:
            actor = worker.actor
            remote = getattr(getattr(actor, "execute_chunk", None), "remote", None)
            if not callable(remote):
                raise RuntimeError(
                    "dynamic chunk placement requires Ray actor workers; "
                    f"worker {worker.worker_id!r} has no remote execute_chunk",
                )
            methods[worker.worker_id] = remote
        return methods


def _require_correlated_result(
    result: ChunkExecutionResult,
    envelope_by_chunk_key: dict[str, ChunkExecutionEnvelope],
) -> ChunkExecutionEnvelope:
    """Require a worker result to match a submitted request and chunk."""

    envelope = envelope_by_chunk_key.get(result.chunk.chunk_key)
    if envelope is None:
        raise RuntimeError(
            "distributed rollout returned an unknown chunk "
            f"(worker_id={result.worker_id}, chunk={result.chunk.chunk_key})",
        )
    expected_request_id = envelope.request.request_id
    if result.request_id != expected_request_id:
        raise RuntimeError(
            "distributed rollout request_id mismatch "
            f"(worker_id={result.worker_id}, chunk={result.chunk.chunk_key}, "
            f"expected={expected_request_id!r}, actual={result.request_id!r})",
        )
    return envelope


def _is_oom_error(message: str) -> bool:
    """Match CUDA/HIP allocator failures flattened to text by a worker."""

    return "out of memory" in message.lower()


__all__ = ["RayGenerationExecutor"]
