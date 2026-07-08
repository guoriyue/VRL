"""Ray-backed generation executor that gathers chunk results."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from vrl.generation.execution.chunk_placement import (
    DistributedExecutionPlanner,
    build_chunk_memory_shadow,
)
from vrl.generation.execution.ids import build_sample_rows
from vrl.generation.execution.planner import attach_engine_plan
from vrl.generation.execution.types import (
    ChunkExecutionResult,
    DistributedWorkerHandle,
    StaleSlotDiscard,
)
from vrl.generation.protocols import ChunkGatherer, ChunkResult
from vrl.generation.types import GenerationOutput, GenerationRequest
from vrl.ray.actor_pool import RayActorJob, run_actor_jobs

logger = logging.getLogger(__name__)


class RayGenerationExecutor:
    """Execute one GenerationRequest across generation workers."""

    def __init__(
        self,
        planner: DistributedExecutionPlanner,
        workers: list[DistributedWorkerHandle],
        gatherer: ChunkGatherer,
        *,
        max_inflight_chunks_per_worker: int = 1,
        pipelined: bool = False,
    ) -> None:
        if not workers:
            raise ValueError("RayGenerationExecutor requires at least one worker")
        if max_inflight_chunks_per_worker < 1:
            raise ValueError("max_inflight_chunks_per_worker must be >= 1")
        self.planner = planner
        self.workers = list(workers)
        self.gatherer = gatherer
        self.max_inflight_chunks_per_worker = int(max_inflight_chunks_per_worker)
        # Opt-in single-worker pipelined rollout. Default False = unchanged per-chunk
        # dispatch. Only valid with exactly one worker; ignored otherwise.
        self.pipelined = bool(pipelined)

    async def execute(self, request: GenerationRequest) -> GenerationOutput:
        import time

        from vrl.utils.profiling import record_function

        _gen_start = time.perf_counter()
        sample_rows = build_sample_rows(request)
        with record_function("engine.plan"):
            generation_plan = self.planner.plan_with_engine(
                request,
                self.workers,
                sample_rows=sample_rows,
            )
        assignments = list(generation_plan.assignments)
        engine_plan = generation_plan.engine_plan
        if self.pipelined and len(self.workers) == 1:
            out = await self._execute_request_pipelined(request, engine_plan, sample_rows)
            logger.info(
                "generation wall: path=per_request_pipelined chunks=%d wall_s=%.3f",
                len(engine_plan.chunks), time.perf_counter() - _gen_start,
            )
            return out
        worker_by_id = {worker.worker_id: worker for worker in self.workers}
        strategy = self.planner.policy.strategy
        remote_jobs: list[RayActorJob] = []
        result_pairs: list[tuple[int, ChunkExecutionResult]] = []
        schedule_rows: list[dict[str, Any]] = []

        for job_index, assignment in enumerate(assignments):
            if assignment.envelope is None:
                raise RuntimeError("distributed rollout assignment is missing execution envelope")
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
            if actor is None:
                raise RuntimeError(f"worker {worker.worker_id!r} has no actor")
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
                await run_actor_jobs(
                    remote_jobs,
                    max_inflight_per_actor=self.max_inflight_chunks_per_worker,
                    worker_methods=worker_methods,
                    schedule=schedule_rows,
                ),
            )

        results = [result for _, result in sorted(result_pairs, key=lambda pair: pair[0])]

        if len(results) != len(assignments):
            raise RuntimeError(
                "distributed rollout returned wrong number of chunks: "
                f"{len(results)} != {len(assignments)}",
            )

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
            envelope_by_chunk_key={
                assignment.envelope.chunk_key: assignment.envelope
                for assignment in assignments
            },
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
        attach_engine_plan(output, engine_plan)
        output.extra["ray_chunk_metrics"] = [dict(result.metrics) for result in results]
        # Raw per-chunk memory readings (drift monitor for the startup
        # chunk-size probe). Telemetry only — nothing here changes chunk sizing.
        memory_shadow = build_chunk_memory_shadow(
            [result.metrics for result in results],
        )
        if memory_shadow:
            output.extra["chunk_memory_shadow"] = memory_shadow
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
        runtime_debug_on = bool(request.metadata.get("_runtime_debug"))
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
            output.extra["ray_chunk_schedule"] = schedule_summary
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
        if oom_splits:
            output.extra["ray_chunk_oom_splits"] = oom_splits
        worker_debug_rows = [
            result.metrics for result in results if result.metrics.get("runtime_debug")
        ]
        debug_payload: dict[str, Any] = {}
        if worker_debug_rows:
            debug_payload["ray_chunks"] = worker_debug_rows
        if runtime_debug_on and schedule_summary:
            debug_payload["chunk_schedule"] = schedule_summary
        if runtime_debug_on and oom_splits:
            debug_payload["chunk_oom_splits"] = oom_splits
        if debug_payload:
            output.extra["runtime_debug"] = debug_payload
        logger.info(
            "generation wall: path=per_chunk_dispatch chunks=%d wall_s=%.3f",
            len(assignments), time.perf_counter() - _gen_start,
        )
        return output

    async def _execute_request_pipelined(
        self,
        request: GenerationRequest,
        engine_plan: Any,
        sample_rows: Any,
    ) -> GenerationOutput:
        """Single-worker stage-overlap path (opt-in, ``pipelined=True``):
        the whole request's chunks run software-pipelined on one worker
        (``forward_plan_pipelined``), returning the already-gathered
        GenerationOutput. Skips the per-chunk dispatch + OOM-split (the pipeline
        keeps depth 1 = ~2 chunks resident, raising peak vs single-chunk — only use
        when 2 chunks fit). Version safety is enforced in the worker (slot
        activation / StaleSlotDiscard); a stale request raises and is counted as a
        graceful discard upstream, never trained off-policy."""

        worker = self.workers[0]
        actor = worker.actor
        if actor is None:
            raise RuntimeError(f"worker {worker.worker_id!r} has no actor")
        call = actor.execute_request_pipelined
        remote = getattr(call, "remote", None)
        if callable(remote):
            return await remote(request, engine_plan, sample_rows)
        return call(request, engine_plan, sample_rows)

    async def _degrade_oom_chunks(
        self,
        results: list[ChunkExecutionResult],
        *,
        envelope_by_chunk_key: dict[str, Any],
        worker_by_id: dict[str, DistributedWorkerHandle],
    ) -> tuple[list[ChunkExecutionResult], list[dict[str, Any]]]:
        """Split OOM chunks in half and re-run until success or single sample.

        The retry lives on the driver so vrl/ray stays chunk-agnostic, and the
        gatherer reassembles by (prompt_index, sample_start) metadata, so the
        extra child results need no positional bookkeeping. Children rebind to
        the worker that OOMed: with max_inflight 1 the two halves run
        sequentially there instead of landing concurrently on the GPU that
        just proved too full.
        """

        final: list[ChunkExecutionResult] = []
        pending = list(results)
        splits: list[dict[str, Any]] = []
        while pending:
            retry_jobs: list[RayActorJob] = []
            local_calls: list[tuple[Any, Any]] = []
            for result in pending:
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
                parent_envelope = envelope_by_chunk_key[chunk.chunk_key]
                worker = worker_by_id.get(result.worker_id)
                if worker is None or worker.actor is None:
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
                pairs = await run_actor_jobs(
                    retry_jobs,
                    max_inflight_per_actor=self.max_inflight_chunks_per_worker,
                )
                pending.extend(result for _, result in pairs)
            pending.extend(call(envelope) for call, envelope in local_calls)
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

def _is_oom_error(message: str) -> bool:
    """Match CUDA/HIP allocator failures flattened to str by the worker."""

    return "out of memory" in message.lower()


__all__ = ["RayGenerationExecutor"]
