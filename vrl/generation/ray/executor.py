"""Ray-backed generation executor that gathers batch results.

The per-request slice of the Ray adapter: split one ``EnginePlan`` across the
engine fleet, await the batch RPCs (with stall deadlines and pipelined
progress probing), and reassemble outputs through the model-free gatherer,
driver-side for per-batch dispatch or on a finalizer actor for the per-request
path. It deliberately owns no lifecycle — admission, terminal failure, and
shutdown belong to ``RayGenerationRuntime``, while the live actors and this
executor are held together by ``RayGenerationSession``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from vrl.generation.execution.batch_placement import DistributedExecutionPlanner
from vrl.generation.execution.planner import EnginePlan
from vrl.generation.execution.types import (
    BatchSizeProbeResult,
    GenerationBatchEnvelope,
    GenerationBatchResult,
    PipelinedBatchRefs,
    PipelinedRequestOutOfMemory,
    StaleSlotDiscard,
)
from vrl.generation.protocols import BatchPayload, GenerationBatchGatherer
from vrl.generation.ray.engine import RayGenerationEngine
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
from vrl.utils.cuda_memory import is_cuda_out_of_memory
from vrl.utils.deadline import require_timeout

logger = logging.getLogger(__name__)

# Bounds progress-probe traffic on the shared health concurrency group. This is
# a wire cadence, not a user-facing generation SLA.
_PIPELINED_PROGRESS_POLL_INTERVAL_S = 1.0


class RayGenerationExecutor:
    """Execute one GenerationRequest across generation engines."""

    def __init__(
        self,
        planner: DistributedExecutionPlanner,
        engines: list[RayGenerationEngine],
        gatherer: GenerationBatchGatherer,
        *,
        actor_dispatcher: RayActorDispatcher,
        generation_stall_timeout_s: float,
        pipelined: bool = False,
        finalizers: Sequence[RayActorHandle] = (),
    ) -> None:
        if not engines:
            raise ValueError("RayGenerationExecutor requires at least one engine")
        self.planner = planner
        self.engines = list(engines)
        self.gatherer = gatherer
        expected_engine_ids = tuple(engine.engine_id for engine in self.engines)
        if actor_dispatcher.worker_ids != expected_engine_ids:
            raise ValueError(
                "RayGenerationExecutor actor dispatcher does not own its engine fleet: "
                f"{actor_dispatcher.worker_ids} != {expected_engine_ids}",
            )
        self.actor_dispatcher = actor_dispatcher
        self.generation_stall_timeout_s = require_timeout(
            generation_stall_timeout_s,
            name="generation_stall_timeout_s",
        )
        self.pipelined = bool(pipelined)
        # The per-request path stages batch payloads in the object store and
        # merges them on a finalizer actor, never on the rank or the driver.
        self.finalizers = tuple(finalizers)
        if self.pipelined and not self.finalizers:
            raise ValueError(
                "pipelined Ray generation requires at least one finalizer actor "
                "to merge staged batch payloads",
            )
        self._next_finalizer = 0

    def _engine_for_result_id(self, worker_id: str) -> RayGenerationEngine | None:
        """Map a result's producing rank id (or an engine id) to its engine."""

        for engine in self.engines:
            if engine.engine_id == worker_id:
                return engine
            if any(rank.worker_id == worker_id for rank in engine.ranks):
                return engine
        return None

    async def execute(self, request: GenerationRequest) -> GenerationOutput:
        """Execute one request.

        Admission is the fleet dispatcher's: each engine exposes one slot with
        FIFO waiters, and a call's stall deadline starts only after its slot is
        acquired, so a later request never spends its budget queued behind an
        earlier one. Requests may therefore be submitted concurrently; the
        per-request path releases an engine's slot as soon as its batches are
        staged, before the request is merged.
        """

        return await self._execute(request)

    @staticmethod
    def _select_request_rank_result(
        results: list[Any],
    ) -> PipelinedBatchRefs | PipelinedRequestOutOfMemory:
        """Return an OOM reported by any rank; otherwise keep the primary's refs."""

        if not all(
            isinstance(result, (PipelinedBatchRefs, PipelinedRequestOutOfMemory))
            for result in results
        ):
            raise TypeError(
                "pipelined engine ranks must return PipelinedBatchRefs or "
                "PipelinedRequestOutOfMemory"
            )
        if any(result.request_id != results[0].request_id for result in results[1:]):
            raise RuntimeError("pipelined engine ranks returned different request identities")
        for result in results:
            if isinstance(result, PipelinedRequestOutOfMemory):
                return result
        return results[0]

    async def probe_batch_sizes(
        self,
        request: GenerationRequest,
        *,
        max_samples: int,
    ) -> list[BatchSizeProbeResult]:
        """Probe every engine through the same actor admission as generation."""

        return await self._probe_batch_sizes(request, max_samples=max_samples)

    async def _probe_batch_sizes(
        self,
        request: GenerationRequest,
        *,
        max_samples: int,
    ) -> list[BatchSizeProbeResult]:
        for engine in self.engines:
            if len(engine.ranks) != 1:
                raise ValueError(
                    "automatic batch-size probing requires single-rank engines; "
                    f"engine {engine.engine_id!r} has {len(engine.ranks)} ranks. "
                    "Set an explicit samples_per_generation_batch: multi-rank "
                    "probe trials do not yet coordinate memory/OOM decisions.",
                )
        result_pairs: list[tuple[int, Any]] = []
        remote_jobs: list[RayActorJob] = []
        for job_index, engine in enumerate(self.engines):
            probe = getattr(engine.primary.actor, "probe_batch_size", None)
            if probe is None:
                raise RuntimeError(
                    f"engine {engine.engine_id!r} does not support the "
                    "batch-size probe required by samples_per_generation_batch: auto",
                )
            remote_jobs.append(
                RayActorJob(
                    job_index=job_index,
                    worker_id=engine.engine_id,
                    remote_method=engine.remote("probe_batch_size"),
                    payload=request,
                    keyword_args={"max_samples": max_samples},
                ),
            )
        result_pairs.extend(
            await self.actor_dispatcher.run(
                remote_jobs,
                operation="rollout.generation.batch_size_probe",
                call_timeout_s=self.generation_stall_timeout_s,
            ),
        )
        results = [result for _, result in sorted(result_pairs, key=lambda pair: pair[0])]
        for engine, result in zip(self.engines, results, strict=True):
            if not isinstance(result, BatchSizeProbeResult):
                raise TypeError(
                    f"engine {engine.engine_id!r} returned invalid batch-size probe "
                    f"result {type(result).__name__}",
                )
        return results

    async def _execute(self, request: GenerationRequest) -> GenerationOutput:
        import time

        from vrl.utils.profiling import profile_range

        _gen_start = time.perf_counter()
        sample_rows = request.sample_rows()
        with profile_range("engine.plan"):
            generation_plan = self.planner.plan_with_engine(
                request,
                tuple(engine.engine_id for engine in self.engines),
            )
        assignments = list(generation_plan.assignments)
        engine_plan = generation_plan.engine_plan
        pipelined_oom: PipelinedRequestOutOfMemory | None = None
        if self.pipelined and len(engine_plan.sample_batches) >= 2:
            pipelined_result = await self._execute_request_pipelined(
                request,
                engine_plan,
                sample_rows,
            )
            if isinstance(pipelined_result, GenerationOutput):
                logger.info(
                    "generation wall: path=per_request_pipelined batches=%d wall_s=%.3f",
                    len(engine_plan.sample_batches),
                    time.perf_counter() - _gen_start,
                )
                return pipelined_result
            pipelined_oom = pipelined_result
            logger.warning(
                "pipelined generation request %s OOMed on rank %s; retrying "
                "through per-batch split admission: %s",
                request.request_id,
                pipelined_oom.worker_id,
                pipelined_oom.error,
            )
        engine_by_id = {engine.engine_id: engine for engine in self.engines}
        strategy = self.planner.strategy
        runtime_debug_on = request.runtime_debug
        remote_jobs: list[RayActorJob] = []
        result_pairs: list[tuple[int, GenerationBatchResult]] = []
        schedule_rows: list[dict[str, Any]] = []

        for job_index, assignment in enumerate(assignments):
            if assignment.engine_id is None:
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
            engine = engine_by_id[assignment.engine_id]
            remote_jobs.append(
                RayActorJob(
                    job_index=job_index,
                    worker_id=engine.engine_id,
                    remote_method=engine.execute_batch(),
                    payload=assignment.envelope,
                ),
            )

        if remote_jobs:
            worker_methods = None
            if any(job.worker_id is None for job in remote_jobs):
                worker_methods = self._remote_engine_methods()
            result_pairs.extend(
                await self.actor_dispatcher.run(
                    remote_jobs,
                    operation="rollout.generation.batch",
                    call_timeout_s=self.generation_stall_timeout_s,
                    worker_methods=worker_methods,
                    schedule=schedule_rows if runtime_debug_on else None,
                ),
            )

        results = [result for _, result in sorted(result_pairs, key=lambda pair: pair[0])]

        if len(results) != len(assignments):
            raise RuntimeError(
                "distributed rollout returned wrong number of batches: "
                f"{len(results)} != {len(assignments)}",
            )

        envelope_by_batch_key = {
            assignment.envelope.batch_key: assignment.envelope for assignment in assignments
        }
        for result in results:
            self._validate_result_identity(result, envelope_by_batch_key)

        results, oom_splits = await self._degrade_oom_chunks(
            results,
            envelope_by_batch_key=envelope_by_batch_key,
        )

        for result in results:
            if (
                request.policy_version is not None
                and result.policy_version != request.policy_version
            ):
                raise RuntimeError(
                    "distributed rollout policy_version mismatch "
                    f"(rank={result.worker_id}, "
                    f"expected={request.policy_version}, "
                    f"actual={result.policy_version})",
                )

        batch_outputs: list[BatchPayload] = []
        for result in results:
            if result.output is None:
                raise RuntimeError(
                    f"distributed rollout batch returned no output: {result}",
                )
            batch_outputs.append(result.output)

        output = self.gatherer.merge_generation_batches(request, sample_rows, batch_outputs)
        # Log measured peaks without changing the probe's batch-size decision.
        for result in results:
            reading = result.memory
            if reading is None:
                continue
            logger.info(
                "batch memory: batch=%s n=%d peak=%.0fMB "
                "(denoise=%.0fMB decode=%.0fMB baseline=%.0fMB) "
                "budget=%.0fMB non_torch=%.0fMB",
                result.batch.batch_key,
                reading.sample_count,
                reading.peak_bytes / 2**20,
                reading.denoise_peak_bytes / 2**20,
                reading.decode_peak_bytes / 2**20,
                reading.baseline_allocated_bytes / 2**20,
                reading.budget_bytes / 2**20,
                reading.non_torch_bytes / 2**20,
            )
        schedule_summary: list[dict[str, Any]] = []
        if schedule_rows:
            by_index = {row["job_index"]: row for row in schedule_rows}
            schedule_summary = [
                {
                    "batch_key": assignment.batch.batch_key,
                    "sample_count": assignment.batch.sample_count,
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
                    "ray batch schedule [%s]: batch=%s worker=%s samples=%d "
                    "cost=%.0f queue_wait=%.3fs exec=%.3fs",
                    strategy,
                    row["batch_key"],
                    row["assigned_worker"],
                    row["sample_count"],
                    row["estimated_cost"],
                    row["queue_wait_s"],
                    row["execution_s"],
                )
        rank_debug_rows: list[dict[str, Any]] = []
        if runtime_debug_on:
            rank_by_id = {rank.worker_id: rank for engine in self.engines for rank in engine.ranks}
            for result in results:
                for worker_id, metrics in result.rank_metrics.items():
                    rank = rank_by_id[worker_id]
                    rank_debug_rows.append(
                        {
                            "worker_id": worker_id,
                            "node_ip": rank.node_ip,
                            "gpu_ids": list(rank.gpu_ids),
                            "policy_version": result.policy_version,
                            "batch_key": result.batch.batch_key,
                            **metrics,
                        },
                    )
        debug_payload: dict[str, Any] = {}
        if rank_debug_rows:
            debug_payload["ray_chunks"] = rank_debug_rows
        if runtime_debug_on and schedule_summary:
            debug_payload["chunk_schedule"] = schedule_summary
        if runtime_debug_on and oom_splits:
            debug_payload["batch_oom_splits"] = oom_splits
        if debug_payload:
            output.runtime_debug = debug_payload
        logger.info(
            "generation wall: path=%s batches=%d wall_s=%.3f",
            (
                "per_batch_dispatch_after_pipelined_oom"
                if pipelined_oom is not None
                else "per_batch_dispatch"
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
        """Per-request path (opt-in, ``pipelined=True``).

        Every engine receives its round-robin share of the request's batches in
        ONE call and stages each batch payload into the object store as it is
        produced; a finalizer actor then merges the references into the
        ``GenerationOutput`` off the ranks' critical path. A typed OOM from any
        engine falls back to the normal per-batch dispatch and split admission
        path. Version safety is enforced in the rank (slot activation /
        StaleSlotDiscard); a stale request raises and is counted as a graceful
        discard upstream, never trained off-policy.
        """

        engine_batches = self._engine_batch_subsets(engine_plan)
        engine_results = await self._gather_cancel_on_error(
            [
                self._execute_engine_pipelined(request, engine, batches)
                for engine, batches in engine_batches
            ],
        )
        for result in engine_results:
            if isinstance(result, StaleSlotDiscard):
                raise result
        refs_by_key: dict[str, Any] = {}
        for (engine, batches), result in zip(engine_batches, engine_results, strict=True):
            if not isinstance(result, (PipelinedBatchRefs, PipelinedRequestOutOfMemory)):
                raise TypeError(
                    "pipelined generation engine returned unsupported result "
                    f"{type(result).__name__}",
                )
            if result.request_id != request.request_id:
                raise RuntimeError(
                    "pipelined generation request_id mismatch: "
                    f"{result.request_id!r} != {request.request_id!r}",
                )
            if result.worker_id not in {rank.worker_id for rank in engine.ranks}:
                raise RuntimeError(
                    "pipelined generation rank mismatch: "
                    f"{result.worker_id!r} is not a rank of engine {engine.engine_id!r}",
                )
            if isinstance(result, PipelinedRequestOutOfMemory):
                return result
            expected_keys = tuple(batch.batch_key for batch in batches)
            if result.batch_keys != expected_keys:
                raise RuntimeError(
                    f"pipelined engine {engine.engine_id!r} staged batches "
                    f"{result.batch_keys} for plan {expected_keys}",
                )
            refs_by_key.update(zip(result.batch_keys, result.batch_refs, strict=True))
        ordered_refs = [refs_by_key[batch.batch_key] for batch in engine_plan.sample_batches]
        return await self._finalize_request(request, sample_rows, ordered_refs)

    def _engine_batch_subsets(
        self,
        engine_plan: EnginePlan,
    ) -> list[tuple[RayGenerationEngine, tuple[Any, ...]]]:
        """Round-robin the plan's batches over the engines; skip idle engines."""

        subsets: list[tuple[RayGenerationEngine, tuple[Any, ...]]] = []
        for index, engine in enumerate(self.engines):
            batches = tuple(engine_plan.sample_batches[index :: len(self.engines)])
            if batches:
                subsets.append((engine, batches))
        return subsets

    async def _execute_engine_pipelined(
        self,
        request: GenerationRequest,
        engine: RayGenerationEngine,
        batches: tuple[Any, ...],
    ) -> PipelinedBatchRefs | PipelinedRequestOutOfMemory | StaleSlotDiscard:
        primary = engine.primary
        # Progress is a rank-0 read on the health concurrency group.
        progress = getattr(primary.actor, "pipelined_progress", None)
        progress_remote = getattr(progress, "remote", None)
        if not callable(progress_remote):
            raise PipelinedProgressError(
                "pipelined Ray generation requires rank progress reporting",
            )
        engine_plan = EnginePlan(sample_batches=batches)

        async def await_pipelined_result(
            result_ref: Any,
            initial_deadline: RayCallDeadline,
        ) -> Any:
            try:
                return await self._await_pipelined_result(
                    result_ref=result_ref,
                    progress_remote=progress_remote,
                    request_id=request.request_id,
                    total_batches=len(batches),
                    initial_deadline=initial_deadline,
                )
            except StaleSlotDiscard as error:
                # A stale version is a known, graceful business outcome. Let
                # the fleet dispatcher release the actor slot before the
                # public executor re-raises the typed discard.
                return error

        return await self.actor_dispatcher.run_one(
            RayActorJob(
                job_index=0,
                worker_id=engine.engine_id,
                remote_method=engine.remote(
                    "execute_request_pipelined", combine=self._select_request_rank_result
                ),
                payload=request,
                keyword_args={"engine_plan": engine_plan},
            ),
            operation="rollout.generation.pipelined",
            call_timeout_s=self.generation_stall_timeout_s,
            await_result=await_pipelined_result,
        )

    @staticmethod
    async def _gather_cancel_on_error(coroutines: list[Any]) -> list[Any]:
        """Await all engine calls; a failure cancels the siblings before raising."""

        tasks = [asyncio.ensure_future(coroutine) for coroutine in coroutines]
        try:
            return list(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def _finalize_request(
        self,
        request: GenerationRequest,
        sample_rows: list[GenerationSampleRow],
        batch_refs: list[Any],
    ) -> GenerationOutput:
        """Merge staged batch references on the next finalizer actor."""

        finalizer = self.finalizers[self._next_finalizer % len(self.finalizers)]
        self._next_finalizer += 1
        deadline = RayCallDeadline(
            "rollout.generation.finalize",
            self.generation_stall_timeout_s,
            context=f"finalizer={finalizer.worker_id}, request_id={request.request_id}",
        )
        ref = finalizer.actor.merge_request.remote(request, list(sample_rows), list(batch_refs))
        task = asyncio.ensure_future(ref)
        try:
            output = await asyncio.wait_for(task, timeout=deadline.remaining_s())
        except TimeoutError as cause:
            cancel_ray_refs(None, [ref], root_error=None)
            raise deadline.timeout_error() from cause
        except asyncio.CancelledError:
            cancel_ray_refs(None, [ref], root_error=None)
            raise
        if not isinstance(output, GenerationOutput):
            raise TypeError(
                f"finalizer {finalizer.worker_id!r} returned {type(output).__name__}, "
                "expected GenerationOutput",
            )
        if output.request_id != request.request_id:
            raise RuntimeError(
                "finalized generation request_id mismatch: "
                f"{output.request_id!r} != {request.request_id!r}",
            )
        return output

    async def _await_pipelined_result(
        self,
        *,
        result_ref: Any,
        progress_remote: Any,
        request_id: str,
        total_batches: int,
        initial_deadline: RayCallDeadline,
    ) -> Any:
        """Reset one stall deadline only when the worker completes a new batch."""

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
        completed_batches = 0
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
                        f"pipelined rank returned invalid progress {type(snapshot).__name__}",
                    )
                if snapshot.request_id != request_id:
                    raise PipelinedProgressError(
                        "pipelined progress request_id mismatch: "
                        f"{snapshot.request_id!r} != {request_id!r}",
                    )
                if snapshot.total_batches != total_batches:
                    raise PipelinedProgressError(
                        "pipelined progress total_batches mismatch: "
                        f"{snapshot.total_batches} != {total_batches}",
                    )
                if snapshot.completed_batches < completed_batches:
                    raise PipelinedProgressError(
                        "pipelined progress regressed: "
                        f"{snapshot.completed_batches} < {completed_batches}",
                    )
                if snapshot.completed_batches > completed_batches:
                    completed_batches = snapshot.completed_batches
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
        results: list[GenerationBatchResult],
        *,
        envelope_by_batch_key: dict[str, GenerationBatchEnvelope],
    ) -> tuple[list[GenerationBatchResult], list[dict[str, Any]]]:
        """Split OOM batches in half and re-run until success or single sample.

        The retry lives on the driver so vrl/ray stays batch-agnostic, and the
        gatherer reassembles by (prompt_index, sample_start) metadata, so the
        extra child results need no positional bookkeeping. Children rebind to
        the engine that OOMed: the fleet-owned dispatcher exposes one real
        slot per engine, so the two halves run sequentially instead
        of landing concurrently on the GPUs that just proved too full.
        """

        final: list[GenerationBatchResult] = []
        pending = list(results)
        splits: list[dict[str, Any]] = []
        while pending:
            # A slot can be evicted before initial execution or between OOM
            # retries. In either case discard the whole request before treating
            # errors as retryable OOMs or accepting partial successful output.
            stale = [result for result in pending if result.stale_slot]
            if stale:
                evicted = stale[0]
                raise StaleSlotDiscard(
                    "distributed rollout discarded a stale trainable-state slot "
                    f"(rank={evicted.worker_id}, "
                    f"policy_version={evicted.policy_version}, "
                    f"batches={len(stale)}/{len(pending)}): {evicted.error}",
                )
            retry_jobs: list[RayActorJob] = []
            for result in pending:
                parent_envelope = envelope_by_batch_key[result.batch.batch_key]
                if not result.error:
                    final.append(result)
                    continue
                batch = result.batch
                if not is_cuda_out_of_memory(result.error) or batch.sample_count <= 1:
                    raise RuntimeError(
                        "distributed rollout batch failed "
                        f"(rank={result.worker_id}, batch={batch}): "
                        f"{result.error}",
                    )
                engine = self._engine_for_result_id(result.worker_id)
                if engine is None:
                    raise RuntimeError(
                        "distributed rollout batch OOMed on unknown rank "
                        f"{result.worker_id!r}: {result.error}",
                    )
                children = batch.split()
                logger.warning(
                    "ray batch %s OOMed on engine %s; splitting %d samples into %s",
                    batch.batch_key,
                    engine.engine_id,
                    batch.sample_count,
                    [child.batch_key for child in children],
                )
                splits.append(
                    {
                        "batch_key": batch.batch_key,
                        "worker_id": result.worker_id,
                        "sample_count": batch.sample_count,
                        "children": [child.batch_key for child in children],
                    },
                )
                for child in children:
                    child_envelope = replace(parent_envelope, batch=child)
                    envelope_by_batch_key[child_envelope.batch_key] = child_envelope
                    retry_jobs.append(
                        RayActorJob(
                            job_index=len(retry_jobs),
                            worker_id=engine.engine_id,
                            remote_method=engine.execute_batch(),
                            payload=child_envelope,
                        ),
                    )
            pending = []
            if retry_jobs:
                pairs = await self.actor_dispatcher.run(
                    retry_jobs,
                    operation="rollout.generation.batch",
                    call_timeout_s=self.generation_stall_timeout_s,
                )
                pending.extend(result for _, result in pairs)
            for result in pending:
                self._validate_result_identity(result, envelope_by_batch_key)
        return final, splits

    @staticmethod
    def _validate_result_identity(
        result: GenerationBatchResult,
        envelope_by_batch_key: dict[str, GenerationBatchEnvelope],
    ) -> None:
        """Require a rank result to match a submitted request and batch."""

        envelope = envelope_by_batch_key.get(result.batch.batch_key)
        if envelope is None:
            raise RuntimeError(
                "distributed rollout returned an unknown batch "
                f"(rank={result.worker_id}, batch={result.batch.batch_key})",
            )
        expected_request_id = envelope.request.request_id
        if result.request_id != expected_request_id:
            raise RuntimeError(
                "distributed rollout request_id mismatch "
                f"(rank={result.worker_id}, batch={result.batch.batch_key}, "
                f"expected={expected_request_id!r}, actual={result.request_id!r})",
            )

    def _remote_engine_methods(self) -> dict[str, Any]:
        """Collect per-engine execute_batch submitters for pull-based dispatch."""

        methods: dict[str, Any] = {}
        for engine in self.engines:
            probe = getattr(engine.primary.actor, "execute_batch", None)
            if not callable(getattr(probe, "remote", None)):
                raise RuntimeError(
                    "dynamic batch placement requires Ray actor ranks; "
                    f"engine {engine.engine_id!r} has no remote execute_batch",
                )
            methods[engine.engine_id] = engine.execute_batch()
        return methods


__all__ = ["RayGenerationExecutor"]
