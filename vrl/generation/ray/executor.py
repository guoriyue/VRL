"""Ray-backed generation executor that gathers batch results.

The per-request slice of the Ray adapter: split one ``EnginePlan`` across the
engine fleet, await the batch RPCs under stall deadlines, and reassemble outputs through the model-free gatherer,
driver-side for per-batch dispatch or on a finalizer actor for the per-request
path. It deliberately owns no lifecycle — admission, terminal failure, and
shutdown belong to ``RayGenerationRuntime``, while the live actors and this
executor are held together by ``RayGenerationSession``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from vrl.generation.execution.planner import EnginePlan
from vrl.generation.execution.types import (
    GenerationBatchEnvelope,
    GenerationBatchResult,
    RequestBatchOutOfMemory,
    StagedBatchRefs,
    StaleSlotDiscard,
)
from vrl.generation.protocols import BatchPayload, GenerationBatchGatherer
from vrl.generation.ray.engine import RayGenerationEngine
from vrl.generation.types import GenerationOutput, GenerationRequest, GenerationSampleRow
from vrl.ray.actor_group import RayActorHandle
from vrl.ray.actor_pool import RayActorDispatcher, RayActorJob
from vrl.utils.cuda_memory import is_cuda_out_of_memory
from vrl.utils.deadline import require_timeout

logger = logging.getLogger(__name__)


class RayGenerationExecutor:
    """Execute one GenerationRequest across generation engines."""

    def __init__(
        self,
        engines: list[RayGenerationEngine],
        gatherer: GenerationBatchGatherer,
        *,
        actor_dispatcher: RayActorDispatcher,
        generation_stall_timeout_s: float,
        finalizers: Sequence[RayActorHandle] = (),
    ) -> None:
        if not engines:
            raise ValueError("RayGenerationExecutor requires at least one engine")
        self.engines = list(engines)
        self.gatherer = gatherer
        self.actor_dispatcher = actor_dispatcher
        self.generation_stall_timeout_s = require_timeout(
            generation_stall_timeout_s,
            name="generation_stall_timeout_s",
        )
        # The per-request path stages batch payloads in the object store and
        # merges them on a finalizer actor, never on the rank or the driver.
        # Finalizers get the same admission as engines: one slot each, FIFO
        # waiters, and a call's deadline starts only once it holds a slot, so
        # a request queued behind a slow merge does not burn its budget waiting.
        self.finalizers = tuple(finalizers)
        # Per-request (pipelined) execution exists exactly when the fleet was
        # launched with finalizer actors to merge the staged batch payloads.
        self.pipelined = bool(self.finalizers)
        self._finalizer_dispatcher = (
            RayActorDispatcher(tuple(finalizer.worker_id for finalizer in self.finalizers))
            if self.finalizers
            else None
        )

    def _engine_for_result_id(self, worker_id: str) -> RayGenerationEngine | None:
        """Map a result's producing rank id (or an engine id) to its engine."""

        for engine in self.engines:
            if engine.engine_id == worker_id:
                return engine
            if any(rank.worker_id == worker_id for rank in engine.ranks):
                return engine
        return None

    @staticmethod
    def _select_request_rank_result(
        results: list[Any],
    ) -> StagedBatchRefs | RequestBatchOutOfMemory | StaleSlotDiscard:
        """Return a stale discard, then an OOM, reported by any rank; else the primary's refs."""

        if not all(
            isinstance(result, (StagedBatchRefs, RequestBatchOutOfMemory, StaleSlotDiscard))
            for result in results
        ):
            raise TypeError(
                "pipelined engine ranks must return StagedBatchRefs, "
                "RequestBatchOutOfMemory, or StaleSlotDiscard"
            )
        for result in results:
            if isinstance(result, StaleSlotDiscard):
                return result
        if any(result.request_id != results[0].request_id for result in results[1:]):
            raise RuntimeError("pipelined engine ranks returned different request identities")
        for result in results:
            if isinstance(result, RequestBatchOutOfMemory):
                return result
        return results[0]

    async def execute(self, request: GenerationRequest) -> GenerationOutput:
        """Execute one request.

        Admission is the fleet dispatcher's: each engine exposes one slot with
        FIFO waiters, and a call's stall deadline starts only after its slot is
        acquired, so a later request never spends its budget queued behind an
        earlier one. Requests may therefore be submitted concurrently; the
        per-request path releases an engine's slot as soon as its batches are
        staged, before the request is merged.
        """

        import time

        from vrl.utils.profiling import profile_range

        _gen_start = time.perf_counter()
        sample_rows = request.sample_rows()
        with profile_range("engine.plan"):
            engine_plan = EnginePlan.from_request(request)
        pipelined_oom: RequestBatchOutOfMemory | None = None
        if self.pipelined and len(engine_plan.sample_batches) >= 2:
            pipelined_result = await self._execute_request_batches(
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
        runtime_debug_on = request.runtime_debug
        remote_jobs: list[RayActorJob] = []
        result_pairs: list[tuple[int, GenerationBatchResult]] = []
        schedule_rows: list[dict[str, Any]] = []
        envelope_by_batch_key: dict[str, GenerationBatchEnvelope] = {}

        for job_index, batch in enumerate(engine_plan.sample_batches):
            engine = self.engines[job_index % len(self.engines)]
            envelope = GenerationBatchEnvelope(request=request, batch=batch)
            envelope_by_batch_key[batch.batch_key] = envelope
            remote_jobs.append(
                RayActorJob(
                    job_index=job_index,
                    worker_id=engine.engine_id,
                    remote_method=engine.execute_batch(),
                    payload=envelope,
                ),
            )

        if remote_jobs:
            result_pairs.extend(
                await self.actor_dispatcher.run(
                    remote_jobs,
                    operation="rollout.generation.batch",
                    call_timeout_s=self.generation_stall_timeout_s,
                    schedule=schedule_rows if runtime_debug_on else None,
                ),
            )

        results = [result for _, result in sorted(result_pairs, key=lambda pair: pair[0])]

        if len(results) != len(engine_plan.sample_batches):
            raise RuntimeError(
                "distributed rollout returned wrong number of batches: "
                f"{len(results)} != {len(engine_plan.sample_batches)}",
            )

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
        # Log each batch's measured memory peaks.
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
                    "batch_key": batch.batch_key,
                    "sample_count": batch.sample_count,
                    "assigned_worker": by_index[job_index]["worker_id"],
                    "queue_wait_s": by_index[job_index]["queue_wait_s"],
                    "execution_s": by_index[job_index]["execution_s"],
                }
                for job_index, batch in enumerate(engine_plan.sample_batches)
                if job_index in by_index
            ]
        if runtime_debug_on:
            for row in schedule_summary:
                logger.info(
                    "ray batch schedule: batch=%s worker=%s samples=%d "
                    "queue_wait=%.3fs exec=%.3fs",
                    row["batch_key"],
                    row["assigned_worker"],
                    row["sample_count"],
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
            len(engine_plan.sample_batches),
            time.perf_counter() - _gen_start,
        )
        return output

    async def _execute_request_batches(
        self,
        request: GenerationRequest,
        engine_plan: EnginePlan,
        sample_rows: list[GenerationSampleRow],
    ) -> GenerationOutput | RequestBatchOutOfMemory:
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
        pairs = await self.actor_dispatcher.run(
            [
                RayActorJob(
                    job_index=job_index,
                    worker_id=engine.engine_id,
                    remote_method=engine.remote(
                        "execute_request_batches",
                        combine=self._select_request_rank_result,
                    ),
                    payload=request,
                    keyword_args={"engine_plan": EnginePlan(sample_batches=batches)},
                )
                for job_index, (engine, batches) in enumerate(engine_batches)
            ],
            operation="rollout.generation.pipelined",
            # One call produces every batch of an engine's share, so its stall
            # budget covers each of those batches.
            call_timeout_s=self.generation_stall_timeout_s
            * max(len(batches) for _, batches in engine_batches),
        )
        engine_results = [result for _, result in pairs]
        for result in engine_results:
            if isinstance(result, StaleSlotDiscard):
                raise result
        refs_by_key: dict[str, Any] = {}
        for (engine, batches), result in zip(engine_batches, engine_results, strict=True):
            if not isinstance(result, (StagedBatchRefs, RequestBatchOutOfMemory)):
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
            if isinstance(result, RequestBatchOutOfMemory):
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

    async def _finalize_request(
        self,
        request: GenerationRequest,
        sample_rows: list[GenerationSampleRow],
        batch_refs: list[Any],
    ) -> GenerationOutput:
        """Merge staged batch references on whichever finalizer is free.

        With several engines a request's references come from every engine, so
        the chosen finalizer reads part of its input from other nodes' object
        stores; per-engine placement only guarantees locality for single-engine
        fleets. Choosing by data location is a measurement away, not a design
        decision taken here.
        """

        dispatcher = self._finalizer_dispatcher
        if dispatcher is None:
            raise RuntimeError("per-request generation has no finalizer to merge on")
        pairs = await dispatcher.run(
            [
                RayActorJob(
                    job_index=0,
                    worker_id=None,
                    remote_method=None,
                    payload=request,
                    keyword_args={
                        "sample_rows": list(sample_rows),
                        "batch_refs": list(batch_refs),
                    },
                ),
            ],
            operation="rollout.generation.finalize",
            call_timeout_s=self.generation_stall_timeout_s,
            worker_methods={
                finalizer.worker_id: finalizer.actor.merge_request.remote
                for finalizer in self.finalizers
            },
        )
        output = pairs[0][1]
        if not isinstance(output, GenerationOutput):
            raise TypeError(
                f"finalizer returned {type(output).__name__}, expected GenerationOutput",
            )
        if output.request_id != request.request_id:
            raise RuntimeError(
                "finalized generation request_id mismatch: "
                f"{output.request_id!r} != {request.request_id!r}",
            )
        return output

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


__all__ = ["RayGenerationExecutor"]
