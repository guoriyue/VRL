"""Distributed generation executor that gathers Ray chunk results."""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any

from vrl.generation.execution.ids import GenerationIdFactory
from vrl.generation.execution.planner import attach_engine_plan
from vrl.generation.protocols import ChunkGatherer, PipelineChunkResult
from vrl.generation.ray.dependencies import require_ray
from vrl.generation.ray.planner import DistributedExecutionPlanner
from vrl.generation.ray.types import (
    RayChunkExecutionEnvelope,
    RayChunkResult,
    RayWorkerHandle,
)
from vrl.generation.types import GenerationOutput, GenerationRequest


class DistributedGenerationExecutor:
    """Execute one GenerationRequest across generation workers."""

    def __init__(
        self,
        planner: DistributedExecutionPlanner,
        workers: list[RayWorkerHandle],
        gatherer: ChunkGatherer,
        *,
        id_factory: GenerationIdFactory | None = None,
        max_inflight_chunks_per_worker: int = 1,
    ) -> None:
        if not workers:
            raise ValueError("DistributedGenerationExecutor requires at least one worker")
        if max_inflight_chunks_per_worker < 1:
            raise ValueError("max_inflight_chunks_per_worker must be >= 1")
        self.planner = planner
        self.workers = list(workers)
        self.gatherer = gatherer
        self.id_factory = id_factory or GenerationIdFactory()
        self.max_inflight_chunks_per_worker = int(max_inflight_chunks_per_worker)

    async def execute(self, request: GenerationRequest) -> GenerationOutput:
        from vrl.utils.profiling import record_function

        sample_rows = self.id_factory.build_sample_rows(request)
        with record_function("engine.plan"):
            generation_plan = self.planner.plan_with_engine(
                request,
                self.workers,
                sample_rows=sample_rows,
            )
        assignments = list(generation_plan.assignments)
        engine_plan = generation_plan.engine_plan
        worker_by_id = {worker.worker_id: worker for worker in self.workers}
        remote_jobs: list[tuple[int, Any, RayWorkerHandle, RayChunkExecutionEnvelope]] = []
        result_pairs: list[tuple[int, RayChunkResult]] = []

        for job_index, assignment in enumerate(assignments):
            worker = worker_by_id[assignment.worker_id]
            actor = worker.actor
            if actor is None:
                raise RuntimeError(f"worker {worker.worker_id!r} has no actor")
            if assignment.envelope is None:
                raise RuntimeError("distributed rollout assignment is missing execution envelope")
            execute_chunk = actor.execute_chunk
            remote = getattr(execute_chunk, "remote", None)
            if callable(remote):
                remote_jobs.append((job_index, remote, worker, assignment.envelope))
            else:
                result_pairs.append(
                    (job_index, execute_chunk(assignment.envelope))
                )

        if remote_jobs:
            result_pairs.extend(await self._run_remote_jobs(remote_jobs))

        results = [result for _, result in sorted(result_pairs, key=lambda pair: pair[0])]

        if len(results) != len(assignments):
            raise RuntimeError(
                "distributed rollout returned wrong number of chunks: "
                f"{len(results)} != {len(assignments)}",
            )

        for result in results:
            if result.error:
                raise RuntimeError(
                    "distributed rollout chunk failed "
                    f"(worker_id={result.worker_id}, chunk={result.chunk}): "
                    f"{result.error}",
                )
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

        chunk_outputs: list[PipelineChunkResult] = []
        for result in results:
            if result.output is None:
                raise RuntimeError(
                    f"distributed rollout chunk returned no output: {result}",
                )
            chunk_outputs.append(result.output)

        output = self.gatherer.gather_chunks(request, sample_rows, chunk_outputs)
        attach_engine_plan(output, engine_plan)
        output.extra["ray_chunk_metrics"] = [dict(result.metrics) for result in results]
        runtime_debug = [
            result.metrics for result in results if result.metrics.get("runtime_debug")
        ]
        if runtime_debug:
            output.extra["runtime_debug"] = {"ray_chunks": runtime_debug}
        return output

    async def _run_remote_jobs(
        self,
        jobs: list[tuple[int, Any, RayWorkerHandle, RayChunkExecutionEnvelope]],
    ) -> list[tuple[int, RayChunkResult]]:
        ray = require_ray()
        pending = deque(jobs)
        inflight_by_worker = {worker.worker_id: 0 for _, _, worker, _ in jobs}
        ref_to_job: dict[Any, tuple[int, str]] = {}
        result_pairs: list[tuple[int, RayChunkResult]] = []

        def _submit_ready() -> None:
            made_progress = True
            while pending and made_progress:
                made_progress = False
                for _ in range(len(pending)):
                    job_index, remote, worker, envelope = pending.popleft()
                    if inflight_by_worker[worker.worker_id] >= self.max_inflight_chunks_per_worker:
                        pending.append((job_index, remote, worker, envelope))
                        continue
                    ref = remote(envelope)
                    ref_to_job[ref] = (job_index, worker.worker_id)
                    inflight_by_worker[worker.worker_id] += 1
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

        return result_pairs


DistributedRolloutExecutor = DistributedGenerationExecutor


__all__ = ["DistributedGenerationExecutor", "DistributedRolloutExecutor"]
