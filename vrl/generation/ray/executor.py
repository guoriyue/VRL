"""Ray-backed generation executor that gathers chunk results."""

from __future__ import annotations

from typing import Any

from vrl.generation.execution.ids import build_sample_rows
from vrl.generation.execution.planner import attach_engine_plan
from vrl.generation.execution.scheduler import DistributedExecutionPlanner
from vrl.generation.execution.types import (
    ChunkExecutionResult,
    DistributedWorkerHandle,
)
from vrl.generation.protocols import ChunkGatherer, ChunkResult
from vrl.generation.types import GenerationOutput, GenerationRequest
from vrl.ray.actor_pool import RayActorJob, run_actor_jobs


class RayGenerationExecutor:
    """Execute one GenerationRequest across generation workers."""

    def __init__(
        self,
        planner: DistributedExecutionPlanner,
        workers: list[DistributedWorkerHandle],
        gatherer: ChunkGatherer,
        *,
        max_inflight_chunks_per_worker: int = 1,
    ) -> None:
        if not workers:
            raise ValueError("RayGenerationExecutor requires at least one worker")
        if max_inflight_chunks_per_worker < 1:
            raise ValueError("max_inflight_chunks_per_worker must be >= 1")
        self.planner = planner
        self.workers = list(workers)
        self.gatherer = gatherer
        self.max_inflight_chunks_per_worker = int(max_inflight_chunks_per_worker)

    async def execute(self, request: GenerationRequest) -> GenerationOutput:
        from vrl.utils.profiling import record_function

        sample_rows = build_sample_rows(request)
        with record_function("engine.plan"):
            generation_plan = self.planner.plan_with_engine(
                request,
                self.workers,
                sample_rows=sample_rows,
            )
        assignments = list(generation_plan.assignments)
        engine_plan = generation_plan.engine_plan
        worker_by_id = {worker.worker_id: worker for worker in self.workers}
        remote_jobs: list[RayActorJob] = []
        result_pairs: list[tuple[int, ChunkExecutionResult]] = []

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
            result_pairs.extend(
                await self._run_remote_jobs(remote_jobs),
            )

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
        runtime_debug = [
            result.metrics for result in results if result.metrics.get("runtime_debug")
        ]
        if runtime_debug:
            output.extra["runtime_debug"] = {"ray_chunks": runtime_debug}
        return output

    async def _run_remote_jobs(
        self,
        jobs: list[RayActorJob],
    ) -> list[tuple[int, Any]]:
        return await run_actor_jobs(
            jobs,
            max_inflight_per_actor=self.max_inflight_chunks_per_worker,
        )

__all__ = ["RayGenerationExecutor"]
