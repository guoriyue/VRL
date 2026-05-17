"""Distributed execution planning for large rollout chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vrl.distributed.ray.rollout.types import RayChunkExecutionEnvelope, RayWorkerHandle
from vrl.engine.core.capabilities import family_capability_from_value
from vrl.engine.core.types import GenerationRequest, GenerationSampleRow
from vrl.engine.execution.microbatching import (
    MicroBatchSample,
)
from vrl.engine.execution.planner import EnginePlan, ExecutionStage, build_engine_plan


@dataclass(frozen=True, slots=True)
class DeviceAssignment:
    """Map one logical chunk to one rollout worker."""

    worker_id: str
    node_id: str
    gpu_ids: tuple[int, ...]
    chunk: MicroBatchSample
    execution_stage: ExecutionStage | None = None
    envelope: RayChunkExecutionEnvelope | None = None

    @property
    def execution_unit(self) -> ExecutionStage | None:
        return self.execution_stage


@dataclass(frozen=True, slots=True)
class DistributedRolloutPlan:
    """Driver-side plan plus worker placement for one rollout request."""

    engine_plan: EnginePlan
    assignments: tuple[DeviceAssignment, ...]


class DistributedExecutionPlanner:
    """Plan chunk placement across Ray rollout workers."""

    def __init__(self, capability: Any | None = None) -> None:
        self.capability = family_capability_from_value(capability)

    def plan(
        self,
        request: GenerationRequest,
        workers: list[RayWorkerHandle],
    ) -> list[DeviceAssignment]:
        return list(self.plan_with_engine(request, workers).assignments)

    def plan_with_engine(
        self,
        request: GenerationRequest,
        workers: list[RayWorkerHandle],
        *,
        sample_rows: list[GenerationSampleRow] | None = None,
    ) -> DistributedRolloutPlan:
        if not workers:
            raise ValueError("DistributedExecutionPlanner requires at least one worker")
        max_samples = int(
            request.sampling.get("sample_batch_size", request.samples_per_prompt),
        )
        capability = self.capability
        if capability is None:
            capability = family_capability_from_value(
                request.metadata.get("family_capability"),
            )
        if capability is None:
            raise ValueError(
                "GenerationRequest.metadata['family_capability'] is required for "
                "distributed rollout planning",
            )
        engine_plan = build_engine_plan(
            request,
            sample_rows,
            capability=capability,
            max_samples_per_microbatch=max(1, max_samples),
        )
        plan_summary = engine_plan.summary()
        capability_summary = engine_plan.capability.to_dict()
        assignments: list[DeviceAssignment] = []
        for idx, chunk in enumerate(engine_plan.micro_batches):
            worker = workers[idx % len(workers)]
            chunk_stage = engine_plan.chunk_stage_for(chunk)
            envelope = RayChunkExecutionEnvelope(
                request=request,
                chunk=chunk,
                plan_id=engine_plan.request_id,
                execution_stage=chunk_stage,
                profiler_label=chunk_stage.profiler_name,
                capability_summary=capability_summary,
                plan_summary=plan_summary,
            )
            assignments.append(
                DeviceAssignment(
                    worker_id=worker.worker_id,
                    node_id=worker.node_id,
                    gpu_ids=worker.gpu_ids,
                    chunk=chunk,
                    execution_stage=chunk_stage,
                    envelope=envelope,
                )
            )
        return DistributedRolloutPlan(
            engine_plan=engine_plan,
            assignments=tuple(assignments),
        )


__all__ = [
    "DeviceAssignment",
    "DistributedExecutionPlanner",
    "DistributedRolloutPlan",
]
