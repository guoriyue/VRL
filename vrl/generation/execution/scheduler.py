"""Distributed execution planning for large generation chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vrl.generation.capabilities import family_capability_from_value
from vrl.generation.execution.chunks import SampleChunk
from vrl.generation.execution.planner import EnginePlan, ExecutionStage, build_engine_plan
from vrl.generation.execution.types import (
    ChunkExecutionEnvelope,
    DistributedWorkerHandle,
)
from vrl.generation.types import GenerationRequest, GenerationSampleRow


@dataclass(frozen=True, slots=True)
class DeviceAssignment:
    """Map one logical chunk to one generation worker."""

    worker_id: str
    node_id: str
    gpu_ids: tuple[int, ...]
    chunk: SampleChunk
    execution_stage: ExecutionStage | None = None
    envelope: ChunkExecutionEnvelope | None = None

    @property
    def execution_unit(self) -> ExecutionStage | None:
        return self.execution_stage


@dataclass(frozen=True, slots=True)
class DistributedGenerationPlan:
    """Driver-side plan plus worker placement for one generation request."""

    engine_plan: EnginePlan
    assignments: tuple[DeviceAssignment, ...]


class DistributedExecutionPlanner:
    """Plan chunk placement across generation workers."""

    def __init__(self, capability: Any | None = None) -> None:
        self.capability = family_capability_from_value(capability)

    def plan(
        self,
        request: GenerationRequest,
        workers: list[DistributedWorkerHandle],
    ) -> list[DeviceAssignment]:
        return list(self.plan_with_engine(request, workers).assignments)

    def plan_with_engine(
        self,
        request: GenerationRequest,
        workers: list[DistributedWorkerHandle],
        *,
        sample_rows: list[GenerationSampleRow] | None = None,
    ) -> DistributedGenerationPlan:
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
                "distributed generation planning",
            )
        engine_plan = build_engine_plan(
            request,
            sample_rows,
            capability=capability,
            max_samples_per_chunk=max(1, max_samples),
        )
        plan_summary = engine_plan.summary()
        capability_summary = engine_plan.capability.to_dict()
        assignments: list[DeviceAssignment] = []
        for idx, chunk in enumerate(engine_plan.chunks):
            worker = workers[idx % len(workers)]
            chunk_stage = engine_plan.chunk_stage_for(chunk)
            envelope = ChunkExecutionEnvelope(
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
                ),
            )
        return DistributedGenerationPlan(
            engine_plan=engine_plan,
            assignments=tuple(assignments),
        )

__all__ = [
    "DeviceAssignment",
    "DistributedExecutionPlanner",
    "DistributedGenerationPlan",
]
