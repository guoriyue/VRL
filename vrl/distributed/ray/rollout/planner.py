"""Distributed execution planning for large rollout chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vrl.distributed.ray.rollout.types import RayChunkExecutionEnvelope, RayWorkerHandle
from vrl.engine.core.capabilities import family_capability_from_value
from vrl.engine.core.types import GenerationRequest, GenerationSampleSpec
from vrl.engine.execution.microbatching import (
    MicroBatchPlan,
)
from vrl.engine.execution.planner import EnginePlan, ExecutionUnit, build_engine_plan


@dataclass(frozen=True, slots=True)
class DeviceAssignment:
    """Map one logical chunk to one rollout worker."""

    worker_id: str
    node_id: str
    gpu_ids: tuple[int, ...]
    chunk: MicroBatchPlan
    execution_unit: ExecutionUnit | None = None
    envelope: RayChunkExecutionEnvelope | None = None


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
        sample_specs: list[GenerationSampleSpec] | None = None,
    ) -> DistributedRolloutPlan:
        if not workers:
            raise ValueError("DistributedExecutionPlanner requires at least one worker")
        max_samples = int(
            request.sampling.get("sample_batch_size", request.samples_per_prompt),
        )
        capability = self.capability or _capability_from_request(request)
        engine_plan = build_engine_plan(
            request,
            sample_specs,
            capability=capability,
            max_samples_per_microbatch=max(1, max_samples),
        )
        plan_summary = engine_plan.summary()
        capability_summary = engine_plan.capability.to_dict()
        assignments: list[DeviceAssignment] = []
        for idx, chunk in enumerate(engine_plan.micro_batches):
            worker = workers[idx % len(workers)]
            chunk_unit = engine_plan.chunk_unit_for(chunk)
            envelope = RayChunkExecutionEnvelope(
                request=request,
                chunk=chunk,
                plan_id=engine_plan.request_id,
                execution_unit=chunk_unit,
                profiler_label=chunk_unit.profiler_name,
                capability_summary=capability_summary,
                plan_summary=plan_summary,
            )
            assignments.append(
                DeviceAssignment(
                    worker_id=worker.worker_id,
                    node_id=worker.node_id,
                    gpu_ids=worker.gpu_ids,
                    chunk=chunk,
                    execution_unit=chunk_unit,
                    envelope=envelope,
                )
            )
        return DistributedRolloutPlan(
            engine_plan=engine_plan,
            assignments=tuple(assignments),
        )


def _capability_from_request(request: GenerationRequest) -> object:
    raw = request.metadata.get("family_capability")
    capability = family_capability_from_value(raw)
    if capability is not None:
        return capability
    raise ValueError(
        "GenerationRequest.metadata['family_capability'] is required for "
        "distributed rollout planning",
    )


__all__ = [
    "DeviceAssignment",
    "DistributedExecutionPlanner",
    "DistributedRolloutPlan",
]
