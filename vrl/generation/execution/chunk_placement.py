"""Chunk placement planning for distributed generation."""

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

_PLACEMENT_STRATEGIES = ("round_robin", "dynamic")


@dataclass(frozen=True, slots=True)
class ChunkPlacementPolicy:
    """How chunks bind to generation workers.

    ``round_robin`` binds at plan time (the historical baseline).
    ``dynamic`` leaves chunks unbound: the dispatch loop pulls the next
    pending chunk onto the first worker with a free slot, submitting by
    ``estimated_cost`` descending (LPT) so a large chunk never anchors the
    tail. With one worker the two strategies are equivalent.
    """

    strategy: str = "round_robin"

    def __post_init__(self) -> None:
        if self.strategy not in _PLACEMENT_STRATEGIES:
            raise ValueError(
                "chunk placement strategy must be one of "
                f"{', '.join(_PLACEMENT_STRATEGIES)}; got {self.strategy!r}",
            )


@dataclass(frozen=True, slots=True)
class DeviceAssignment:
    """Map one logical chunk to one generation worker.

    Worker fields are ``None`` under dynamic placement — binding then
    happens at dispatch time in the actor pool, not at plan time.
    """

    worker_id: str | None
    node_id: str | None
    gpu_ids: tuple[int, ...]
    chunk: SampleChunk
    execution_stage: ExecutionStage | None = None
    envelope: ChunkExecutionEnvelope | None = None
    estimated_cost: float = 0.0


@dataclass(frozen=True, slots=True)
class DistributedGenerationPlan:
    """Driver-side plan plus worker placement for one generation request."""

    engine_plan: EnginePlan
    assignments: tuple[DeviceAssignment, ...]


def estimate_chunk_cost(request: GenerationRequest, chunk: SampleChunk) -> float:
    """Family-neutral relative cost of one chunk (scheduler hint only).

    Diffusion work scales with samples x denoise steps; AR with samples x
    new tokens. This must never influence sample identity or gather order —
    it only orders dispatch under dynamic placement and labels telemetry.
    """

    steps = request.sampling.get("num_steps") or request.sampling.get(
        "max_new_tokens",
    )
    return float(chunk.sample_count * max(1, int(steps or 1)))


class DistributedExecutionPlanner:
    """Plan chunk placement across generation workers."""

    def __init__(
        self,
        capability: Any | None = None,
        *,
        policy: ChunkPlacementPolicy | None = None,
    ) -> None:
        self.capability = family_capability_from_value(capability)
        self.policy = policy or ChunkPlacementPolicy()

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
        bind_at_plan_time = self.policy.strategy == "round_robin"
        assignments: list[DeviceAssignment] = []
        for idx, chunk in enumerate(engine_plan.chunks):
            worker = workers[idx % len(workers)] if bind_at_plan_time else None
            chunk_stage = engine_plan.chunk_stage_for(chunk)
            envelope = ChunkExecutionEnvelope(
                request=request,
                chunk=chunk,
                plan_id=engine_plan.request_id,
                execution_stage=chunk_stage,
                profiler_label=chunk_stage.profiler_name,
            )
            assignments.append(
                DeviceAssignment(
                    worker_id=worker.worker_id if worker else None,
                    node_id=worker.node_id if worker else None,
                    gpu_ids=worker.gpu_ids if worker else (),
                    chunk=chunk,
                    execution_stage=chunk_stage,
                    envelope=envelope,
                    estimated_cost=estimate_chunk_cost(request, chunk),
                ),
            )
        return DistributedGenerationPlan(
            engine_plan=engine_plan,
            assignments=tuple(assignments),
        )

__all__ = [
    "ChunkPlacementPolicy",
    "DeviceAssignment",
    "DistributedExecutionPlanner",
    "DistributedGenerationPlan",
    "estimate_chunk_cost",
]
