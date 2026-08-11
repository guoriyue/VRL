"""Chunk placement planning for distributed generation.

The fleet-only layer above planner.py: planner builds the runtime-neutral
chunk list every executor consumes, while this module decides which worker
runs each chunk — a question that only exists for the Ray runtime
(vrl/generation/ray), never for the direct in-process path. Chunk memory
sizing (probe fit, occupancy snapshots, drift shadow) lives in
``chunk_memory.py``; placement consumes none of it — the chunk width is
already resolved by the time a request reaches planning.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import get_args

from vrl.generation.execution.chunks import SampleChunk
from vrl.generation.execution.planner import EnginePlan, build_engine_plan
from vrl.generation.execution.types import (
    ChunkExecutionEnvelope,
    ChunkPlacementStrategy,
)
from vrl.generation.types import GenerationRequest


@dataclass(frozen=True, slots=True)
class DeviceAssignment:
    """Map one logical chunk to one generation worker.

    ``worker_id`` is ``None`` under dynamic placement — binding then happens at
    dispatch time in the actor pool, not at plan time. The envelope is the wire
    payload and single source of truth for chunk identity.
    """

    worker_id: str | None
    envelope: ChunkExecutionEnvelope
    estimated_cost: float = 0.0

    @property
    def chunk(self) -> SampleChunk:
        """Return the chunk carried by the authoritative wire envelope."""

        return self.envelope.chunk


@dataclass(frozen=True, slots=True)
class DistributedGenerationPlan:
    """Driver-side plan plus worker placement for one generation request."""

    engine_plan: EnginePlan
    assignments: tuple[DeviceAssignment, ...]


class DistributedExecutionPlanner:
    """Plan chunk placement across generation workers.

    ``round_robin`` binds at plan time. ``dynamic`` leaves chunks unbound so
    the dispatch loop can pull the highest-cost pending chunk onto a free worker.
    """

    def __init__(
        self,
        *,
        strategy: ChunkPlacementStrategy = "round_robin",
    ) -> None:
        allowed = get_args(ChunkPlacementStrategy)
        if strategy not in allowed:
            raise ValueError(
                f"chunk placement strategy must be one of {', '.join(allowed)}; got {strategy!r}",
            )
        self.strategy = strategy

    def plan_with_engine(
        self,
        request: GenerationRequest,
        worker_ids: Sequence[str],
    ) -> DistributedGenerationPlan:
        worker_ids = tuple(worker_ids)
        if not worker_ids:
            raise ValueError("DistributedExecutionPlanner requires at least one worker")
        if any(not worker_id for worker_id in worker_ids):
            raise ValueError("DistributedExecutionPlanner worker IDs must be non-empty")
        if len(set(worker_ids)) != len(worker_ids):
            raise ValueError("DistributedExecutionPlanner worker IDs must be unique")
        engine_plan = build_engine_plan(request)
        bind_at_plan_time = self.strategy == "round_robin"
        steps = request.sampling.get("num_steps") or request.sampling.get(
            "max_new_tokens",
        )
        cost_per_sample = max(1, int(steps or 1))
        assignments: list[DeviceAssignment] = []
        for idx, chunk in enumerate(engine_plan.chunks):
            worker_id = worker_ids[idx % len(worker_ids)] if bind_at_plan_time else None
            envelope = ChunkExecutionEnvelope(
                request=request,
                chunk=chunk,
            )
            assignments.append(
                DeviceAssignment(
                    worker_id=worker_id,
                    envelope=envelope,
                    estimated_cost=float(chunk.sample_count * cost_per_sample),
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
