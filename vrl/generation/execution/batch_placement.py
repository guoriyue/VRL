"""Batch placement planning for distributed generation.

The fleet-only layer above planner.py: planner builds the runtime-neutral
batch list every executor consumes, while this module decides which engine
runs each batch — a question that only exists for the Ray runtime
(vrl/generation/ray), never for the direct in-process path. Batch memory
sizing (probe fit, occupancy snapshots, drift shadow) lives in
``batch_memory.py``; placement consumes none of it — the batch width is
already resolved by the time a request reaches planning.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import get_args

from vrl.generation.execution.planner import EnginePlan
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.execution.types import (
    BatchPlacementStrategy,
    GenerationBatchEnvelope,
)
from vrl.generation.types import GenerationRequest


@dataclass(frozen=True, slots=True)
class DeviceAssignment:
    """Map one logical batch to one generation engine.

    ``engine_id`` is ``None`` under dynamic placement — binding then happens at
    dispatch time in the actor pool, not at plan time. The envelope is the wire
    payload and single source of truth for batch identity.
    """

    engine_id: str | None
    envelope: GenerationBatchEnvelope
    estimated_cost: float = 0.0

    @property
    def batch(self) -> GenerationSampleBatch:
        """Return the batch carried by the authoritative wire envelope."""

        return self.envelope.batch


@dataclass(frozen=True, slots=True)
class DistributedGenerationPlan:
    """Driver-side plan plus engine placement for one generation request."""

    engine_plan: EnginePlan
    assignments: tuple[DeviceAssignment, ...]


class DistributedExecutionPlanner:
    """Plan batch placement across generation engines.

    ``round_robin`` binds at plan time. ``dynamic`` leaves batches unbound so
    the dispatch loop can pull the highest-cost pending batch onto a free engine.
    """

    def __init__(
        self,
        *,
        strategy: BatchPlacementStrategy = "round_robin",
    ) -> None:
        allowed = get_args(BatchPlacementStrategy)
        if strategy not in allowed:
            raise ValueError(
                f"batch placement strategy must be one of {', '.join(allowed)}; got {strategy!r}",
            )
        self.strategy = strategy

    def plan_with_engine(
        self,
        request: GenerationRequest,
        engine_ids: Sequence[str],
    ) -> DistributedGenerationPlan:
        engine_ids = tuple(engine_ids)
        if not engine_ids:
            raise ValueError("DistributedExecutionPlanner requires at least one engine")
        if any(not engine_id for engine_id in engine_ids):
            raise ValueError("DistributedExecutionPlanner engine IDs must be non-empty")
        if len(set(engine_ids)) != len(engine_ids):
            raise ValueError("DistributedExecutionPlanner engine IDs must be unique")
        engine_plan = EnginePlan.from_request(request)
        bind_at_plan_time = self.strategy == "round_robin"
        # Diffusion requests carry num_steps; AR requests cost one unit per
        # sample (no request key names their token budget).
        cost_per_sample = max(1, int(request.sampling.get("num_steps") or 1))
        assignments: list[DeviceAssignment] = []
        for idx, batch in enumerate(engine_plan.sample_batches):
            engine_id = engine_ids[idx % len(engine_ids)] if bind_at_plan_time else None
            envelope = GenerationBatchEnvelope(
                request=request,
                batch=batch,
            )
            assignments.append(
                DeviceAssignment(
                    engine_id=engine_id,
                    envelope=envelope,
                    estimated_cost=float(batch.sample_count * cost_per_sample),
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
