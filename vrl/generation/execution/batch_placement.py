"""Batch placement planning for distributed generation.

The fleet-only layer above planner.py: planner builds the runtime-neutral
batch list every executor consumes, while this module decides which engine
runs each batch — a question that only exists for the Ray runtime
(vrl/generation/ray), never for the direct in-process path. Batches are bound
round-robin at plan time: within one request every batch shares the prompt
group's shape and step count, so their costs are equal and a static rotation
already balances the engines. Batch memory sizing (probe fit, occupancy
snapshots, drift shadow) lives in ``batch_memory.py``; placement consumes
none of it — the batch width is already resolved by the time a request
reaches planning.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from vrl.generation.execution.planner import EnginePlan
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.execution.types import GenerationBatchEnvelope
from vrl.generation.types import GenerationRequest


@dataclass(frozen=True, slots=True)
class DeviceAssignment:
    """Map one logical batch to one generation engine.

    The envelope is the wire payload and single source of truth for batch
    identity.
    """

    engine_id: str
    envelope: GenerationBatchEnvelope

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
    """Plan batch placement across generation engines: round-robin at plan time."""

    def plan_with_engine(
        self,
        request: GenerationRequest,
        engine_ids: Sequence[str],
    ) -> DistributedGenerationPlan:
        if isinstance(engine_ids, (str, bytes)):
            raise ValueError(
                "DistributedExecutionPlanner engine IDs must be a sequence of strings"
            )
        engine_ids = tuple(engine_ids)
        if not engine_ids:
            raise ValueError("DistributedExecutionPlanner requires at least one engine")
        if any(not isinstance(engine_id, str) or not engine_id for engine_id in engine_ids):
            raise ValueError("DistributedExecutionPlanner engine IDs must be non-empty strings")
        if len(set(engine_ids)) != len(engine_ids):
            raise ValueError("DistributedExecutionPlanner engine IDs must be unique")
        engine_plan = EnginePlan.from_request(request)
        assignments = tuple(
            DeviceAssignment(
                engine_id=engine_ids[idx % len(engine_ids)],
                envelope=GenerationBatchEnvelope(request=request, batch=batch),
            )
            for idx, batch in enumerate(engine_plan.sample_batches)
        )
        return DistributedGenerationPlan(
            engine_plan=engine_plan,
            assignments=assignments,
        )


__all__ = [
    "DeviceAssignment",
    "DistributedExecutionPlanner",
    "DistributedGenerationPlan",
]
