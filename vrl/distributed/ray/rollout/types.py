"""Types shared by Ray-backed rollout execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vrl.engine.core.protocols import PipelineChunkResult
from vrl.engine.core.types import GenerationRequest
from vrl.engine.execution.microbatching import MicroBatchPlan
from vrl.engine.execution.planner import ExecutionUnit


@dataclass(frozen=True, slots=True)
class RayWorkerHandle:
    """Scheduler-visible metadata for one rollout worker actor."""

    worker_id: str
    node_id: str
    gpu_ids: tuple[int, ...] = ()
    actor: Any | None = None


@dataclass(frozen=True, slots=True)
class RayChunkExecutionEnvelope:
    """Authoritative chunk execution payload sent from the driver to a worker."""

    request: GenerationRequest
    chunk: MicroBatchPlan
    plan_id: str
    execution_unit: ExecutionUnit | None = None
    profiler_label: str | None = None
    capability_summary: dict[str, Any] = field(default_factory=dict)
    plan_summary: dict[str, Any] = field(default_factory=dict)

    @property
    def unit_id(self) -> str | None:
        return None if self.execution_unit is None else self.execution_unit.unit_id

    @property
    def unit_name(self) -> str | None:
        return None if self.execution_unit is None else self.execution_unit.name

    @property
    def chunk_key(self) -> str:
        return self.chunk.chunk_key


@dataclass(slots=True)
class RayChunkResult:
    """Envelope returned by a rollout worker for one generation chunk."""

    request_id: str
    worker_id: str
    chunk: MicroBatchPlan
    output: PipelineChunkResult | None
    metrics: dict[str, Any] = field(default_factory=dict)
    plan_id: str | None = None
    unit_id: str | None = None
    unit_name: str | None = None
    profiler_label: str | None = None
    chunk_key: str | None = None
    policy_version: int | None = None
    error: str | None = None


__all__ = [
    "RayChunkExecutionEnvelope",
    "RayChunkResult",
    "RayWorkerHandle",
]
