"""Types shared by Ray-backed generation execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vrl.generation.execution.microbatching import MicroBatchSample
from vrl.generation.execution.planner import ExecutionStage
from vrl.generation.protocols import PipelineChunkResult
from vrl.generation.types import GenerationRequest


@dataclass(frozen=True, slots=True)
class RayWorkerHandle:
    """Scheduler-visible metadata for one generation worker actor."""

    worker_id: str
    node_id: str
    gpu_ids: tuple[int, ...] = ()
    actor: Any | None = None


@dataclass(frozen=True, slots=True)
class RayChunkExecutionEnvelope:
    """Authoritative chunk execution payload sent from the driver to a worker."""

    request: GenerationRequest
    chunk: MicroBatchSample
    plan_id: str
    execution_stage: ExecutionStage | None = None
    profiler_label: str | None = None
    capability_summary: dict[str, Any] = field(default_factory=dict)
    plan_summary: dict[str, Any] = field(default_factory=dict)

    @property
    def stage_id(self) -> str | None:
        return None if self.execution_stage is None else self.execution_stage.stage_id

    @property
    def stage_name(self) -> str | None:
        return None if self.execution_stage is None else self.execution_stage.name

    @property
    def execution_unit(self) -> ExecutionStage | None:
        return self.execution_stage

    @property
    def unit_id(self) -> str | None:
        return self.stage_id

    @property
    def unit_name(self) -> str | None:
        return self.stage_name

    @property
    def chunk_key(self) -> str:
        return self.chunk.chunk_key


@dataclass(slots=True)
class RayChunkResult:
    """Envelope returned by a generation worker for one generation chunk."""

    request_id: str
    worker_id: str
    chunk: MicroBatchSample
    output: PipelineChunkResult | None
    metrics: dict[str, Any] = field(default_factory=dict)
    plan_id: str | None = None
    stage_id: str | None = None
    stage_name: str | None = None
    profiler_label: str | None = None
    chunk_key: str | None = None
    policy_version: int | None = None
    error: str | None = None

    @property
    def unit_id(self) -> str | None:
        return self.stage_id

    @property
    def unit_name(self) -> str | None:
        return self.stage_name


__all__ = [
    "RayChunkExecutionEnvelope",
    "RayChunkResult",
    "RayWorkerHandle",
]
