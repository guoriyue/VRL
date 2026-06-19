"""Types for distributed generation execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vrl.generation.execution.chunks import SampleChunk
from vrl.generation.execution.planner import ExecutionStage
from vrl.generation.protocols import ChunkResult
from vrl.generation.types import GenerationRequest


@dataclass(frozen=True, slots=True)
class DistributedWorkerHandle:
    """Scheduler-visible metadata for one generation worker actor."""

    worker_id: str
    node_id: str
    gpu_ids: tuple[int, ...] = ()
    actor: Any | None = None


@dataclass(frozen=True, slots=True)
class ChunkExecutionEnvelope:
    """Authoritative chunk execution payload sent from the driver to a worker."""

    request: GenerationRequest
    chunk: SampleChunk
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
    def chunk_key(self) -> str:
        return self.chunk.chunk_key


@dataclass(slots=True)
class ChunkExecutionResult:
    """Envelope returned by a generation worker for one generation chunk."""

    request_id: str
    worker_id: str
    chunk: SampleChunk
    output: ChunkResult | None
    metrics: dict[str, Any] = field(default_factory=dict)
    plan_id: str | None = None
    stage_id: str | None = None
    stage_name: str | None = None
    profiler_label: str | None = None
    chunk_key: str | None = None
    policy_version: int | None = None
    error: str | None = None
    # Set when the chunk could not run because the worker no longer retains a
    # trainable-state slot for ``request.policy_version`` (the request outlived
    # the slot-retention window under a non-draining weight sync). Distinct from a
    # real generation failure so the caller counts it as a stale discard, not an
    # error — see SPRINT_shadow_model_weight_sync.md.
    stale_slot: bool = False


__all__ = [
    "ChunkExecutionEnvelope",
    "ChunkExecutionResult",
    "DistributedWorkerHandle",
]
