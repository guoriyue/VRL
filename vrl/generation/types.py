"""Typed generation runtime payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vrl.generation.execution.planner import EnginePlan
    from vrl.trajectory import TrajectoryBatch


@dataclass(slots=True)
class GenerationMetrics:
    """Runtime-only generation metrics.

    These metrics describe engine execution. Reward and trainer metrics stay
    outside this module.
    """

    queue_wait_s: float = 0.0
    execution_s: float = 0.0
    peak_memory_mb: float | None = None
    num_prompts: int = 0
    num_samples: int = 0
    num_steps: int | None = None
    chunks: int = 0
    trajectory_kind: str | None = None
    execution_stages: tuple[str, ...] = ()
    engine_plan_id: str | None = None
    engine_counters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationRequest:
    """One generation request submitted to the engine."""

    request_id: str
    family: str
    task: str
    prompts: list[str]
    samples_per_prompt: int
    sampling: dict[str, Any] = field(default_factory=dict)
    return_artifacts: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    policy_version: int | None = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("GenerationRequest.request_id must be non-empty")
        if not self.family:
            raise ValueError("GenerationRequest.family must be non-empty")
        if not self.task:
            raise ValueError("GenerationRequest.task must be non-empty")
        if not self.prompts:
            raise ValueError("GenerationRequest.prompts must be non-empty")
        if self.samples_per_prompt < 1:
            raise ValueError("GenerationRequest.samples_per_prompt must be >= 1")
        if self.policy_version is not None and self.policy_version < 0:
            raise ValueError("GenerationRequest.policy_version must be >= 0")


@dataclass(slots=True)
class GenerationSampleRow:
    """Expanded sample-level unit inside a generation request."""

    prompt_index: int
    sample_index: int
    prompt: str
    prompt_id: str
    group_id: str
    sample_id: str
    trajectory_id: str
    seed: int | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationOutput:
    """Engine runtime output batch.

    This is the generation-side output, not the trainer-side RolloutBatch.
    Reward, advantage, and GRPO group semantics stay outside this type.
    """

    request_id: str
    family: str
    task: str
    prompts: list[str]
    sample_rows: list[GenerationSampleRow]
    output: Any
    trajectory: TrajectoryBatch | None = None
    engine_plan: EnginePlan | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    metrics: GenerationMetrics | None = None
    peak_memory_mb: float = 0.0
    error: str | None = None


__all__ = [
    "GenerationMetrics",
    "GenerationOutput",
    "GenerationRequest",
    "GenerationSampleRow",
]
