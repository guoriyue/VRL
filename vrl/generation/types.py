"""Typed generation runtime payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vrl.generation.capabilities import FamilyCapability
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
    micro_batches: int = 0
    trajectory_kind: str | None = None
    execution_stages: tuple[str, ...] = ()
    engine_plan_id: str | None = None
    engine_counters: dict[str, Any] = field(default_factory=dict)

    @property
    def execution_units(self) -> tuple[str, ...]:
        return self.execution_stages

    @execution_units.setter
    def execution_units(self, value: tuple[str, ...]) -> None:
        self.execution_stages = tuple(value)


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


@dataclass(frozen=True, slots=True)
class WorkloadSignature:
    """Batch grouping key for generation requests."""

    family: str
    task: str
    height: int | None
    width: int | None
    num_frames: int | None
    num_steps: int | None
    artifact_mode: tuple[str, ...]
    max_new_tokens: int | None = None
    trajectory_kind: str | None = None
    axis_kinds: tuple[tuple[str, str], ...] = ()
    capability_key: tuple[Any, ...] = ()

    @classmethod
    def from_request(cls, request: GenerationRequest) -> WorkloadSignature:
        sampling = request.sampling
        height = sampling.get("height")
        width = sampling.get("width")
        num_frames = sampling.get("num_frames", sampling.get("frame_count"))
        num_steps = sampling.get("num_steps", sampling.get("num_inference_steps"))
        max_new_tokens = sampling.get(
            "max_new_tokens",
            sampling.get("max_new_image_tokens"),
        )
        return cls(
            family=request.family,
            task=request.task,
            height=None if height is None else int(height),
            width=None if width is None else int(width),
            num_frames=None if num_frames is None else int(num_frames),
            num_steps=None if num_steps is None else int(num_steps),
            artifact_mode=tuple(sorted(request.return_artifacts)),
            max_new_tokens=None if max_new_tokens is None else int(max_new_tokens),
        )

    @classmethod
    def from_request_and_capability(
        cls,
        request: GenerationRequest,
        capability: FamilyCapability,
    ) -> WorkloadSignature:
        signature = cls.from_request(request)
        return cls(
            family=signature.family,
            task=signature.task,
            height=signature.height,
            width=signature.width,
            num_frames=signature.num_frames,
            num_steps=signature.num_steps,
            artifact_mode=signature.artifact_mode,
            max_new_tokens=signature.max_new_tokens,
            trajectory_kind=capability.trajectory_kind,
            axis_kinds=tuple((axis.name, axis.kind) for axis in capability.expected_axes),
            capability_key=capability.batch_signature(),
        )


@dataclass(slots=True)
class OutputBatch:
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


GenerationOutput = OutputBatch


__all__ = [
    "GenerationMetrics",
    "GenerationOutput",
    "GenerationRequest",
    "GenerationSampleRow",
    "OutputBatch",
    "WorkloadSignature",
]
