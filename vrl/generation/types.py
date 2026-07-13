"""Typed generation runtime payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vrl.trajectory import TrajectoryBatch


@dataclass(slots=True)
class GenerationInput:
    """One prompt and its functional conditioning inputs."""

    prompt: str
    task_type: str | None = None
    reference_image: str | None = None
    reference_video: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError("GenerationInput.prompt must be non-empty")
        for name in ("reference_image", "reference_video"):
            if getattr(self, name) == "":
                raise ValueError(f"GenerationInput.{name} must be None or non-empty")


@dataclass(slots=True, init=False)
class GenerationRequest:
    """One generation request submitted to the engine."""

    request_id: str
    family: str
    task: str
    inputs: list[GenerationInput]
    samples_per_prompt: int
    sampling: dict[str, Any] = field(default_factory=dict)
    return_artifacts: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    policy_version: int | None = None

    def __init__(
        self,
        request_id: str,
        family: str,
        task: str,
        inputs: list[GenerationInput | str],
        samples_per_prompt: int,
        sampling: dict[str, Any] | None = None,
        return_artifacts: set[str] | None = None,
        metadata: dict[str, Any] | None = None,
        priority: int = 0,
        policy_version: int | None = None,
    ) -> None:
        normalized_inputs: list[GenerationInput] = []
        for value in inputs:
            if isinstance(value, GenerationInput):
                normalized_inputs.append(value)
            elif isinstance(value, str):
                normalized_inputs.append(GenerationInput(prompt=value))
            else:
                raise TypeError(
                    "GenerationRequest.inputs must contain GenerationInput or str",
                )
        self.request_id = request_id
        self.family = family
        self.task = task
        self.inputs = normalized_inputs
        self.samples_per_prompt = samples_per_prompt
        self.sampling = dict(sampling or {})
        self.return_artifacts = set(return_artifacts or ())
        self.metadata = dict(metadata or {})
        self.priority = priority
        self.policy_version = policy_version
        self.__post_init__()

    @property
    def prompts(self) -> list[str]:
        """Text-only view used by family-agnostic execution code."""

        return [value.prompt for value in self.inputs]

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("GenerationRequest.request_id must be non-empty")
        if not self.family:
            raise ValueError("GenerationRequest.family must be non-empty")
        if not self.task:
            raise ValueError("GenerationRequest.task must be non-empty")
        if not self.inputs:
            raise ValueError("GenerationRequest.inputs must be non-empty")
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
    extra: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


__all__ = [
    "GenerationInput",
    "GenerationOutput",
    "GenerationRequest",
    "GenerationSampleRow",
]
