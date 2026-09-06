"""Typed generation runtime payloads.

The data half of the collector <-> engine contract in protocols.py. These
dataclasses cross two boundaries at once — the collector-facing runtime API
and the driver -> Ray-worker wire (a ``GenerationRequest`` rides inside every
``GenerationBatchEnvelope``) — so they live apart from any runtime and
validate themselves at construction. The module is torch-free at import and
imports ``TrajectoryBatch`` lazily, because config parsing reaches this
package long before any model exists.

The reward-side dual is vrl/rewards/types.py (``RewardSample`` /
``RewardOutput``); trainer-side batch semantics stay outside (see
``GenerationOutput``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from vrl.trajectory import TrajectoryBatch, TrajectoryStoragePolicy


@dataclass(slots=True)
class GenerationInput:
    """One prompt and its functional conditioning inputs."""

    prompt: str
    task_type: str | None = None
    reference_image: str | None = None
    reference_video: str | None = None

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError("GenerationInput.prompt must be non-empty")
        for name in ("reference_image", "reference_video"):
            if getattr(self, name) == "":
                raise ValueError(f"GenerationInput.{name} must be None or non-empty")


@dataclass(slots=True)
class DenoiseRequest:
    """Backend-neutral parameters for one image or video generation call.

    Geometry and schedule have no defaults: the sampling config is their single
    source (``DiffusionRequestLayout.parse_sampling_params`` and the eval
    scripts always pass them). ``fps`` is ``None`` for image families and for
    video families that resolve their own rate; models read ``request.fps or
    <family default>``.
    """

    width: int
    height: int
    frame_count: int
    num_steps: int
    guidance_scale: float
    negative_prompt: str = ""
    seed: int | None = None
    fps: int | None = None


@dataclass(slots=True, init=False)
class GenerationRequest:
    """One generation request submitted to the engine."""

    request_id: str
    # display/provenance-only: names the fleet this request was built for. The
    # worker verifies its executor against the launch contract, not against
    # this token; ``task`` is behavior-consumed (reward modality selection).
    family: str
    task: str
    inputs: list[GenerationInput]
    samples_per_prompt: int
    sampling: dict[str, Any] = field(default_factory=dict)
    # Engine-level knobs read by family-neutral code (planner, executor,
    # trajectory builders). They are request fields, not sampling keys:
    # ``samples_per_generation_batch`` is the planner batch width (``"auto"``
    # until the Ray runtime's startup probe rewrites it to an int),
    # ``train_segments`` marks which multi-segment outputs are trainable, and
    # ``trajectory_storage`` is applied worker-side before tensors cross the wire.
    samples_per_generation_batch: int | Literal["auto"] | None = None
    train_segments: dict[str, bool] | None = None
    trajectory_storage: TrajectoryStoragePolicy | None = None
    runtime_debug: bool = False
    policy_version: int | None = None

    def __init__(
        self,
        request_id: str,
        family: str,
        task: str,
        inputs: list[GenerationInput | str],
        samples_per_prompt: int,
        sampling: dict[str, Any] | None = None,
        samples_per_generation_batch: int | Literal["auto"] | None = None,
        train_segments: dict[str, bool] | None = None,
        trajectory_storage: TrajectoryStoragePolicy | None = None,
        runtime_debug: bool = False,
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
        self.samples_per_generation_batch = samples_per_generation_batch
        self.train_segments = None if train_segments is None else dict(train_segments)
        self.trajectory_storage = trajectory_storage
        self.runtime_debug = runtime_debug
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
        width = self.samples_per_generation_batch
        if width is not None and width != "auto" and width < 1:
            raise ValueError(
                "GenerationRequest.samples_per_generation_batch must be >= 1 or 'auto'",
            )
        if not isinstance(self.runtime_debug, bool):
            raise TypeError("GenerationRequest.runtime_debug must be a bool")
        if self.policy_version is not None and self.policy_version < 0:
            raise ValueError("GenerationRequest.policy_version must be >= 0")

    def sample_rows(self) -> list[GenerationSampleRow]:
        """Mint the deterministic per-sample identity rows for this request.

        Sample identity is minted exactly once, here: the driver derives the
        rows before dispatching batches, and everything downstream joins on
        them — batch gatherers check exact coverage against these rows,
        ``TrajectoryBatch.sample_rows`` records them, and reward scoring keys
        ``RewardSample.sample_id`` off them. That cross-package join is why
        the derivation is request-owned and must stay deterministic
        (tests/generation/execution/test_generation_contracts.py pins this).
        """

        rows: list[GenerationSampleRow] = []
        for prompt_index, request_input in enumerate(self.inputs):
            prompt = request_input.prompt
            prompt_id = f"{self.request_id}:prompt:{prompt_index}"
            for sample_index in range(self.samples_per_prompt):
                sample_id = f"{prompt_id}:sample:{sample_index}"
                rows.append(
                    GenerationSampleRow(
                        prompt_index=prompt_index,
                        sample_index=sample_index,
                        prompt=prompt,
                        sample_id=sample_id,
                    )
                )
        return rows


@dataclass(slots=True)
class GenerationSampleRow:
    """Expanded sample-level unit inside a generation request."""

    prompt_index: int
    sample_index: int
    prompt: str
    sample_id: str


@dataclass(slots=True)
class GenerationOutput:
    """Engine runtime output batch.

    This is the generation-side output, not the trainer-side RolloutBatch.
    Reward, advantage, and GRPO group semantics stay outside this type.
    """

    output: Any
    trajectory: TrajectoryBatch
    # Display/provenance-only: optional scheduler/worker diagnostics requested
    # explicitly by GenerationRequest.runtime_debug.
    runtime_debug: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        from vrl.trajectory import TrajectoryBatch

        if not isinstance(self.trajectory, TrajectoryBatch):
            raise TypeError("GenerationOutput.trajectory must be a TrajectoryBatch")

    @property
    def request_id(self) -> str:
        """Return the request identity owned by the trajectory record."""

        return self.trajectory.request_id

    @property
    def sample_rows(self) -> list[GenerationSampleRow]:
        """Return the sample identities owned by the trajectory record."""

        return self.trajectory.sample_rows


__all__ = [
    "DenoiseRequest",
    "GenerationInput",
    "GenerationOutput",
    "GenerationRequest",
    "GenerationSampleRow",
]
