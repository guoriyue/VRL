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

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from vrl.utils.validation import require_int

if TYPE_CHECKING:
    from vrl.generation.steps.denoise.config import DenoiseRequestOptions
    from vrl.trajectory.storage import TrajectoryStoragePolicy
    from vrl.trajectory.types import TrajectoryBatch


@dataclass(slots=True)
class GenerationInput:
    """One prompt and its functional conditioning inputs."""

    prompt: str
    task_type: str | None = None
    reference_video: str | None = None
    # Ordered conditioning images. Single-image families (Wan I2V, Cosmos
    # Video2World, MAGI-1 i2v) require exactly one; Qwen-Image-2.1 takes any number.
    reference_images: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError("GenerationInput.prompt must be non-empty")
        if self.reference_video == "":
            raise ValueError("GenerationInput.reference_video must be None or non-empty")
        if not isinstance(self.reference_images, list) or any(
            not isinstance(path, str) or not path.strip() for path in self.reference_images
        ):
            raise ValueError("GenerationInput.reference_images must be a list of non-empty paths")
        self.reference_images = list(self.reference_images)


@dataclass(slots=True)
class DenoiseRequest:
    """Backend-neutral parameters for one image or video generation call.

    Geometry and schedule have no defaults: the sampling config is their single
    source (``DenoiseRequestLayout.parse_sampling_params`` and the eval
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

    def __post_init__(self) -> None:
        for name in ("width", "height", "frame_count", "num_steps"):
            require_int(getattr(self, name), path=f"DenoiseRequest.{name}", minimum=1)
        if self.fps is not None:
            require_int(self.fps, path="DenoiseRequest.fps", minimum=1)
        if self.seed is not None:
            require_int(self.seed, path="DenoiseRequest.seed")


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
    # until the Ray runtime's startup probe rewrites it to an int) and
    # ``trajectory_storage`` is applied worker-side before tensors cross the wire.
    samples_per_generation_batch: int | Literal["auto"] | None = None
    trajectory_storage: TrajectoryStoragePolicy | None = None
    # Rollout-owned denoise knobs (rollout.* / rollout.sde.*), projected once by
    # the collector. ``None`` on hand-built requests means the option defaults.
    denoise: DenoiseRequestOptions | None = None
    # Request-owned fallback randomness for the SDE window. Copies sent to
    # separate workers retain it without imposing a latent-noise sampling seed.
    sde_window_seed: int | None = None
    # One seed per sample row (``sample_rows()`` order: prompt-major), each
    # naming the row's INITIAL latent only. The rollout layer plans them: rows
    # that must start from the same latent carry the same seed (a GRPO group
    # under ``rollout.group_shared_noise``), others carry distinct ones. ``None``
    # leaves every row to the family's own draw from ``sampling.seed``. Per
    # row, not per batch, so an OOM split keeps each row's start. The SDE step
    # noise is seeded separately and stays per sample either way.
    initial_noise_seeds: tuple[int, ...] | None = None
    runtime_debug: bool = False
    policy_version: int | None = None
    # Online reward transport: Ray actors return boxed media references instead
    # of decoded tensors. Direct/in-process generation retains its tensor API.
    reward_media_refs: bool = False

    def __init__(
        self,
        request_id: str,
        family: str,
        task: str,
        inputs: list[GenerationInput | str],
        samples_per_prompt: int,
        *,
        sampling: dict[str, Any] | None = None,
        samples_per_generation_batch: int | Literal["auto"] | None = None,
        trajectory_storage: TrajectoryStoragePolicy | None = None,
        denoise: DenoiseRequestOptions | None = None,
        runtime_debug: bool = False,
        policy_version: int | None = None,
        sde_window_seed: int | None = None,
        initial_noise_seeds: tuple[int, ...] | list[int] | None = None,
        reward_media_refs: bool = False,
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
        self.trajectory_storage = trajectory_storage
        self.denoise = denoise
        self.sde_window_seed = sde_window_seed
        self.initial_noise_seeds = (
            None if initial_noise_seeds is None else tuple(initial_noise_seeds)
        )
        self.reward_media_refs = bool(reward_media_refs)
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
        if type(self.samples_per_prompt) is not int or self.samples_per_prompt < 1:
            raise ValueError("GenerationRequest.samples_per_prompt must be an integer >= 1")
        width = self.samples_per_generation_batch
        if width is not None and width != "auto" and (type(width) is not int or width < 1):
            raise ValueError(
                "GenerationRequest.samples_per_generation_batch must be >= 1 or 'auto'",
            )
        if not isinstance(self.runtime_debug, bool):
            raise TypeError("GenerationRequest.runtime_debug must be a bool")
        if self.policy_version is not None:
            require_int(
                self.policy_version,
                path="GenerationRequest.policy_version",
                minimum=0,
            )

        if self.sde_window_seed is not None:
            require_int(self.sde_window_seed, path="GenerationRequest.sde_window_seed", minimum=0)
        elif (
            self.denoise is not None
            and self.denoise.sde_window_size > 0
            and self.sampling.get("seed") is None
        ):
            self.sde_window_seed = random.getrandbits(64)
        seeds = self.initial_noise_seeds
        if seeds is not None:
            expected = len(self.inputs) * self.samples_per_prompt
            if len(seeds) != expected:
                raise ValueError(
                    "GenerationRequest.initial_noise_seeds must carry one seed per sample "
                    f"row ({expected}), got {len(seeds)}",
                )
            for index, seed in enumerate(seeds):
                require_int(
                    seed, path=f"GenerationRequest.initial_noise_seeds[{index}]", minimum=0
                )

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

    def validate_batch_range(
        self,
        *,
        prompt_index: int,
        sample_start: int,
        sample_count: int,
    ) -> None:
        """Validate a batch's prompt/sample range against its source request."""

        for name, value in (
            ("prompt_index", prompt_index),
            ("sample_start", sample_start),
            ("sample_count", sample_count),
        ):
            if type(value) is not int:
                raise ValueError(f"batch.{name} must be an integer, got {value!r}")
        if prompt_index < 0 or prompt_index >= len(self.inputs):
            raise ValueError(f"batch.prompt_index={prompt_index} is out of range")
        sample_end = sample_start + sample_count
        if sample_start < 0 or sample_count < 1:
            raise ValueError(
                "batch sample range must have non-negative start and positive count",
            )
        if sample_end > self.samples_per_prompt:
            raise ValueError(
                "batch sample range exceeds self.samples_per_prompt: "
                f"{sample_start}:{sample_end} > {self.samples_per_prompt}",
            )


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

    # Decoded media, or sample-ordered boxed MediaReferences for online scoring.
    output: Any
    trajectory: TrajectoryBatch
    # Display/provenance-only: optional scheduler/worker diagnostics requested
    # explicitly by GenerationRequest.runtime_debug.
    runtime_debug: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        from vrl.trajectory.types import TrajectoryBatch

        if not isinstance(self.trajectory, TrajectoryBatch):
            raise TypeError("GenerationOutput.trajectory must be a TrajectoryBatch")

    @property
    def reward_media(self) -> Any:
        """Expose gathered media to the Ray transport adapter."""

        return self.output

    @reward_media.setter
    def reward_media(self, value: Any) -> None:
        self.output = value

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
