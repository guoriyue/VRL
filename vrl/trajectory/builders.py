"""Builders that project family outputs into the trajectory contract."""

from __future__ import annotations

import logging
from typing import Any

import torch

from vrl.generation.types import GenerationRequest, GenerationSampleRow
from vrl.models.families.semantics import task_modality
from vrl.trajectory.types import (
    ReplayInput,
    TrajectoryAxis,
    TrajectoryBatch,
    TrajectorySegment,
    TrajectoryTensor,
)
from vrl.trajectory.validation import (
    TrajectoryValidator,
    tensor_ref,
    validate_shape_prefix,
)
from vrl.trajectory.views import RewardInputSpec

logger = logging.getLogger(__name__)

# Sentinel marking a context value that cannot be serialized into a trajectory
# record; it is dropped rather than stored.
_DROP = object()


def build_diffusion_trajectory(
    *,
    request: GenerationRequest,
    sample_rows: list[GenerationSampleRow],
    observations: Any,
    actions: Any,
    old_log_prob: Any,
    timesteps: Any,
    replay_tensors: dict[str, Any],
    context: dict[str, Any],
) -> TrajectoryBatch:
    """Build a denoise-step trajectory from shared diffusion rollout tensors."""

    batch_size = len(sample_rows)
    timestep_count = int(old_log_prob.shape[1])
    tensors: dict[str, TrajectoryTensor] = {
        "observations": TrajectoryTensor(
            "observations",
            observations,
            ("sample", "denoise"),
            "observation",
        ),
        "actions": TrajectoryTensor(
            "actions",
            actions,
            ("sample", "denoise"),
            "action",
        ),
        "old_log_prob": TrajectoryTensor(
            "old_log_prob",
            old_log_prob,
            ("sample", "denoise"),
            "old_log_prob",
        ),
        "mask": TrajectoryTensor(
            "mask",
            torch.ones_like(old_log_prob),
            ("sample", "denoise"),
            "mask",
        ),
        "timesteps": TrajectoryTensor(
            "timesteps",
            timesteps,
            ("sample", "denoise"),
            "replay_input",
        ),
    }
    replay_tensor_names: list[str] = []
    dropped: list[str] = []
    for name, value in replay_tensors.items():
        if name in tensors:
            continue
        if not _sample_aligned(value, batch_size):
            dropped.append(name)
            continue
        tensors[name] = TrajectoryTensor(name, value, ("sample",), "replay_input")
        replay_tensor_names.append(name)
    if dropped:
        # Scalars and static tables are legitimately exported alongside the
        # per-sample tensors and live in the batch context instead; record
        # what was left out so a family's mis-shaped export is diagnosable.
        logger.debug(
            "build_diffusion_trajectory: replay tensors %s are not sample-aligned "
            "(leading dim != %d) and were left out of the trajectory",
            sorted(dropped),
            batch_size,
        )
    replay_tensor_refs = (
        tensor_ref("denoise", "observations"),
        tensor_ref("denoise", "actions"),
        tensor_ref("denoise", "timesteps"),
        *(tensor_ref("denoise", name) for name in replay_tensor_names),
    )

    reward_modality = task_modality(request.task)
    trajectory = TrajectoryBatch(
        request_id=request.request_id,
        family=request.family,
        task=request.task,
        sample_rows=list(sample_rows),
        axes={
            "sample": TrajectoryAxis("sample", "sample", batch_size),
            "denoise": TrajectoryAxis("denoise", "denoise_step", timestep_count),
        },
        segments={
            "denoise": TrajectorySegment(
                name="denoise",
                modality="latent",
                trainable=True,
                distribution="flow_matching",
                tensors=tensors,
                reward_view=reward_modality,
                replay_inputs={
                    "logprob": ReplayInput(
                        name="logprob",
                        tensor_refs=replay_tensor_refs,
                    ),
                },
            )
        },
        primary_segment="denoise",
        reward_views={
            reward_modality: RewardInputSpec(
                name=reward_modality,
                value_range="unit",
                metadata={"output_ref": "GenerationOutput.output"},
            )
        },
        context=_serializable_context(context),
    )
    return TrajectoryValidator(trajectory).validate_batch()


def build_chunk_autoregressive_denoise_trajectory(
    *,
    request: GenerationRequest,
    sample_rows: list[GenerationSampleRow],
    observations: Any,
    actions: Any,
    old_log_prob: Any,
    mask: Any,
    timesteps: Any,
    finalized_chunk_latents: Any,
    replay_tensors: dict[str, TrajectoryTensor],
    context: dict[str, Any],
) -> TrajectoryBatch:
    """Build a trainable chunk-autoregressive denoise trajectory.

    Unlike a full-sequence denoise trace, the policy decisions here have two
    independent logical axes: which temporal chunk is being produced and which
    stochastic transition inside that chunk is replayed.  Every trainable fact
    therefore starts with ``[sample, temporal_chunk, denoise_transition]``.
    Terminal chunk latents are kept separately at ``[sample, temporal_chunk]``
    so replay can reconstruct the cache/conditioning boundary between chunks.
    Extra replay tensors carry their own logical axes in TrajectoryTensor records;
    equal dimension lengths do not establish chunk or transition semantics.
    """

    batch_size = len(sample_rows)
    chunk_count, transition_count = _chunk_denoise_shape(old_log_prob, batch_size)
    for name, value in (
        ("observations", observations),
        ("actions", actions),
        ("mask", mask),
        ("timesteps", timesteps),
    ):
        validate_shape_prefix(
            name,
            value,
            (batch_size, chunk_count, transition_count),
        )
    validate_shape_prefix(
        "finalized_chunk_latents",
        finalized_chunk_latents,
        (batch_size, chunk_count),
    )

    transition_axes = ("sample", "temporal_chunk", "denoise_transition")
    tensors: dict[str, TrajectoryTensor] = {
        "observations": TrajectoryTensor(
            "observations",
            observations,
            transition_axes,
            "observation",
        ),
        "actions": TrajectoryTensor(
            "actions",
            actions,
            transition_axes,
            "action",
        ),
        "old_log_prob": TrajectoryTensor(
            "old_log_prob",
            old_log_prob,
            transition_axes,
            "old_log_prob",
        ),
        "mask": TrajectoryTensor(
            "mask",
            mask,
            transition_axes,
            "mask",
        ),
        "timesteps": TrajectoryTensor(
            "timesteps",
            timesteps,
            transition_axes,
            "replay_input",
        ),
        "finalized_chunk_latents": TrajectoryTensor(
            "finalized_chunk_latents",
            finalized_chunk_latents,
            ("sample", "temporal_chunk"),
            "replay_input",
        ),
    }
    replay_tensor_names: list[str] = []
    for name, tensor in replay_tensors.items():
        if name in tensors:
            raise ValueError(f"chunk replay tensor {name!r} conflicts with a built-in tensor")
        if tensor.role != "replay_input":
            raise ValueError(f"chunk replay tensor {name!r} must have replay_input role")
        if tensor.axes[:1] != ("sample",):
            raise ValueError(f"chunk replay tensor {name!r} axes must start with 'sample'")
        tensors[name] = tensor
        replay_tensor_names.append(name)

    reward_modality = task_modality(request.task)
    replay_tensor_refs = (
        tensor_ref("denoise", "observations"),
        tensor_ref("denoise", "actions"),
        tensor_ref("denoise", "timesteps"),
        tensor_ref("denoise", "finalized_chunk_latents"),
        *(tensor_ref("denoise", name) for name in replay_tensor_names),
    )
    trajectory = TrajectoryBatch(
        request_id=request.request_id,
        family=request.family,
        task=request.task,
        sample_rows=list(sample_rows),
        axes={
            "sample": TrajectoryAxis("sample", "sample", batch_size),
            "temporal_chunk": TrajectoryAxis(
                "temporal_chunk",
                "temporal_chunk",
                chunk_count,
            ),
            "denoise_transition": TrajectoryAxis(
                "denoise_transition",
                "denoise_transition",
                transition_count,
            ),
        },
        segments={
            "denoise": TrajectorySegment(
                name="denoise",
                modality="latent",
                trainable=True,
                distribution="gaussian",
                tensors=tensors,
                reward_view=reward_modality,
                replay_inputs={
                    "logprob": ReplayInput(
                        name="logprob",
                        tensor_refs=replay_tensor_refs,
                    ),
                },
            )
        },
        primary_segment="denoise",
        reward_views={
            reward_modality: RewardInputSpec(
                name=reward_modality,
                value_range="unit",
                metadata={"output_ref": "GenerationOutput.output"},
            )
        },
        context=_serializable_context(context),
    )
    return TrajectoryValidator(trajectory).validate_batch()


def build_chunk_autoregressive_generation_trajectory(
    *,
    request: GenerationRequest,
    sample_rows: list[GenerationSampleRow],
    output: Any,
    temporal_chunk_count: int,
    context: dict[str, Any],
) -> TrajectoryBatch:
    """Record chunk-autoregressive generation without inventing RL actions.

    Some upstream runtimes expose the generated artifact and temporal chunk
    count but not the stochastic transition facts needed for exact replay.  In
    that case the artifact remains useful for reward/evaluation, while the
    segment is explicitly non-trainable and contains no fake log-probability.
    """

    batch_size = len(sample_rows)
    if temporal_chunk_count < 1:
        raise ValueError("temporal_chunk_count must be >= 1")
    output_rows = _leading_length(output)
    if output_rows is not None and output_rows != batch_size:
        raise ValueError(
            f"output has {output_rows} rows, expected {batch_size}",
        )
    reward_modality = task_modality(request.task)
    trajectory = TrajectoryBatch(
        request_id=request.request_id,
        family=request.family,
        task=request.task,
        sample_rows=list(sample_rows),
        axes={
            "sample": TrajectoryAxis("sample", "sample", batch_size),
            "temporal_chunk": TrajectoryAxis(
                "temporal_chunk",
                "temporal_chunk",
                temporal_chunk_count,
            ),
        },
        segments={
            "generated_chunks": TrajectorySegment(
                name="generated_chunks",
                modality=reward_modality,
                trainable=False,
                distribution="deterministic",
                # Decoded media belongs to GenerationOutput, not replay state.
                # Keeping another reference here defeats file-only transport.
                tensors={},
                reward_view=reward_modality,
            )
        },
        primary_segment=None,
        reward_views={
            reward_modality: RewardInputSpec(
                name=reward_modality,
                value_range="unit",
                metadata={"output_ref": "GenerationOutput.output"},
            )
        },
        context=_serializable_context(context),
    )
    return TrajectoryValidator(trajectory).validate_batch()


def _chunk_denoise_shape(value: Any, batch_size: int) -> tuple[int, int]:
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) < 3:
        raise ValueError(
            "old_log_prob must have leading [sample, temporal_chunk, "
            "denoise_transition] dimensions",
        )
    if int(shape[0]) != batch_size:
        raise ValueError(
            f"old_log_prob has {shape[0]} rows, expected {batch_size}",
        )
    chunk_count = int(shape[1])
    transition_count = int(shape[2])
    if chunk_count < 1 or transition_count < 1:
        raise ValueError(
            "old_log_prob temporal_chunk and denoise_transition dimensions must be >= 1",
        )
    return chunk_count, transition_count


def _leading_length(value: Any) -> int | None:
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[0])
    if isinstance(value, (list, tuple)):
        return len(value)
    return None


def _sample_aligned(value: Any, batch_size: int) -> bool:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return len(shape) >= 1 and int(shape[0]) == batch_size
    return isinstance(value, (list, tuple)) and len(value) == batch_size


def _serializable_context(value: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, inner in value.items():
        serializable = _serializable_value(inner)
        if serializable is not _DROP:
            out[key] = serializable
    return out


def _serializable_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return value
    if isinstance(value, tuple):
        values = [_serializable_value(inner) for inner in value]
        return tuple(inner for inner in values if inner is not _DROP)
    if isinstance(value, list):
        values = [_serializable_value(inner) for inner in value]
        return [inner for inner in values if inner is not _DROP]
    if isinstance(value, dict):
        return _serializable_context(value)
    return _DROP


__all__ = [
    "build_chunk_autoregressive_denoise_trajectory",
    "build_chunk_autoregressive_generation_trajectory",
    "build_diffusion_trajectory",
]
