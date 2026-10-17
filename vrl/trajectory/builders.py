"""Builders that project family outputs into the trajectory contract."""

from __future__ import annotations

import logging
from typing import Any

import torch

from vrl.generation.types import GenerationRequest, GenerationSampleRow
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
)

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
        shape = getattr(value, "shape", None)
        if shape is not None:
            sample_aligned = len(shape) >= 1 and int(shape[0]) == batch_size
        else:
            sample_aligned = isinstance(value, (list, tuple)) and len(value) == batch_size
        if not sample_aligned:
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
                replay_inputs={
                    "logprob": ReplayInput(
                        name="logprob",
                        tensor_refs=replay_tensor_refs,
                    ),
                },
            )
        },
        primary_segment="denoise",
        context=_serializable_context(context),
    )
    return TrajectoryValidator(trajectory).validate_batch()


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
    "build_diffusion_trajectory",
]
