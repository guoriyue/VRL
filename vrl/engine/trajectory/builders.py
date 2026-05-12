"""Builders that project family outputs into the trajectory contract."""

from __future__ import annotations

from typing import Any

import torch

from vrl.engine.core.types import GenerationRequest, GenerationSampleSpec
from vrl.engine.trajectory.axes import AxisSpec
from vrl.engine.trajectory.types import (
    ReplayInput,
    TrajectoryBatch,
    TrajectoryMetrics,
    TrajectorySegment,
    TrajectoryTensor,
)
from vrl.engine.trajectory.validation import (
    tensor_ref,
    validate_trajectory_batch,
)
from vrl.engine.trajectory.views import RewardView


def build_diffusion_trajectory(
    *,
    request: GenerationRequest,
    sample_specs: list[GenerationSampleSpec],
    observations: Any,
    actions: Any,
    old_log_prob: Any,
    timesteps: Any,
    kl: Any,
    training_extras: dict[str, Any],
    context: dict[str, Any],
) -> TrajectoryBatch:
    """Build a denoise-step trajectory from shared diffusion rollout tensors."""

    batch_size = len(sample_specs)
    timestep_count = int(old_log_prob.shape[1])
    device = getattr(old_log_prob, "device", None)
    tensors: dict[str, TrajectoryTensor] = {
        "observations": TrajectoryTensor(
            "observations",
            observations,
            ("sample", "timestep"),
            "observation",
        ),
        "actions": TrajectoryTensor(
            "actions",
            actions,
            ("sample", "timestep"),
            "action",
        ),
        "old_log_prob": TrajectoryTensor(
            "old_log_prob",
            old_log_prob,
            ("sample", "timestep"),
            "old_log_prob",
        ),
        "mask": TrajectoryTensor(
            "mask",
            torch.ones_like(old_log_prob),
            ("sample", "timestep"),
            "mask",
        ),
        "timesteps": TrajectoryTensor(
            "timesteps",
            timesteps,
            ("sample", "timestep"),
            "replay_input",
        ),
        "kl": TrajectoryTensor(
            "kl",
            kl,
            ("sample", "timestep"),
            "replay_input",
        ),
    }
    for name, value in training_extras.items():
        if name in tensors:
            continue
        if not _sample_aligned_or_scalar(value, batch_size):
            continue
        tensors[name] = TrajectoryTensor(name, value, ("sample",), "replay_input")

    trajectory = TrajectoryBatch(
        request_id=request.request_id,
        family=request.family,
        task=request.task,
        sample_specs=list(sample_specs),
        group_ids=_prompt_group_ids(sample_specs, device=device),
        axes={
            "sample": AxisSpec("sample", "sample", batch_size),
            "timestep": AxisSpec("timestep", "denoise_step", timestep_count),
        },
        segments={
            "denoise": TrajectorySegment(
                name="denoise",
                modality="latent",
                trainable=True,
                distribution="flow_matching",
                tensors=tensors,
                reward_view="image",
                replay_inputs={
                    "logprob": ReplayInput(
                        name="logprob",
                        tensor_refs=(
                            tensor_ref("denoise", "observations"),
                            tensor_ref("denoise", "actions"),
                            tensor_ref("denoise", "timesteps"),
                        ),
                    ),
                },
            )
        },
        reward_views={
            "image": RewardView(
                name="image",
                modality="image",
                metadata={"output_ref": "OutputBatch.output"},
            )
        },
        metrics=TrajectoryMetrics(
            num_samples=batch_size,
            axis_lengths={"sample": batch_size, "timestep": timestep_count},
            values={"num_steps": timestep_count},
        ),
        context=_serializable_context(context),
    )
    return validate_trajectory_batch(trajectory)


def build_ar_discrete_trajectory(
    *,
    request: GenerationRequest,
    sample_specs: list[GenerationSampleSpec],
    token_ids: Any,
    token_log_probs: Any,
    token_mask: Any,
    prompt_input_ids: Any,
    prompt_attention_mask: Any,
    uncond_input_ids: Any,
    uncond_attention_mask: Any,
    context: dict[str, Any],
) -> TrajectoryBatch:
    """Build a discrete image-token trajectory from Janus-Pro rollout tensors."""

    batch_size = len(sample_specs)
    token_count = int(token_ids.shape[1])
    device = getattr(token_ids, "device", None)
    trajectory = TrajectoryBatch(
        request_id=request.request_id,
        family=request.family,
        task=request.task,
        sample_specs=list(sample_specs),
        group_ids=_prompt_group_ids(sample_specs, device=device),
        axes={
            "sample": AxisSpec("sample", "sample", batch_size),
            "token": AxisSpec("token", "discrete_token", token_count),
        },
        segments={
            "image_tokens": TrajectorySegment(
                name="image_tokens",
                modality="image",
                trainable=True,
                distribution="categorical",
                tensors={
                    "token_ids": TrajectoryTensor(
                        "token_ids",
                        token_ids,
                        ("sample", "token"),
                        "action",
                    ),
                    "old_log_prob": TrajectoryTensor(
                        "old_log_prob",
                        token_log_probs,
                        ("sample", "token"),
                        "old_log_prob",
                    ),
                    "token_mask": TrajectoryTensor(
                        "token_mask",
                        token_mask,
                        ("sample", "token"),
                        "mask",
                    ),
                    "prompt_input_ids": TrajectoryTensor(
                        "prompt_input_ids",
                        prompt_input_ids,
                        ("sample",),
                        "replay_input",
                    ),
                    "prompt_attention_mask": TrajectoryTensor(
                        "prompt_attention_mask",
                        prompt_attention_mask,
                        ("sample",),
                        "replay_input",
                    ),
                    "uncond_input_ids": TrajectoryTensor(
                        "uncond_input_ids",
                        uncond_input_ids,
                        ("sample",),
                        "replay_input",
                    ),
                    "uncond_attention_mask": TrajectoryTensor(
                        "uncond_attention_mask",
                        uncond_attention_mask,
                        ("sample",),
                        "replay_input",
                    ),
                },
                reward_view="image",
                replay_inputs={
                    "logprob": ReplayInput(
                        name="logprob",
                        tensor_refs=(
                            tensor_ref("image_tokens", "token_ids"),
                            tensor_ref("image_tokens", "prompt_input_ids"),
                            tensor_ref("image_tokens", "prompt_attention_mask"),
                            tensor_ref("image_tokens", "uncond_input_ids"),
                            tensor_ref("image_tokens", "uncond_attention_mask"),
                        ),
                    ),
                },
            )
        },
        reward_views={
            "image": RewardView(
                name="image",
                modality="image",
                metadata={"output_ref": "OutputBatch.output"},
            )
        },
        metrics=TrajectoryMetrics(
            num_samples=batch_size,
            axis_lengths={"sample": batch_size, "token": token_count},
            values={"num_tokens": token_count},
        ),
        context=_serializable_context(context),
    )
    return validate_trajectory_batch(trajectory)


def _prompt_group_ids(
    sample_specs: list[GenerationSampleSpec],
    *,
    device: Any,
) -> Any:
    return torch.tensor(
        [spec.prompt_index for spec in sample_specs],
        dtype=torch.long,
        device=device,
    )


def _sample_aligned_or_scalar(value: Any, batch_size: int) -> bool:
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return True
    shape = getattr(value, "shape", None)
    if shape is None:
        return False
    return len(shape) == 0 or int(shape[0]) == batch_size


def _serializable_context(value: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, inner in value.items():
        serializable = _serializable_value(inner)
        if serializable is not _DROP:
            out[key] = serializable
    return out


_DROP = object()


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
    "build_ar_discrete_trajectory",
    "build_diffusion_trajectory",
]
