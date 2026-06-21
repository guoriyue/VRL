"""Builders that project family outputs into the trajectory contract."""

from __future__ import annotations

from typing import Any

import torch

from vrl.generation.types import GenerationRequest, GenerationSampleRow
from vrl.trajectory.types import (
    ReplayInput,
    TrajectoryAxis,
    TrajectoryBatch,
    TrajectoryMetrics,
    TrajectorySegment,
    TrajectoryTensor,
)
from vrl.trajectory.validation import (
    TrajectoryValidator,
    tensor_ref,
)
from vrl.trajectory.views import RewardView


def build_diffusion_trajectory(
    *,
    request: GenerationRequest,
    sample_rows: list[GenerationSampleRow],
    observations: Any,
    actions: Any,
    old_log_prob: Any,
    timesteps: Any,
    kl: Any,
    replay_tensors: dict[str, Any],
    context: dict[str, Any],
) -> TrajectoryBatch:
    """Build a denoise-step trajectory from shared diffusion rollout tensors."""

    batch_size = len(sample_rows)
    timestep_count = int(old_log_prob.shape[1])
    device = getattr(old_log_prob, "device", None)
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
        "kl": TrajectoryTensor(
            "kl",
            kl,
            ("sample", "denoise"),
            "replay_input",
        ),
    }
    replay_tensor_names: list[str] = []
    for name, value in replay_tensors.items():
        if name in tensors:
            continue
        if not _sample_aligned_or_scalar(value, batch_size):
            continue
        tensors[name] = TrajectoryTensor(name, value, ("sample",), "replay_input")
        replay_tensor_names.append(name)
    replay_tensor_refs = (
        tensor_ref("denoise", "observations"),
        tensor_ref("denoise", "actions"),
        tensor_ref("denoise", "timesteps"),
        *(tensor_ref("denoise", name) for name in replay_tensor_names),
    )

    reward_modality = _reward_modality_for_task(request.task)
    trajectory = TrajectoryBatch(
        request_id=request.request_id,
        family=request.family,
        task=request.task,
        sample_rows=list(sample_rows),
        group_ids=_prompt_group_ids(sample_rows, device=device),
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
        reward_views={
            reward_modality: RewardView(
                name=reward_modality,
                modality=reward_modality,
                value_range="unit",
                metadata={"output_ref": "GenerationOutput.output"},
            )
        },
        metrics=TrajectoryMetrics(
            num_samples=batch_size,
            axis_lengths={"sample": batch_size, "denoise": timestep_count},
            values={"num_steps": timestep_count},
        ),
        context=_serializable_context(context),
    )
    return TrajectoryValidator(trajectory).validate_batch()


def build_ar_discrete_trajectory(
    *,
    request: GenerationRequest,
    sample_rows: list[GenerationSampleRow],
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

    batch_size = len(sample_rows)
    token_count = int(token_ids.shape[1])
    device = getattr(token_ids, "device", None)
    trajectory = TrajectoryBatch(
        request_id=request.request_id,
        family=request.family,
        task=request.task,
        sample_rows=list(sample_rows),
        group_ids=_prompt_group_ids(sample_rows, device=device),
        axes={
            "sample": TrajectoryAxis("sample", "sample", batch_size),
            "token": TrajectoryAxis("token", "discrete_token", token_count),
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
                value_range="tanh",
                metadata={"output_ref": "GenerationOutput.output"},
            )
        },
        metrics=TrajectoryMetrics(
            num_samples=batch_size,
            axis_lengths={"sample": batch_size, "token": token_count},
            values={"num_tokens": token_count},
        ),
        context=_serializable_context(context),
    )
    return TrajectoryValidator(trajectory).validate_batch()


def build_ar_continuous_trajectory(
    *,
    request: GenerationRequest,
    sample_rows: list[GenerationSampleRow],
    tokens: Any,
    saved_noise: Any,
    token_log_probs: Any,
    token_mask: Any,
    prompt_input_ids: Any,
    prompt_attention_mask: Any,
    uncond_input_ids: Any,
    uncond_attention_mask: Any,
    images_for_reward: Any | None,
    context: dict[str, Any],
) -> TrajectoryBatch:
    """Build a continuous image-token trajectory from NextStep rollout tensors."""

    batch_size = len(sample_rows)
    token_count = int(token_log_probs.shape[1])
    device = getattr(token_log_probs, "device", None)
    segments: dict[str, TrajectorySegment] = {
        "image_tokens": TrajectorySegment(
            name="image_tokens",
            modality="image",
            trainable=True,
            distribution="gaussian",
            tensors={
                "tokens": TrajectoryTensor(
                    "tokens",
                    tokens,
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
                "saved_noise": TrajectoryTensor(
                    "saved_noise",
                    saved_noise,
                    ("sample", "token"),
                    "replay_input",
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
                        tensor_ref("image_tokens", "tokens"),
                        tensor_ref("image_tokens", "saved_noise"),
                        tensor_ref("image_tokens", "prompt_input_ids"),
                        tensor_ref("image_tokens", "prompt_attention_mask"),
                        tensor_ref("image_tokens", "uncond_input_ids"),
                        tensor_ref("image_tokens", "uncond_attention_mask"),
                    ),
                ),
            },
        )
    }
    reward_refs: tuple[str, ...] = ()
    if images_for_reward is not None:
        segments["decoded"] = TrajectorySegment(
            name="decoded",
            modality="image",
            trainable=False,
            distribution="deterministic",
            tensors={
                "images_for_reward": TrajectoryTensor(
                    "images_for_reward",
                    images_for_reward,
                    ("sample",),
                    "replay_input",
                )
            },
        )
        reward_refs = (tensor_ref("decoded", "images_for_reward"),)

    trajectory = TrajectoryBatch(
        request_id=request.request_id,
        family=request.family,
        task=request.task,
        sample_rows=list(sample_rows),
        group_ids=_prompt_group_ids(sample_rows, device=device),
        axes={
            "sample": TrajectoryAxis("sample", "sample", batch_size),
            "token": TrajectoryAxis("token", "continuous_token", token_count),
        },
        segments=segments,
        reward_views={
            "image": RewardView(
                name="image",
                modality="image",
                tensor_refs=reward_refs,
                value_range="tanh",
                metadata={"output_ref": "GenerationOutput.output"},
            )
        },
        metrics=TrajectoryMetrics(
            num_samples=batch_size,
            axis_lengths={"sample": batch_size, "token": token_count},
            values={"num_tokens": token_count},
        ),
        context=_serializable_context(context),
    )
    return TrajectoryValidator(trajectory).validate_batch()


def build_ar_multisegment_trajectory(
    *,
    request: GenerationRequest,
    sample_rows: list[GenerationSampleRow],
    segments: dict[str, dict[str, Any]],
    decoded_outputs: dict[str, Any],
    primary_segment: str,
    reward_segments: tuple[str, ...] | None = None,
    context: dict[str, Any],
) -> TrajectoryBatch:
    """Build a multi-segment categorical trajectory without flattening segments."""

    if not segments:
        raise ValueError("segments must be non-empty")
    batch_size = len(sample_rows)
    first_segment = next(iter(segments.values()))
    device = getattr(first_segment.get("token_ids"), "device", None)

    axes = {"sample": TrajectoryAxis("sample", "sample", batch_size)}
    trajectory_segments: dict[str, TrajectorySegment] = {}
    for name, payload in segments.items():
        token_ids = payload["token_ids"]
        old_log_prob = payload.get("token_log_probs")
        if old_log_prob is None:
            old_log_prob = payload.get("old_log_probs")
        if old_log_prob is None:
            old_log_prob = torch.zeros_like(token_ids, dtype=torch.float32)
        token_mask = payload.get("token_mask")
        if token_mask is None:
            token_mask = torch.ones_like(old_log_prob, dtype=torch.float32)

        axis_name = f"{name}_token"
        token_count = int(token_ids.shape[1])
        axes[axis_name] = TrajectoryAxis(axis_name, "discrete_token", token_count)
        modality = "image" if bool(payload.get("visual", not name.endswith("_text"))) else "text"
        trainable = _segment_trainable(request.sampling.get("train_segments"), name, payload)
        trajectory_segments[name] = TrajectorySegment(
            name=name,
            modality=modality,
            trainable=trainable,
            distribution="categorical",
            tensors={
                "token_ids": TrajectoryTensor(
                    "token_ids",
                    token_ids,
                    ("sample", axis_name),
                    "action",
                ),
                "old_log_prob": TrajectoryTensor(
                    "old_log_prob",
                    old_log_prob,
                    ("sample", axis_name),
                    "old_log_prob",
                ),
                "token_mask": TrajectoryTensor(
                    "token_mask",
                    token_mask,
                    ("sample", axis_name),
                    "mask",
                ),
                "prompt_embeds": TrajectoryTensor(
                    "prompt_embeds",
                    payload["prompt_embeds"],
                    ("sample",),
                    "replay_input",
                ),
                "attention_mask": TrajectoryTensor(
                    "attention_mask",
                    payload["attention_mask"],
                    ("sample",),
                    "replay_input",
                ),
                "prompt_attention_mask": TrajectoryTensor(
                    "prompt_attention_mask",
                    payload.get("prompt_attention_mask", payload["attention_mask"]),
                    ("sample",),
                    "replay_input",
                ),
            },
            reward_view="image" if name == primary_segment else None,
            advantage_scope="segment",
            replay_inputs={
                "logprob": ReplayInput(
                    name="logprob",
                    tensor_refs=(
                        tensor_ref(name, "token_ids"),
                        tensor_ref(name, "prompt_embeds"),
                        tensor_ref(name, "attention_mask"),
                    ),
                ),
            },
            metadata={
                "visual": bool(payload.get("visual", modality == "image")),
                "cfg": bool(payload.get("cfg", False)),
                "train": trainable,
            },
        )

    decoded_tensors = {
        name: TrajectoryTensor(
            name,
            value,
            ("sample",),
            "replay_input",
        )
        for name, value in decoded_outputs.items()
        if value is not None
    }
    reward_segment_names = reward_segments or (primary_segment,)
    reward_refs: tuple[str, ...] = ()
    if decoded_tensors:
        trajectory_segments["decoded"] = TrajectorySegment(
            name="decoded",
            modality="mixed",
            trainable=False,
            distribution="deterministic",
            tensors=decoded_tensors,
        )
        reward_refs = tuple(
            tensor_ref("decoded", name)
            for name in reward_segment_names
            if name in decoded_tensors
        )

    axis_lengths = {name: axis.length for name, axis in axes.items() if axis.length is not None}
    trajectory = TrajectoryBatch(
        request_id=request.request_id,
        family=request.family,
        task=request.task,
        sample_rows=list(sample_rows),
        group_ids=_prompt_group_ids(sample_rows, device=device),
        axes=axes,
        segments=trajectory_segments,
        reward_views={
            "image": RewardView(
                name="image",
                modality="image",
                tensor_refs=reward_refs,
                value_range="tanh",
                metadata={"output_ref": "GenerationOutput.output"},
            )
        },
        metrics=TrajectoryMetrics(
            num_samples=batch_size,
            axis_lengths=axis_lengths,
            values={"num_segments": len(segments)},
        ),
        context={
            **_serializable_context(context),
            "primary_segment": primary_segment,
            "segment_names": tuple(segments),
        },
    )
    return TrajectoryValidator(trajectory).validate_batch()


def _prompt_group_ids(
    sample_rows: list[GenerationSampleRow],
    *,
    device: Any,
) -> Any:
    return torch.tensor(
        [row.prompt_index for row in sample_rows],
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


def _reward_modality_for_task(task: str) -> str:
    # i2v is image-conditioned but still emits video frames, so it scores as video.
    if task in {"t2v", "i2v", "v2w", "t2w"}:
        return "video"
    return "image"


def _segment_trainable(value: Any, name: str, payload: dict[str, Any]) -> bool:
    if "train" in payload:
        return bool(payload["train"])
    if "enabled" in payload:
        return bool(payload["enabled"])
    if value is None:
        return bool(payload.get("visual", True))
    if isinstance(value, dict):
        return bool(value.get(name, False))
    if isinstance(value, str):
        return name == value
    if isinstance(value, (list, tuple, set, frozenset)):
        return name in value
    return bool(value)


__all__ = [
    "build_ar_continuous_trajectory",
    "build_ar_discrete_trajectory",
    "build_ar_multisegment_trajectory",
    "build_diffusion_trajectory",
]
