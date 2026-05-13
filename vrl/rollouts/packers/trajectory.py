"""Generic trajectory-backed OutputBatch to RolloutBatch packing."""

from __future__ import annotations

from typing import Any

import torch

from vrl.engine import OutputBatch
from vrl.engine.trajectory import (
    TrajectoryBatch,
    TrajectorySegment,
    TrajectoryTensor,
    build_training_view,
    require_output_trajectory,
)
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.packers.base import RolloutPackContext


class TrajectoryRolloutPacker:
    """Pack migrated trajectory records into the legacy trainer batch."""

    def reward_outputs(
        self,
        output: OutputBatch,
        context: RolloutPackContext,
    ) -> Any:
        trajectory = require_output_trajectory(output)
        segment = _primary_trainable_segment(
            trajectory,
            preferred=_primary_segment_name(trajectory),
        )
        reward_output = _reward_output_from_trajectory(output, trajectory)
        if segment.distribution == "categorical" and context.rescale_to_unit:
            reward_output = (reward_output + 1.0) * 0.5
            reward_output = reward_output.clamp(0.0, 1.0)
        return reward_output

    def reward_prompts(
        self,
        output: OutputBatch,
        context: RolloutPackContext,
    ) -> list[str]:
        del context
        return [spec.prompt for spec in output.sample_specs]

    async def pack(
        self,
        output: OutputBatch,
        rewards_raw: torch.Tensor,
        context: RolloutPackContext,
    ) -> RolloutBatch:
        trajectory = require_output_trajectory(output)
        trainable = _trainable_segments(trajectory)
        if _is_multisegment_categorical(trajectory, trainable):
            return _pack_ar_multisegment(output, trajectory, trainable, rewards_raw, context)
        segment = _primary_trainable_segment(
            trajectory,
            preferred=_primary_segment_name(trajectory),
        )
        if segment.distribution == "flow_matching":
            return _pack_diffusion(output, trajectory, segment, rewards_raw, context)
        if segment.distribution == "categorical":
            return _pack_ar_discrete(output, trajectory, segment, rewards_raw, context)
        if segment.distribution == "gaussian":
            return _pack_ar_continuous(output, trajectory, segment, rewards_raw, context)
        raise NotImplementedError(
            "TrajectoryRolloutPacker does not support distribution="
            f"{segment.distribution!r}",
        )


def _pack_diffusion(
    output: OutputBatch,
    trajectory: TrajectoryBatch,
    segment: TrajectorySegment,
    rewards_raw: torch.Tensor,
    context: RolloutPackContext,
) -> RolloutBatch:
    observations = _role_tensor(segment, "observation").value
    actions = _role_tensor(segment, "action").value
    log_probs = _role_tensor(segment, "old_log_prob").value
    timesteps = _named_tensor(segment, "timesteps").value
    kl_tensor = _named_tensor(segment, "kl").value
    device = observations.device

    if context.kl_reward > 0:
        rewards_adjusted = rewards_raw.to(device) - context.kl_reward * kl_tensor.sum(dim=1)
    else:
        rewards_adjusted = rewards_raw.to(device)

    extras: dict[str, Any] = {
        "log_probs": log_probs,
        "timesteps": timesteps,
        "kl": kl_tensor,
        "reward_before_kl": rewards_raw.to(device),
    }
    for name, tensor in segment.tensors.items():
        if tensor.role != "replay_input" or name in {"timesteps", "kl"}:
            continue
        extras[name] = tensor.value

    rollout_context = dict(trajectory.context)
    if context.metadata:
        rollout_context["reward_metadata"] = dict(context.metadata)
    runtime_debug = output.extra.get("runtime_debug")
    if runtime_debug is not None:
        rollout_context["runtime_debug"] = runtime_debug

    return RolloutBatch(
        observations=observations,
        actions=actions,
        rewards=rewards_adjusted,
        dones=torch.ones(observations.shape[0], dtype=torch.bool, device=device),
        group_ids=_group_ids_from_trajectory(trajectory, device=device),
        extras=extras,
        context=rollout_context,
        videos=output.output,
        prompts=[spec.prompt for spec in output.sample_specs],
        trajectory=trajectory,
        training_view=build_training_view(trajectory, primary_segment=segment.name),
    )


def _pack_ar_discrete(
    output: OutputBatch,
    trajectory: TrajectoryBatch,
    segment: TrajectorySegment,
    rewards_raw: torch.Tensor,
    context: RolloutPackContext,
) -> RolloutBatch:
    token_ids = _role_tensor(segment, "action").value
    token_log_probs = _role_tensor(segment, "old_log_prob").value
    token_mask = _role_tensor(segment, "mask").value
    prompt_ids = _named_tensor(segment, "prompt_input_ids").value
    prompt_mask = _named_tensor(segment, "prompt_attention_mask").value
    uncond_ids = _named_tensor(segment, "uncond_input_ids").value
    uncond_mask = _named_tensor(segment, "uncond_attention_mask").value
    device = context.device or prompt_ids.device
    images = output.output

    return RolloutBatch(
        observations=prompt_ids.unsqueeze(1),
        actions=token_ids,
        rewards=rewards_raw.to(device),
        dones=torch.ones(len(output.sample_specs), dtype=torch.bool, device=device),
        group_ids=_group_ids_from_trajectory(trajectory, device=device),
        extras={
            "log_probs": token_log_probs.detach().unsqueeze(1),
            "prompt_attention_mask": prompt_mask,
            "uncond_input_ids": uncond_ids,
            "uncond_attention_mask": uncond_mask,
            "token_mask": token_mask,
        },
        context=dict(trajectory.context),
        videos=images.unsqueeze(2),
        prompts=[spec.prompt for spec in output.sample_specs],
        trajectory=trajectory,
        training_view=build_training_view(trajectory, primary_segment=segment.name),
    )


def _pack_ar_continuous(
    output: OutputBatch,
    trajectory: TrajectoryBatch,
    segment: TrajectorySegment,
    rewards_raw: torch.Tensor,
    context: RolloutPackContext,
) -> RolloutBatch:
    tokens = _role_tensor(segment, "action").value
    token_log_probs = _role_tensor(segment, "old_log_prob").value
    token_mask = _role_tensor(segment, "mask").value
    saved_noise = _named_tensor(segment, "saved_noise").value
    prompt_ids = _named_tensor(segment, "prompt_input_ids").value
    prompt_mask = _named_tensor(segment, "prompt_attention_mask").value
    uncond_ids = _named_tensor(segment, "uncond_input_ids").value
    uncond_mask = _named_tensor(segment, "uncond_attention_mask").value
    device = context.device or prompt_ids.device
    images = output.output

    return RolloutBatch(
        observations=prompt_ids.unsqueeze(1),
        actions=tokens,
        rewards=rewards_raw.to(device),
        dones=torch.ones(len(output.sample_specs), dtype=torch.bool, device=device),
        group_ids=_group_ids_from_trajectory(trajectory, device=device),
        extras={
            "log_probs": token_log_probs.detach().unsqueeze(1),
            "prompt_attention_mask": prompt_mask,
            "uncond_input_ids": uncond_ids,
            "uncond_attention_mask": uncond_mask,
            "token_mask": token_mask,
            "saved_noise": saved_noise,
        },
        context=dict(trajectory.context),
        videos=images.unsqueeze(2),
        prompts=[spec.prompt for spec in output.sample_specs],
        trajectory=trajectory,
        training_view=build_training_view(trajectory, primary_segment=segment.name),
    )


def _pack_ar_multisegment(
    output: OutputBatch,
    trajectory: TrajectoryBatch,
    trainable: list[TrajectorySegment],
    rewards_raw: torch.Tensor,
    context: RolloutPackContext,
) -> RolloutBatch:
    primary_name = _primary_segment_name(trajectory) or "final_image"
    primary = trajectory.segments.get(primary_name)
    if primary is None or not primary.trainable:
        primary = trainable[-1]
        primary_name = primary.name

    token_ids = _role_tensor(primary, "action").value
    token_log_probs = _role_tensor(primary, "old_log_prob").value
    token_mask = _role_tensor(primary, "mask").value
    prompt_ids = _optional_named_tensor(primary, "prompt_input_ids")
    if prompt_ids is None:
        prompt_ids = torch.zeros(
            token_ids.shape[0],
            1,
            dtype=torch.long,
            device=token_ids.device,
        )
    prompt_mask = _optional_named_tensor(primary, "prompt_attention_mask")
    if prompt_mask is None:
        prompt_mask = _optional_named_tensor(primary, "attention_mask")
    if prompt_mask is None:
        prompt_mask = torch.ones_like(prompt_ids)

    device = context.device or prompt_ids.device
    final_image = _decoded_tensor(trajectory, "final_image")
    if final_image is None:
        final_image = output.output
    segments = {
        name: _segment_payload_from_trajectory(segment)
        for name, segment in trajectory.segments.items()
        if segment.distribution == "categorical"
    }
    decoded_initial = _decoded_tensor(trajectory, "initial_image")
    decoded_selfcheck = _decoded_tensor(trajectory, "selfcheck")

    extras: dict[str, Any] = {
        "log_probs": token_log_probs.detach().unsqueeze(1),
        "prompt_attention_mask": prompt_mask,
        "token_mask": token_mask,
        "r1_segments": segments,
        "primary_segment": primary_name,
        "initial_image": decoded_initial,
        "final_image": final_image,
        "selfcheck": decoded_selfcheck,
    }
    rollout_context = dict(trajectory.context)
    rollout_context.pop("primary_segment", None)
    rollout_context.pop("segment_names", None)
    return RolloutBatch(
        observations=prompt_ids.unsqueeze(1),
        actions=token_ids,
        rewards=rewards_raw.to(device),
        dones=torch.ones(len(output.sample_specs), dtype=torch.bool, device=device),
        group_ids=_group_ids_from_trajectory(trajectory, device=device),
        extras=extras,
        context={**rollout_context, "r1_segment_names": tuple(segments)},
        videos=final_image.unsqueeze(2),
        prompts=[spec.prompt for spec in output.sample_specs],
        trajectory=trajectory,
        training_view=build_training_view(trajectory, primary_segment=primary_name),
    )


def _trainable_segments(trajectory: TrajectoryBatch) -> list[TrajectorySegment]:
    return [segment for segment in trajectory.segments.values() if segment.trainable]


def _primary_trainable_segment(
    trajectory: TrajectoryBatch,
    *,
    preferred: str | None = None,
) -> TrajectorySegment:
    if preferred is not None:
        segment = trajectory.segments.get(preferred)
        if segment is not None and segment.trainable:
            return segment
    for segment in trajectory.segments.values():
        if segment.trainable:
            return segment
    raise RuntimeError("TrajectoryBatch has no trainable segment")


def _role_tensor(segment: TrajectorySegment, role: str) -> TrajectoryTensor:
    matches = [tensor for tensor in segment.tensors.values() if tensor.role == role]
    if len(matches) != 1:
        raise RuntimeError(
            f"segment {segment.name!r} requires exactly one role {role!r}, "
            f"found {len(matches)}",
        )
    return matches[0]


def _named_tensor(segment: TrajectorySegment, name: str) -> TrajectoryTensor:
    try:
        return segment.tensors[name]
    except KeyError as exc:
        raise RuntimeError(f"segment {segment.name!r} is missing tensor {name!r}") from exc


def _optional_named_tensor(segment: TrajectorySegment, name: str) -> Any | None:
    tensor = segment.tensors.get(name)
    return None if tensor is None else tensor.value


def _group_ids_from_trajectory(
    trajectory: TrajectoryBatch,
    *,
    device: Any,
) -> torch.Tensor:
    group_ids = trajectory.group_ids
    if isinstance(group_ids, torch.Tensor):
        return group_ids.to(device=device, dtype=torch.long)
    return torch.tensor(group_ids, dtype=torch.long, device=device)


def _primary_segment_name(trajectory: TrajectoryBatch) -> str | None:
    value = trajectory.context.get("primary_segment")
    return value if isinstance(value, str) else None


def _is_multisegment_categorical(
    trajectory: TrajectoryBatch,
    trainable: list[TrajectorySegment],
) -> bool:
    if trajectory.family == "janus_pro_r1" or trajectory.task == "ar_t2i_r1":
        return True
    return len(trainable) > 1 and all(segment.distribution == "categorical" for segment in trainable)


def _reward_output_from_trajectory(output: OutputBatch, trajectory: TrajectoryBatch) -> Any:
    for view in trajectory.reward_views.values():
        if view.tensor_refs:
            ref = view.tensor_refs[0]
            segment_name, tensor_name = ref.split(".", 1)
            return trajectory.segments[segment_name].tensors[tensor_name].value
    return output.output


def _decoded_tensor(trajectory: TrajectoryBatch, name: str) -> Any | None:
    decoded = trajectory.segments.get("decoded")
    if decoded is None:
        return None
    tensor = decoded.tensors.get(name)
    return None if tensor is None else tensor.value


def _segment_payload_from_trajectory(segment: TrajectorySegment) -> dict[str, Any]:
    payload = {
        "name": segment.name,
        "token_ids": _role_tensor(segment, "action").value,
        "token_log_probs": _role_tensor(segment, "old_log_prob").value,
        "token_mask": _role_tensor(segment, "mask").value,
        "visual": bool(segment.metadata.get("visual", segment.modality == "image")),
        "cfg": bool(segment.metadata.get("cfg", False)),
        "train": bool(segment.metadata.get("train", segment.trainable)),
        "modality": segment.modality,
    }
    for key in ("prompt_embeds", "attention_mask", "prompt_attention_mask", "prompt_input_ids"):
        value = _optional_named_tensor(segment, key)
        if value is not None:
            payload[key] = value
    return payload


__all__ = ["TrajectoryRolloutPacker"]
