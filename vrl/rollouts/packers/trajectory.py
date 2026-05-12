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
        segment = _primary_trainable_segment(trajectory)
        reward_output = output.output
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
        segment = _primary_trainable_segment(trajectory)
        if segment.distribution == "flow_matching":
            return _pack_diffusion(output, trajectory, segment, rewards_raw, context)
        if segment.distribution == "categorical":
            return _pack_ar_discrete(output, trajectory, segment, rewards_raw, context)
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


def _primary_trainable_segment(trajectory: TrajectoryBatch) -> TrajectorySegment:
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


def _group_ids_from_trajectory(
    trajectory: TrajectoryBatch,
    *,
    device: Any,
) -> torch.Tensor:
    group_ids = trajectory.group_ids
    if isinstance(group_ids, torch.Tensor):
        return group_ids.to(device=device, dtype=torch.long)
    return torch.tensor(group_ids, dtype=torch.long, device=device)


__all__ = ["TrajectoryRolloutPacker"]
