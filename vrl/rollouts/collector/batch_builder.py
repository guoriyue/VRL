"""Build trainer rollout batches from trajectory-backed engine outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from vrl.engine import OutputBatch
from vrl.engine.trajectory import (
    TrajectoryBatch,
    TrajectorySegment,
    TrajectoryTensor,
    build_training_view,
)
from vrl.rollouts.batch import RolloutBatch


@dataclass(slots=True)
class RolloutBatchBuildContext:
    """Non-engine metadata needed while building a trainer RolloutBatch."""

    metadata: dict[str, Any]
    device: Any | None = None
    kl_reward: float = 0.0
    rescale_to_unit: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def reward_outputs_from_trajectory(
    output: OutputBatch,
    context: RolloutBatchBuildContext,
) -> Any:
    """Return the generated artifact that the reward scorer should inspect."""

    trajectory = _require_output_trajectory(output)
    segment = _primary_trainable_segment(
        trajectory,
        preferred=_primary_segment_name(trajectory),
    )
    reward_output = _reward_output_from_trajectory(output, trajectory)
    if segment.distribution == "categorical" and context.rescale_to_unit:
        reward_output = (reward_output + 1.0) * 0.5
        reward_output = reward_output.clamp(0.0, 1.0)
    return reward_output


def reward_prompts_from_output(
    output: OutputBatch,
    context: RolloutBatchBuildContext,
) -> list[str]:
    """Return prompts aligned with reward outputs."""

    del context
    return [row.prompt for row in output.sample_rows]


def rollout_batch_from_trajectory(
    output: OutputBatch,
    rewards_raw: torch.Tensor,
    context: RolloutBatchBuildContext,
) -> RolloutBatch:
    """Convert a trajectory-backed engine output into the trainer batch shape."""

    trajectory = _require_output_trajectory(output)
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
        "trajectory rollout collection does not support distribution="
        f"{segment.distribution!r}",
    )


def _pack_diffusion(
    output: OutputBatch,
    trajectory: TrajectoryBatch,
    segment: TrajectorySegment,
    rewards_raw: torch.Tensor,
    context: RolloutBatchBuildContext,
) -> RolloutBatch:
    observations = _role_tensor(segment, "observation").value
    actions = _role_tensor(segment, "action").value
    kl_tensor = _named_tensor(segment, "kl").value
    device = observations.device

    if context.kl_reward > 0:
        rewards_adjusted = rewards_raw.to(device) - context.kl_reward * kl_tensor.sum(dim=1)
    else:
        rewards_adjusted = rewards_raw.to(device)

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
        extras={},
        context=rollout_context,
        videos=output.output,
        prompts=[row.prompt for row in output.sample_rows],
        trajectory=trajectory,
        training_view=build_training_view(trajectory, primary_segment=segment.name),
    )


def _pack_ar_discrete(
    output: OutputBatch,
    trajectory: TrajectoryBatch,
    segment: TrajectorySegment,
    rewards_raw: torch.Tensor,
    context: RolloutBatchBuildContext,
) -> RolloutBatch:
    token_ids = _role_tensor(segment, "action").value
    prompt_ids = _named_tensor(segment, "prompt_input_ids").value
    device = context.device or prompt_ids.device
    images = output.output

    return RolloutBatch(
        observations=prompt_ids.unsqueeze(1),
        actions=token_ids,
        rewards=rewards_raw.to(device),
        dones=torch.ones(len(output.sample_rows), dtype=torch.bool, device=device),
        group_ids=_group_ids_from_trajectory(trajectory, device=device),
        extras={},
        context=dict(trajectory.context),
        videos=images.unsqueeze(2),
        prompts=[row.prompt for row in output.sample_rows],
        trajectory=trajectory,
        training_view=build_training_view(trajectory, primary_segment=segment.name),
    )


def _pack_ar_continuous(
    output: OutputBatch,
    trajectory: TrajectoryBatch,
    segment: TrajectorySegment,
    rewards_raw: torch.Tensor,
    context: RolloutBatchBuildContext,
) -> RolloutBatch:
    tokens = _role_tensor(segment, "action").value
    prompt_ids = _named_tensor(segment, "prompt_input_ids").value
    device = context.device or prompt_ids.device
    images = output.output

    return RolloutBatch(
        observations=prompt_ids.unsqueeze(1),
        actions=tokens,
        rewards=rewards_raw.to(device),
        dones=torch.ones(len(output.sample_rows), dtype=torch.bool, device=device),
        group_ids=_group_ids_from_trajectory(trajectory, device=device),
        extras={},
        context=dict(trajectory.context),
        videos=images.unsqueeze(2),
        prompts=[row.prompt for row in output.sample_rows],
        trajectory=trajectory,
        training_view=build_training_view(trajectory, primary_segment=segment.name),
    )


def _pack_ar_multisegment(
    output: OutputBatch,
    trajectory: TrajectoryBatch,
    trainable: list[TrajectorySegment],
    rewards_raw: torch.Tensor,
    context: RolloutBatchBuildContext,
) -> RolloutBatch:
    primary_name = _primary_segment_name(trajectory) or "final_image"
    primary = trajectory.segments.get(primary_name)
    if primary is None or not primary.trainable:
        primary = trainable[-1]
        primary_name = primary.name

    token_ids = _role_tensor(primary, "action").value
    prompt_ids = _optional_named_tensor(primary, "prompt_input_ids")
    if prompt_ids is None:
        prompt_ids = torch.zeros(
            token_ids.shape[0],
            1,
            dtype=torch.long,
            device=token_ids.device,
        )
    device = context.device or prompt_ids.device
    final_image = _decoded_tensor(trajectory, "final_image")
    if final_image is None:
        final_image = output.output

    rollout_context = dict(trajectory.context)
    rollout_context.pop("primary_segment", None)
    rollout_context.pop("segment_names", None)
    return RolloutBatch(
        observations=prompt_ids.unsqueeze(1),
        actions=token_ids,
        rewards=rewards_raw.to(device),
        dones=torch.ones(len(output.sample_rows), dtype=torch.bool, device=device),
        group_ids=_group_ids_from_trajectory(trajectory, device=device),
        extras={},
        context={**rollout_context, "r1_segment_names": tuple(
            name
            for name, segment in trajectory.segments.items()
            if segment.distribution == "categorical"
        )},
        videos=final_image.unsqueeze(2),
        prompts=[row.prompt for row in output.sample_rows],
        trajectory=trajectory,
        training_view=build_training_view(trajectory, primary_segment=primary_name),
    )


def _trainable_segments(trajectory: TrajectoryBatch) -> list[TrajectorySegment]:
    return [segment for segment in trajectory.segments.values() if segment.trainable]


def _require_output_trajectory(output: OutputBatch) -> TrajectoryBatch:
    trajectory = output.trajectory
    if not isinstance(trajectory, TrajectoryBatch):
        raise RuntimeError(f"OutputBatch {output.request_id!r} is missing TrajectoryBatch")
    return trajectory


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


__all__ = [
    "RolloutBatchBuildContext",
    "reward_outputs_from_trajectory",
    "reward_prompts_from_output",
    "rollout_batch_from_trajectory",
]
