"""Rollout batch operations shared by collectors, schedules, and trainers."""

from __future__ import annotations

from typing import Any

import torch

from vrl.rollouts.batch import RolloutBatch, stack_batches
from vrl.trajectory.ops import move_trajectory_batch, select_trajectory_batch


def select_batch(batch: RolloutBatch, selector: torch.Tensor) -> RolloutBatch:
    """Select ``RolloutBatch`` rows by a boolean mask or long indices."""

    selector = selector.detach()
    new_extras: dict[str, Any] = {}
    batch_size = batch.rewards.shape[0]
    for key, value in batch.extras.items():
        new_extras[key] = select_tensor_tree(value, selector, batch_size)
    videos = (
        batch.videos[selector.to(batch.videos.device)]
        if batch.videos is not None
        else None
    )
    if batch.prompts is not None:
        selector_cpu = selector.cpu()
        if selector_cpu.dtype == torch.bool:
            positions = torch.where(selector_cpu)[0].tolist()
        else:
            positions = [int(i) for i in selector_cpu.reshape(-1).tolist()]
        prompts = [batch.prompts[i] for i in positions]
    else:
        prompts = None
    return RolloutBatch(
        observations=batch.observations[selector.to(batch.observations.device)],
        actions=batch.actions[selector.to(batch.actions.device)],
        rewards=batch.rewards[selector.to(batch.rewards.device)],
        dones=batch.dones[selector.to(batch.dones.device)],
        group_ids=batch.group_ids[selector.to(batch.group_ids.device)],
        extras=new_extras,
        context=batch.context,
        videos=videos,
        prompts=prompts,
        trajectory=select_trajectory_batch(batch.trajectory, selector),
        training_view=batch.training_view,
    )


def select_tensor_tree(value: Any, selector: torch.Tensor, batch_size: int) -> Any:
    """Select per-sample tensor leaves inside nested rollout metadata."""

    if isinstance(value, torch.Tensor):
        if value.dim() > 0 and value.shape[0] == batch_size:
            return value[selector.to(value.device)]
        return value
    if isinstance(value, dict):
        return {
            key: select_tensor_tree(inner, selector, batch_size)
            for key, inner in value.items()
        }
    if isinstance(value, list):
        return [select_tensor_tree(inner, selector, batch_size) for inner in value]
    if isinstance(value, tuple):
        return tuple(select_tensor_tree(inner, selector, batch_size) for inner in value)
    return value


def apply_sample_mask(batch: RolloutBatch, mask: torch.Tensor) -> RolloutBatch:
    """Filter ``RolloutBatch`` along the sample dimension."""

    return select_batch(batch, mask)


def nonzero_advantage_mask(advantages: torch.Tensor) -> torch.Tensor:
    """Return Flow-GRPO's mask for samples with non-zero total advantage."""

    adv_abs = advantages.detach().abs()
    if adv_abs.dim() <= 1:
        return adv_abs != 0
    reduce_dims = tuple(range(1, adv_abs.dim()))
    return adv_abs.sum(dim=reduce_dims) != 0


def pad_zero_advantage_mask(mask: torch.Tensor, num_batches: int) -> torch.Tensor:
    """Pad zero-advantage samples back in so rebatching divides evenly."""

    if num_batches <= 0:
        return mask
    padded = mask.clone()
    true_count = int(padded.sum().item())
    if true_count == 0 or true_count % num_batches == 0:
        return padded
    false_indices = torch.where(~padded)[0]
    num_to_change = num_batches - (true_count % num_batches)
    if false_indices.numel() >= num_to_change:
        random_indices = torch.randperm(false_indices.numel(), device=false_indices.device)[
            :num_to_change
        ]
        padded[false_indices[random_indices]] = True
    return padded


def shuffle_and_rebatch_batches(
    batches: list[RolloutBatch],
    advantages: list[torch.Tensor],
    *,
    num_batches: int,
) -> tuple[list[RolloutBatch], list[torch.Tensor]]:
    """Shuffle across all rollout rows and split into Flow-GRPO microbatches."""

    combined = stack_batches(batches)
    adv_all = torch.cat(advantages)
    total_batch_size = int(combined.rewards.shape[0])
    if total_batch_size == 0:
        return [], []
    if num_batches <= 0 or total_batch_size % num_batches != 0:
        raise ValueError(
            "Flow-GRPO rebatch requires total samples divisible by rollout batches: "
            f"total_batch_size={total_batch_size}, num_batches={num_batches}",
        )

    perm = torch.randperm(total_batch_size, device=combined.rewards.device)
    combined = select_batch(combined, perm)
    adv_all = adv_all[perm.to(adv_all.device)]

    microbatch_size = total_batch_size // num_batches
    rebatches: list[RolloutBatch] = []
    rebatch_advs: list[torch.Tensor] = []
    for start in range(0, total_batch_size, microbatch_size):
        idx = torch.arange(
            start,
            start + microbatch_size,
            device=combined.rewards.device,
        )
        rebatches.append(select_batch(combined, idx))
        rebatch_advs.append(adv_all[idx.to(adv_all.device)])
    return rebatches, rebatch_advs


def split_batch_by_group(batch: RolloutBatch) -> list[RolloutBatch]:
    """Split a rollout batch into group-local batches for bounded training memory."""

    group_ids = batch.group_ids
    ordered_ids: list[int] = []
    seen: set[int] = set()
    for group_id in group_ids.detach().cpu().tolist():
        gid = int(group_id)
        if gid not in seen:
            seen.add(gid)
            ordered_ids.append(gid)
    if len(ordered_ids) <= 1:
        return [batch]
    return [apply_sample_mask(batch, group_ids == group_id) for group_id in ordered_ids]


def remap_group_ids_(batch: RolloutBatch, global_prompt_indices: list[int]) -> None:
    """Map collector-local prompt groups back to trainer-global prompt indices."""

    if not global_prompt_indices:
        return
    remapped = batch.group_ids.clone()
    for local_idx, global_idx in enumerate(global_prompt_indices):
        remapped[batch.group_ids == local_idx] = global_idx
    batch.group_ids = remapped
    if batch.trajectory is not None:
        trajectory_group_ids = batch.trajectory.group_ids
        device = getattr(trajectory_group_ids, "device", remapped.device)
        batch.trajectory.group_ids = remapped.to(device)


def move_tensor_tree(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_tensor_tree(inner, device) for key, inner in value.items()}
    if isinstance(value, list):
        return [move_tensor_tree(inner, device) for inner in value]
    if isinstance(value, tuple):
        return tuple(move_tensor_tree(inner, device) for inner in value)
    return value


def move_training_batch_to_device(
    batch: RolloutBatch,
    device: torch.device,
    *,
    defer_replay_tensors: bool = False,
) -> RolloutBatch:
    """Move trainer-owned tensors to the trainer device.

    Diffusion CPU-offload keeps timestep-indexed replay tensors on their
    storage device until the evaluator slices the current denoise step.
    """

    return RolloutBatch(
        observations=batch.observations if defer_replay_tensors else batch.observations.to(device),
        actions=batch.actions if defer_replay_tensors else batch.actions.to(device),
        rewards=batch.rewards.to(device),
        dones=batch.dones.to(device),
        group_ids=batch.group_ids.to(device),
        extras=batch.extras if defer_replay_tensors else move_tensor_tree(batch.extras, device),
        context=batch.context if defer_replay_tensors else move_tensor_tree(batch.context, device),
        videos=batch.videos,
        prompts=batch.prompts,
        trajectory=batch.trajectory
        if defer_replay_tensors
        else move_trajectory_batch(batch.trajectory, device),
        training_view=batch.training_view,
    )


__all__ = [
    "apply_sample_mask",
    "move_tensor_tree",
    "move_training_batch_to_device",
    "nonzero_advantage_mask",
    "pad_zero_advantage_mask",
    "remap_group_ids_",
    "select_batch",
    "select_tensor_tree",
    "shuffle_and_rebatch_batches",
    "split_batch_by_group",
]
