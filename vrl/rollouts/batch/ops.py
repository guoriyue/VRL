"""Rollout batch operations shared by collectors, schedules, and trainers."""

from __future__ import annotations

from typing import Any

import torch

from vrl.rollouts.batch import RolloutBatch
from vrl.trajectory.device import map_tensor_tree
from vrl.trajectory.ops import move_trajectory_batch, select_trajectory_batch


def select_batch(batch: RolloutBatch, selector: torch.Tensor) -> RolloutBatch:
    """Select ``RolloutBatch`` rows by a boolean mask or long indices."""

    selector = selector.detach()
    new_extras: dict[str, Any] = {}
    batch_size = batch.rewards.shape[0]
    for key, value in batch.extras.items():
        new_extras[key] = _select_tensor_tree(value, selector, batch_size)
    return RolloutBatch(
        rewards=batch.rewards[selector.to(batch.rewards.device)],
        group_ids=batch.group_ids[selector.to(batch.group_ids.device)],
        extras=new_extras,
        context=batch.context,
        trajectory=select_trajectory_batch(batch.trajectory, selector),
    )


def _select_tensor_tree(value: Any, selector: torch.Tensor, batch_size: int) -> Any:
    """Select per-sample tensor leaves inside nested rollout metadata."""

    def _select(leaf: torch.Tensor) -> torch.Tensor:
        if leaf.dim() > 0 and leaf.shape[0] == batch_size:
            return leaf[selector.to(leaf.device)]
        return leaf

    return map_tensor_tree(
        value,
        _select,
        is_leaf=lambda v: isinstance(v, torch.Tensor),
    )


def nonzero_advantage_mask(advantages: torch.Tensor) -> torch.Tensor:
    """Return Flow-GRPO's mask for samples with non-zero total advantage."""

    adv_abs = advantages.detach().abs()
    if adv_abs.dim() <= 1:
        return adv_abs != 0
    reduce_dims = tuple(range(1, adv_abs.dim()))
    return adv_abs.sum(dim=reduce_dims) != 0


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
    return [select_batch(batch, group_ids == group_id) for group_id in ordered_ids]


def remap_group_ids_(batch: RolloutBatch, global_prompt_indices: list[int]) -> None:
    """Map collector-local prompt groups back to trainer-global prompt indices."""

    if not global_prompt_indices:
        return
    remapped = batch.group_ids.clone()
    for local_idx, global_idx in enumerate(global_prompt_indices):
        remapped[batch.group_ids == local_idx] = global_idx
    batch.group_ids = remapped


def _move_tensor_tree(value: Any, device: torch.device) -> Any:
    return map_tensor_tree(
        value,
        lambda leaf: leaf.to(device),
        is_leaf=lambda v: isinstance(v, torch.Tensor),
    )


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
        rewards=batch.rewards.to(device),
        group_ids=batch.group_ids.to(device),
        extras=batch.extras if defer_replay_tensors else _move_tensor_tree(batch.extras, device),
        context=batch.context
        if defer_replay_tensors
        else _move_tensor_tree(batch.context, device),
        trajectory=batch.trajectory
        if defer_replay_tensors
        else move_trajectory_batch(batch.trajectory, device),
    )


__all__ = [
    "move_training_batch_to_device",
    "nonzero_advantage_mask",
    "remap_group_ids_",
    "select_batch",
    "split_batch_by_group",
]
