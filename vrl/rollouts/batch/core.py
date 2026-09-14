"""Trainer-side batches produced from rollout generation outputs.

``RolloutBatch`` is the single trainer-facing contract every engine converges
to: diffusion and token trajectories are both packed into it by the
collector's batch builder, and schedules, trainers, and evaluators consume it
without knowing which engine produced it. The type imports no torch at
module level so it stays a dependency-light leaf; its operations import
lazily.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from vrl.trajectory.types import TrajectoryBatch


@dataclass
class RolloutBatch:
    """Trainer-ready batch collected from model rollouts.

    Reward scoring finishes before this boundary. The collector retains only
    tensors and trajectory facts consumed by replay or training. Tensor and
    trajectory annotations are type-checking-only so this leaf stays torch-free
    at import.
    """

    rewards: torch.Tensor  # [B] scalar rewards per sample
    group_ids: torch.Tensor  # [B] prompt group assignment (for per-prompt normalization)
    extras: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)  # shared metadata (not stacked)
    # The collector always attaches the trajectory; None only exists for
    # synthetic batches (tests, batch-op fixtures) that never reach replay.
    trajectory: TrajectoryBatch | None = None

    def estimated_payload_bytes(self) -> int:
        """Estimate queued rewards, groups, extras, and replay payload bytes.

        Repeated objects count once; distinct views may still share storage.
        This preserves the queue's admission heuristic, not allocator/RSS
        measurement. Shared batch context and Python object overhead are excluded.
        """
        from vrl.trajectory.storage import trajectory_tensor_bytes

        return trajectory_tensor_bytes(
            {
                "rewards": self.rewards,
                "group_ids": self.group_ids,
                "extras": self.extras,
                "trajectory": self.trajectory,
            }
        )

    def select(self, selector: torch.Tensor) -> RolloutBatch:
        """Select rows by a boolean mask or long indices."""

        import torch

        from vrl.trajectory.device import map_tensor_tree

        selector = selector.detach()
        batch_size = self.rewards.shape[0]

        def select_sample_tensor(leaf: torch.Tensor) -> torch.Tensor:
            if leaf.dim() > 0 and leaf.shape[0] == batch_size:
                return leaf[selector.to(leaf.device)]
            return leaf

        return RolloutBatch(
            rewards=self.rewards[selector.to(self.rewards.device)],
            group_ids=self.group_ids[selector.to(self.group_ids.device)],
            extras=map_tensor_tree(
                self.extras,
                select_sample_tensor,
                is_leaf=lambda value: isinstance(value, torch.Tensor),
            ),
            context=self.context,
            trajectory=self.trajectory.select_samples(selector)
            if self.trajectory is not None
            else None,
        )

    def split_by_group(self) -> list[RolloutBatch]:
        """Split into group-local batches for bounded training memory."""

        ordered_ids = list(
            dict.fromkeys(int(group_id) for group_id in self.group_ids.detach().cpu().tolist())
        )
        if len(ordered_ids) <= 1:
            return [self]
        return [self.select(self.group_ids == group_id) for group_id in ordered_ids]

    def remap_group_ids_(self, global_prompt_indices: list[int]) -> None:
        """Map collector-local prompt groups back to trainer-global prompt indices."""

        if not global_prompt_indices:
            return
        remapped = self.group_ids.clone()
        for local_idx, global_idx in enumerate(global_prompt_indices):
            remapped[self.group_ids == local_idx] = global_idx
        self.group_ids = remapped

    def to_device(
        self, device: torch.device, *, defer_replay_tensors: bool = False
    ) -> RolloutBatch:
        """Move trainer-owned tensors to ``device``.

        Denoise CPU-offload keeps timestep-indexed replay tensors on their
        storage device until the evaluator slices the current denoise step.
        """

        import torch

        from vrl.trajectory.device import map_tensor_tree

        def move(value: Any) -> Any:
            return map_tensor_tree(
                value,
                lambda leaf: leaf.to(device),
                is_leaf=lambda v: isinstance(v, torch.Tensor),
            )

        return RolloutBatch(
            rewards=self.rewards.to(device),
            group_ids=self.group_ids.to(device),
            extras=self.extras if defer_replay_tensors else move(self.extras),
            context=self.context if defer_replay_tensors else move(self.context),
            trajectory=self.trajectory
            if defer_replay_tensors or self.trajectory is None
            else self.trajectory.to_device(device),
        )


__all__ = ["RolloutBatch"]
