"""Training strategy seam: the boundary between the trainer and how it runs.

The trainer drives the GRPO loop; *how* a step executes on the hardware —
backward, grad clipping, and trainable-state export/load — goes through a
``TrainingStrategy`` so the trainer never hard-codes single-process vs FSDP2.

This readiness sprint ships only ``SingleProcessStrategy`` (current behavior
moved behind the protocol, byte-for-byte). The FSDP2 strategy — DTensor-aware
clip and full-state export, a real ``barrier`` — lands in
``SPRINT_multi_gpu_training.md`` and slots in here without touching the trainer.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

import torch
from torch import nn

from vrl.trainers.distributed import DistributedTrainingContext


class TrainingStrategy(Protocol):
    """How one training step executes; the only seam the trainer depends on."""

    context: DistributedTrainingContext

    def backward(self, loss: torch.Tensor, *, grad_scaler: Any | None = None) -> None:
        """Run the backward pass (scaled when an fp16 GradScaler is active)."""
        ...

    def clip_grad_norm(
        self,
        parameters: Iterable[nn.Parameter],
        max_norm: float,
    ) -> float:
        """Clip gradients in place and return the pre-clip total norm."""
        ...

    def export_trainable_state(self, bundle: Any) -> dict[str, dict[str, Any]]:
        """Checkpoint-facing trainable state (nested by module name, CPU tensors)."""
        ...

    def export_rollout_state(self, bundle: Any) -> dict[str, Any]:
        """Rollout-facing flat trainable state (unwrapped, policy-facing keys)."""
        ...

    def load_trainable_state(self, bundle: Any, state: dict[str, Any]) -> None:
        """Load a checkpoint-facing trainable state back into the bundle."""
        ...

    def barrier(self) -> None:
        """Synchronize all training ranks (no-op for single process)."""
        ...


class SingleProcessStrategy:
    """The current single-GPU behavior, moved behind the strategy protocol.

    Every method here is the existing trainer / checkpoint / weight-sync logic
    verbatim; this installs the seam without changing what a single-process run
    does. ``context`` defaults to a rank0/world1 identity.
    """

    def __init__(self, context: DistributedTrainingContext | None = None) -> None:
        self.context = context or _single_process_context()

    def backward(self, loss: torch.Tensor, *, grad_scaler: Any | None = None) -> None:
        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()
        else:
            loss.backward()

    def clip_grad_norm(
        self,
        parameters: Iterable[nn.Parameter],
        max_norm: float,
    ) -> float:
        return float(nn.utils.clip_grad_norm_(parameters, max_norm))

    def export_trainable_state(self, bundle: Any) -> dict[str, dict[str, Any]]:
        from vrl.trainers.checkpointing import export_trainable_state

        return export_trainable_state(bundle)

    def export_rollout_state(self, bundle: Any) -> dict[str, Any]:
        from vrl.trainers.weight_sync import build_trainable_state_sync_getter

        return build_trainable_state_sync_getter(bundle)()

    def load_trainable_state(self, bundle: Any, state: dict[str, Any]) -> None:
        from vrl.trainers.checkpointing import load_trainable_state

        load_trainable_state(bundle, state)

    def barrier(self) -> None:
        return None


def _single_process_context() -> DistributedTrainingContext:
    return DistributedTrainingContext(
        strategy="single_process",
        distributed=False,
        rank=0,
        local_rank=0,
        world_size=1,
        is_primary=True,
        device=torch.device("cpu"),
    )


__all__ = ["SingleProcessStrategy", "TrainingStrategy"]
