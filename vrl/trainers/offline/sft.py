"""Offline supervised flow-matching trainer over cached clean latents.

One step noises a batch of clean target latents at a drawn table position,
runs the family forward once and penalizes the prediction error under the
batch's token weights. The forward adapter is family-specific and lives with
the recipe (``vrl/scripts/families/qwen_image_layered/train_sft.py``); the
batch carries whatever conditioning that adapter reads.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from vrl.models.precision import model_autocast
from vrl.trainers.offline.common import OfflineTrainer
from vrl.trainers.offline.config import OfflineTrainerConfig

if TYPE_CHECKING:
    from vrl.algorithms.sft import DiffusionSFT


@dataclass(slots=True)
class SFTBatch:
    """Clean target latents plus the conditioning the family forward reads.

    ``latents`` is ``[B, ...]`` in the family's latent layout; ``weights`` is
    ``None`` or broadcastable against it with mean one. ``conditioning`` is
    opaque here: the recipe's forward adapter owns its keys.
    """

    latents: torch.Tensor
    conditioning: Any
    weights: torch.Tensor | None = None


@dataclass(slots=True)
class SFTStepMetrics:
    """One training-step's metrics."""

    loss: float = 0.0
    mse: float = 0.0
    grad_norm: float = 0.0


# ``(model, noisy_latents, timesteps, conditioning) -> prediction``.
ForwardFn = Callable[..., torch.Tensor]


class OfflineSFTTrainer(OfflineTrainer):
    """Supervised flow matching over ``SFTBatch`` samples.

    Caller responsibilities:
      * Build the policy ``model`` (typically a LoRA-wrapped backbone).
      * Provide ``forward_fn``, the family forward over ``batch.conditioning``.
      * Provide ``noise_scheduler`` with its timestep table set: the draw
        covers ``algorithm.config.timestep_index_range`` of that table.
    """

    def __init__(
        self,
        model: nn.Module,
        algorithm: DiffusionSFT,
        forward_fn: ForwardFn,
        noise_scheduler: Any,
        config: OfflineTrainerConfig | None = None,
        device: torch.device | str = "cuda",
    ) -> None:
        super().__init__(model, noise_scheduler, config=config, device=device)
        self.algorithm = algorithm
        self.forward_fn = forward_fn

    def step(self, batch: SFTBatch) -> SFTStepMetrics:
        """Train on one microbatch; update at the accumulation boundary."""
        self.model.train()
        latents = batch.latents.to(self.device)
        bsz = latents.shape[0]
        weights = None if batch.weights is None else batch.weights.to(self.device)

        noise = torch.randn_like(latents)
        timestep_indices = self._sample_timestep_indices(
            bsz, index_range=self.algorithm.config.timestep_index_range
        )
        timesteps = self.noise_scheduler.timesteps.to(self.device)[timestep_indices]
        noisy_latents, target = self._inject_noise(
            latents, noise, timesteps, timestep_indices=timestep_indices
        )

        with model_autocast(self.model, self.device):
            model_pred = self.forward_fn(self.model, noisy_latents, timesteps, batch.conditioning)

        # The reduction is protected FP32 math, not transformer forward.
        stats = self.algorithm.compute_loss(model_pred.float(), target.float(), weights)
        loss = stats["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite SFT loss at global step {self.global_step}")
        grad_norm = self._optimizer_update(loss)
        return SFTStepMetrics(
            loss=float(loss.detach()),
            mse=float(stats["mse"]),
            grad_norm=grad_norm,
        )


__all__ = ["ForwardFn", "OfflineSFTTrainer", "SFTBatch", "SFTStepMetrics"]
