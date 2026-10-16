"""Offline Diffusion-DPO trainer.

Generic over diffusion model family: the caller supplies the forward adapter,
the encoders, and the scheduler, so nothing here names a family. The
optimizer, the timestep draw and the noise/target convention per
``prediction_type`` come from ``OfflineTrainer``; what is DPO's own is the
winner-then-loser batch, the twin policy/reference forwards and the pairwise
loss.

For Wan video models, image-only datasets (Pick-a-Pic) are handled by the
caller's ``encode_pixels`` replicating each image along the temporal dim
before VAE encoding (see ``vrl/scripts/families/wan_2_1/train_dpo.py``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from vrl.models.precision import model_autocast
from vrl.trainers.data.preferences import PreferenceBatch
from vrl.trainers.offline.common import OfflineTrainer
from vrl.trainers.offline.config import OfflineTrainerConfig

if TYPE_CHECKING:
    from vrl.algorithms.dpo import DiffusionDPO


@dataclass(slots=True)
class DPOStepMetrics:
    """One training-step's metrics."""

    loss: float = 0.0
    raw_model_loss: float = 0.0
    raw_ref_loss: float = 0.0
    model_diff: float = 0.0
    ref_diff: float = 0.0
    implicit_acc: float = 0.0
    sft_loss: float = 0.0
    grad_norm: float = 0.0


# ---------------------------------------------------------------------------
# Forward adapters — caller plugs in a model-specific forward function.
# ---------------------------------------------------------------------------

# A ForwardFn takes ``(model, noisy_latents, timesteps, encoder_hidden_states)``
# and returns the prediction tensor. Concrete adapters live with
# the family recipe that selects them (e.g. ``wan_forward`` in
# ``vrl/scripts/families/wan_2_1/train_dpo.py``), never here: this module must
# stay family-neutral.
ForwardFn = Callable[..., torch.Tensor]


# ---------------------------------------------------------------------------
# OfflineDPOTrainer
# ---------------------------------------------------------------------------


class OfflineDPOTrainer(OfflineTrainer):
    """Offline DPO trainer over preference pairs.

    Caller responsibilities:
      * Build the policy ``model`` (typically a LoRA-wrapped backbone).
      * Build the frozen ``ref_model`` (or pass ``None`` and rely on
        LoRA adapter-disable inside ``forward_fn``).
      * Provide ``encode_pixels`` — turns ``[2B, 3, H, W]`` pixels into
        latents preserving the 2B samples (handles VAE + temporal
        replication for video models).
      * Provide ``encode_text`` — turns a list of captions into the
        ``encoder_hidden_states`` tensor shaped ``[B, ..., D]``. The trainer
        duplicates this batch in winner-then-loser order.
      * Provide ``forward_fn`` — the model-family-specific forward; it lives
        with the family recipe, not in this module.
      * Provide ``noise_scheduler`` for sampling timesteps + injecting
        noise. Flow-matching schedulers use ``scale_noise``; epsilon
        schedulers use ``add_noise``.
    """

    def __init__(
        self,
        model: nn.Module,
        algorithm: DiffusionDPO,
        ref_model: nn.Module | None,
        forward_fn: ForwardFn,
        noise_scheduler: Any,
        encode_pixels: Callable[[torch.Tensor], torch.Tensor],
        encode_text: Callable[[list[str]], torch.Tensor],
        config: OfflineTrainerConfig | None = None,
        device: torch.device | str = "cuda",
    ) -> None:
        super().__init__(model, noise_scheduler, config=config, device=device)
        self.algorithm = algorithm
        self.ref_model = ref_model
        self.forward_fn = forward_fn
        self.encode_pixels = encode_pixels
        self.encode_text = encode_text

        if self.ref_model is not None:
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------
    # One training step
    # ------------------------------------------------------------------

    def step(self, batch: PreferenceBatch) -> DPOStepMetrics:
        """Train on one preference microbatch; update at the accumulation boundary."""
        self.model.train()

        bsz_pair = batch.pixel_values.shape[0]
        if len(batch.captions) != bsz_pair:
            raise ValueError("preference batch requires one caption per image pair")

        # 1. Stack winner-then-loser → [2B, 3, H, W]
        pixels = batch.stacked_winner_then_loser().to(self.device)

        # 2. Pixels → latents (caller handles VAE + temporal replication)
        with torch.no_grad():
            latents = self.encode_pixels(pixels)
            if latents.shape[0] != 2 * bsz_pair:
                raise ValueError("encode_pixels must preserve the image batch size")
            # 3. Match the winner block, then the loser block: [A, B, A, B].
            encoder_hidden_states = self.encode_text(batch.captions)
            if encoder_hidden_states.shape[0] != bsz_pair:
                raise ValueError("encode_text must return one embedding per caption")
            encoder_hidden_states = torch.cat(
                [encoder_hidden_states, encoder_hidden_states],
                dim=0,
            )

        # 4. Sample shared noise + timestep across each pair
        noise = torch.randn(
            (bsz_pair, *tuple(latents.shape[1:])),
            device=latents.device,
            dtype=latents.dtype,
        ).repeat(2, *([1] * (latents.ndim - 1)))
        timestep_indices = self._sample_timestep_indices(bsz_pair).repeat(2)
        timestep_table = self.noise_scheduler.timesteps.to(self.device)
        timesteps = timestep_table[timestep_indices]

        noisy_latents, target = self._inject_noise(
            latents,
            noise,
            timesteps,
            timestep_indices=timestep_indices,
        )

        # 5. Forward — policy + frozen reference
        with model_autocast(self.model, self.device):
            model_pred = self.forward_fn(
                self.model,
                noisy_latents,
                timesteps,
                encoder_hidden_states,
            )
        with torch.no_grad(), model_autocast(self.model, self.device):
            ref_pred = self._reference_forward(
                noisy_latents,
                timesteps,
                encoder_hidden_states,
            ).detach()

        # DPO/SFT reductions are protected FP32 math, not transformer forward.
        stats = self.algorithm.compute_loss(
            model_pred=model_pred.float(),
            ref_pred=ref_pred.float(),
            target=target.float(),
        )
        loss = stats["loss"]

        sft_loss_val = torch.tensor(0.0, device=self.device)
        sft_weight = self.algorithm.sft_weight
        if sft_weight > 0:
            bsz = model_pred.shape[0] // 2
            sft_loss_val = self.algorithm.sft_loss(model_pred[:bsz].float(), target[:bsz].float())
            loss = loss + sft_weight * sft_loss_val

        # 6. Backward + step (with optional accumulation)
        grad_norm = self._optimizer_update(loss)

        return DPOStepMetrics(
            loss=float(loss.detach()),
            raw_model_loss=float(stats["raw_model_loss"].detach()),
            raw_ref_loss=float(stats["raw_ref_loss"].detach()),
            model_diff=float(stats["model_diff"].detach()),
            ref_diff=float(stats["ref_diff"].detach()),
            implicit_acc=float(stats["implicit_acc"].detach()),
            sft_loss=float(sft_loss_val.detach()),
            grad_norm=grad_norm,
        )

    def _reference_forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Forward through the reference policy.

        If a separate ``ref_model`` was provided, use it directly. Otherwise
        assume the policy is a LoRA-wrapped backbone and disable adapters
        for the reference pass (PEFT convention).
        """
        if self.ref_model is not None:
            return self.forward_fn(
                self.ref_model,
                noisy_latents,
                timesteps,
                encoder_hidden_states,
            )
        if hasattr(self.model, "disable_adapter"):
            with self.model.disable_adapter():
                return self.forward_fn(
                    self.model,
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                )
        raise RuntimeError(
            "no ref_model and policy has no ``disable_adapter`` — "
            "cannot compute reference prediction"
        )


__all__ = ["DPOStepMetrics", "ForwardFn", "OfflineDPOTrainer"]
