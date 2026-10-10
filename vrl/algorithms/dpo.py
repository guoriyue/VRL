"""Diffusion-DPO loss — Direct Preference Optimization for diffusion models.

Ports the loss formulation from Wallace et al. (arXiv:2311.12908),
reference implementation at github.com/SalesforceAIResearch/DiffusionDPO
(see ``train.py:1119-1145``).

DPO is offline preference learning, fundamentally different from GRPO/PPO
(no rollouts, no advantages): ``DiffusionDPO`` is an ``OfflineAlgorithm``
whose loss the offline DPO trainer calls on each preference batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vrl.algorithms.base import OfflineAlgorithm

# Torch is a call-time dependency, not an import-time one: this module's config
# dataclass is what ``algorithm.kind`` dispatch loads during config parsing,
# and every annotation is a string under PEP 563.
if TYPE_CHECKING:
    import torch


@dataclass(slots=True)
class DiffusionDPOConfig:
    """Hyper-parameters for Diffusion-DPO.

    ``beta`` controls KL strength. Reference values from the paper:
      * SD 1.5 pixel-space: 5000
      * SDXL latent-space: 2000
      * Larger β → tighter ref-policy anchor.
    """

    beta: float = 5000.0
    sft_weight: float = 0.0  # optional auxiliary SFT-on-winner loss


class DiffusionDPO(OfflineAlgorithm):
    """The offline Diffusion-DPO algorithm.

    ``compute_loss`` is the paper's pairwise objective over the policy's and
    the frozen reference's predictions; ``sft_loss`` the optional winner-only
    regularizer. The class also restricts the config sections the recipe
    consumes.
    """

    sft_source = "preference_winner"
    config: DiffusionDPOConfig
    consumed_sections = (
        (
            "actor",
            frozenset(
                {
                    "gradient_accumulation_steps",
                    "gradient_checkpointing",
                    "max_norm",
                    "optim",
                    "prediction_type",
                    "scale_lr",
                    "train_batch_size",
                    "use_adafactor",
                }
            ),
        ),
        (
            "trainer",
            frozenset(
                {
                    "checkpointing_steps",
                    "entrypoint",
                    "log_interval",
                    "max_train_steps",
                    "output_dir",
                    "resume_from",
                }
            ),
        ),
        ("rollout", frozenset()),
        ("reward", None),
    )

    @property
    def sft_weight(self) -> float:  # type: ignore[override]
        return float(self.config.sft_weight)

    def compute_loss(
        self,
        model_pred: torch.Tensor,
        ref_pred: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute the Diffusion-DPO loss for one batch of preference pairs.

        Tensor layout: each tensor has leading dim ``2*B`` where the first ``B``
        entries are the *winner* (preferred) samples and the last ``B`` are the
        *loser* (dispreferred) samples — matching the reference repo's
        ``batch(2)`` convention.

        Works for both 4-D (image UNet) and 5-D (video transformer) tensors —
        per-sample MSE reduces over all non-batch dims.

        Args:
          model_pred: policy prediction, shape ``[2B, ...]``
          ref_pred:   reference prediction (frozen), shape ``[2B, ...]``
          target:     ground-truth signal (noise for ε-pred, velocity for
                      flow-matching), shape ``[2B, ...]``

        Returns dict with:
          ``loss``           — scalar, the DPO objective to backprop
          ``raw_model_loss`` — diagnostic: average MSE under policy
          ``raw_ref_loss``   — diagnostic: average MSE under reference
          ``model_diff``     — winner_loss - loser_loss under policy
          ``ref_diff``       — winner_loss - loser_loss under reference
          ``implicit_acc``   — fraction of pairs where the policy ranks winner
                               above loser more strongly than the reference does
        """
        import torch
        import torch.nn.functional as F

        if model_pred.shape != ref_pred.shape or model_pred.shape != target.shape:
            raise ValueError(
                f"shape mismatch: model_pred={tuple(model_pred.shape)} "
                f"ref_pred={tuple(ref_pred.shape)} target={tuple(target.shape)}"
            )
        if model_pred.shape[0] % 2 != 0:
            raise ValueError(
                f"leading dim must be 2*B (winner-then-loser); got {model_pred.shape[0]}"
            )

        reduce_dims = tuple(range(1, model_pred.ndim))

        # Per-sample MSE — shape [2B]
        model_losses = (model_pred - target).pow(2).mean(dim=reduce_dims)
        model_losses_w, model_losses_l = model_losses.chunk(2)

        with torch.no_grad():
            ref_losses = (ref_pred - target).pow(2).mean(dim=reduce_dims)
            ref_losses_w, ref_losses_l = ref_losses.chunk(2)

        model_diff = model_losses_w - model_losses_l
        ref_diff = ref_losses_w - ref_losses_l

        # Lower MSE on winner → smaller model_diff. We want policy to push
        # this even further below ref_diff, so inside_term should be > 0.
        inside_term = -0.5 * float(self.config.beta) * (model_diff - ref_diff)
        loss = -F.logsigmoid(inside_term).mean()

        implicit_acc = (inside_term > 0).float().mean()

        return {
            "loss": loss,
            "raw_model_loss": 0.5 * (model_losses_w.mean() + model_losses_l.mean()),
            "raw_ref_loss": ref_losses.mean(),
            "model_diff": model_diff.mean(),
            "ref_diff": ref_diff.mean(),
            "implicit_acc": implicit_acc,
        }

    @staticmethod
    def sft_loss(model_pred_winner: torch.Tensor, target_winner: torch.Tensor) -> torch.Tensor:
        """Plain MSE on the winner halves (``model_pred[:B]``, ``target[:B]``)."""

        import torch.nn.functional as F

        return F.mse_loss(model_pred_winner.float(), target_winner.float(), reduction="mean")


__all__ = ["DiffusionDPO", "DiffusionDPOConfig"]
