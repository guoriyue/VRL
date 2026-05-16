"""Diffusion-DPO loss — Direct Preference Optimization for diffusion models.

Ports the loss formulation from Wallace et al. (arXiv:2311.12908),
reference implementation at github.com/SalesforceAIResearch/DiffusionDPO
(see ``train.py:1119-1145``).

DPO is offline preference learning, fundamentally different from GRPO/PPO
(no rollouts, no advantages). The module keeps the pure functional loss for
offline trainers and exposes ``DiffusionDPO`` for the shared algorithm adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from vrl.algorithms.base import Algorithm
from vrl.algorithms.trajectory import AlgorithmInput
from vrl.algorithms.types import TrainStepMetrics


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


class DiffusionDPO(Algorithm):
    """Algorithm wrapper for Diffusion-DPO preference loss.

    DPO does not compute reward-normalized advantages. It consumes preference
    tensors from ``AlgorithmInput.metadata``:

    - ``model_pred``: policy prediction, shape ``[2B, ...]``
    - ``ref_pred``: frozen reference prediction, shape ``[2B, ...]``
    - ``target``: ground-truth noise/velocity target, shape ``[2B, ...]``
    """

    def __init__(self, config: DiffusionDPOConfig | None = None) -> None:
        self.config = config or DiffusionDPOConfig()

    def compute_advantages_from_tensors(
        self,
        rewards: Any,
        group_ids: Any,
    ) -> Any:
        del rewards, group_ids
        raise RuntimeError("DiffusionDPO is an offline preference objective")

    def compute_loss(
        self,
        inputs: AlgorithmInput,
    ) -> tuple[Any, TrainStepMetrics]:
        model_pred = inputs.metadata.get("model_pred")
        ref_pred = inputs.metadata.get("ref_pred")
        target = inputs.metadata.get("target")
        if model_pred is None or ref_pred is None or target is None:
            raise RuntimeError(
                "DiffusionDPO AlgorithmInput.metadata requires "
                "model_pred, ref_pred, and target",
            )

        beta = float(inputs.metadata.get("beta", self.config.beta))
        out = diffusion_dpo_loss(model_pred, ref_pred, target, beta=beta)
        dpo_loss = out["loss"]
        loss = dpo_loss
        if self.config.sft_weight > 0:
            half = model_pred.shape[0] // 2
            loss = loss + float(self.config.sft_weight) * diffusion_sft_loss(
                model_pred[:half],
                target[:half],
            )

        return loss, TrainStepMetrics(
            loss=float(loss.detach().item()),
            policy_loss=float(dpo_loss.detach().item()),
            approx_kl=float(out["raw_model_loss"].detach().item()),
            clip_fraction=float(out["implicit_acc"].detach().item()),
        )


def diffusion_dpo_loss(
    model_pred: torch.Tensor,
    ref_pred: torch.Tensor,
    target: torch.Tensor,
    beta: float,
) -> dict[str, torch.Tensor]:
    """Compute the Diffusion-DPO loss for one batch of preference pairs.

    Tensor layout: each tensor has leading dim ``2*B`` where the first ``B``
    entries are the *winner* (preferred) samples and the last ``B`` are the
    *loser* (dispreferred) samples — matching the reference repo's
    ``chunk(2)`` convention.

    Works for both 4-D (image UNet) and 5-D (video transformer) tensors —
    per-sample MSE reduces over all non-batch dims.

    Args:
      model_pred: policy prediction, shape ``[2B, ...]``
      ref_pred:   reference prediction (frozen), shape ``[2B, ...]``
      target:     ground-truth signal (noise for ε-pred, velocity for
                  flow-matching), shape ``[2B, ...]``
      beta:       DPO temperature

    Returns dict with:
      ``loss``           — scalar, the DPO objective to backprop
      ``raw_model_loss`` — diagnostic: average MSE under policy
      ``raw_ref_loss``   — diagnostic: average MSE under reference
      ``model_diff``     — winner_loss - loser_loss under policy
      ``ref_diff``       — winner_loss - loser_loss under reference
      ``implicit_acc``   — fraction of pairs where the policy ranks winner
                           above loser more strongly than the reference does
    """
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
    inside_term = -0.5 * beta * (model_diff - ref_diff)
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


def diffusion_sft_loss(
    model_pred_winner: torch.Tensor,
    target_winner: torch.Tensor,
) -> torch.Tensor:
    """Plain MSE on the winner only — useful as auxiliary regulariser.

    Pass ``model_pred[:B]`` and ``target[:B]`` (the winner halves).
    """
    return F.mse_loss(model_pred_winner.float(), target_winner.float(), reduction="mean")


__all__ = [
    "DiffusionDPO",
    "DiffusionDPOConfig",
    "diffusion_dpo_loss",
    "diffusion_sft_loss",
]
