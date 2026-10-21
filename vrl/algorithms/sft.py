"""Supervised flow matching on clean latents: the SFT that precedes RL.

The loss is the family's own pretraining objective (velocity for flow
matching, noise or v for the DDPM ladders) on fixed targets, weighted per
token so a dataset can raise the regions it cares about. The GRPO
``sft_weight`` term uses the same target construction as a regularizer; this
algorithm uses it as the whole objective, so a run can warm the policy on
rendered truth before any rollout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vrl.algorithms.base import OfflineAlgorithm

# Torch is a call-time dependency: ``algorithm.kind`` dispatch loads this module
# during config parsing.
if TYPE_CHECKING:
    import torch


@dataclass(slots=True)
class DiffusionSFTConfig:
    """Hyper-parameters for supervised flow matching.

    ``timestep_index_range`` restricts the noise level to positions ``[low,
    high)`` of the scheduler's timestep table; ``None`` draws from the whole
    table. Qwen-Image-Layered trains over the first 48 of its 50 sigmas: the
    last two are so close to the data that their velocity carries no signal.
    """

    timestep_index_range: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.timestep_index_range is None:
            return
        low, high = (int(value) for value in self.timestep_index_range)
        if low < 0 or high <= low:
            raise ValueError(
                "algorithm.timestep_index_range must satisfy 0 <= low < high, "
                f"got {self.timestep_index_range!r}",
            )
        self.timestep_index_range = (low, high)


class DiffusionSFT(OfflineAlgorithm):
    """Offline supervised flow matching over cached clean latents."""

    sft_source = "unsupported"
    config: DiffusionSFTConfig
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
                    "offload",
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
                    "seed",
                }
            ),
        ),
        ("rollout", frozenset()),
        ("reward", None),
    )

    @staticmethod
    def compute_loss(
        model_pred: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Token-weighted squared error between the prediction and the target.

        ``weights`` broadcasts against the squared error and is expected to
        average one, so the loss stays on the plain-MSE scale whatever the
        dataset raises. Returns the optimized ``loss`` and the unweighted
        ``mse`` for logging.
        """

        if model_pred.shape != target.shape:
            raise ValueError(
                f"shape mismatch: model_pred={tuple(model_pred.shape)} "
                f"target={tuple(target.shape)}"
            )
        error = (model_pred.float() - target.float()).square()
        loss = error.mean() if weights is None else (error * weights.float()).mean()
        return {"loss": loss, "mse": error.detach().mean()}


__all__ = ["DiffusionSFT", "DiffusionSFTConfig"]
