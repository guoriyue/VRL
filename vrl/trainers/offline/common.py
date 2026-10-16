"""What the offline trainers share: the optimizer, the forward process, the step boundary.

An offline trainer runs **synchronously** on a fixed dataset: no rollout
collection, so none of the OnlineTrainer's async machinery. DPO and SFT differ
only in what one step forwards and which loss it takes; the optimizer, the
timestep draw, the noise injection per ``prediction_type``, gradient
accumulation and resumable state live here once. Their config is
``vrl.trainers.offline.config.OfflineTrainerConfig``.

Noise/target convention, one branch of ``inject_noise`` per ``prediction_type``:

  * ``epsilon`` / ``v_prediction`` (``DDPMScheduler.add_noise``)
  * ``flow_matching`` velocity (``scheduler.scale_noise``, or a sigma lookup)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from vrl.models.precision import apply_float32_precision
from vrl.trainers.offline.config import OfflineTrainerConfig
from vrl.utils.validation import require_int


class OfflineTrainer:
    """The synchronous training loop's shared half: optimizer, noise, step boundary.

    ``model`` is the policy (typically a LoRA-wrapped backbone) stamped with its
    ``precision`` role. A subclass defines ``step`` over its own batch type,
    draws timesteps with ``_sample_timestep_indices``, builds the forward
    process with ``_inject_noise`` and ends the micro-step with
    ``_optimizer_update``.
    """

    def __init__(
        self,
        model: nn.Module,
        noise_scheduler: Any,
        *,
        config: OfflineTrainerConfig | None = None,
        device: torch.device | str = "cuda",
    ) -> None:
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.precision = model.precision
        self.config = config or OfflineTrainerConfig()
        self.device = torch.device(device) if isinstance(device, str) else device
        apply_float32_precision(self.precision.float32_precision)
        self.global_step = 0
        self._gradient_accumulation_micro_step = 0

        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("model has no trainable parameters — wire up LoRA / unfreeze first")
        cfg = self.config
        if cfg.use_adafactor:
            try:
                from transformers.optimization import Adafactor
            except ImportError as exc:
                raise ImportError(
                    "Install transformers for Adafactor: pip install transformers"
                ) from exc
            self._optimizer = Adafactor(
                trainable,
                lr=cfg.lr,
                scale_parameter=False,
                relative_step=False,
                warmup_init=False,
                weight_decay=cfg.adam_weight_decay,
            )
        else:
            self._optimizer = torch.optim.AdamW(
                trainable,
                lr=cfg.lr,
                betas=(cfg.adam_beta1, cfg.adam_beta2),
                weight_decay=cfg.adam_weight_decay,
                eps=cfg.adam_epsilon,
            )

    # ------------------------------------------------------------------
    # Noise injection — branch on prediction_type
    # ------------------------------------------------------------------

    def _sample_timestep_indices(
        self,
        bsz: int,
        *,
        index_range: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Uniform positions in the scheduler's timestep table, or in ``index_range`` of it."""

        # Resolve the timestep range explicitly. Silently using
        # ``num_train_timesteps`` when ``scheduler.timesteps`` is empty
        # would mask the common bug of forgetting to call
        # ``scheduler.set_timesteps(...)`` before training, which can
        # silently change the sampling distribution.
        ts = getattr(self.noise_scheduler, "timesteps", None)
        if ts is None or len(ts) == 0:
            raise RuntimeError(
                "noise_scheduler.timesteps is empty/missing. "
                "Call scheduler.set_timesteps(num_inference_steps) "
                "before training."
            )
        lo, hi = 0, len(ts)
        if index_range is not None:
            lo, hi = index_range
            if hi > len(ts):
                raise ValueError(
                    f"timestep_index_range {index_range!r} exceeds the scheduler's "
                    f"{len(ts)}-entry timestep table",
                )
        return torch.randint(lo, hi, (bsz,), device=self.device)

    def _inject_noise(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        timestep_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (noisy_latents, target).

        For epsilon prediction:    target = noise
        For flow-matching:         target = noise - latents (velocity)
        """
        pt = self.config.prediction_type
        if pt == "epsilon":
            noisy = self.noise_scheduler.add_noise(latents, noise, timesteps)
            return noisy, noise
        if pt == "v_prediction":
            noisy = self.noise_scheduler.add_noise(latents, noise, timesteps)
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
            return noisy, target
        if pt == "flow_matching":
            # Flow-matching schedulers expose ``scale_noise(sample, timestep, noise)``.
            # Forward process: x_t = (1 - sigma) * x_0 + sigma * noise.
            if hasattr(self.noise_scheduler, "scale_noise"):
                noisy = self.noise_scheduler.scale_noise(latents, timesteps, noise)
            else:
                # The sampled table index also identifies its sigma exactly.
                sigmas = self.noise_scheduler.sigmas.to(latents.device)
                view = (-1,) + (1,) * (latents.ndim - 1)
                sigma = sigmas[timestep_indices].view(*view)
                noisy = (1.0 - sigma) * latents + sigma * noise
            target = noise - latents
            return noisy, target
        raise ValueError(f"unknown prediction_type: {pt}")

    # ------------------------------------------------------------------
    # Step boundary
    # ------------------------------------------------------------------

    def _optimizer_update(self, loss: torch.Tensor) -> float:
        """Backward one micro-step; clip and step at the accumulation boundary.

        Returns the pre-clip gradient norm at a boundary, 0 otherwise.
        """

        cfg = self.config
        (loss / cfg.gradient_accumulation_steps).backward()
        grad_norm = 0.0
        if self._mark_gradient_accumulation_step():
            if cfg.max_grad_norm > 0:
                gn = nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                grad_norm = float(gn)
            self._optimizer.step()
            self._optimizer.zero_grad(set_to_none=True)
        self.global_step += 1
        return grad_norm

    def _mark_gradient_accumulation_step(self) -> bool:
        accumulation_steps = self.config.gradient_accumulation_steps
        self._gradient_accumulation_micro_step += 1
        if self._gradient_accumulation_micro_step < accumulation_steps:
            return False
        self._gradient_accumulation_micro_step = 0
        return True

    def state_dict(self) -> dict[str, Any]:
        """Return resumable trainer state."""

        return {
            "global_step": self.global_step,
            "optimizer": self._optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore resumable trainer state."""

        if not isinstance(state, dict):
            raise TypeError(f"{type(self).__name__}.load_state_dict expects a dict")
        global_step = require_int(
            state.get("global_step", 0), path="trainer_state.global_step", minimum=0
        )
        if "optimizer" not in state:
            raise ValueError("checkpoint missing optimizer state")
        self._optimizer.load_state_dict(state["optimizer"])
        # Neither accumulated gradients nor a partial accumulation window resume.
        self._optimizer.zero_grad(set_to_none=True)
        self.global_step = global_step
        self._gradient_accumulation_micro_step = 0


__all__ = ["OfflineTrainer"]
