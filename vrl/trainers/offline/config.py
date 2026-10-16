"""Configuration owned by the offline training loops (DPO and SFT)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vrl.utils.validation import require_int

if TYPE_CHECKING:
    from vrl.config.schema import RootConfig


@dataclass(slots=True)
class OfflineTrainerConfig:
    """Configuration for the offline training loops (DPO and SFT)."""

    # --- optimizer ---
    # The caller resolves lr scaling (e.g. by effective batch size) before
    # constructing this config; the trainer applies ``lr`` as-is.
    lr: float = 1e-8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    use_adafactor: bool = False
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    # --- noise / schedule ---
    prediction_type: str = "flow_matching"  # "epsilon" | "v_prediction" | "flow_matching"

    def __post_init__(self) -> None:
        require_int(
            self.gradient_accumulation_steps,
            path="gradient_accumulation_steps",
            minimum=1,
        )

    @classmethod
    def from_root(cls, root: RootConfig) -> OfflineTrainerConfig:
        """Project the public actor section into the offline trainer config."""
        from vrl.trainers.core.types import OptimConfig

        actor = root.actor

        def required(name: str) -> Any:
            value = None if actor is None else getattr(actor, name)
            if value is None:
                raise ValueError(f"config missing required field: actor.{name}")
            return value

        train_batch_size = required("train_batch_size")
        gradient_accumulation_steps = required("gradient_accumulation_steps")
        optim: OptimConfig = required("optim")
        if optim.optim_8bit:
            raise ValueError(
                "actor.optim.optim_8bit=true is not supported by the offline trainers; "
                "use AdamW/Adafactor without 8-bit optimizer state",
            )
        use_adafactor = required("use_adafactor")
        if use_adafactor:
            # An AdamW-only knob moved off its default would be silently ignored
            # under Adafactor; refuse rather than train with a no-op setting.
            defaults = OptimConfig(lr=optim.lr)
            adam_only_keys = sorted(
                key
                for key in ("adam_beta1", "adam_beta2", "eps")
                if getattr(optim, key) != getattr(defaults, key)
            )
            if adam_only_keys:
                paths = ", ".join(f"actor.optim.{key}" for key in adam_only_keys)
                raise ValueError(
                    f"actor.use_adafactor=true does not consume AdamW-only key(s): {paths}",
                )

        scale_lr = required("scale_lr")
        effective_batch_size = train_batch_size * gradient_accumulation_steps
        lr = float(optim.lr) * effective_batch_size if scale_lr else float(optim.lr)
        max_grad_norm = actor.max_norm if actor is not None else None
        if max_grad_norm is None:
            max_grad_norm = cls().max_grad_norm
        return cls(
            lr=lr,
            adam_beta1=float(optim.adam_beta1),
            adam_beta2=float(optim.adam_beta2),
            adam_weight_decay=float(optim.weight_decay),
            adam_epsilon=float(optim.eps),
            max_grad_norm=float(max_grad_norm),
            gradient_accumulation_steps=gradient_accumulation_steps,
            prediction_type=required("prediction_type"),
            use_adafactor=use_adafactor,
        )


__all__ = ["OfflineTrainerConfig"]
