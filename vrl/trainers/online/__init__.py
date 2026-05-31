"""Online training loop package."""

from vrl.trainers.online.trainer import OnlineTrainer, _validate_ema_state_shapes

__all__ = [
    "OnlineTrainer",
    "_validate_ema_state_shapes",
]
