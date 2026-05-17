"""Core trainer contracts — shared by online and offline trainers."""

from vrl.trainers.core.base import Trainer
from vrl.trainers.core.types import (
    DebugConfig,
    EMAConfig,
    OptimConfig,
    TorchProfilerConfig,
    TrainerConfig,
    TrainState,
)

__all__ = [
    "DebugConfig",
    "EMAConfig",
    "OptimConfig",
    "TorchProfilerConfig",
    "TrainState",
    "Trainer",
    "TrainerConfig",
]
