"""Core trainer contracts — shared by online and offline trainers."""

from vrl.trainers.core.base import Trainer
from vrl.trainers.core.types import (
    ContinuousRolloutConfig,
    DebugConfig,
    EMAConfig,
    OptimConfig,
    RolloutOrchestrationConfig,
    TorchProfilerConfig,
    TrainerConfig,
    TrainState,
)

__all__ = [
    "ContinuousRolloutConfig",
    "DebugConfig",
    "EMAConfig",
    "OptimConfig",
    "RolloutOrchestrationConfig",
    "TorchProfilerConfig",
    "TrainState",
    "Trainer",
    "TrainerConfig",
]
