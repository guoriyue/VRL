"""Core trainer contracts — shared by online and offline trainers."""

from vrl.trainers.core.base import Trainer
from vrl.trainers.core.types import (
    ContinuousRolloutConfig,
    DebugConfig,
    EMAConfig,
    OptimConfig,
    RolloutOrchestrationConfig,
    TrainState,
)
from vrl.utils.profiling import TorchProfilerConfig

__all__ = [
    "ContinuousRolloutConfig",
    "DebugConfig",
    "EMAConfig",
    "OptimConfig",
    "RolloutOrchestrationConfig",
    "TorchProfilerConfig",
    "TrainState",
    "Trainer",
]
