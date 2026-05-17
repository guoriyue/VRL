"""RL trainers: training loop orchestration and weight sync."""

from vrl.trainers.core.base import Trainer
from vrl.trainers.core.types import TrainerConfig, TrainState
from vrl.trainers.data import DistributedKRepeatSampler, TextPromptDataset
from vrl.trainers.online import OnlineTrainer
from vrl.trainers.online.ema import EMAModuleWrapper
from vrl.trainers.weight_sync import (
    InMemoryWeightSyncer,
    RayRuntimeWeightSyncer,
    WeightSyncer,
    build_runtime_weight_syncer,
    build_trainable_state_sync_getter,
    flatten_trainable_module_state,
)

__all__ = [
    "DistributedKRepeatSampler",
    "EMAModuleWrapper",
    "InMemoryWeightSyncer",
    "OnlineTrainer",
    "RayRuntimeWeightSyncer",
    "TextPromptDataset",
    "TrainState",
    "Trainer",
    "TrainerConfig",
    "WeightSyncer",
    "build_runtime_weight_syncer",
    "build_trainable_state_sync_getter",
    "flatten_trainable_module_state",
]
