"""RL trainers: public training contracts and lazy implementations."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vrl.trainers.core.base import Trainer as Trainer
    from vrl.trainers.core.types import (
        RolloutOrchestrationConfig as RolloutOrchestrationConfig,
    )
    from vrl.trainers.core.types import TrainState as TrainState
    from vrl.trainers.online.config import OnlineBatchPlan as OnlineBatchPlan
    from vrl.trainers.online.config import TrainerConfig as TrainerConfig
    from vrl.trainers.online.ema import EMAModuleWrapper as EMAModuleWrapper
    from vrl.trainers.online.trainer import OnlineTrainer as OnlineTrainer
    from vrl.trainers.weight_sync import (
        RayRuntimeWeightSyncer as RayRuntimeWeightSyncer,
    )
    from vrl.trainers.weight_sync import (
        WeightSyncer as WeightSyncer,
    )
    from vrl.trainers.weight_sync import (
        build_runtime_weight_syncer as build_runtime_weight_syncer,
    )
    from vrl.trainers.weight_sync import (
        build_trainable_state_sync_getter as build_trainable_state_sync_getter,
    )
    from vrl.trainers.weight_sync import (
        flatten_trainable_module_state as flatten_trainable_module_state,
    )


# This table is the public lazy-import boundary: keeping the exports together
# avoids importing torch-heavy trainer implementations during config parsing.
_PUBLIC_EXPORTS = {
    "EMAModuleWrapper": ("vrl.trainers.online.ema", "EMAModuleWrapper"),
    "OnlineBatchPlan": ("vrl.trainers.online.config", "OnlineBatchPlan"),
    "OnlineTrainer": ("vrl.trainers.online.trainer", "OnlineTrainer"),
    "RayRuntimeWeightSyncer": ("vrl.trainers.weight_sync", "RayRuntimeWeightSyncer"),
    "RolloutOrchestrationConfig": (
        "vrl.trainers.core.types",
        "RolloutOrchestrationConfig",
    ),
    "TrainState": ("vrl.trainers.core.types", "TrainState"),
    "Trainer": ("vrl.trainers.core.base", "Trainer"),
    "TrainerConfig": ("vrl.trainers.online.config", "TrainerConfig"),
    "WeightSyncer": ("vrl.trainers.weight_sync", "WeightSyncer"),
    "build_runtime_weight_syncer": (
        "vrl.trainers.weight_sync",
        "build_runtime_weight_syncer",
    ),
    "build_trainable_state_sync_getter": (
        "vrl.trainers.weight_sync",
        "build_trainable_state_sync_getter",
    ),
    "flatten_trainable_module_state": (
        "vrl.trainers.weight_sync",
        "flatten_trainable_module_state",
    ),
}

__all__ = list(_PUBLIC_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load a public trainer symbol only when it is requested."""

    try:
        module_name, symbol_name = _PUBLIC_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), symbol_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
