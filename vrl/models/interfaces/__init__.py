"""Shared public model interfaces."""

from vrl.models.interfaces.generation_memory import (
    GenerationMemoryPolicy,
    VaeDecodeMemory,
)
from vrl.models.interfaces.replay import (
    ReplayModel,
    RuntimeModel,
    require_replay_model,
    require_runtime_model,
)
from vrl.models.interfaces.runtime import (
    ModelBuild,
    RolloutBuildOptions,
    RuntimeBundle,
    checkpoint_owned_state_names,
)

__all__ = [
    "GenerationMemoryPolicy",
    "ModelBuild",
    "ReplayModel",
    "RolloutBuildOptions",
    "RuntimeBundle",
    "RuntimeModel",
    "VaeDecodeMemory",
    "checkpoint_owned_state_names",
    "require_replay_model",
    "require_runtime_model",
]
