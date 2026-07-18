"""Shared public model interfaces."""

from vrl.models.interfaces.replay import (
    ReplayModel,
    ReplayRequest,
    ReplayResult,
    ReplaySegmentResult,
    RuntimeModel,
    require_replay_model,
    require_runtime_model,
)
from vrl.models.interfaces.runtime import (
    ModelBuild,
    ResolvedForwardPrecision,
    RolloutBuildOptions,
    RuntimeBundle,
)

__all__ = [
    "ModelBuild",
    "ReplayModel",
    "ReplayRequest",
    "ReplayResult",
    "ReplaySegmentResult",
    "ResolvedForwardPrecision",
    "RolloutBuildOptions",
    "RuntimeBundle",
    "RuntimeModel",
    "require_replay_model",
    "require_runtime_model",
]
