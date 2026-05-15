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
from vrl.models.interfaces.runtime import RuntimeBuildSpec, RuntimeBundle

__all__ = [
    "ReplayModel",
    "ReplayRequest",
    "ReplayResult",
    "ReplaySegmentResult",
    "RuntimeBuildSpec",
    "RuntimeBundle",
    "RuntimeModel",
    "require_replay_model",
    "require_runtime_model",
]
