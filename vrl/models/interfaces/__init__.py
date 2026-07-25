"""Shared public model interfaces."""

from vrl.models.interfaces.replay import (
    ReplayModel,
    ReplayRequest,
    ReplayResult,
    ReplaySegmentResult,
    RuntimeModel,
    replay_context_image_size,
    require_replay_model,
    require_replay_segments,
    require_runtime_model,
    require_zero_replay_timestep,
    resolve_image_token_replay,
    single_segment_result,
)
from vrl.models.interfaces.runtime import (
    ModelBuild,
    RolloutBuildOptions,
    RuntimeBundle,
    checkpoint_owned_state_names,
    register_checkpoint_owned_state,
)

__all__ = [
    "ModelBuild",
    "ReplayModel",
    "ReplayRequest",
    "ReplayResult",
    "ReplaySegmentResult",
    "RolloutBuildOptions",
    "RuntimeBundle",
    "RuntimeModel",
    "checkpoint_owned_state_names",
    "register_checkpoint_owned_state",
    "replay_context_image_size",
    "require_replay_model",
    "require_replay_segments",
    "require_runtime_model",
    "require_zero_replay_timestep",
    "resolve_image_token_replay",
    "single_segment_result",
]
