"""Shared public model interfaces."""

from vrl.models.interfaces.generation_memory import (
    GenerationMemoryPolicy,
    VaeDecodeMemory,
)
from vrl.models.interfaces.replay import (
    ReplayModel,
    ReplayRequest,
    ReplayRequestContract,
    ReplayResult,
    ReplaySegmentResult,
    RuntimeModel,
    replay_context_image_size,
    require_replay_model,
    require_runtime_model,
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
    "GenerationMemoryPolicy",
    "ModelBuild",
    "ReplayModel",
    "ReplayRequest",
    "ReplayRequestContract",
    "ReplayResult",
    "ReplaySegmentResult",
    "RolloutBuildOptions",
    "RuntimeBundle",
    "RuntimeModel",
    "VaeDecodeMemory",
    "checkpoint_owned_state_names",
    "register_checkpoint_owned_state",
    "replay_context_image_size",
    "require_replay_model",
    "require_runtime_model",
    "single_segment_result",
]
