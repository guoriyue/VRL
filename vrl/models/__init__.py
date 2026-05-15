"""Model interfaces and family implementations."""

from vrl.models.interfaces import (
    ReplayModel,
    ReplayRequest,
    ReplayResult,
    ReplaySegmentResult,
    RuntimeBuildSpec,
    RuntimeBundle,
    RuntimeModel,
    require_replay_model,
    require_runtime_model,
)
from vrl.models.replay_loading import (
    FULL_GENERATION_RUNTIME_ROLE,
    MINIMAL_REPLAY_RUNTIME_ROLE,
    ReplayModuleLoadingProfile,
    bundle_loads_full_generation_modules,
    full_generation_bundle_metadata,
    minimal_replay_bundle_metadata,
    module_loading_profile_from_metadata,
    require_minimal_replay_bundle,
)

__all__ = [
    "FULL_GENERATION_RUNTIME_ROLE",
    "MINIMAL_REPLAY_RUNTIME_ROLE",
    "ReplayModel",
    "ReplayModuleLoadingProfile",
    "ReplayRequest",
    "ReplayResult",
    "ReplaySegmentResult",
    "RuntimeBuildSpec",
    "RuntimeBundle",
    "RuntimeModel",
    "bundle_loads_full_generation_modules",
    "full_generation_bundle_metadata",
    "minimal_replay_bundle_metadata",
    "module_loading_profile_from_metadata",
    "require_minimal_replay_bundle",
    "require_replay_model",
    "require_runtime_model",
]
