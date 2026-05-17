"""Compatibility exports for generation runtime factory."""

from vrl.generation.runtime.factory import (
    DRIVER_CUDA_OWNERSHIP_ERROR,
    ReleasableRayGenerationRuntime,
    ReleasableRayRolloutBackend,
    build_generation_runtime_from_cfg,
    build_rollout_backend_from_cfg,
    validate_generation_runtime_config,
    validate_rollout_backend_config,
)

__all__ = [
    "DRIVER_CUDA_OWNERSHIP_ERROR",
    "ReleasableRayGenerationRuntime",
    "ReleasableRayRolloutBackend",
    "build_generation_runtime_from_cfg",
    "build_rollout_backend_from_cfg",
    "validate_generation_runtime_config",
    "validate_rollout_backend_config",
]
