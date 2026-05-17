"""Compatibility exports for rollout runtime backend wiring."""

from vrl.generation.runtime.factory import (
    DRIVER_CUDA_OWNERSHIP_ERROR,
    ReleasableRayRolloutBackend,
    build_rollout_backend_from_cfg,
    validate_rollout_backend_config,
)

__all__ = [
    "DRIVER_CUDA_OWNERSHIP_ERROR",
    "ReleasableRayRolloutBackend",
    "build_rollout_backend_from_cfg",
    "validate_rollout_backend_config",
]
