"""Thin Ray adapter for generation runtimes."""

from vrl.generation.ray.config import (
    DRIVER_CUDA_OWNERSHIP_ERROR,
    RayGenerationConfig,
)
from vrl.generation.ray.launcher import (
    RayGenerationLauncher,
    RayGenerationLaunchInputs,
)
from vrl.generation.ray.runtime import (
    RayGenerationRuntime,
    ReleasableRayGenerationRuntime,
)

__all__ = [
    "DRIVER_CUDA_OWNERSHIP_ERROR",
    "RayGenerationConfig",
    "RayGenerationLaunchInputs",
    "RayGenerationLauncher",
    "RayGenerationRuntime",
    "ReleasableRayGenerationRuntime",
]
