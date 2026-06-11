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
from vrl.generation.ray.pipeline_runner import (
    RayPipelineRunner,
    RayPipelineStageHandle,
)
from vrl.generation.ray.stage_worker import RayPipelineStageWorker

__all__ = [
    "DRIVER_CUDA_OWNERSHIP_ERROR",
    "RayGenerationConfig",
    "RayGenerationLaunchInputs",
    "RayGenerationLauncher",
    "RayGenerationRuntime",
    "RayPipelineRunner",
    "RayPipelineStageHandle",
    "RayPipelineStageWorker",
    "ReleasableRayGenerationRuntime",
]
