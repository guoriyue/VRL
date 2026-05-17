"""Generation runtime factory wiring."""

from vrl.generation.runtime.config import GenerationRuntimeConfig
from vrl.generation.runtime.factory import (
    DRIVER_CUDA_OWNERSHIP_ERROR,
    ReleasableRayGenerationRuntime,
    build_generation_runtime_from_cfg,
    validate_generation_runtime_config,
)
from vrl.generation.runtime.launch_inputs import (
    GenerationRuntimeInputs,
    build_generation_runtime_inputs,
)

__all__ = [
    "DRIVER_CUDA_OWNERSHIP_ERROR",
    "GenerationRuntimeConfig",
    "GenerationRuntimeInputs",
    "ReleasableRayGenerationRuntime",
    "build_generation_runtime_from_cfg",
    "build_generation_runtime_inputs",
    "validate_generation_runtime_config",
]
