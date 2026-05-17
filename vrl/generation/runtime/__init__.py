"""Generation runtime factory wiring."""

from vrl.generation.runtime.config import GenerationRuntimeConfig, RolloutBackendConfig
from vrl.generation.runtime.factory import (
    DRIVER_CUDA_OWNERSHIP_ERROR,
    ReleasableRayGenerationRuntime,
    build_generation_runtime_from_cfg,
    build_rollout_backend_from_cfg,
    validate_generation_runtime_config,
    validate_rollout_backend_config,
)
from vrl.generation.runtime.launch_inputs import (
    GenerationRuntimeInputs,
    RolloutRuntimeInputs,
    build_generation_runtime_inputs,
)

__all__ = [
    "DRIVER_CUDA_OWNERSHIP_ERROR",
    "GenerationRuntimeConfig",
    "GenerationRuntimeInputs",
    "ReleasableRayGenerationRuntime",
    "RolloutBackendConfig",
    "RolloutRuntimeInputs",
    "build_generation_runtime_from_cfg",
    "build_generation_runtime_inputs",
    "build_rollout_backend_from_cfg",
    "validate_generation_runtime_config",
    "validate_rollout_backend_config",
]
