"""Core engine contracts."""

from vrl.engine.core.backend import RolloutBackend
from vrl.engine.core.capabilities import (
    AxisCapability,
    ExecutionUnitCapability,
    FamilyCapability,
    ar_continuous_family_capability,
    ar_discrete_family_capability,
    diffusion_family_capability,
    family_capability_from_value,
)
from vrl.engine.core.protocols import (
    ChunkedFamilyPipelineExecutor,
    FamilyPipelineExecutor,
    PipelineChunkResult,
)
from vrl.engine.core.runtime_spec import GenerationRuntimeSpec
from vrl.engine.core.types import (
    GenerationMetrics,
    GenerationRequest,
    GenerationSampleSpec,
    OutputBatch,
    WorkloadSignature,
)

__all__ = [
    "AxisCapability",
    "ChunkedFamilyPipelineExecutor",
    "ExecutionUnitCapability",
    "FamilyCapability",
    "FamilyPipelineExecutor",
    "GenerationMetrics",
    "GenerationRequest",
    "GenerationRuntimeSpec",
    "GenerationSampleSpec",
    "OutputBatch",
    "PipelineChunkResult",
    "RolloutBackend",
    "WorkloadSignature",
    "ar_continuous_family_capability",
    "ar_discrete_family_capability",
    "diffusion_family_capability",
    "family_capability_from_value",
]
