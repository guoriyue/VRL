"""Generation runtime public API.

This package owns generation execution contracts and runtime helpers. RL
rollout collection, rewards, and trainer-ready batches live outside this layer.
"""

from vrl.generation.capabilities import (
    ExecutionStageCapability,
    FamilyCapability,
    family_capability_from_value,
)
from vrl.generation.execution.ids import build_sample_rows
from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.generation.protocols import (
    ChunkGatherer,
    ChunkResult,
    GenerationRuntime,
    PipelineExecutor,
)
from vrl.generation.types import (
    GenerationMetrics,
    GenerationOutput,
    GenerationRequest,
    GenerationSampleRow,
    WorkloadSignature,
)

__all__ = [
    "ChunkGatherer",
    "ChunkResult",
    "ExecutionStageCapability",
    "FamilyCapability",
    "GenerationMetrics",
    "GenerationOutput",
    "GenerationRequest",
    "GenerationRuntime",
    "GenerationRuntimeLaunchContract",
    "GenerationSampleRow",
    "PipelineExecutor",
    "WorkloadSignature",
    "build_sample_rows",
    "family_capability_from_value",
]
