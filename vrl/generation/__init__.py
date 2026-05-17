"""Generation runtime public API.

This package owns generation execution contracts and runtime helpers. RL
rollout collection, rewards, and trainer-ready batches live outside this layer.
"""

from vrl.generation.capabilities import (
    ExecutionStageCapability,
    FamilyCapability,
    family_capability_from_value,
)
from vrl.generation.execution.ids import GenerationIdFactory
from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.generation.protocols import (
    ChunkedFamilyPipelineExecutor,
    ChunkGatherer,
    FamilyPipelineExecutor,
    GenerationRuntime,
    PipelineChunkResult,
    RolloutBackend,
)
from vrl.generation.types import (
    GenerationMetrics,
    GenerationOutput,
    GenerationRequest,
    GenerationSampleRow,
    OutputBatch,
    WorkloadSignature,
)

__all__ = [
    "ChunkGatherer",
    "ChunkedFamilyPipelineExecutor",
    "ExecutionStageCapability",
    "FamilyCapability",
    "FamilyPipelineExecutor",
    "GenerationIdFactory",
    "GenerationMetrics",
    "GenerationOutput",
    "GenerationRequest",
    "GenerationRuntime",
    "GenerationRuntimeLaunchContract",
    "GenerationSampleRow",
    "OutputBatch",
    "PipelineChunkResult",
    "RolloutBackend",
    "WorkloadSignature",
    "family_capability_from_value",
]
