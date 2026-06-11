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
from vrl.generation.pipeline import (
    PipelineStage,
    PipelineStagePayload,
    PipelineStageResult,
    PipelineStageRuntimePolicy,
    PipelineStageWorkerCore,
    PipelineTopology,
    SerialPipelineRunner,
)
from vrl.generation.protocols import (
    ChunkGatherer,
    ChunkResult,
    GenerationChunkExecutor,
    GenerationRuntime,
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
    "GenerationChunkExecutor",
    "GenerationMetrics",
    "GenerationOutput",
    "GenerationRequest",
    "GenerationRuntime",
    "GenerationRuntimeLaunchContract",
    "GenerationSampleRow",
    "PipelineStage",
    "PipelineStagePayload",
    "PipelineStageResult",
    "PipelineStageRuntimePolicy",
    "PipelineStageWorkerCore",
    "PipelineTopology",
    "SerialPipelineRunner",
    "WorkloadSignature",
    "build_sample_rows",
    "family_capability_from_value",
]
