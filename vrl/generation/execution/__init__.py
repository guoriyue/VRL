"""Engine execution planning and request batches."""

from vrl.generation.execution.chunk_placement import (
    ChunkPlacementPolicy,
    DeviceAssignment,
    DistributedExecutionPlanner,
    DistributedGenerationPlan,
)
from vrl.generation.execution.chunks import (
    SampleChunk,
    SampleChunkSchedule,
    build_prompt_chunk_schedule,
    run_sample_chunks_with_oom_retry,
)
from vrl.generation.execution.ids import build_sample_rows
from vrl.generation.execution.planner import (
    EnginePlan,
    EnginePlanner,
    ExecutionStage,
    ResolvedAxis,
    attach_engine_plan,
    build_engine_plan,
)
from vrl.generation.execution.types import (
    ChunkExecutionEnvelope,
    ChunkExecutionResult,
    DistributedWorkerHandle,
)
from vrl.generation.execution.worker import GenerationWorkerCore

__all__ = [
    "ChunkExecutionEnvelope",
    "ChunkExecutionResult",
    "ChunkPlacementPolicy",
    "DeviceAssignment",
    "DistributedExecutionPlanner",
    "DistributedGenerationPlan",
    "DistributedWorkerHandle",
    "EnginePlan",
    "EnginePlanner",
    "ExecutionStage",
    "GenerationWorkerCore",
    "ResolvedAxis",
    "SampleChunk",
    "SampleChunkSchedule",
    "attach_engine_plan",
    "build_engine_plan",
    "build_prompt_chunk_schedule",
    "build_sample_rows",
    "run_sample_chunks_with_oom_retry",
]
