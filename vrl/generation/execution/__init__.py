"""Engine execution planning and request batches."""

from vrl.generation.execution.chunks import (
    SampleChunk,
    SampleChunkSchedule,
    build_prompt_chunk_schedule,
    run_sample_chunks_with_oom_retry,
)
from vrl.generation.execution.ids import GenerationIdFactory
from vrl.generation.execution.planner import (
    EnginePlan,
    EnginePlanner,
    ExecutionStage,
    ResolvedAxis,
    attach_engine_plan,
    build_engine_plan,
    resolve_executor_capability,
)
from vrl.generation.execution.request_batch import RequestBatch
from vrl.generation.execution.scheduler import (
    DeviceAssignment,
    DistributedExecutionPlanner,
    DistributedGenerationPlan,
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
    "DeviceAssignment",
    "DistributedExecutionPlanner",
    "DistributedGenerationPlan",
    "DistributedWorkerHandle",
    "EnginePlan",
    "EnginePlanner",
    "ExecutionStage",
    "GenerationIdFactory",
    "GenerationWorkerCore",
    "RequestBatch",
    "ResolvedAxis",
    "SampleChunk",
    "SampleChunkSchedule",
    "attach_engine_plan",
    "build_engine_plan",
    "build_prompt_chunk_schedule",
    "resolve_executor_capability",
    "run_sample_chunks_with_oom_retry",
]
