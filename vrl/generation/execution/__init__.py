"""Engine execution planning and request batches."""

from vrl.generation.execution.ids import GenerationIdFactory
from vrl.generation.execution.chunks import (
    SampleChunk,
    SampleChunkSchedule,
    build_prompt_chunk_schedule,
    run_sample_chunks_with_oom_retry,
)
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
from vrl.generation.execution.stage_plan import GenerationStagePlan, StagePlacement

__all__ = [
    "EnginePlan",
    "EnginePlanner",
    "ExecutionStage",
    "GenerationIdFactory",
    "GenerationStagePlan",
    "SampleChunk",
    "SampleChunkSchedule",
    "RequestBatch",
    "ResolvedAxis",
    "StagePlacement",
    "attach_engine_plan",
    "build_engine_plan",
    "build_prompt_chunk_schedule",
    "resolve_executor_capability",
    "run_sample_chunks_with_oom_retry",
]
