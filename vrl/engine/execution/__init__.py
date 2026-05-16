"""Engine execution planning, request batches, and gathering."""

from vrl.engine.core.protocols import RolloutBackend
from vrl.engine.execution.gather import (
    ChunkGatherer,
    gather_pipeline_chunks,
    require_chunk_gatherer,
    require_chunked_executor,
)
from vrl.engine.execution.ids import GenerationIdFactory
from vrl.engine.execution.microbatching import (
    MicroBatchSample,
    MicroBatchSchedule,
    build_prompt_microbatch_schedule,
    run_microbatch_samples_with_oom_retry,
)
from vrl.engine.execution.planner import (
    EnginePlan,
    ExecutionUnit,
    ResolvedAxis,
    attach_engine_plan,
    build_engine_plan,
    profiler_label_for_unit,
    resolve_executor_capability,
)
from vrl.engine.execution.request_batch import RequestBatch

__all__ = [
    "ChunkGatherer",
    "EnginePlan",
    "ExecutionUnit",
    "GenerationIdFactory",
    "MicroBatchSample",
    "MicroBatchSchedule",
    "RequestBatch",
    "ResolvedAxis",
    "RolloutBackend",
    "attach_engine_plan",
    "build_engine_plan",
    "build_prompt_microbatch_schedule",
    "gather_pipeline_chunks",
    "profiler_label_for_unit",
    "require_chunk_gatherer",
    "require_chunked_executor",
    "resolve_executor_capability",
    "run_microbatch_samples_with_oom_retry",
]
