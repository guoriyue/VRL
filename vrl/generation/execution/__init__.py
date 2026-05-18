"""Engine execution planning and request batches."""

from vrl.generation.execution.ids import GenerationIdFactory
from vrl.generation.execution.microbatching import (
    MicroBatchSample,
    MicroBatchSchedule,
    build_prompt_microbatch_schedule,
    run_microbatch_samples_with_oom_retry,
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
    "MicroBatchSample",
    "MicroBatchSchedule",
    "RequestBatch",
    "ResolvedAxis",
    "StagePlacement",
    "attach_engine_plan",
    "build_engine_plan",
    "build_prompt_microbatch_schedule",
    "resolve_executor_capability",
    "run_microbatch_samples_with_oom_retry",
]
