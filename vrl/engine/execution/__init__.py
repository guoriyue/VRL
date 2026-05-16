"""Engine execution planning and request batches."""

from vrl.engine.core.protocols import RolloutBackend
from vrl.engine.execution.ids import GenerationIdFactory
from vrl.engine.execution.microbatching import (
    MicroBatchSample,
    MicroBatchSchedule,
    build_prompt_microbatch_schedule,
    run_microbatch_samples_with_oom_retry,
)
from vrl.engine.execution.planner import (
    EnginePlan,
    EnginePlanner,
    ExecutionUnit,
    ResolvedAxis,
    attach_engine_plan,
    build_engine_plan,
    resolve_executor_capability,
)
from vrl.engine.execution.request_batch import RequestBatch

__all__ = [
    "EnginePlan",
    "EnginePlanner",
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
    "resolve_executor_capability",
    "run_microbatch_samples_with_oom_retry",
]
