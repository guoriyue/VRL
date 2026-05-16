"""Engine execution planning, batching, and gathering."""

from vrl.engine.core.backend import RolloutBackend
from vrl.engine.execution.batching import forward_batch_by_merging_prompts
from vrl.engine.execution.gather import (
    ChunkGatherer,
    gather_pipeline_chunks,
    require_chunk_gatherer,
    require_chunked_executor,
)
from vrl.engine.execution.ids import GenerationIdFactory
from vrl.engine.execution.microbatching import (
    ExecutionPlan,
    MicroBatchPlan,
    RolloutShardPlan,
    plan_prompt_group_microbatches,
    run_microbatches_with_oom_retry,
)
from vrl.engine.execution.planner import (
    AxisPlan,
    EnginePlan,
    ExecutionUnit,
    attach_engine_plan,
    build_engine_plan,
    profiler_label_for_unit,
    resolve_executor_capability,
)

__all__ = [
    "AxisPlan",
    "ChunkGatherer",
    "EnginePlan",
    "ExecutionPlan",
    "ExecutionUnit",
    "GenerationIdFactory",
    "MicroBatchPlan",
    "RolloutBackend",
    "RolloutShardPlan",
    "attach_engine_plan",
    "build_engine_plan",
    "forward_batch_by_merging_prompts",
    "gather_pipeline_chunks",
    "plan_prompt_group_microbatches",
    "profiler_label_for_unit",
    "require_chunk_gatherer",
    "require_chunked_executor",
    "resolve_executor_capability",
    "run_microbatches_with_oom_retry",
]
