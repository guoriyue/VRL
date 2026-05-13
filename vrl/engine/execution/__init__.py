"""Engine execution planning, batching, gathering, and worker runtime."""

from vrl.engine.execution.batching import forward_batch_by_merging_prompts
from vrl.engine.execution.gather import (
    ChunkGatherer,
    DiffusionChunkGatherer,
    gather_diffusion_chunks,
    gather_pipeline_chunks,
    require_chunk_gatherer,
    require_chunked_executor,
)
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
from vrl.engine.execution.runtime import GenerationRuntime, RolloutBackend
from vrl.engine.execution.worker import GenerationIdFactory, GenerationWorker

__all__ = [
    "AxisPlan",
    "ChunkGatherer",
    "DiffusionChunkGatherer",
    "EnginePlan",
    "ExecutionPlan",
    "ExecutionUnit",
    "GenerationIdFactory",
    "GenerationRuntime",
    "GenerationWorker",
    "MicroBatchPlan",
    "RolloutBackend",
    "RolloutShardPlan",
    "attach_engine_plan",
    "build_engine_plan",
    "forward_batch_by_merging_prompts",
    "gather_diffusion_chunks",
    "gather_pipeline_chunks",
    "plan_prompt_group_microbatches",
    "profiler_label_for_unit",
    "require_chunk_gatherer",
    "require_chunked_executor",
    "resolve_executor_capability",
    "run_microbatches_with_oom_retry",
]
