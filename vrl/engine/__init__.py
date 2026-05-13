"""Visual generation engine primitives."""

from vrl.engine.core.capabilities import (
    AxisCapability,
    ExecutionUnitCapability,
    FamilyCapability,
    ar_continuous_family_capability,
    ar_discrete_family_capability,
    diffusion_family_capability,
    family_capability_from_value,
)
from vrl.engine.core.protocols import (
    CapabilityAwareFamilyPipelineExecutor,
    ChunkedFamilyPipelineExecutor,
    FamilyPipelineExecutor,
    PipelineChunkResult,
    PlanAwareBatchedFamilyPipelineExecutor,
    PlanAwareFamilyPipelineExecutor,
)
from vrl.engine.core.registry import ExecutorKey, FamilyPipelineRegistry
from vrl.engine.core.runtime_spec import GenerationRuntimeSpec
from vrl.engine.core.types import (
    GenerationMetrics,
    GenerationRequest,
    GenerationSampleSpec,
    OutputBatch,
    WorkloadSignature,
)
from vrl.engine.execution.batching import forward_batch_by_merging_prompts
from vrl.engine.execution.gather import (
    gather_pipeline_chunks,
    require_chunked_executor,
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
from vrl.engine.execution.runtime import (
    GenerationRuntime,
    RolloutBackend,
)
from vrl.engine.execution.worker import GenerationIdFactory, GenerationWorker

__all__ = [
    "AxisCapability",
    "AxisPlan",
    "CapabilityAwareFamilyPipelineExecutor",
    "ChunkedFamilyPipelineExecutor",
    "EnginePlan",
    "ExecutionUnit",
    "ExecutionUnitCapability",
    "ExecutorKey",
    "FamilyCapability",
    "FamilyPipelineExecutor",
    "FamilyPipelineRegistry",
    "GenerationIdFactory",
    "GenerationMetrics",
    "GenerationRequest",
    "GenerationRuntime",
    "GenerationRuntimeSpec",
    "GenerationSampleSpec",
    "GenerationWorker",
    "OutputBatch",
    "PipelineChunkResult",
    "PlanAwareBatchedFamilyPipelineExecutor",
    "PlanAwareFamilyPipelineExecutor",
    "RolloutBackend",
    "WorkloadSignature",
    "ar_continuous_family_capability",
    "ar_discrete_family_capability",
    "attach_engine_plan",
    "build_engine_plan",
    "diffusion_family_capability",
    "family_capability_from_value",
    "forward_batch_by_merging_prompts",
    "gather_pipeline_chunks",
    "profiler_label_for_unit",
    "require_chunked_executor",
    "resolve_executor_capability",
]
