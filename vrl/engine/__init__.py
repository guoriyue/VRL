"""Visual generation engine primitives."""

from vrl.engine.batching import forward_batch_by_merging_prompts
from vrl.engine.core.capabilities import (
    AxisCapability,
    ExecutionUnitCapability,
    FamilyCapability,
    ar_continuous_family_capability,
    ar_discrete_family_capability,
    diffusion_family_capability,
    family_capability_from_value,
    generic_family_capability,
)
from vrl.engine.core.planner import (
    AxisPlan,
    EnginePlan,
    ExecutionUnit,
    attach_engine_plan,
    build_engine_plan,
    profiler_label_for_unit,
    resolve_executor_capability,
)
from vrl.engine.core.protocols import (
    BatchedFamilyPipelineExecutor,
    CapabilityAwareFamilyPipelineExecutor,
    ChunkedFamilyPipelineExecutor,
    FamilyPipelineExecutor,
    PipelineChunkResult,
)
from vrl.engine.core.registry import ExecutorKey, FamilyPipelineRegistry
from vrl.engine.core.runtime import (
    GenerationRuntime,
    RolloutBackend,
)
from vrl.engine.core.runtime_spec import GenerationRuntimeSpec
from vrl.engine.core.types import (
    GenerationMetrics,
    GenerationRequest,
    GenerationSampleSpec,
    OutputBatch,
    RolloutDebugTensors,
    RolloutDenoisingEnv,
    RolloutDitTrajectory,
    RolloutTrajectoryData,
    WorkloadSignature,
)
from vrl.engine.core.worker import GenerationIdFactory, GenerationWorker
from vrl.engine.gather import (
    gather_pipeline_chunks,
    require_chunked_executor,
)

__all__ = [
    "AxisCapability",
    "AxisPlan",
    "BatchedFamilyPipelineExecutor",
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
    "RolloutBackend",
    "RolloutDebugTensors",
    "RolloutDenoisingEnv",
    "RolloutDitTrajectory",
    "RolloutTrajectoryData",
    "WorkloadSignature",
    "ar_continuous_family_capability",
    "ar_discrete_family_capability",
    "attach_engine_plan",
    "build_engine_plan",
    "diffusion_family_capability",
    "family_capability_from_value",
    "forward_batch_by_merging_prompts",
    "gather_pipeline_chunks",
    "generic_family_capability",
    "profiler_label_for_unit",
    "require_chunked_executor",
    "resolve_executor_capability",
]
