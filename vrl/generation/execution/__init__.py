"""Engine execution planning and request batches."""

from vrl.generation.execution.chunk_placement import (
    DeviceAssignment,
    DistributedExecutionPlanner,
    DistributedGenerationPlan,
)
from vrl.generation.execution.chunks import (
    SampleChunk,
    build_prompt_chunks,
    run_sample_chunks_with_oom_retry,
)
from vrl.generation.execution.ids import build_sample_rows
from vrl.generation.execution.planner import (
    EnginePlan,
    build_engine_plan,
)
from vrl.generation.execution.types import (
    ChunkExecutionEnvelope,
    ChunkExecutionResult,
    ChunkMemoryReading,
    ChunkPlacementStrategy,
    ChunkSizeProbeResult,
    ChunkSizeProbeTrial,
    WorkerMemoryParkingSnapshot,
)
from vrl.generation.execution.worker import GenerationWorkerCore

__all__ = [
    "ChunkExecutionEnvelope",
    "ChunkExecutionResult",
    "ChunkMemoryReading",
    "ChunkPlacementStrategy",
    "ChunkSizeProbeResult",
    "ChunkSizeProbeTrial",
    "DeviceAssignment",
    "DistributedExecutionPlanner",
    "DistributedGenerationPlan",
    "EnginePlan",
    "GenerationWorkerCore",
    "SampleChunk",
    "WorkerMemoryParkingSnapshot",
    "build_engine_plan",
    "build_prompt_chunks",
    "build_sample_rows",
    "run_sample_chunks_with_oom_retry",
]
