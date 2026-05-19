"""Generation-specific distributed execution primitives."""

from vrl.generation.execution.distributed.planner import (
    DeviceAssignment,
    DistributedExecutionPlanner,
    DistributedGenerationPlan,
    DistributedRolloutPlan,
)
from vrl.generation.execution.distributed.types import (
    ChunkExecutionEnvelope,
    ChunkExecutionResult,
    DistributedWorkerHandle,
)

__all__ = [
    "ChunkExecutionEnvelope",
    "ChunkExecutionResult",
    "DeviceAssignment",
    "DistributedExecutionPlanner",
    "DistributedGenerationPlan",
    "DistributedRolloutPlan",
    "DistributedWorkerHandle",
]
