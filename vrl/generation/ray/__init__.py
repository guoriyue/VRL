"""Ray generation runtime components."""

from __future__ import annotations

from vrl.generation.ray.executor import DistributedGenerationExecutor, DistributedRolloutExecutor
from vrl.generation.ray.launcher import (
    RayGenerationLauncher,
    RayPlacement,
    RayRolloutLauncher,
    create_generation_placement_group,
    create_rollout_placement_group,
)
from vrl.generation.ray.planner import (
    DeviceAssignment,
    DistributedExecutionPlanner,
    DistributedGenerationPlan,
    DistributedRolloutPlan,
)
from vrl.generation.ray.runtime import RayDistributedRuntime, RayGenerationRuntime
from vrl.generation.ray.types import (
    RayChunkExecutionEnvelope,
    RayChunkResult,
    RayWorkerHandle,
)
from vrl.generation.ray.weight_sync import (
    GenerationWeightSync,
    RayGenerationWeightSync,
    RayRolloutWeightSync,
    RolloutWeightSync,
)
from vrl.generation.ray.worker import RayGenerationWorker, RayRolloutWorker

__all__ = [
    "DeviceAssignment",
    "DistributedExecutionPlanner",
    "DistributedGenerationExecutor",
    "DistributedGenerationPlan",
    "DistributedRolloutExecutor",
    "DistributedRolloutPlan",
    "GenerationWeightSync",
    "RayChunkExecutionEnvelope",
    "RayChunkResult",
    "RayDistributedRuntime",
    "RayGenerationLauncher",
    "RayGenerationRuntime",
    "RayGenerationWeightSync",
    "RayGenerationWorker",
    "RayPlacement",
    "RayRolloutLauncher",
    "RayRolloutWeightSync",
    "RayRolloutWorker",
    "RayWorkerHandle",
    "RolloutWeightSync",
    "create_generation_placement_group",
    "create_rollout_placement_group",
]
