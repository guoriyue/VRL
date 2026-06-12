"""Trajectory contract types for generation RL."""

from vrl.trajectory.builders import (
    build_ar_continuous_trajectory,
    build_ar_discrete_trajectory,
    build_ar_multisegment_trajectory,
    build_diffusion_trajectory,
)
from vrl.trajectory.resolver import (
    TrajectoryResolver,
    TrajectoryResolverError,
)
from vrl.trajectory.storage import (
    TrajectoryStoragePolicy,
    apply_trajectory_storage_policy,
    trajectory_storage_policy_from_cfg,
    trajectory_tensor_bytes,
)
from vrl.trajectory.types import (
    AdvantageScope,
    AxisKind,
    DistributionKind,
    ReplayInput,
    ReplaySignalKind,
    SegmentModality,
    TensorRole,
    TrajectoryAxis,
    TrajectoryBatch,
    TrajectoryMetrics,
    TrajectorySegment,
    TrajectoryTensor,
)
from vrl.trajectory.validation import (
    TrajectoryValidationError,
    TrajectoryValidator,
    replay_input_ref,
    tensor_ref,
)
from vrl.trajectory.views import (
    AlgorithmFamily,
    LossUnit,
    RewardModality,
    RewardView,
    TrainingView,
    build_training_view,
    role_tensor,
)

__all__ = [
    "AdvantageScope",
    "AlgorithmFamily",
    "AxisKind",
    "DistributionKind",
    "LossUnit",
    "ReplayInput",
    "ReplaySignalKind",
    "RewardModality",
    "RewardView",
    "SegmentModality",
    "TensorRole",
    "TrainingView",
    "TrajectoryAxis",
    "TrajectoryBatch",
    "TrajectoryMetrics",
    "TrajectoryResolver",
    "TrajectoryResolverError",
    "TrajectorySegment",
    "TrajectoryStoragePolicy",
    "TrajectoryTensor",
    "TrajectoryValidationError",
    "TrajectoryValidator",
    "apply_trajectory_storage_policy",
    "build_ar_continuous_trajectory",
    "build_ar_discrete_trajectory",
    "build_ar_multisegment_trajectory",
    "build_diffusion_trajectory",
    "build_training_view",
    "replay_input_ref",
    "role_tensor",
    "tensor_ref",
    "trajectory_storage_policy_from_cfg",
    "trajectory_tensor_bytes",
]
