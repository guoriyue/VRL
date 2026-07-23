"""Trajectory contract types for generation RL."""

from vrl.trajectory.builders import (
    build_ar_continuous_trajectory,
    build_ar_discrete_trajectory,
    build_ar_multisegment_trajectory,
    build_chunk_autoregressive_denoise_trajectory,
    build_chunk_autoregressive_generation_trajectory,
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
    tensor_ref,
)
from vrl.trajectory.views import (
    RewardView,
    named_tensor,
    role_tensor,
)

__all__ = [
    "AdvantageScope",
    "AxisKind",
    "DistributionKind",
    "ReplayInput",
    "RewardView",
    "SegmentModality",
    "TensorRole",
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
    "build_chunk_autoregressive_denoise_trajectory",
    "build_chunk_autoregressive_generation_trajectory",
    "build_diffusion_trajectory",
    "named_tensor",
    "role_tensor",
    "tensor_ref",
    "trajectory_storage_policy_from_cfg",
    "trajectory_tensor_bytes",
]
