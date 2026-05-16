"""Trajectory contract types for generation RL."""

from vrl.engine.trajectory.axes import AxisKind, TrajectoryAxis
from vrl.engine.trajectory.builders import (
    build_ar_continuous_trajectory,
    build_ar_discrete_trajectory,
    build_ar_multisegment_trajectory,
    build_diffusion_trajectory,
)
from vrl.engine.trajectory.resolver import (
    ResolvedContextValue,
    ResolvedLossUnit,
    ResolvedReplayInput,
    ResolvedTrainingView,
    ResolvedTrajectoryTensor,
    TrajectoryResolver,
    TrajectoryResolverError,
)
from vrl.engine.trajectory.types import (
    AdvantageScope,
    DistributionKind,
    ReplayInput,
    ReplaySignalKind,
    SegmentModality,
    TensorRole,
    TrajectoryBatch,
    TrajectoryMetrics,
    TrajectorySegment,
    TrajectoryTensor,
)
from vrl.engine.trajectory.validation import (
    TrajectoryValidationError,
    TrajectoryValidator,
    replay_input_ref,
    tensor_ref,
)
from vrl.engine.trajectory.views import (
    AlgorithmFamily,
    LossUnit,
    RewardModality,
    RewardView,
    TrainingView,
    build_training_view,
)

__all__ = [
    "AdvantageScope",
    "AlgorithmFamily",
    "AxisKind",
    "DistributionKind",
    "LossUnit",
    "ReplayInput",
    "ReplaySignalKind",
    "ResolvedContextValue",
    "ResolvedLossUnit",
    "ResolvedReplayInput",
    "ResolvedTrainingView",
    "ResolvedTrajectoryTensor",
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
    "TrajectoryTensor",
    "TrajectoryValidationError",
    "TrajectoryValidator",
    "build_ar_continuous_trajectory",
    "build_ar_discrete_trajectory",
    "build_ar_multisegment_trajectory",
    "build_diffusion_trajectory",
    "build_training_view",
    "replay_input_ref",
    "tensor_ref",
]
