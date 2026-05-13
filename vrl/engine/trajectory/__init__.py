"""Trajectory contract types for generation RL."""

from vrl.engine.trajectory.axes import AxisKind, AxisSpec
from vrl.engine.trajectory.builders import (
    build_ar_continuous_trajectory,
    build_ar_discrete_trajectory,
    build_ar_multisegment_trajectory,
    build_diffusion_trajectory,
)
from vrl.engine.trajectory.compat import (
    TRAJECTORY_EXTRA_KEY,
    get_output_trajectory,
    require_output_trajectory,
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
    replay_input_ref,
    tensor_ref,
    validate_loss_unit,
    validate_reward_view,
    validate_training_view,
    validate_trajectory_batch,
)
from vrl.engine.trajectory.view_builders import build_training_view
from vrl.engine.trajectory.views import (
    AlgorithmFamily,
    LossUnit,
    RewardModality,
    RewardView,
    TrainingView,
)

__all__ = [
    "TRAJECTORY_EXTRA_KEY",
    "AdvantageScope",
    "AlgorithmFamily",
    "AxisKind",
    "AxisSpec",
    "DistributionKind",
    "LossUnit",
    "ReplayInput",
    "ReplaySignalKind",
    "RewardModality",
    "RewardView",
    "SegmentModality",
    "TensorRole",
    "TrainingView",
    "TrajectoryBatch",
    "TrajectoryMetrics",
    "TrajectorySegment",
    "TrajectoryTensor",
    "TrajectoryValidationError",
    "build_ar_continuous_trajectory",
    "build_ar_discrete_trajectory",
    "build_ar_multisegment_trajectory",
    "build_diffusion_trajectory",
    "build_training_view",
    "get_output_trajectory",
    "replay_input_ref",
    "require_output_trajectory",
    "tensor_ref",
    "validate_loss_unit",
    "validate_reward_view",
    "validate_training_view",
    "validate_trajectory_batch",
]
