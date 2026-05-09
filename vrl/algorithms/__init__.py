"""RL algorithms: advantage estimation and policy gradient losses."""

from vrl.algorithms.base import Algorithm
from vrl.algorithms.flow_matching import (
    SDEStepResult,
    compute_kl_divergence,
    sde_step_with_logprob,
)
from vrl.algorithms.grpo.continuous import GRPO, GRPOConfig
from vrl.algorithms.grpo.multisegment import (
    MultiSegmentTokenGRPO,
    MultiSegmentTokenGRPOConfig,
)
from vrl.algorithms.grpo.token import TokenGRPO, TokenGRPOConfig
from vrl.algorithms.types import (
    Rollout,
    TrainStepMetrics,
    Trajectory,
    TrajectoryStep,
)

__all__ = [
    "GRPO",
    "Algorithm",
    "GRPOConfig",
    "MultiSegmentTokenGRPO",
    "MultiSegmentTokenGRPOConfig",
    "Rollout",
    "SDEStepResult",
    "TokenGRPO",
    "TokenGRPOConfig",
    "TrainStepMetrics",
    "Trajectory",
    "TrajectoryStep",
    "compute_kl_divergence",
    "sde_step_with_logprob",
]
