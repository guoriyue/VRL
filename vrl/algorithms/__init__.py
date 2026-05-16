"""RL algorithms: advantage estimation and policy gradient losses."""

from vrl.algorithms.base import Algorithm
from vrl.algorithms.diffusion_nft import DiffusionNFT, DiffusionNFTConfig
from vrl.algorithms.grpo.continuous import GRPO, GRPOConfig
from vrl.algorithms.grpo.multisegment import (
    MultiSegmentTokenGRPO,
    MultiSegmentTokenGRPOConfig,
)
from vrl.algorithms.grpo.token import TokenGRPO, TokenGRPOConfig
from vrl.algorithms.trajectory import AlgorithmAdapter, AlgorithmInput
from vrl.algorithms.types import (
    Rollout,
    TrainStepMetrics,
    Trajectory,
    TrajectoryStep,
)

__all__ = [
    "GRPO",
    "Algorithm",
    "AlgorithmAdapter",
    "AlgorithmInput",
    "DiffusionNFT",
    "DiffusionNFTConfig",
    "GRPOConfig",
    "MultiSegmentTokenGRPO",
    "MultiSegmentTokenGRPOConfig",
    "Rollout",
    "TokenGRPO",
    "TokenGRPOConfig",
    "TrainStepMetrics",
    "Trajectory",
    "TrajectoryStep",
]
