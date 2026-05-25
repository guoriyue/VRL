"""Generic Ray actor substrate for reward inference (model-agnostic)."""

from vrl.rewards.ray.model import RewardModel
from vrl.rewards.ray.runtime import RayRewardRuntime
from vrl.rewards.ray.worker import RewardModelWorker

__all__ = [
    "RayRewardRuntime",
    "RewardModel",
    "RewardModelWorker",
]
