"""Generic Ray actor substrate for reward inference (model-agnostic)."""

from vrl.rewards.ray.model import RewardModel
from vrl.rewards.ray.runtime import build_reward_actor_runtime, score_reward_request
from vrl.rewards.ray.worker import RewardModelWorker

__all__ = [
    "RewardModel",
    "RewardModelWorker",
    "build_reward_actor_runtime",
    "score_reward_request",
]
