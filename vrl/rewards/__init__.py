"""Reward functions for RL training."""

from typing import Any

from vrl.rewards.base import RewardFunction
from vrl.rewards.functions.registry import MultiReward, get_reward, register_reward
from vrl.rewards.types import RewardRollout, RewardTrajectory, RewardTrajectoryStep


def __getattr__(name: str) -> Any:
    if name == "VideoReward":
        from vrl.rewards.functions.video_reward import VideoReward

        return VideoReward
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "MultiReward",
    "RewardFunction",
    "RewardRollout",
    "RewardTrajectory",
    "RewardTrajectoryStep",
    "VideoReward",
    "get_reward",
    "register_reward",
]
