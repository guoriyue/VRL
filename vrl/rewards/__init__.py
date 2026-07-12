"""Reward functions for RL training."""

from typing import Any

from vrl.rewards.base import RewardFunction
from vrl.rewards.functions.registry import MultiReward, get_reward
from vrl.rewards.types import RewardRollout


def __getattr__(name: str) -> Any:
    if name == "KlingVideoReward":
        from vrl.rewards.functions.kling_video_reward import KlingVideoReward

        return KlingVideoReward
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "KlingVideoReward",
    "MultiReward",
    "RewardFunction",
    "RewardRollout",
    "get_reward",
]
