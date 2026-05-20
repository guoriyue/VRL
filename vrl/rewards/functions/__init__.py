"""Concrete reward function implementations."""

from __future__ import annotations

from typing import Any

from vrl.rewards.functions.composite import CompositeReward
from vrl.rewards.functions.registry import MultiReward, get_reward, register_reward


def __getattr__(name: str) -> Any:
    if name == "VideoReward":
        from vrl.rewards.functions.video_reward import VideoReward

        return VideoReward
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CompositeReward",
    "MultiReward",
    "VideoReward",
    "get_reward",
    "register_reward",
]
