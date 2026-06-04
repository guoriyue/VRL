"""Concrete reward function implementations."""

from __future__ import annotations

from typing import Any

from vrl.rewards.functions.registry import MultiReward, get_reward


def __getattr__(name: str) -> Any:
    if name == "KlingVideoReward":
        from vrl.rewards.functions.kling_video_reward import KlingVideoReward

        return KlingVideoReward
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "KlingVideoReward",
    "MultiReward",
    "get_reward",
]
