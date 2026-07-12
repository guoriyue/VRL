"""Standalone reward service public facade with lazy endpoint imports."""

from __future__ import annotations

from typing import Any

__all__ = [
    "HttpRewardRuntime",
    "RemoteRewardServiceError",
    "RewardService",
    "RewardServiceInfo",
]


def __getattr__(name: str) -> Any:
    # Keep ``python -m vrl.rewards.service.server`` from importing the target
    # module through its package first, which otherwise triggers runpy's eager
    # import warning and executes endpoint setup twice.
    if name == "HttpRewardRuntime":
        from vrl.rewards.service.client import HttpRewardRuntime

        return HttpRewardRuntime
    if name == "RewardService":
        from vrl.rewards.service.server import RewardService

        return RewardService
    if name in {"RemoteRewardServiceError", "RewardServiceInfo"}:
        from vrl.rewards.service.protocol import (
            RemoteRewardServiceError,
            RewardServiceInfo,
        )

        return {
            "RemoteRewardServiceError": RemoteRewardServiceError,
            "RewardServiceInfo": RewardServiceInfo,
        }[name]
    raise AttributeError(name)
