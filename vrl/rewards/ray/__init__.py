"""Thin Ray adapter for reward inference runtimes."""

from vrl.rewards.ray.launcher import build_reward_ray_runtime
from vrl.rewards.ray.runtime import RewardInferenceActorRuntime

__all__ = ["RewardInferenceActorRuntime", "build_reward_ray_runtime"]
