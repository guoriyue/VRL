"""Rollout collector construction for RL training."""

from vrl.rollouts.collector.config import RolloutCollectorConfig
from vrl.rollouts.collector.core import (
    RolloutCollector,
    build_rollout_collector,
)

__all__ = [
    "RolloutCollector",
    "RolloutCollectorConfig",
    "build_rollout_collector",
]
