"""Rollout collector construction for RL training."""

from vrl.rollouts.collector.config import RolloutCollectorConfig
from vrl.rollouts.collector.core import (
    RolloutCollector,
)

__all__ = [
    "RolloutCollector",
    "RolloutCollectorConfig",
]
