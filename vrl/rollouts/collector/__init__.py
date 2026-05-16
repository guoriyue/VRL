"""Rollout collector construction for RL training."""

from vrl.rollouts.collector.core import (
    LAST_COLLECT_PHASES,
    RolloutCollector,
    build_rollout_collector,
)

__all__ = [
    "LAST_COLLECT_PHASES",
    "RolloutCollector",
    "build_rollout_collector",
]
