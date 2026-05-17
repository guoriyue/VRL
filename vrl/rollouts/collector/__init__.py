"""Rollout collector construction for RL training."""

from vrl.rollouts.collector.config import (
    RolloutConfig,
    build_rollout_config_from_cfg,
)
from vrl.rollouts.collector.core import (
    LAST_COLLECT_PHASES,
    RolloutCollector,
    build_rollout_collector,
)

__all__ = [
    "LAST_COLLECT_PHASES",
    "RolloutCollector",
    "RolloutConfig",
    "build_rollout_collector",
    "build_rollout_config_from_cfg",
]
