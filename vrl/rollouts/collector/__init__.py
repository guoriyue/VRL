"""Rollout collector construction for RL training."""

from vrl.rollouts.collector.base import Collector
from vrl.rollouts.collector.core import RolloutCollector
from vrl.rollouts.collector.factory import (
    COLLECTOR_REGISTRY,
    LAST_COLLECT_PHASES,
    CollectorRegistryEntry,
    build_rollout_collector,
)

__all__ = [
    "COLLECTOR_REGISTRY",
    "LAST_COLLECT_PHASES",
    "Collector",
    "CollectorRegistryEntry",
    "RolloutCollector",
    "build_rollout_collector",
]
