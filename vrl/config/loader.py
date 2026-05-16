"""Backward-compatible config API exports."""

from __future__ import annotations

from vrl.config.builders import (
    build_algorithm_config,
    build_configs,
    build_reward_config,
    build_trainer_config,
)
from vrl.config.loading import CONFIGS_ROOT, load_config
from vrl.config.validation import (
    optional_none,
    require,
    resolve_algorithm_kind,
    validate_reward_config,
    validate_training_config,
)

__all__ = [
    "CONFIGS_ROOT",
    "build_algorithm_config",
    "build_configs",
    "build_reward_config",
    "build_trainer_config",
    "load_config",
    "optional_none",
    "require",
    "resolve_algorithm_kind",
    "validate_reward_config",
    "validate_training_config",
]
