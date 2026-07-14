"""Rollout config projection from user YAML configs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from vrl.utils.config import cfg_path, to_builtin_deep


@dataclass(frozen=True, slots=True)
class RolloutCollectorConfig:
    """Resolved rollout config projected from user YAML configs."""

    values: dict[str, Any]

    def get(self, name: str, default: Any = None) -> Any:
        """Return a resolved config value or ``default`` when absent."""

        return self.values.get(name, default)

    def request_sampling(self) -> dict[str, Any]:
        """Return config values that should be serialized into an engine request."""

        return {
            key: value
            for key, value in self.values.items()
            if key not in _REQUEST_SAMPLING_EXCLUDES
        }


def build_rollout_config_from_cfg(cfg: Any) -> RolloutCollectorConfig:
    """Project rollout and sampling values from merged YAML."""

    values: dict[str, Any] = {}
    _merge_flat_section_values(values, cfg, "rollout")
    sde_values = _cfg_mapping(cfg, "rollout.sde")
    for source_key, target_key in {
        "type": "sde_type",
        "window_size": "sde_window_size",
        "window_range": "sde_window_range",
    }.items():
        if source_key in sde_values and sde_values[source_key] is not None:
            values[target_key] = to_builtin_deep(sde_values[source_key])
    _merge_flat_section_values(values, cfg, "sampling")
    _copy_value(values, cfg, "kl_reward_coef", "algorithm.kl_reward_coef")
    _copy_value(
        values,
        cfg,
        "train_segments",
        "algorithm.train_segments",
    )
    _copy_value(values, cfg, "trajectory_storage", "rollout.trajectory_storage")
    has_sde_values = any(
        name in values for name in ("sde_type", "sde_window_size", "sde_window_range")
    )
    if "kl_reward_coef" in values and has_sde_values:
        values["return_kl"] = float(values.get("kl_reward_coef", 0.0)) > 0.0
    return RolloutCollectorConfig(values=values)


def _merge_flat_section_values(
    values: dict[str, Any],
    cfg: Any,
    section: str,
) -> None:
    section_values = _cfg_mapping(cfg, section)
    for key, value in section_values.items():
        name = str(key)
        normalized = to_builtin_deep(value)
        if isinstance(normalized, dict):
            continue
        if name in values:
            raise ValueError(
                f"rollout request key {name!r} has multiple config owners; "
                f"remove the duplicate from {section}",
            )
        values[name] = normalized


def _copy_value(
    values: dict[str, Any],
    cfg: Any,
    name: str,
    path: str,
) -> None:
    value = cfg_path(cfg, path, _MISSING)
    if value is not _MISSING and value is not None:
        values[name] = to_builtin_deep(value)


def _cfg_mapping(cfg: Any, path: str) -> dict[str, Any]:
    value = cfg_path(cfg, path, _MISSING)
    if value is _MISSING or value is None:
        return {}
    value = to_builtin_deep(value)
    if isinstance(value, Mapping):
        return {str(key): inner for key, inner in value.items()}
    raise ValueError(f"{path} config must be a mapping")


_MISSING = object()

_REQUEST_SAMPLING_EXCLUDES = {
    "host_memory_budget_fraction",
    "kl_reward_coef",
    "microbatch_size",
    "n_samples_per_prompt",
    "prompts_per_batch",
}


__all__ = [
    "RolloutCollectorConfig",
    "build_rollout_config_from_cfg",
]
