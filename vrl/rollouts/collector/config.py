"""Rollout config projection from user YAML configs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf


@dataclass(frozen=True, slots=True)
class RolloutConfig:
    """Resolved rollout config projected from user YAML configs."""

    family: str
    values: dict[str, Any] = field(default_factory=dict)

    def get(self, name: str, default: Any = None) -> Any:
        """Return a resolved config value or ``default`` when absent."""

        return self.values.get(name, default)

    def require(self, name: str) -> Any:
        """Return a resolved config value or fail with a clear family-scoped error."""

        try:
            return self.values[name]
        except KeyError as exc:
            raise ValueError(
                f"{self.family} rollout config missing required field {name!r}",
            ) from exc

    def request_sampling(self) -> dict[str, Any]:
        """Return config values that should be serialized into an engine request."""

        return {
            key: value
            for key, value in self.values.items()
            if key not in _REQUEST_SAMPLING_EXCLUDES
        }


def build_rollout_config_from_cfg(
    cfg: Any,
    *,
    family: str,
) -> RolloutConfig:
    """Resolve one rollout family's config from YAML."""

    values: dict[str, Any] = {}
    _merge_flat_section_values(values, cfg, "rollout")
    _merge_sde_values(values, cfg)
    _merge_flat_section_values(values, cfg, "sampling")
    _copy_first_present(values, cfg, "kl_reward", ("algorithm.kl_reward",))
    _copy_first_present(
        values,
        cfg,
        "final_image_policy",
        ("rollout.final_image_policy", "sampling.r1.final_image_policy"),
    )
    _copy_first_present(
        values,
        cfg,
        "train_segments",
        (
            "rollout.train_segments",
            "algorithm.train_segments",
            "sampling.r1.train_segments",
        ),
    )
    _copy_first_present(values, cfg, "trajectory_storage", ("rollout.trajectory_storage",))
    _copy_first_present(values, cfg, "reward_artifact", ("rollout.reward_artifact",))
    _add_derived_values(values)
    return RolloutConfig(family=family, values=values)


def _merge_flat_section_values(
    values: dict[str, Any],
    cfg: Any,
    section: str,
) -> None:
    section_values = _cfg_mapping(cfg, section)
    for key, value in section_values.items():
        name = str(key)
        normalized = _normalize_config_value(name, value)
        if isinstance(normalized, dict):
            continue
        values[name] = normalized


def _merge_sde_values(values: dict[str, Any], cfg: Any) -> None:
    sde_values = _cfg_mapping(cfg, "rollout.sde")
    aliases = {
        "type": "sde_type",
        "window_size": "sde_window_size",
        "window_range": "sde_window_range",
    }
    for source_key, target_key in aliases.items():
        if source_key in sde_values and sde_values[source_key] is not None:
            values[target_key] = _normalize_config_value(
                target_key,
                sde_values[source_key],
            )


def _copy_first_present(
    values: dict[str, Any],
    cfg: Any,
    name: str,
    paths: tuple[str, ...],
) -> None:
    for path in paths:
        value = _cfg_select(cfg, path, _MISSING)
        if value is not _MISSING and value is not None:
            values[name] = _normalize_config_value(name, value)
            return


def _add_derived_values(values: dict[str, Any]) -> None:
    if "kl_reward" in values and _has_sde_sampling(values):
        values["return_kl"] = float(values.get("kl_reward", 0.0)) > 0.0


def _has_sde_sampling(values: dict[str, Any]) -> bool:
    return any(
        name in values
        for name in ("sde_type", "sde_window_size", "sde_window_range")
    )


def _normalize_config_value(name: str, value: Any) -> Any:
    value = _to_builtin(value)
    if name == "sde_window_range":
        return tuple(value)
    return value


def _cfg_mapping(cfg: Any, path: str) -> dict[str, Any]:
    value = _cfg_select(cfg, path, _MISSING)
    if value is _MISSING or value is None:
        return {}
    value = _to_builtin(value)
    if isinstance(value, Mapping):
        return {str(key): inner for key, inner in value.items()}
    raise ValueError(f"{path} config must be a mapping")


def _cfg_select(cfg: Any, path: str, default: Any) -> Any:
    if isinstance(cfg, DictConfig):
        return OmegaConf.select(cfg, path, default=default)
    node = cfg
    for key in path.split("."):
        node = _cfg_get(node, key, _MISSING)
        if node is _MISSING:
            return default
    return node


def _cfg_get(node: Any, key: str, default: Any) -> Any:
    if node is None:
        return default
    getter = getattr(node, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass
    try:
        return node[key]
    except (KeyError, IndexError, TypeError):
        pass
    return getattr(node, key, default)


def _to_builtin(value: Any) -> Any:
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {str(key): _to_builtin(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_to_builtin(inner) for inner in value]
    if isinstance(value, tuple):
        return tuple(_to_builtin(inner) for inner in value)
    return value


_MISSING = object()

_REQUEST_SAMPLING_EXCLUDES = {
    "kl_reward",
    "n",
    "n_samples_per_prompt",
    "reward_view",
    "reward_view_name",
    "rollout_batch_size",
}


__all__ = [
    "RolloutConfig",
    "build_rollout_config_from_cfg",
]
