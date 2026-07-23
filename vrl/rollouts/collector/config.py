"""Rollout config projection from user YAML configs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from vrl.config.algorithm import resolve_kl_reward_coef
from vrl.config.schema import (
    SamplingConfig,
    generation_request_rollout_fields,
)
from vrl.trajectory import (
    TrajectoryStoragePolicy,
    trajectory_storage_policy_from_cfg,
)
from vrl.utils.config import cfg_path, to_builtin_deep


@dataclass(frozen=True, slots=True)
class RolloutCollectorConfig:
    """Collector-local policy plus the fail-closed generation request projection."""

    request_sampling: dict[str, Any] = field(default_factory=dict)
    kl_reward_coef: float = 0.0
    trajectory_storage: TrajectoryStoragePolicy = field(
        default_factory=TrajectoryStoragePolicy,
    )

    def get(self, name: str, default: Any = None) -> Any:
        """Adapt typed local fields and request values to generic config readers."""

        if name in {"kl_reward_coef", "trajectory_storage"}:
            return getattr(self, name)
        return self.request_sampling.get(name, default)

    def generation_sampling(self) -> dict[str, Any]:
        """Build the request payload, deriving wire storage from the typed policy."""

        sampling = dict(self.request_sampling)
        default_storage = TrajectoryStoragePolicy()
        if self.trajectory_storage != default_storage:
            sampling["trajectory_storage"] = {
                "device": self.trajectory_storage.device,
                "dtype": self.trajectory_storage.dtype,
            }
        return sampling


def build_rollout_config_from_cfg(cfg: Any) -> RolloutCollectorConfig:
    """Project validated public fields into collector-local and request state."""

    request_sampling: dict[str, Any] = {}
    _merge_flat_section_values(
        request_sampling,
        cfg,
        "rollout",
        allowed=generation_request_rollout_fields(),
    )
    sde_values = _cfg_mapping(cfg, "rollout.sde")
    for source_key, target_key in {
        "type": "sde_type",
        "window_size": "sde_window_size",
        "window_range": "sde_window_range",
    }.items():
        if source_key in sde_values and sde_values[source_key] is not None:
            request_sampling[target_key] = to_builtin_deep(sde_values[source_key])
    _merge_flat_section_values(
        request_sampling,
        cfg,
        "sampling",
        allowed=frozenset(SamplingConfig.model_fields),
    )
    _copy_value(
        request_sampling,
        cfg,
        "train_segments",
        "algorithm.train_segments",
    )
    kl_reward_coef = resolve_kl_reward_coef(
        cfg_path(cfg, "algorithm.kl_reward_coef", None),
    )
    trajectory_storage = trajectory_storage_policy_from_cfg(
        cfg_path(cfg, "rollout.trajectory_storage", None),
    )
    has_sde_values = any(
        name in request_sampling for name in ("sde_type", "sde_window_size", "sde_window_range")
    )
    if has_sde_values:
        request_sampling["return_kl"] = kl_reward_coef > 0.0
    return RolloutCollectorConfig(
        request_sampling=request_sampling,
        kl_reward_coef=kl_reward_coef,
        trajectory_storage=trajectory_storage,
    )


def _merge_flat_section_values(
    values: dict[str, Any],
    cfg: Any,
    section: str,
    *,
    allowed: frozenset[str],
) -> None:
    section_values = _cfg_mapping(cfg, section)
    for key, value in section_values.items():
        name = str(key)
        if name not in allowed:
            continue
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
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        raw = model_dump(
            mode="python",
            exclude_none=True,
            exclude_unset=True,
        )
        if isinstance(raw, dict):
            return {str(key): inner for key, inner in raw.items()}
        raise ValueError(f"{path} config must be a mapping")
    value = to_builtin_deep(value)
    if isinstance(value, Mapping):
        return {str(key): inner for key, inner in value.items()}
    raise ValueError(f"{path} config must be a mapping")


_MISSING = object()


__all__ = [
    "RolloutCollectorConfig",
    "build_rollout_config_from_cfg",
]
