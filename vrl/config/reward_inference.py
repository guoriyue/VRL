"""Domain-neutral deployment configuration for reward inference runtimes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, Literal
from urllib.parse import urlparse

from vrl.utils.config import cfg_get


@dataclass(frozen=True, slots=True)
class RewardInferenceConfig:
    """Where one reward component executes.

    This structure owns only the transport/deployment decision so resource
    planning can exclude operator-owned HTTP services before assigning local
    GPUs. In-process model construction remains component-owned; HTTP model and
    device configuration belongs to the standalone service.
    """

    kind: Literal["in_process", "http"] = "in_process"
    endpoint: str = ""
    timeout_s: float = 1800.0
    expected_model: str = ""

    def __post_init__(self) -> None:
        if self.kind not in {"in_process", "http"}:
            raise ValueError(
                f"reward inference.kind must be 'in_process' or 'http', got {self.kind!r}",
            )
        timeout_s = float(self.timeout_s)
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("reward inference.timeout_s must be a finite number > 0")
        endpoint = self.endpoint.strip()
        expected_model = self.expected_model.strip()
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "timeout_s", timeout_s)
        object.__setattr__(self, "expected_model", expected_model)
        if self.kind == "in_process":
            if endpoint or expected_model:
                raise ValueError(
                    "reward inference.kind=in_process cannot set endpoint or expected_model",
                )
            return
        parsed = urlparse(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(
                "reward inference.kind=http requires an absolute http(s) endpoint",
            )
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError(
                "reward inference.endpoint cannot contain credentials, query, or fragment",
            )
        if parsed.path not in {"", "/"}:
            raise ValueError(
                "reward inference.endpoint must be an origin URL without a path",
            )
        if not expected_model:
            raise ValueError(
                "reward inference.kind=http requires expected_model for startup identity validation",
            )


_INFERENCE_FIELDS = frozenset(field.name for field in fields(RewardInferenceConfig))


def parse_reward_inference_config(
    value: Mapping[str, Any] | RewardInferenceConfig | None,
    *,
    context: str,
) -> RewardInferenceConfig:
    """Parse one component's config without duplicating its typed field list."""

    if value is None:
        return RewardInferenceConfig()
    if isinstance(value, RewardInferenceConfig):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping, got {type(value).__name__}")
    payload = dict(value)
    unknown = sorted(set(payload) - _INFERENCE_FIELDS)
    if unknown:
        raise ValueError(f"unsupported {context} keys: {unknown}")
    return RewardInferenceConfig(**payload)


def reward_inference_configs_from_cfg(cfg: Any) -> dict[str, RewardInferenceConfig]:
    """Resolve every configured component's deployment before GPU placement."""

    reward = cfg_get(cfg, "reward", None)
    components = cfg_get(reward, "components", {}) if reward is not None else {}
    kwargs = cfg_get(reward, "kwargs", {}) if reward is not None else {}
    if not isinstance(components, Mapping):
        return {}
    kwargs = kwargs if isinstance(kwargs, Mapping) else {}
    resolved: dict[str, RewardInferenceConfig] = {}
    for raw_name in components:
        name = str(raw_name)
        component_kwargs = cfg_get(kwargs, name, {}) or {}
        inference = cfg_get(component_kwargs, "inference", None)
        resolved[name] = parse_reward_inference_config(
            inference,
            context=f"reward.kwargs.{name}.inference",
        )
    return resolved


__all__ = [
    "RewardInferenceConfig",
    "parse_reward_inference_config",
    "reward_inference_configs_from_cfg",
]
