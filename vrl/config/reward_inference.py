"""Domain-neutral deployment configuration for reward inference runtimes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, Literal
from urllib.parse import urlparse


def validate_http_origin(url: str, *, context: str) -> str:
    """Validate one operator-service origin URL and return it normalized.

    The trainer-side client and this config are the two producers of service
    base URLs; both must enforce the same shape (absolute http(s) origin, no
    credentials/query/fragment/path) or their accepted inputs drift apart.
    """

    origin = str(url).strip()
    parsed = urlparse(origin)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            f"{context} must be an absolute http(s) origin URL, got {url!r}",
        )
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            f"{context} cannot contain credentials, query, or fragment",
        )
    if parsed.path not in {"", "/"}:
        raise ValueError(f"{context} must be an origin URL without a path")
    return origin.rstrip("/")


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
    expected_model_version: str = ""

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
        expected_model_version = self.expected_model_version.strip()
        object.__setattr__(self, "timeout_s", timeout_s)
        object.__setattr__(self, "expected_model", expected_model)
        object.__setattr__(self, "expected_model_version", expected_model_version)
        if self.kind == "in_process":
            if endpoint or expected_model or expected_model_version:
                raise ValueError(
                    "reward inference.kind=in_process cannot set endpoint, expected_model, "
                    "or expected_model_version",
                )
            object.__setattr__(self, "endpoint", endpoint)
            return
        endpoint = validate_http_origin(endpoint, context="reward inference.endpoint")
        object.__setattr__(self, "endpoint", endpoint)
        if not expected_model:
            raise ValueError(
                "reward inference.kind=http requires expected_model for startup identity validation",
            )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | RewardInferenceConfig | None,
        *,
        context: str,
    ) -> RewardInferenceConfig:
        """Parse one component's config without duplicating its typed field list."""

        if value is None:
            return cls()
        if isinstance(value, RewardInferenceConfig):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(f"{context} must be a mapping, got {type(value).__name__}")
        payload = dict(value)
        unknown = sorted(set(payload) - _INFERENCE_FIELDS)
        if unknown:
            raise ValueError(f"unsupported {context} keys: {unknown}")
        return cls(**payload)


_INFERENCE_FIELDS = frozenset(field.name for field in fields(RewardInferenceConfig))


__all__ = [
    "RewardInferenceConfig",
    "RewardInferenceConfig",
    "validate_http_origin",
]
