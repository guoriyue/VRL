"""Domain-neutral deployment configuration for reward inference runtimes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, Literal
from urllib.parse import urlparse

from vrl.utils.deadline import require_timeout


def require_http_origin(url: str, *, context: str) -> str:
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

    # ray (default): one driver-owned Ray actor per component, with device
    #          ownership and memory parking derived from the run topology.
    # http: an operator-owned service the driver connects to (endpoint +
    #       expected_model required).
    # in_process: the component builds and scores its model inside the driver.
    #          Not admitted for online training (a reward in the trainer
    #          process competes with kernel launch for the interpreter); kept
    #          for evaluation scripts and tests that construct rewards directly.
    kind: Literal["in_process", "http", "ray"] = "ray"
    endpoint: str = ""
    timeout_s: float = 1800.0
    expected_model: str = ""
    expected_model_version: str = ""

    def __post_init__(self) -> None:
        if self.kind == "service":
            raise ValueError(
                "reward inference.kind=service was replaced by kind=ray; remove "
                "the override for a run-owned actor, or use kind=http with an "
                "endpoint and expected_model for an operator-owned service",
            )
        if self.kind not in {"in_process", "http", "ray"}:
            raise ValueError(
                f"reward inference.kind must be 'in_process', 'http', or 'ray', got {self.kind!r}",
            )
        timeout_s = require_timeout(self.timeout_s, name="reward inference.timeout_s")
        endpoint = self.endpoint.strip()
        expected_model = self.expected_model.strip()
        expected_model_version = self.expected_model_version.strip()
        object.__setattr__(self, "timeout_s", timeout_s)
        object.__setattr__(self, "expected_model", expected_model)
        object.__setattr__(self, "expected_model_version", expected_model_version)
        if self.kind in {"in_process", "ray"}:
            if endpoint or expected_model or expected_model_version:
                raise ValueError(
                    f"reward inference.kind={self.kind} cannot set endpoint, expected_model, "
                    "or expected_model_version",
                )
            object.__setattr__(self, "endpoint", endpoint)
            return
        endpoint = require_http_origin(endpoint, context="reward inference.endpoint")
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
        """Build one component's inference config from its mapping, rejecting unknown keys."""

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
    "require_http_origin",
]
