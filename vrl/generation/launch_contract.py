"""Serializable generation runtime launch contract.

The one object the driver (vrl/run.py) hands a Ray generation actor at
construction: everything a worker process needs to rebuild the family
executor on its own GPU. It lives apart from both runtimes because the two
sides of that process boundary share no other runtime code — only this
contract. Construction validates pickle-serializability and primitives-only
content so a live driver object (callable, model, tensor) fails fast on the
driver instead of inside actor deserialization.

The reward-side twin is ``RewardWorkerLaunchContract``
(vrl/rewards/launch_contract.py).
"""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class GenerationRuntimeLaunchContract:
    """Worker-side executor construction contract for generation runtimes.

    The canonical family identifies worker wiring. Only per-run primitive model,
    expected model identity, executor, profiler, and lifecycle values cross the
    Ray boundary.
    """

    family: str
    model_build: dict[str, Any]
    expected_model_identity: dict[str, Any]
    executor_kwargs: dict[str, Any] = field(default_factory=dict)
    policy_version: int | None = None
    torch_profiler: dict[str, Any] = field(default_factory=dict)
    sleep_offload: bool = False
    # Continuous rollout may sync while older requests remain in flight, so its
    # workers retain versioned CPU policy payloads. Strict on-policy drains every
    # request before syncing and must overwrite in place; retaining full-parameter
    # snapshots there wastes one model-sized host allocation per slot.
    versioned_weight_sync: bool = False

    def __post_init__(self) -> None:
        if not self.family:
            raise ValueError("GenerationRuntimeLaunchContract.family must be non-empty")
        object.__setattr__(
            self,
            "model_build",
            self._normalize_config_mapping(self.model_build, "model_build"),
        )
        expected_model_identity = self._normalize_config_mapping(
            self.expected_model_identity,
            "expected_model_identity",
        )
        if not expected_model_identity:
            raise ValueError(
                "GenerationRuntimeLaunchContract.expected_model_identity must be non-empty",
            )
        object.__setattr__(
            self,
            "expected_model_identity",
            expected_model_identity,
        )
        object.__setattr__(
            self,
            "executor_kwargs",
            self._normalize_config_mapping(self.executor_kwargs, "executor_kwargs"),
        )
        object.__setattr__(
            self,
            "torch_profiler",
            self._normalize_config_mapping(self.torch_profiler, "torch_profiler"),
        )
        if not isinstance(self.sleep_offload, bool):
            raise TypeError(
                f"sleep_offload must be a bool, got {type(self.sleep_offload).__name__}",
            )
        if self.policy_version is not None:
            object.__setattr__(self, "policy_version", int(self.policy_version))
        if not isinstance(self.versioned_weight_sync, bool):
            raise TypeError("versioned_weight_sync must be a bool")

        try:
            pickle.dumps(self)
        except Exception as exc:
            raise TypeError(
                "GenerationRuntimeLaunchContract must be pickle-serializable",
            ) from exc

    @classmethod
    def _normalize_config_mapping(cls, value: Any, path: str) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise TypeError(f"{path} must be a dict, got {type(value).__name__}")
        normalized = dict(value)
        cls._validate_serializable_config(normalized, path)
        return normalized

    @classmethod
    def _validate_serializable_config(cls, value: Any, path: str) -> None:
        if value is None or isinstance(value, (str, int, float)):
            return

        if callable(value):
            raise TypeError(f"{path} must not contain live callable objects")

        if isinstance(value, Mapping):
            for key, inner in value.items():
                if not isinstance(key, str):
                    raise TypeError(f"{path} keys must be strings, got {type(key).__name__}")
                cls._validate_serializable_config(inner, f"{path}.{key}")
            return

        if isinstance(value, list):
            for index, inner in enumerate(value):
                cls._validate_serializable_config(inner, f"{path}[{index}]")
            return

        if isinstance(value, tuple):
            for index, inner in enumerate(value):
                cls._validate_serializable_config(inner, f"{path}[{index}]")
            return

        raise TypeError(
            f"{path} must contain only primitive config values, lists, tuples, and dicts; "
            f"got {type(value).__name__}",
        )


__all__ = ["GenerationRuntimeLaunchContract"]
