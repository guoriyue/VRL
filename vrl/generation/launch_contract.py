"""Serializable generation runtime launch contract."""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any, Literal, get_args

GenerationKind = Literal["diffusion", "ar"]


@dataclass(frozen=True, slots=True)
class GenerationRuntimeLaunchContract:
    """Worker-side executor construction contract for generation runtimes.

    The contract intentionally carries import paths and serializable config only.
    Distributed runtimes own model/executor construction; callers must not pass
    live executor, policy, or pipeline objects through this boundary.
    """

    family: str
    task: str
    generation_kind: GenerationKind
    model_config: dict[str, Any] = field(default_factory=dict)
    executor_kwargs: dict[str, Any] = field(default_factory=dict)
    policy_version: int | None = None
    runtime_builder: str | None = None
    executor_cls: str | None = None
    model_build: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.family:
            raise ValueError("GenerationRuntimeLaunchContract.family must be non-empty")
        if not self.task:
            raise ValueError("GenerationRuntimeLaunchContract.task must be non-empty")
        if self.generation_kind not in get_args(GenerationKind):
            raise ValueError(f"unsupported generation_kind: {self.generation_kind!r}")
        object.__setattr__(
            self,
            "model_config",
            self._normalize_config_mapping(self.model_config, "model_config"),
        )
        object.__setattr__(
            self,
            "executor_kwargs",
            self._normalize_config_mapping(self.executor_kwargs, "executor_kwargs"),
        )
        if self.model_build is not None:
            object.__setattr__(
                self,
                "model_build",
                self._normalize_config_mapping(self.model_build, "model_build"),
            )
        object.__setattr__(
            self,
            "extra",
            self._normalize_config_mapping(self.extra, "extra"),
        )
        if self.policy_version is not None:
            object.__setattr__(self, "policy_version", int(self.policy_version))

        self._validate_builder_import_paths()
        self._assert_pickle_serializable()

    @classmethod
    def from_value(
        cls,
        value: GenerationRuntimeLaunchContract | Mapping[str, Any],
    ) -> GenerationRuntimeLaunchContract:
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls.from_dict(value)
        raise TypeError(
            "launch_contract must be a GenerationRuntimeLaunchContract or dict, "
            f"got {type(value).__name__}",
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GenerationRuntimeLaunchContract:
        field_names = {field.name for field in fields(cls)}
        unknown = sorted(set(value).difference(field_names))
        if unknown:
            keys = ", ".join(repr(key) for key in unknown)
            raise ValueError(
                "unsupported GenerationRuntimeLaunchContract keys: "
                f"{keys}. Put model payload under model_config or executor_kwargs.",
            )

        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "task": self.task,
            "generation_kind": self.generation_kind,
            "model_config": dict(self.model_config),
            "executor_kwargs": dict(self.executor_kwargs),
            "policy_version": self.policy_version,
            "runtime_builder": self.runtime_builder,
            "executor_cls": self.executor_cls,
            "model_build": None if self.model_build is None else dict(self.model_build),
            "extra": dict(self.extra),
        }

    def model_build_payload(self) -> dict[str, Any]:
        if self.model_build is not None:
            return dict(self.model_build)
        return dict(self.model_config)

    def _validate_builder_import_paths(self) -> None:
        for attr in ("runtime_builder", "executor_cls"):
            value = getattr(self, attr)
            if value is not None and not isinstance(value, str):
                raise TypeError(
                    f"{attr} must be an import path string, got {type(value).__name__}"
                )
        if self.runtime_builder is None or self.executor_cls is None:
            raise ValueError(
                "GenerationRuntimeLaunchContract requires runtime_builder and "
                "executor_cls import paths",
            )

    def _assert_pickle_serializable(self) -> None:
        try:
            pickle.dumps(self)
        except Exception as exc:
            raise TypeError(
                "GenerationRuntimeLaunchContract must be pickle-serializable",
            ) from exc

    @classmethod
    def _normalize_config_mapping(cls, value: Any, path: str) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError(f"{path} must be a dict, got {type(value).__name__}")
        normalized = dict(value)
        cls._validate_serializable_config(normalized, path)
        return normalized

    @classmethod
    def _validate_serializable_config(cls, value: Any, path: str) -> None:
        if value is None or isinstance(value, (str, int, float)):
            return

        if cls._is_torch_tensor(value):
            raise TypeError(f"{path} must not contain live torch.Tensor objects")
        if cls._is_torch_module(value):
            raise TypeError(f"{path} must not contain live torch.nn.Module objects")
        if cls._is_diffusers_pipeline(value):
            raise TypeError(f"{path} must not contain live diffusers pipeline objects")
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

    @staticmethod
    def _is_torch_tensor(value: Any) -> bool:
        return (
            type(value).__module__.split(".", 1)[0] == "torch"
            and value.__class__.__name__ == "Tensor"
        )

    @staticmethod
    def _is_torch_module(value: Any) -> bool:
        module_cls = getattr(getattr(value, "__class__", None), "__mro__", ())
        return any(
            cls.__module__ == "torch.nn.modules.module" and cls.__name__ == "Module"
            for cls in module_cls
        )

    @staticmethod
    def _is_diffusers_pipeline(value: Any) -> bool:
        value_type = type(value)
        return value_type.__module__.startswith("diffusers.") and "Pipeline" in value_type.__name__


__all__ = ["GenerationKind", "GenerationRuntimeLaunchContract"]
