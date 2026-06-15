"""Weight synchronisation between trainer and inference workers."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any

import torch

from vrl.trajectory.device import map_tensor_tree

TrainableStateGetter = Callable[[], dict[str, Any]]


class WeightSyncer(ABC):
    """Pushes updated weights from the trainer to inference workers."""

    @abstractmethod
    async def push(self, state_dict: dict[str, Any]) -> None:
        """Send updated weights to inference workers."""

    @property
    def current_policy_version(self) -> int | None:
        """Policy version of the last pushed weights (PolicyVersionProvider).

        ``None`` means this syncer does not track a version. Orchestration asks
        through this property instead of reaching into syncer internals.
        """
        return None


class RayRuntimeWeightSyncer(WeightSyncer):
    """Bridge ``OnlineTrainer`` weight pushes to a Ray rollout runtime."""

    def __init__(
        self,
        runtime: Any,
        *,
        initial_policy_version: int | None = None,
    ) -> None:
        update_weights = getattr(runtime, "update_weights", None)
        if not callable(update_weights):
            raise TypeError("runtime must expose async update_weights(state, version)")
        self.runtime = runtime
        self._next_policy_version = _resolve_next_policy_version(
            runtime,
            initial_policy_version,
        )
        self._push_lock = asyncio.Lock()

    async def push(self, state_dict: dict[str, Any]) -> None:
        state = _cpu_state_dict(state_dict)
        async with self._push_lock:
            policy_version = self._next_policy_version
            await self.runtime.update_weights(state, policy_version)
            self._next_policy_version = policy_version + 1

    @property
    def current_policy_version(self) -> int | None:
        # The syncer owns its runtime, so reading the runtime's version here is
        # a legal internal access (callers used to probe syncer.runtime from
        # outside).
        return self.runtime.current_policy_version


def build_runtime_weight_syncer(
    runtime: Any,
    *,
    initial_policy_version: int | None = None,
) -> WeightSyncer | None:
    """Return a syncer when a rollout runtime supports weight updates."""

    if not callable(getattr(runtime, "update_weights", None)):
        return None
    if getattr(runtime, "weight_sync", None) is None:
        return None
    return RayRuntimeWeightSyncer(
        runtime,
        initial_policy_version=initial_policy_version,
    )


def build_trainable_state_sync_getter(bundle: Any) -> TrainableStateGetter:
    """Build the flattened trainable-state getter used by rollout sync.

    Checkpoints store trainable modules as a nested mapping keyed by module
    name. Rollout policies consume a flat policy-state payload so a worker can
    call ``policy.load_trainable_state(payload)`` without knowing bundle shape.
    """

    modules = require_trainable_modules(bundle)

    def _getter() -> dict[str, Any]:
        return flatten_trainable_module_state(modules)

    return _getter


def flatten_trainable_module_state(modules: Mapping[str, Any]) -> dict[str, Any]:
    """Return trainable ``module_name.parameter_name`` keys for rollout sync."""

    if not isinstance(modules, Mapping) or not modules:
        raise ValueError("trainable modules must be a non-empty mapping")
    state: dict[str, Any] = {}
    for module_name, module in modules.items():
        name = str(module_name)
        if not name:
            raise ValueError("trainable module names must be non-empty")
        # Sync payload keys live in the policy's uncompiled, unwrapped namespace:
        # torch.compile exposes the inner module as ``_orig_mod``, DDP / FSDP1 as
        # ``.module``. The receiver unwraps its own compile wrapper the same way
        # (models/utils.py load_weights_into), so neither wrapper prefix may leak
        # into the rollout payload. Loop because wrapper nesting/order varies
        # (e.g. compile(DDP(m)) vs DDP(compile(m))).
        while True:
            unwrapped = getattr(module, "_orig_mod", module)
            unwrapped = getattr(unwrapped, "module", unwrapped)
            if unwrapped is module:
                break
            module = unwrapped
        state_dict = getattr(module, "state_dict", None)
        if not callable(state_dict):
            raise TypeError(f"trainable module {name!r} does not expose state_dict()")
        module_state = state_dict()
        if not isinstance(module_state, Mapping):
            raise TypeError(f"trainable module {name!r} state_dict() must return a mapping")
        trainable_names = _trainable_parameter_names(module, name)
        missing = sorted(trainable_names - set(module_state))
        if missing:
            preview = ", ".join(missing[:5])
            suffix = " ..." if len(missing) > 5 else ""
            raise ValueError(
                f"trainable module {name!r} state_dict() is missing trainable "
                f"parameters: {preview}{suffix}",
            )
        for key, value in module_state.items():
            if str(key) in trainable_names:
                state[f"{name}.{key}"] = value
    if not state:
        raise ValueError("trainable module state is empty")
    return state


def _resolve_next_policy_version(
    runtime: Any,
    initial_policy_version: int | None,
) -> int:
    if initial_policy_version is not None:
        return int(initial_policy_version) + 1
    current = getattr(runtime, "current_policy_version", None)
    if current is None:
        return 1
    return int(current) + 1


def require_trainable_modules(bundle: Any) -> Mapping[str, Any]:
    modules = getattr(bundle, "trainable_modules", None)
    if not isinstance(modules, Mapping) or not modules:
        raise ValueError("RuntimeBundle.trainable_modules must be a non-empty mapping")
    return modules


def _trainable_parameter_names(module: Any, module_name: str) -> set[str]:
    named_parameters = getattr(module, "named_parameters", None)
    if not callable(named_parameters):
        raise TypeError(
            f"trainable module {module_name!r} must expose named_parameters() "
            "for trainable-only rollout sync",
        )
    names = {
        str(name)
        for name, parameter in named_parameters()
        if bool(getattr(parameter, "requires_grad", False))
    }
    if not names:
        raise ValueError(f"trainable module {module_name!r} has no trainable parameters")
    return names


def _cpu_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {key: to_cpu(value) for key, value in state_dict.items()}


def to_cpu(value: Any) -> Any:
    return map_tensor_tree(
        value,
        lambda leaf: leaf.detach().cpu(),
        is_leaf=lambda v: isinstance(v, torch.Tensor),
    )
