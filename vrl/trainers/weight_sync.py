"""Weight synchronisation between trainer and inference workers."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from typing import Any

import torch

from vrl.models.interfaces import RuntimeBundle
from vrl.models.weight_utils import unwrap_compile_and_ddp
from vrl.trajectory.device import map_tensor_tree

TrainableStateGetter = Callable[[], dict[str, Any]]


class RayRuntimeWeightSyncer:
    """Bridge ``OnlineTrainer`` weight pushes to a ``GenerationRuntime``.

    Every runtime installs weights through ``update_weights``. The runtime is
    the one owner of the policy version: each push carries the version after
    the one the runtime currently publishes, so no second counter can drift
    from it.
    """

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self._push_lock = asyncio.Lock()

    async def push(self, state_dict: dict[str, Any]) -> None:
        """Send updated weights to the runtime under the next policy version."""

        # Split the two costs: the device->host copy is trainer-side GPU work,
        # the update_weights await is transport plus worker-side load. They
        # overlap differently once rollout and training stop sharing one GPU, so
        # they must be attributable separately rather than as one "sync" blob.
        from vrl.utils.profiling import profile_range

        with profile_range("weight_sync.state_to_cpu"):
            # to_cpu_snapshot: push() is exactly the "another thread may read
            # later" case its docstring describes, so the copy is required.
            state = to_cpu_snapshot(state_dict)
        async with self._push_lock:
            with profile_range("weight_sync.push"):
                await self.runtime.update_weights(
                    state,
                    self.runtime.current_policy_version + 1,
                )


def require_trainable_modules(bundle: RuntimeBundle) -> Mapping[str, Any]:
    """Return the checkpoint-root mapping of a bundle, refusing an empty one.

    Deliberately a free function, not ``RuntimeBundle.require_trainable_modules``:
    the invariant belongs to the trainer layer (checkpointing / strategy / weight
    sync all read roots through here), and every caller passes the bundle
    structurally — the annotation documents the intended type without demanding
    a fully constructed one.
    """

    modules = bundle.trainable_modules
    if not modules:
        raise ValueError("RuntimeBundle.trainable_modules must be a non-empty mapping")
    return modules


def flatten_trainable_module_state(modules: Mapping[str, Any]) -> dict[str, Any]:
    """Return trainable ``module_name.parameter_name`` keys for rollout sync.

    Checkpoints store trainable modules as a nested mapping keyed by module
    name; rollout policies consume this flat payload, so a worker can call
    ``policy.load_trainable_state(payload)`` without knowing bundle shape. Each
    module's ``state_dict()`` is read here, at call time, so the result tracks
    the live weights rather than any earlier snapshot.
    """

    state: dict[str, Any] = {}
    for module_name, module in modules.items():
        name = str(module_name)
        module = unwrap_compile_and_ddp(module)
        state.update(select_trainable_state(module, name))
    if not state:
        raise ValueError("trainable module state is empty")
    return state


def select_trainable_state(module: Any, name: str) -> dict[str, Any]:
    """Pick a module's trainable-parameter entries from its state dict, prefixed ``name.``.

    ``module`` must already be unwrapped enough that its ``named_parameters()``
    names match its ``state_dict()`` keys.
    """

    if not name:
        raise ValueError("trainable module names must be non-empty")
    module_state = module.state_dict()
    named_parameters = getattr(module, "named_parameters", None)
    if not callable(named_parameters):
        raise TypeError(
            f"trainable module {name!r} must expose named_parameters() "
            "for trainable-only rollout sync",
        )
    trainable_names = {
        str(parameter_name)
        for parameter_name, parameter in named_parameters()
        if bool(getattr(parameter, "requires_grad", False))
    }
    if not trainable_names:
        raise ValueError(f"trainable module {name!r} has no trainable parameters")
    missing = sorted(trainable_names - set(module_state))
    if missing:
        preview = ", ".join(missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise ValueError(
            f"trainable module {name!r} state_dict() is missing trainable "
            f"parameters: {preview}{suffix}",
        )
    return {
        f"{name}.{key}": value
        for key, value in module_state.items()
        if str(key) in trainable_names
    }


def to_cpu_snapshot(value: Any) -> Any:
    """Return detached CPU tensors whose storage cannot alias the live model.

    A plain ``detach().cpu()`` aliases storage when the trainer already lives on
    CPU.  Continuous weight owners may push later from another thread/process, so
    the prepared payload must remain stable if training mutates the live weights.
    """

    return map_tensor_tree(
        value,
        lambda leaf: leaf.detach().to(device="cpu", copy=True),
        is_leaf=lambda v: isinstance(v, torch.Tensor),
    )
