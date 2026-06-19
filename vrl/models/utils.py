"""Shared model-side helpers (weight loading, ...).

Currently holds the rollout weight-sync receiver. It is the inverse of
``trainers.weight_sync.flatten_trainable_module_state``: that flattens a module's
trainable params to ``{name}.{param}`` keys for the sync payload; this loads such
a payload back into the module.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from typing import Any


def load_weights_into(
    module: Any,
    state_dict: Mapping[str, Any],
    *,
    prefix: str,
    label: str,
) -> Any:
    """Load a flattened trainable-weight sync payload into ``module``.

    The payload carries ``{prefix}.{param}`` keys for ``module``'s ``requires_grad``
    parameters only (as produced by the sync sender). Strict by design: rejects any
    key outside ``{prefix}.`` and any mismatch (extra/missing) against the module's
    trainable parameter set, so a malformed sync payload fails loudly instead of
    silently leaving weights stale.
    """

    state = dict(state_dict)
    if not state:
        raise ValueError(f"{label}: load_trainable_state received an empty state dict")

    dot = f"{prefix}."
    bad = sorted(key for key in state if not key.startswith(dot))
    if bad:
        raise ValueError(
            f"{label}: load_trainable_state only accepts trainable keys prefixed "
            f"with {dot!r}; got {bad[:5]}",
        )
    stripped = {key[len(dot):]: value for key, value in state.items()}

    module = getattr(module, "_orig_mod", module)

    named_parameters = getattr(module, "named_parameters", None)
    if not callable(named_parameters):
        raise TypeError(f"{label} must expose named_parameters()")
    trainable_keys = {
        name for name, parameter in named_parameters() if bool(getattr(parameter, "requires_grad", False))
    }
    if not trainable_keys:
        raise ValueError(f"{label} has no trainable parameters to sync")

    extra = sorted(set(stripped) - trainable_keys)
    missing = sorted(trainable_keys - set(stripped))
    if extra or missing:
        raise ValueError(
            f"{label}: load_trainable_state must receive exactly trainable keys; "
            f"missing={missing[:5]}, extra={extra[:5]}",
        )

    return module.load_state_dict(stripped, strict=False)


class TrainableStateSlots:
    """Retain a few policy versions of a model's flat trainable-state payload.

    The continuous rollout worker installs one slot per weight sync, keyed by
    policy version, so a generation request that was stamped under an older
    version can still find its weights after the trainer has advanced — the
    prerequisite for a *non-draining* weight sync (no drain bubble; see
    ``SPRINT_shadow_model_weight_sync.md``).

    It holds ONLY the trainable-state payload dicts (whatever the sync sender
    selected — LoRA/adapter params for the common case, or full-param), never the
    frozen base model. So the extra footprint is ``retained * trainable_bytes``,
    not a model copy. The payloads are the host-side dicts handed to the worker
    (already CPU tensors), so retained slots cost host RAM, not VRAM; only the
    active slot is copied onto the live model by ``activate``.
    """

    def __init__(self, *, max_retained: int = 8) -> None:
        if int(max_retained) < 1:
            raise ValueError("max_retained must be >= 1")
        self.max_retained = int(max_retained)
        self._slots: dict[int, Mapping[str, Any]] = {}

    def install(self, version: int, state: Mapping[str, Any] | None) -> None:
        """Retain ``state`` under ``version``; ``None`` aliases the newest slot.

        A ``None`` payload is a version-only bump (no new weights): the new
        version should resolve to the same live state as the previous version, so
        we alias the most recent slot rather than create an empty one.
        """

        version = int(version)
        if state is None:
            if not self._slots:
                return
            state = self._slots[max(self._slots)]
        self._slots[version] = state
        self._evict()

    def has(self, version: int) -> bool:
        return int(version) in self._slots

    def get(self, version: int) -> Mapping[str, Any]:
        return self._slots[int(version)]

    def versions(self) -> list[int]:
        return sorted(self._slots)

    def _evict(self) -> None:
        # Keep the most recent ``max_retained`` versions. A request older than the
        # window loses its slot and is reported as a stale-slot result rather than
        # silently mixing weights.
        while len(self._slots) > self.max_retained:
            del self._slots[min(self._slots)]


def count_trainable_params(module: Any) -> int:
    """Total number of trainable (``requires_grad``) parameters in ``module``."""

    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def disable_adapter_on(module: Any) -> contextlib.AbstractContextManager[None]:
    """Context manager disabling ``module``'s LoRA/PEFT adapter, or a no-op when absent.

    Used for the reference (adapter-off) forward pass; a module with no attachable
    adapter still satisfies the contract by returning a null context.
    """

    disable = getattr(module, "disable_adapter", None)
    if not callable(disable):
        return contextlib.nullcontext()
    return disable()
