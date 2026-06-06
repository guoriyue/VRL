"""Shared model-side helpers (weight loading, ...).

Currently holds the rollout weight-sync receiver. It is the inverse of
``trainers.weight_sync.flatten_trainable_module_state``: that flattens a module's
trainable params to ``{name}.{param}`` keys for the sync payload; this loads such
a payload back into the module.
"""

from __future__ import annotations

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
