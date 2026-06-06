from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["load_trainable_state_into"]


def load_trainable_state_into(
    module: Any,
    state_dict: Mapping[str, Any],
    *,
    model_label: str,
) -> Any:
    """Load only the trainable parameters of ``module`` from a rollout sync state."""

    state = dict(state_dict)
    if not state:
        raise ValueError("load_trainable_state received an empty state dict")

    trainable_keys = {
        name
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }
    if not trainable_keys:
        raise ValueError(f"{model_label} has no trainable parameters to sync")

    filtered: dict[str, Any] = {}
    for key, value in state.items():
        normalized_key = key.removeprefix("model.").removeprefix("policy.")
        if normalized_key in trainable_keys:
            filtered[normalized_key] = value

    missing = sorted(trainable_keys - set(filtered))
    if missing:
        preview = ", ".join(missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise ValueError(
            f"load_trainable_state missing trainable {model_label} keys: "
            f"{preview}{suffix}",
        )

    return module.load_state_dict(filtered, strict=False)
