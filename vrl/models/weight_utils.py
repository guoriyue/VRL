"""Model-side receiver for the flat trainable-weight sync payload.

The inverse of ``trainers.weight_sync.flatten_trainable_module_state``: that
flattens a module's trainable params to ``{name}.{param}`` keys for the sync
payload; this validates such a payload and loads it back into the module, and
retains a few versions of it (:class:`TrainableStateSlots`).

``unwrap_compile_and_ddp`` lives here because that key namespace is what it
exists to reach: every producer and consumer of those keys — sync sender,
receiver, checkpoint export, FSDP2 gather — peels through it so no wrapper
prefix leaks into the payload. The PEFT adapter helpers in
``vrl.models.peft_adapter`` reuse it for a different reason (reaching the module
that owns the adapter surface).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def unwrap_compile_and_ddp(module: Any) -> Any:
    """Peel torch.compile (``_orig_mod``) and DDP / FSDP1 (``.module``) wrappers.

    Sync payload keys live in the policy's uncompiled, unwrapped namespace. Both
    ends peel through here: the sync sender
    (``trainers.weight_sync.flatten_trainable_module_state``) and the receiver
    (:func:`load_weights_into` / :func:`validate_weights_for` below), so neither
    wrapper prefix may leak into the rollout payload. PEFT is deliberately NOT
    peeled — LoRA keys (``base_model.model.*``) are part of the policy-facing
    namespace. Loop because wrapper nesting/order varies (e.g. compile(DDP(m)) vs
    DDP(compile(m))).

    FSDP2 export reuses this (vrl/trainers/strategy.py) so a sharded gather lands
    in the same namespace as single-process sync: ``get_model_state_dict`` strips
    ``_orig_mod.`` while ``named_parameters()`` keeps it, so selecting trainable
    keys on a still-compiled module would mismatch.
    """

    while True:
        unwrapped = getattr(module, "_orig_mod", module)
        unwrapped = getattr(unwrapped, "module", unwrapped)
        if unwrapped is module:
            return module
        module = unwrapped


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

    stripped = validate_weights_for(
        module,
        state_dict,
        prefix=prefix,
        label=label,
    )
    module = unwrap_compile_and_ddp(module)
    return module.load_state_dict(stripped, strict=False)


def validate_weights_for(
    module: Any,
    state_dict: Mapping[str, Any],
    *,
    prefix: str,
    label: str,
) -> dict[str, Any]:
    """Validate one flattened sync payload without changing live weights."""

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
    stripped = {key[len(dot) :]: value for key, value in state.items()}

    module = unwrap_compile_and_ddp(module)

    named_parameters = getattr(module, "named_parameters", None)
    if not callable(named_parameters):
        raise TypeError(f"{label} must expose named_parameters()")
    trainable = {
        name: parameter
        for name, parameter in named_parameters()
        if bool(getattr(parameter, "requires_grad", False))
    }
    trainable_keys = set(trainable)
    if not trainable_keys:
        raise ValueError(f"{label} has no trainable parameters to sync")

    extra = sorted(set(stripped) - trainable_keys)
    missing = sorted(trainable_keys - set(stripped))
    if extra or missing:
        raise ValueError(
            f"{label}: load_trainable_state must receive exactly trainable keys; "
            f"missing={missing[:5]}, extra={extra[:5]}",
        )
    for name, value in stripped.items():
        parameter = trainable[name]
        if not hasattr(value, "shape") or not hasattr(value, "dtype"):
            raise TypeError(f"{label}: trainable state {name!r} must be a tensor")
        if tuple(value.shape) != tuple(parameter.shape):
            raise ValueError(
                f"{label}: trainable state {name!r} shape mismatch: "
                f"payload={tuple(value.shape)}, runtime={tuple(parameter.shape)}",
            )
        if value.dtype != parameter.dtype:
            raise ValueError(
                f"{label}: trainable state {name!r} dtype mismatch: "
                f"payload={value.dtype}, runtime={parameter.dtype}",
            )
    return stripped


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

    def _evict(self) -> None:
        # Keep the most recent ``max_retained`` versions. A request older than the
        # window loses its slot and is reported as a stale-slot result rather than
        # silently mixing weights.
        while len(self._slots) > self.max_retained:
            del self._slots[min(self._slots)]
