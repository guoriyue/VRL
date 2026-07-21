"""Shared model-side helpers (weight loading, ...).

Currently holds the rollout weight-sync receiver. It is the inverse of
``trainers.weight_sync.flatten_trainable_module_state``: that flattens a module's
trainable params to ``{name}.{param}`` keys for the sync payload; this loads such
a payload back into the module.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping
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

    stripped = validate_weights_for(
        module,
        state_dict,
        prefix=prefix,
        label=label,
    )
    module = getattr(module, "_orig_mod", module)
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

    module = getattr(module, "_orig_mod", module)

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

    Used for the reference (adapter-off) forward pass. Switches on the module
    behind any DDP/``torch.compile`` wrapper (same reason as
    :func:`activate_adapter_on`) and covers both adapter-disable surfaces:

    - PEFT ``PeftModel.disable_adapter()`` — already a context manager;
    - diffusers ``PeftAdapterMixin`` ``disable_adapters()`` / ``enable_adapters()``.

    The plural surface matters: ``WanTransformer3DModel`` / cosmos-predict2 carry
    LoRA via ``PeftAdapterMixin.add_adapter`` and expose ONLY the plural pair, so
    checking the singular method alone silently failed to disable the adapter —
    the reference forward (e.g. the diffusion GRPO KL term in
    ``evaluators/diffusion/sde_logprob.py``) then ran with the policy adapter
    still on. A module exposing neither surface is genuinely adapter-less and
    returns a null context.
    """

    from vrl.trainers.weight_sync import unwrap_compile_and_ddp

    host = unwrap_compile_and_ddp(module)

    disable = getattr(host, "disable_adapter", None)
    if callable(disable):
        return disable()

    disable_adapters = getattr(host, "disable_adapters", None)
    enable_adapters = getattr(host, "enable_adapters", None)
    if callable(disable_adapters) and callable(enable_adapters):
        if not _has_plural_adapter_loaded(host):
            return contextlib.nullcontext()

        @contextlib.contextmanager
        def _disabled() -> Iterator[None]:
            disabled_before = _plural_adapter_disable_flags(host)
            restore_enabled = (
                any(not disabled for disabled in disabled_before) if disabled_before else True
            )
            disable_adapters()
            try:
                yield
            finally:
                if restore_enabled:
                    enable_adapters()

        return _disabled()

    return contextlib.nullcontext()


def _has_plural_adapter_loaded(module: Any) -> bool:
    """Whether a diffusers-style plural adapter surface has real adapters loaded."""

    loaded = getattr(module, "_hf_peft_config_loaded", None)
    if loaded is False:
        return False
    peft_config = getattr(module, "peft_config", None)
    return not (isinstance(peft_config, Mapping) and not peft_config)


def _plural_adapter_disable_flags(module: Any) -> tuple[bool, ...]:
    """Return per-layer disable flags for PEFT tuner layers when exposed."""

    named_modules = getattr(module, "named_modules", None)
    if not callable(named_modules):
        return ()
    flags: list[bool] = []
    for _name, child in named_modules():
        disabled = getattr(child, "disable_adapters", None)
        if isinstance(disabled, bool):
            flags.append(disabled)
            continue
        disabled = getattr(child, "_disable_adapters", None)
        if isinstance(disabled, bool):
            flags.append(disabled)
    return tuple(flags)


def activate_adapter_on(module: Any, name: str) -> contextlib.AbstractContextManager[None]:
    """Context manager activating PEFT adapter ``name``, restoring ``"default"`` on exit.

    The named-adapter sibling of :func:`disable_adapter_on`, used for the
    previous-policy forward pass. The adapter is switched on the module *behind*
    any DDP / ``torch.compile`` wrapper: those wrappers do not proxy PEFT's
    ``set_adapter``, but they wrap the same underlying module, so the switch is
    visible to the wrapped grad forward too. ``set_adapter`` exists on both the
    PEFT ``PeftModel`` and the diffusers ``PeftAdapterMixin`` surfaces. Unlike
    *disabling* — a sensible no-op on an adapter-less module — activating a
    *named* adapter requires one, so a module without ``set_adapter`` raises
    rather than silently forwarding the wrong weights.
    """

    from vrl.trainers.weight_sync import unwrap_compile_and_ddp

    host = unwrap_compile_and_ddp(module)
    set_adapter = getattr(host, "set_adapter", None)
    if not callable(set_adapter):
        raise RuntimeError(
            f"cannot activate adapter {name!r}: {type(host).__name__} has no "
            "set_adapter(); attach a PEFT adapter before requesting it",
        )

    @contextlib.contextmanager
    def _activated() -> Iterator[None]:
        set_adapter(name)
        try:
            yield
        finally:
            set_adapter("default")

    return _activated()
