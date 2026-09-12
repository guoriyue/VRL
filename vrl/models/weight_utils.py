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

from vrl.utils.validation import require_int


def unwrap_compile_and_ddp(module: Any) -> Any:
    """Peel torch.compile (``_orig_mod``) and DDP / FSDP1 (``.module``) wrappers.

    Only framework wrapper instances are peeled; a normal model may own a
    child named ``module`` or ``_orig_mod`` without being a wrapper.

    Sync payload keys live in the policy's uncompiled, unwrapped namespace. Both
    ends peel through here: the sync sender
    (``trainers.weight_sync.flatten_trainable_module_state``) and the receiver
    (:func:`load_weights_into` / :func:`require_weights_for` below), so neither
    wrapper prefix may leak into the rollout payload. PEFT is deliberately NOT
    peeled — LoRA keys (``base_model.model.*``) are part of the policy-facing
    namespace. Loop because wrapper nesting/order varies (e.g. compile(DDP(m)) vs
    DDP(compile(m))).

    FSDP2 export reuses this (vrl/trainers/strategy.py) so a sharded gather lands
    in the same namespace as single-process sync: ``get_model_state_dict`` strips
    ``_orig_mod.`` while ``named_parameters()`` keeps it, so selecting trainable
    keys on a still-compiled module would mismatch.
    """

    from torch._dynamo.eval_frame import OptimizedModule
    from torch.distributed.fsdp import FullyShardedDataParallel
    from torch.nn.parallel import DistributedDataParallel

    while True:
        if isinstance(module, OptimizedModule):
            module = module._orig_mod
        elif isinstance(module, (DistributedDataParallel, FullyShardedDataParallel)):
            module = module.module
        else:
            return module


def load_weights_into(
    module: Any,
    state_dict: Mapping[str, Any],
    *,
    prefix: str,
) -> Any:
    """Load a flattened trainable-weight sync payload into ``module``.

    The payload carries ``{prefix}.{param}`` keys for ``module``'s ``requires_grad``
    parameters only (as produced by the sync sender). Strict by design: rejects any
    key outside ``{prefix}.`` and any mismatch (extra/missing) against the module's
    trainable parameter set, so a malformed sync payload fails loudly instead of
    silently leaving weights stale.
    """

    stripped = require_weights_for(module, state_dict, prefix=prefix)
    module = unwrap_compile_and_ddp(module)
    return module.load_state_dict(stripped, strict=False)


def require_weights_for(
    module: Any,
    state_dict: Mapping[str, Any],
    *,
    prefix: str,
) -> dict[str, Any]:
    """Validate one flattened sync payload without changing live weights.

    Every rejection below names the module the payload failed against. That name
    is the class the checks actually run on, so it is derived from the same
    unwrapped module rather than threaded in by each caller.
    """

    from torch import Tensor

    module = unwrap_compile_and_ddp(module)
    label = type(module).__name__
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
        if not isinstance(value, Tensor):
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


def verify_weights_in(module: Any, state_dict: Mapping[str, Any], *, prefix: str) -> None:
    """Read live trainable parameters and require exact installed payload bytes.

    Acceptance-only: this synchronizes device copies and reads every tensor.
    Never call it implicitly from the normal loader. Distributed/sharded and
    quantized representations need their own explicit comparison contracts.
    """

    import torch
    from torch.distributed.tensor import DTensor

    expected = require_weights_for(module, state_dict, prefix=prefix)
    parameters = dict(unwrap_compile_and_ddp(module).named_parameters())
    for name, value in expected.items():
        actual = parameters[name].detach()
        for tensor in (actual, value):
            if isinstance(tensor, DTensor) or tensor.is_quantized or tensor.is_meta:
                raise NotImplementedError(
                    f"{prefix}.{name}: exact weight verification requires materialized, "
                    "unsharded, nonquantized tensors"
                )
            if tensor.layout != torch.strided:
                raise NotImplementedError(f"{prefix}.{name}: unsupported tensor layout")
        # Byte comparison preserves bf16 bits, signed zero and NaN payloads.
        # A float32 diagnostic digest cannot provide that guarantee for all dtypes.
        observed = actual.contiguous().reshape(-1).view(torch.uint8).cpu()
        wanted = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
        if not torch.equal(observed, wanted):
            raise RuntimeError(f"{prefix}.{name}: installed weight content differs from payload")


def verify_trainable_modules(modules: Mapping[str, Any], state_dict: Mapping[str, Any]) -> None:
    """Verify the existing flat sync namespace across all selected module roots."""

    if not modules or not state_dict:
        raise ValueError("weight verification requires modules and a nonempty payload")
    remaining = set(state_dict)
    for prefix, module in modules.items():
        selected = {
            key: value for key, value in state_dict.items() if key.startswith(f"{prefix}.")
        }
        verify_weights_in(module, selected, prefix=prefix)
        remaining.difference_update(selected)
    if remaining:
        raise ValueError(f"weight verification received unknown roots: {sorted(remaining)[:5]}")


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
        self.max_retained = require_int(max_retained, path="max_retained", minimum=1)
        self._slots: dict[int, Mapping[str, Any]] = {}

    def install(self, version: int, state: Mapping[str, Any] | None) -> None:
        """Retain ``state`` under ``version``; ``None`` aliases the newest slot.

        A ``None`` payload is a version-only bump (no new weights): the new
        version should resolve to the same live state as the previous version, so
        we alias the most recent slot rather than create an empty one.
        """

        version = require_int(version, path="policy version", minimum=0)
        if state is None:
            if not self._slots:
                return
            state = self._slots[max(self._slots)]
        self._slots[version] = state
        self._evict()

    def has(self, version: int) -> bool:
        return require_int(version, path="policy version", minimum=0) in self._slots

    def get(self, version: int) -> Mapping[str, Any]:
        return self._slots[require_int(version, path="policy version", minimum=0)]

    def _evict(self) -> None:
        # Keep the most recent ``max_retained`` versions. A request older than the
        # window loses its slot and is reported as a stale-slot result rather than
        # silently mixing weights.
        while len(self._slots) > self.max_retained:
            del self._slots[min(self._slots)]
