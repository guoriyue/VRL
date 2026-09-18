"""Previous-policy updates and checkpoint ownership beyond PEFT's adapter API.

Model construction and adapter installation use PEFT directly in
``DiffusionModelBase``. These operations remain VRL-owned: PEFT does not define
the algorithms' copy/EMA schedule or checkpoint registration of mutable frozen
parameters. The objectives that consume the frozen mirror live under
``vrl/algorithms``; nothing here depends on which one requested it.
"""

from __future__ import annotations

from typing import Any

from vrl.models.interfaces.runtime import register_checkpoint_owned_state


def copy_adapter_weights(
    module: Any,
    *,
    src: str,
    dst: str,
    decay: float = 0.0,
) -> None:
    """Copy (or EMA-blend) one PEFT adapter's params into another in place.

    ``decay=0`` is an exact copy; ``decay`` in (0, 1] is a soft update
    ``dst <- decay*dst + (1-decay)*src`` (the objectives' ``weight_copy_decay``). Matches
    params by the ``.{src}.`` / ``.{dst}.`` marker PEFT puts in every adapter path.
    """

    named = dict(module.named_parameters())
    copied = 0
    decay = float(decay)
    if not 0.0 <= decay <= 1.0:
        raise ValueError(f"adapter weight copy decay must be in [0, 1], got {decay}")
    for name, param in named.items():
        src_marker = f".{src}."
        if src_marker not in name:
            continue
        dst_name = name.replace(src_marker, f".{dst}.")
        dst_param = named.get(dst_name)
        if dst_param is None:
            continue
        if decay == 0.0:
            dst_param.data.copy_(param.data)
        else:
            dst_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)
        copied += 1
    if copied == 0:
        raise RuntimeError(
            f"failed to copy adapter weights from {src!r} to {dst!r}; "
            "no matching adapter parameters were found",
        )


def freeze_checkpoint_owned_adapter_params(module: Any, adapter: str) -> None:
    """Freeze a mutable PEFT adapter and register it for exact checkpoint resume.

    Used for the ``previous`` adapter: it is only forward-evaluated under
    no_grad and refreshed by weight copy (``sync_previous_policy_adapter``),
    never optimized. PEFT creates adapter params with ``requires_grad=True``, so
    without this DDP's reducer expects a gradient for them that the no_grad
    replay never produces — failing the first backward unless the more expensive
    ``find_unused_parameters=true`` is forced. Freezing it keeps the cheaper
    ``find_unused_parameters=false`` correct.
    """

    marker = f".{adapter}."
    matched = [
        (name, parameter)
        for name, parameter in module.named_parameters(remove_duplicate=False)
        if marker in name
    ]
    if not matched:
        raise RuntimeError(
            f"no parameters found for adapter {adapter!r} to freeze",
        )
    prior_requires_grad = [parameter.requires_grad for _, parameter in matched]
    try:
        for _, parameter in matched:
            parameter.requires_grad_(False)
        register_checkpoint_owned_state(module, (name for name, _ in matched))
    except BaseException:
        for (_, parameter), requires_grad in zip(
            matched,
            prior_requires_grad,
            strict=True,
        ):
            parameter.requires_grad_(requires_grad)
        raise


__all__ = [
    "copy_adapter_weights",
    "freeze_checkpoint_owned_adapter_params",
]
