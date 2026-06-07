"""Unified precision policy: one config surface for all dtype axes.

A run's public precision is one forward dtype plus protected math/frozen defaults:

- ``forward``  : generation rollout + trainer replay transformer forward dtype
- ``math``     : the numerically-sensitive algorithm math *outside* the
                 transformer (the SDE step / log-probability / loss reductions,
                 e.g. ``sde_step_with_logprob``). torch autocast keeps such ops
                 in fp32 automatically for the NN; this axis is the same idea,
                 region-level, for our custom math autocast doesn't cover.
- ``frozen``   : frozen text encoders / VAE

Users normally set a scalar (``precision: fp16`` → rollout/replay forward both
use fp16, ``math`` stays fp32). A mapping may override ``math`` or ``frozen``, but
it must still set one shared ``forward`` dtype. The old public
``precision.compute``/``precision.rollout`` split is intentionally rejected:
rollout/replay split precision is an experimental correction problem, not a
safe default config surface.

This module is torch-free (config layer): it resolves precision *policy* only.
Consumers materialize a canonical axis name into a ``torch.dtype`` via
:func:`vrl.models.dtypes.resolve_torch_dtype`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_MISSING = object()

# The three canonical precision names. ``no`` remains a tolerated spelling for
# low-level parser boundaries, but live YAML should use top-level ``precision: fp32``.
_CANONICAL = ("fp32", "bf16", "fp16")


def normalize_precision(value: Any, *, default: str = "fp32") -> str:
    """Map a precision token to ``"fp32"``/``"bf16"``/``"fp16"``."""

    if value is None:
        return default
    token = str(value).lower().strip()
    if not token:
        return default
    if token == "no":
        return "fp32"
    if token not in _CANONICAL:
        raise ValueError(
            f"precision must be one of {_CANONICAL} (or legacy 'no'); got {value!r}",
        )
    return token


@dataclass(frozen=True, slots=True)
class PrecisionPolicy:
    """Resolved precision for the four dtype axes (canonical string names)."""

    compute: str
    rollout: str
    math: str
    frozen: str

    def __post_init__(self) -> None:
        for axis in (self.compute, self.rollout, self.math, self.frozen):
            if axis not in _CANONICAL:
                raise ValueError(f"invalid precision axis value: {axis!r}")


def _frozen_default(rollout: str) -> str:
    # Frozen modules (encoders/VAE) live on the rollout/generation side; keep the
    # historical "fp32 run -> fp16 frozen (save memory)" behavior, otherwise
    # follow the rollout storage dtype.
    return "fp16" if rollout == "fp32" else rollout


def _policy_from_compute(
    compute: str,
    *,
    rollout: str | None = None,
    math: str = "fp32",
    frozen: str | None = None,
) -> PrecisionPolicy:
    rollout = rollout or compute
    frozen = frozen or _frozen_default(rollout)
    return PrecisionPolicy(compute=compute, rollout=rollout, math=math, frozen=frozen)


def resolve_precision_policy(cfg: Any) -> PrecisionPolicy:
    """Resolve the precision policy from a run config.

    A public config must use the top-level ``precision`` block. The returned
    policy still exposes ``compute`` and ``rollout`` internally so trainer/debug
    code can report both sides, but they are deliberately derived from the same
    forward dtype.
    """

    _reject_legacy_precision_keys(cfg)
    block = _select(cfg, "precision", _MISSING)
    if block is _MISSING:
        raise ValueError(
            "top-level `precision` is required; actor.mixed_precision/actor.bf16 "
            "were removed. Use `precision: fp16`, `precision: bf16`, or "
            "`precision: fp32`.",
        )
    return _from_precision_block(block)


def _from_precision_block(block: Any) -> PrecisionPolicy:
    if isinstance(block, (str, bool)):
        return _policy_from_compute(normalize_precision(block))
    if _select(block, "compute", _MISSING) is not _MISSING:
        raise ValueError(
            "precision.compute is no longer supported; rollout and replay forward "
            "precision must be configured together via `precision: fp16` or "
            "`precision.forward: fp16`.",
        )
    if _select(block, "rollout", _MISSING) is not _MISSING:
        raise ValueError(
            "precision.rollout is no longer supported; rollout and replay forward "
            "precision must be configured together via `precision: fp16` or "
            "`precision.forward: fp16`.",
        )
    forward = normalize_precision(_select(block, "forward", "fp32"))
    math = normalize_precision(_select(block, "math", "fp32"))
    frozen_raw = _select(block, "frozen", None)
    frozen = normalize_precision(frozen_raw) if frozen_raw is not None else _frozen_default(forward)
    return PrecisionPolicy(compute=forward, rollout=forward, math=math, frozen=frozen)


def _reject_legacy_precision_keys(cfg: Any) -> None:
    legacy = [
        path
        for path in (
            "actor.mixed_precision",
            "actor.bf16",
            "+actor.mixed_precision",
            "+actor.bf16",
            "+precision.compute",
            "+precision.rollout",
        )
        if _select(cfg, path, _MISSING) is not _MISSING
    ]
    if legacy:
        raise ValueError(
            "legacy precision config is no longer supported: "
            + ", ".join(legacy)
            + ". Use top-level `precision: fp16`, `precision: bf16`, or "
            "`precision: fp32`.",
        )


def _select(obj: Any, path: str, default: Any = None) -> Any:
    """Read a dotted path from a Mapping/DictConfig/namespace, else ``default``."""

    cur = obj
    for part in path.split("."):
        if cur is None:
            return default
        nxt = None
        if isinstance(cur, Mapping) or hasattr(cur, "get"):
            try:
                nxt = cur.get(part, None)  # type: ignore[union-attr]
            except Exception:
                nxt = None
        if nxt is None:
            nxt = getattr(cur, part, None)
        cur = nxt
    return default if cur is None else cur


__all__ = [
    "PrecisionPolicy",
    "normalize_precision",
    "resolve_precision_policy",
]
