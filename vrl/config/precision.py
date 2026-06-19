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
use fp16, ``math`` stays fp32). A mapping may override ``math`` or ``frozen``, and
it must still set one shared ``forward`` dtype.

A mapping may also set an explicit ``rollout`` dtype that differs from
``forward``. This is the **experimental** split: it is the only way to express an
fp8/fp4 rollout against a bf16/fp32 replay, which is inherently a rollout-vs-replay
precision-correction problem (the collection-time logprob no longer equals the
replay logprob). fp8/fp4 are rollout-only: the scalar form (``precision: fp8``)
and the ``forward``/``math`` axes reject them, because a sub-byte replay/training
forward has no stable gradient path. The split is reachable only by deliberately
writing ``{forward: bf16, rollout: fp8}``, and a run that uses it must arm the
precision drift guard (``trainer.precision_drift_guard``) and/or truncated
importance sampling (``GRPOConfig.tis_mode``) so the resulting drift is measured
and bounded rather than silently biasing the gradient.

This module is torch-free (config layer): it resolves precision *policy* only.
Consumers materialize a canonical axis name into a ``torch.dtype`` via
:func:`vrl.models.dtypes.resolve_torch_dtype`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_MISSING = object()

# The canonical precision names. ``no`` remains a tolerated spelling for low-level
# parser boundaries, but live YAML should use top-level ``precision: fp32``. fp8/fp4
# are rollout-only quantized GEMM dtypes (Hopper/Blackwell); they are valid policy
# tokens but only meaningful on the ``rollout`` axis (see module docstring).
_CANONICAL = ("fp32", "bf16", "fp16", "fp8", "fp4")


def normalize_precision(value: Any, *, default: str = "fp32") -> str:
    """Map a precision token to a canonical name (``fp32``/``bf16``/``fp16``/``fp8``/``fp4``)."""

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
    # follow the rollout storage dtype. A sub-byte rollout (fp8/fp4) quantizes
    # only the policy GEMM; the frozen encoders/VAE have no quantized path, so
    # default them to fp16 rather than an unusable fp8/fp4 storage dtype.
    if rollout in ("fp8", "fp4"):
        return "fp16"
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

    block = _select(cfg, "precision", _MISSING)
    if block is _MISSING:
        raise ValueError(
            "top-level `precision` is required. Use `precision: fp16`, "
            "`precision: bf16`, or `precision: fp32`.",
        )
    return _from_precision_block(block)


def _from_precision_block(block: Any) -> PrecisionPolicy:
    # Unknown keys inside a precision mapping are reported by the whole-tree
    # walker (vrl.config.unknown_keys); this parser only reads the known axes.
    if isinstance(block, (str, bool)):
        scalar = normalize_precision(block)
        if scalar in ("fp8", "fp4"):
            raise ValueError(
                f"precision: {scalar!r} is invalid as a scalar: fp8/fp4 is a "
                "rollout-only quantized GEMM dtype and a scalar would set the "
                "replay forward to it too. Use the mapping form "
                f"`precision: {{forward: bf16, rollout: {scalar}}}`.",
            )
        return _policy_from_compute(scalar)
    forward = normalize_precision(_select(block, "forward", "fp32"))
    # `rollout` is the experimental split (fp8/fp4 rollout vs bf16/fp32 replay);
    # absent, it follows `forward` so the common path stays single-dtype.
    rollout_raw = _select(block, "rollout", None)
    rollout = normalize_precision(rollout_raw) if rollout_raw is not None else forward
    if forward in ("fp8", "fp4"):
        raise ValueError(
            f"precision.forward={forward!r} is invalid: fp8/fp4 is a rollout-only "
            "quantized GEMM dtype. The replay/training forward must stay fp32/bf16/"
            "fp16 for stable gradients. Use `{forward: bf16, rollout: fp8}`.",
        )
    math = normalize_precision(_select(block, "math", "fp32"))
    if math in ("fp8", "fp4"):
        raise ValueError(
            f"precision.math={math!r} is invalid: the SDE/logprob/loss math axis "
            "must stay fp32/bf16/fp16, never sub-byte.",
        )
    frozen_raw = _select(block, "frozen", None)
    frozen = normalize_precision(frozen_raw) if frozen_raw is not None else _frozen_default(rollout)
    return PrecisionPolicy(compute=forward, rollout=rollout, math=math, frozen=frozen)


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
