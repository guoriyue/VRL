"""Unified precision policy: one config surface for all dtype axes.

A run's precision is four orthogonal dtype axes — each answers "what dtype":

- ``compute``  : training + replay transformer fwd/bwd (torch autocast / bf16 weights)
- ``rollout``  : generation/sampling forward (model storage dtype)
- ``math``     : the numerically-sensitive algorithm math *outside* the
                 transformer (the SDE step / log-probability / loss reductions,
                 e.g. ``sde_step_with_logprob``). torch autocast keeps such ops
                 in fp32 automatically for the NN; this axis is the same idea,
                 region-level, for our custom math autocast doesn't cover.
- ``frozen``   : frozen text encoders / VAE

Users set a scalar (``precision: bf16`` → that dtype for compute/rollout/frozen,
``math`` stays fp32) or a dict overriding individual axes. When no
``precision:`` block is present, the policy is derived from the legacy
``actor.mixed_precision`` / ``actor.bf16`` fields so existing configs keep their
exact behavior.

This module is torch-free (config layer). Consumers convert a string axis to a
``torch.dtype`` via :func:`precision_to_torch_dtype`, passing ``torch`` in.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_logger = logging.getLogger(__name__)
_legacy_warned = False

# The three canonical precision names. ``no`` is the one legacy spelling we still
# accept, because existing configs write ``actor.mixed_precision: "no"`` for fp32.
_CANONICAL = ("fp32", "bf16", "fp16")


def normalize_precision(value: Any, *, default: str = "fp32") -> str:
    """Map a precision token to ``"fp32"``/``"bf16"``/``"fp16"``."""

    if value is None:
        return default
    token = str(value).lower().strip()
    if not token:
        return default
    if token == "no":  # legacy actor.mixed_precision spelling for fp32
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

    Precedence: an explicit ``precision:`` block (scalar or dict) wins; otherwise
    derive from the legacy ``actor.mixed_precision`` / ``actor.bf16`` fields so
    pre-existing configs are byte-for-byte unchanged.
    """

    block = _select(cfg, "precision", None)
    if block is not None:
        return _from_precision_block(block)
    return _from_legacy(cfg)


def _from_precision_block(block: Any) -> PrecisionPolicy:
    if isinstance(block, (str, bool)):
        return _policy_from_compute(normalize_precision(block))
    compute = normalize_precision(_select(block, "compute", "fp32"))
    rollout_raw = _select(block, "rollout", None)
    rollout = normalize_precision(rollout_raw) if rollout_raw is not None else compute
    math = normalize_precision(_select(block, "math", "fp32"))
    frozen_raw = _select(block, "frozen", None)
    frozen = normalize_precision(frozen_raw) if frozen_raw is not None else _frozen_default(rollout)
    return PrecisionPolicy(compute=compute, rollout=rollout, math=math, frozen=frozen)


def _from_legacy(cfg: Any) -> PrecisionPolicy:
    global _legacy_warned
    if not _legacy_warned:
        _legacy_warned = True
        _logger.warning(
            "no top-level `precision:` block found; deriving precision from the "
            "deprecated actor.mixed_precision/actor.bf16 fields. Prefer "
            "`precision: bf16` (or fp32/fp16). The legacy fields still work.",
        )
    mixed_precision = _select(cfg, "actor.mixed_precision", "")
    # TrainerConfig defaults bf16=True; mirror that when the field is absent.
    bf16 = _select(cfg, "actor.bf16", True)
    token = str(mixed_precision or "").strip()
    # No explicit mixed_precision -> follow the bf16 flag (TrainerConfig default).
    compute = normalize_precision(token) if token else ("bf16" if bf16 else "fp32")
    # Today storage (weight_dtype) and compute (autocast) are the same knob.
    return _policy_from_compute(compute)


def precision_to_torch_dtype(name: str, torch: Any) -> Any:
    """Convert a canonical precision name (from a ``PrecisionPolicy``) to a dtype."""

    return {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[name]


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
    "precision_to_torch_dtype",
    "resolve_precision_policy",
]
