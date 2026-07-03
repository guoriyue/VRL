"""Unified precision policy: one config surface for all dtype axes.

A run's public precision is one training-forward dtype plus protected math/frozen
defaults:

- ``train``    : trainer replay transformer forward dtype; the generation rollout
                 forward follows it unless ``rollout`` overrides it
- ``math``     : the numerically-sensitive algorithm math *outside* the
                 transformer (the SDE step / log-probability / loss reductions,
                 e.g. ``sde_step_with_logprob``). torch autocast keeps such ops
                 in fp32 automatically for the NN; this axis is the same idea,
                 region-level, for our custom math autocast doesn't cover.
- ``frozen``   : frozen text encoders / VAE

Users normally set a scalar (``precision: fp16`` → rollout/replay forward both
use fp16, ``math`` stays fp32). A mapping may override ``math`` or ``frozen``, and
it must still set one shared ``train`` dtype.

A mapping may also set an explicit ``rollout`` dtype that differs from
``train``. This is the **experimental** split: it is the only way to express an
fp8/fp4 rollout against a bf16/fp32 replay, which is inherently a rollout-vs-replay
precision-correction problem (the collection-time logprob no longer equals the
replay logprob). fp8/fp4 are rollout-only: the scalar form (``precision: fp8``)
and the ``train``/``math`` axes reject them, because a sub-byte replay/training
forward has no stable gradient path. The split is reachable only by deliberately
writing ``{train: bf16, rollout: fp8}``. An optional ``rollout_recipe`` picks the
quantization kernel recipe (fp8: ``rowwise`` default / ``tensorwise`` /
``blockwise`` via vLLM's block kernel). The trainer config builder then derives
the correction path automatically: rollout-recorded logprobs are used as the old
policy anchor, drift metrics are reported, TIS/RS are enabled, and the guard
fails only on catastrophic/non-finite drift unless the user provides an explicit
expert ``trainer.precision_*`` block.

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

    train: str
    rollout: str
    math: str
    frozen: str
    # Quantization kernel recipe for a quantized (fp8/fp4) rollout, or None for
    # the scheme default (fp8: "rowwise"). Only legal alongside a quantized
    # rollout — on a plain-dtype rollout it would be a silent no-op knob. The
    # recipe vocabulary is owned by the kernel layer (Fp8Linear raises on an
    # unknown recipe), not duplicated here. "blockwise" is additionally rejected
    # when model.torch_compile is on (build-time guard in
    # apply_rollout_quantization: the vLLM kernel graph-breaks inductor and the
    # compiled forward is ~10x slower than eager).
    rollout_recipe: str | None = None

    def __post_init__(self) -> None:
        for axis in (self.train, self.rollout, self.math, self.frozen):
            if axis not in _CANONICAL:
                raise ValueError(f"invalid precision axis value: {axis!r}")
        if self.rollout_recipe is not None and self.rollout not in ("fp8", "fp4"):
            raise ValueError(
                f"precision.rollout_recipe={self.rollout_recipe!r} requires a "
                f"quantized rollout (fp8/fp4); rollout={self.rollout!r} is a plain "
                "dtype with no kernel recipe, so the knob would silently do nothing.",
            )


def precision_bridge_fields(policy: PrecisionPolicy) -> dict[str, str]:
    """The TrainerConfig fields bridged from a resolved precision policy.

    Single source for the policy -> derived-fields expansion so the trainer
    builder (dict payload) and the online recipe (TrainerConfig attrs) cannot
    drift. ``train_precision`` is the replay/training forward dtype; rollout and
    math are exposed separately so the precision drift guard can prove both
    sides stayed aligned.
    """
    return {
        "train_precision": policy.train,
        "rollout_precision": policy.rollout,
        "math_precision": policy.math,
    }


def _frozen_default(rollout: str) -> str:
    # Frozen modules (encoders/VAE) live on the rollout/generation side; keep the
    # historical "fp32 run -> fp16 frozen (save memory)" behavior, otherwise
    # follow the rollout storage dtype. A sub-byte rollout (fp8/fp4) quantizes
    # only the policy GEMM; the frozen encoders/VAE have no quantized path, so
    # default them to fp16 rather than an unusable fp8/fp4 storage dtype.
    if rollout in ("fp8", "fp4"):
        return "fp16"
    return "fp16" if rollout == "fp32" else rollout


def _policy_from_train(
    train: str,
    *,
    rollout: str | None = None,
    math: str = "fp32",
    frozen: str | None = None,
) -> PrecisionPolicy:
    rollout = rollout or train
    frozen = frozen or _frozen_default(rollout)
    return PrecisionPolicy(train=train, rollout=rollout, math=math, frozen=frozen)


def resolve_precision_policy(cfg: Any) -> PrecisionPolicy:
    """Resolve the precision policy from a run config.

    A public config must use the top-level ``precision`` block. The returned
    policy still exposes ``train`` and ``rollout`` internally so trainer/debug
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
                f"`precision: {{train: bf16, rollout: {scalar}}}`.",
            )
        return _policy_from_train(scalar)
    train = normalize_precision(_select(block, "train", "fp32"))
    # `rollout` is the experimental split (fp8/fp4 rollout vs bf16/fp32 replay);
    # absent, it follows `train` so the common path stays single-dtype.
    rollout_raw = _select(block, "rollout", None)
    rollout = normalize_precision(rollout_raw) if rollout_raw is not None else train
    if train in ("fp8", "fp4"):
        raise ValueError(
            f"precision.train={train!r} is invalid: fp8/fp4 is a rollout-only "
            "quantized GEMM dtype. The replay/training forward must stay fp32/bf16/"
            "fp16 for stable gradients. Use `{train: bf16, rollout: fp8}`.",
        )
    math = normalize_precision(_select(block, "math", "fp32"))
    if math in ("fp8", "fp4"):
        raise ValueError(
            f"precision.math={math!r} is invalid: the SDE/logprob/loss math axis "
            "must stay fp32/bf16/fp16, never sub-byte.",
        )
    frozen_raw = _select(block, "frozen", None)
    frozen = normalize_precision(frozen_raw) if frozen_raw is not None else _frozen_default(rollout)
    # Kernel recipe for the quantized rollout (e.g. fp8 "rowwise"/"blockwise").
    # Passed through as-is: PrecisionPolicy rejects it on a non-quantized rollout,
    # the swap layer (Fp8Linear) rejects an unknown recipe token.
    recipe_raw = _select(block, "rollout_recipe", None)
    rollout_recipe = str(recipe_raw).lower().strip() if recipe_raw is not None else None
    return PrecisionPolicy(
        train=train, rollout=rollout, math=math, frozen=frozen, rollout_recipe=rollout_recipe,
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
    "precision_bridge_fields",
    "resolve_precision_policy",
]
