"""Resolve public precision config into role-specific runtime policy.

``dtype`` is the ordinary parameter precision for a role; ``outer_autocast``
selects whether the shared diffusion boundary also applies that dtype through
AMP. Selective quantization is a separate kernel policy layered on the dtype:
FP8 replaces eligible attention/MLP Linears, while experimental NVFP4
conservatively replaces eligible MLP Linears only. Unswapped operations keep
the role dtype.

Rollout dtype and outer-autocast behavior default to the training role.
Prompt-encoder dtype defaults to the resolved rollout dtype; the canonical base
preset explicitly selects fp16 to preserve its established memory policy. VAE
precision is family-owned and is not represented by this prompt-encoder axis.
Diffusion math defaults to fp32. FP32 matmul precision is an explicit run-wide
policy because PyTorch process defaults are not a stable trainer/worker contract.

Training quantization is part of the structural schema so the eventual training
runtime will use the same role shape, but policy resolution fails until a real
autograd-capable consumer exists. This module remains torch-free; runtime
boundaries materialize dtype tokens through ``vrl.models.dtypes``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast, get_args

_MISSING = object()

Float32Precision = Literal["ieee", "tf32"]

# Protocol vocabularies are a real config boundary. Keep recipe compatibility in
# one deliberately isolated table so config resolution, defaults, and error text
# cannot drift when a new format lands.
_PLAIN_DTYPES = ("fp32", "bf16", "fp16")


@dataclass(frozen=True, slots=True)
class _QuantizationFormatRules:
    allowed_recipes: tuple[str, ...]
    default_recipe: str | None


_QUANTIZATION_FORMAT_RULES = {
    "fp8": _QuantizationFormatRules(
        allowed_recipes=("rowwise", "tensorwise", "blockwise"),
        default_recipe="rowwise",
    ),
    # NVFP4 is the complete two-level scaling scheme, not an FP8-style recipe.
    "nvfp4": _QuantizationFormatRules(allowed_recipes=(), default_recipe=None),
}
_PRECISION_TOKENS = (*_PLAIN_DTYPES, *_QUANTIZATION_FORMAT_RULES)


# These dataclasses define the public YAML shape consumed by the unknown-key
# walker. The parser below owns behavior, while schema keys are derived from this
# source instead of repeated as hand-maintained allowed-key tuples.
@dataclass(frozen=True, slots=True)
class DTypePrecisionConfig:
    dtype: Any


@dataclass(frozen=True, slots=True)
class QuantizationConfig:
    format: Any
    recipe: Any = None


@dataclass(frozen=True, slots=True)
class TrainingPrecisionConfig:
    dtype: Any
    outer_autocast: Any = True
    quantization: QuantizationConfig | None = None


@dataclass(frozen=True, slots=True)
class PromptEncodersPrecisionConfig:
    dtype: Any


@dataclass(frozen=True, slots=True)
class RolloutPrecisionConfig:
    dtype: Any = None
    outer_autocast: Any = None
    quantization: QuantizationConfig | None = None
    prompt_encoders: PromptEncodersPrecisionConfig | None = None


@dataclass(frozen=True, slots=True)
class PrecisionConfig:
    training: TrainingPrecisionConfig
    float32_precision: Any
    rollout: RolloutPrecisionConfig | None = None
    diffusion_math: DTypePrecisionConfig | None = None


def normalize_precision(value: Any, *, default: str = "fp32") -> str:
    """Normalize a known precision token at a config or tool boundary."""

    if value is None:
        return default
    token = str(value).lower().strip()
    if not token:
        return default
    if token == "no":
        return "fp32"
    if token not in _PRECISION_TOKENS:
        raise ValueError(
            f"precision must be one of {_PRECISION_TOKENS} (or legacy 'no'); got {value!r}",
        )
    return token


def _normalize_plain_dtype(value: Any, *, path: str) -> str:
    token = normalize_precision(value)
    if token not in _PLAIN_DTYPES:
        raise ValueError(
            f"{path} must be one of {_PLAIN_DTYPES}; got {token!r}. FP8/NVFP4 "
            "belongs under a `quantization.format` key, not a `dtype` key.",
        )
    return token


def _normalize_float32_precision(value: Any) -> Float32Precision:
    mode = str(value).lower().strip()
    allowed = get_args(Float32Precision)
    if mode not in allowed:
        raise ValueError(
            f"precision.float32_precision must be one of {allowed}; got {value!r}",
        )
    return cast("Float32Precision", mode)


@dataclass(frozen=True, slots=True)
class QuantizationPolicy:
    """Selective low-precision kernel policy layered on a role's base dtype."""

    format: str
    recipe: str | None = None

    def __post_init__(self) -> None:
        format_name = str(self.format).lower().strip()
        recipe = str(self.recipe).lower().strip() if self.recipe is not None else None
        if format_name == "fp4":
            raise ValueError(
                "quantization.format='fp4' was replaced by the precise scheme name "
                "'nvfp4'; use `format: nvfp4` without a recipe.",
            )
        format_rules = _QUANTIZATION_FORMAT_RULES.get(format_name)
        if format_rules is None:
            raise ValueError(
                "quantization.format must be one of "
                f"{tuple(_QUANTIZATION_FORMAT_RULES)}; got {self.format!r}",
            )
        if recipe is None:
            recipe = format_rules.default_recipe
        elif recipe not in format_rules.allowed_recipes:
            if not format_rules.allowed_recipes:
                raise ValueError(
                    f"quantization.format={format_name!r} does not accept a recipe; "
                    "remove the `recipe` key.",
                )
            raise ValueError(
                f"quantization.recipe={recipe!r} is invalid for format "
                f"{format_name!r}; expected one of {format_rules.allowed_recipes}.",
            )
        object.__setattr__(self, "format", format_name)
        object.__setattr__(self, "recipe", recipe)


@dataclass(frozen=True, slots=True)
class RolePrecision:
    """Transformer precision selected for one trainer or rollout process."""

    dtype: str
    float32_precision: Float32Precision
    quantization: QuantizationPolicy | None = None
    outer_autocast: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dtype",
            _normalize_plain_dtype(self.dtype, path="precision role dtype"),
        )
        object.__setattr__(
            self,
            "float32_precision",
            _normalize_float32_precision(self.float32_precision),
        )
        if not isinstance(self.outer_autocast, bool):
            raise TypeError("role outer_autocast must be a bool")
        if self.quantization is not None and not isinstance(
            self.quantization,
            QuantizationPolicy,
        ):
            raise TypeError("role quantization must be a QuantizationPolicy or None")

    @property
    def label(self) -> str:
        """Stable role label containing base dtype and execution policies."""

        parts = [self.dtype]
        if self.quantization is not None:
            parts.append(self.quantization.format)
        if not self.outer_autocast:
            parts.append("no-autocast")
        return "+".join(parts)


@dataclass(frozen=True, slots=True)
class PrecisionPolicy:
    """Resolved precision for both roles, protected math, and prompt encoders."""

    training: RolePrecision
    rollout: RolePrecision
    diffusion_math: str
    prompt_encoder_dtype: str

    def __post_init__(self) -> None:
        if not isinstance(self.training, RolePrecision):
            raise TypeError("precision.training must resolve to RolePrecision")
        if not isinstance(self.rollout, RolePrecision):
            raise TypeError("precision.rollout must resolve to RolePrecision")
        object.__setattr__(
            self,
            "diffusion_math",
            _normalize_plain_dtype(
                self.diffusion_math,
                path="precision.diffusion_math.dtype",
            ),
        )
        object.__setattr__(
            self,
            "prompt_encoder_dtype",
            _normalize_plain_dtype(
                self.prompt_encoder_dtype,
                path="precision.rollout.prompt_encoders.dtype",
            ),
        )
        if self.training.quantization is not None:
            raise ValueError(
                "precision.training.quantization is unavailable: the trainer has no "
                "quantized Linear forward/backward consumer. Remove the block until "
                "that runtime is implemented; VRL will not accept a silent no-op.",
            )

    @property
    def stages_match(self) -> bool:
        """Whether rollout and replay use the same complete execution policy."""

        return self.training == self.rollout


def resolve_precision_policy(cfg: Any) -> PrecisionPolicy:
    """Resolve the required nested top-level ``precision`` block."""

    legacy_allow_tf32 = _select(cfg, "actor.optim.allow_tf32", _MISSING)
    if legacy_allow_tf32 is not _MISSING:
        raise ValueError(
            "actor.optim.allow_tf32 is no longer supported because FP32 matmul "
            "precision is a forward policy, not an optimizer setting. Replace it "
            "with top-level precision.float32_precision: tf32 (or ieee).",
        )

    block = _select(cfg, "precision", _MISSING)
    if block is _MISSING:
        raise ValueError(
            "top-level `precision` is required. Configure "
            "`precision.training.dtype`; add a `precision.rollout` block only "
            "for a rollout-specific override.",
        )
    if isinstance(block, (str, bool)):
        raise ValueError(
            "scalar `precision` is no longer supported because it hides the "
            "difference between base dtype and quantization. Use "
            "`precision: {training: {dtype: bf16}}`.",
        )
    _reject_legacy_keys(block)

    float32_precision_raw = _select(block, "float32_precision", _MISSING)
    if float32_precision_raw is _MISSING:
        raise ValueError(
            "precision.float32_precision is required; set it to 'tf32' for "
            "TensorFloat-32 FP32 matmuls or 'ieee' for full IEEE FP32 matmuls.",
        )
    float32_precision = _normalize_float32_precision(float32_precision_raw)
    training = _parse_role(
        block,
        "training",
        float32_precision=float32_precision,
        outer_autocast_default=True,
    )
    rollout = _parse_role(
        block,
        "rollout",
        dtype_default=training.dtype,
        float32_precision=float32_precision,
        outer_autocast_default=training.outer_autocast,
    )
    math_raw = _select(block, "diffusion_math.dtype", "fp32")
    prompt_encoder_raw = _select(
        block,
        "rollout.prompt_encoders.dtype",
        rollout.dtype,
    )
    return PrecisionPolicy(
        training=training,
        rollout=rollout,
        diffusion_math=_normalize_plain_dtype(
            math_raw,
            path="precision.diffusion_math.dtype",
        ),
        prompt_encoder_dtype=_normalize_plain_dtype(
            prompt_encoder_raw,
            path="precision.rollout.prompt_encoders.dtype",
        ),
    )


def _parse_role(
    block: Any,
    role: str,
    *,
    float32_precision: Float32Precision,
    outer_autocast_default: bool,
    dtype_default: Any = _MISSING,
) -> RolePrecision:
    dtype_raw = _select(block, f"{role}.dtype", dtype_default)
    if dtype_raw is _MISSING:
        raise ValueError(f"precision.{role}.dtype is required")
    quantization_raw = _select(block, f"{role}.quantization", None)
    quantization = None
    if quantization_raw is not None:
        format_raw = _select(quantization_raw, "format", _MISSING)
        if format_raw is _MISSING:
            raise ValueError(f"precision.{role}.quantization.format is required")
        recipe_raw = _select(quantization_raw, "recipe", None)
        quantization = QuantizationPolicy(
            format=str(format_raw),
            recipe=str(recipe_raw) if recipe_raw is not None else None,
        )
    outer_autocast = _select(
        block,
        f"{role}.outer_autocast",
        outer_autocast_default,
    )
    if not isinstance(outer_autocast, bool):
        raise TypeError(
            f"precision.{role}.outer_autocast must be a bool; got {outer_autocast!r}",
        )
    return RolePrecision(
        dtype=_normalize_plain_dtype(dtype_raw, path=f"precision.{role}.dtype"),
        float32_precision=float32_precision,
        outer_autocast=outer_autocast,
        quantization=quantization,
    )


def _reject_legacy_keys(block: Any) -> None:
    legacy_paths = (
        "train",
        "math",
        "frozen",
        "frozen_components",
        "rollout_recipe",
        "rollout.frozen_components",
    )
    present = [
        f"precision.{path}"
        for path in legacy_paths
        if _select(block, path, _MISSING) is not _MISSING
    ]
    rollout = _select(block, "rollout", None)
    if (
        rollout is not None
        and not isinstance(rollout, Mapping)
        and not hasattr(
            rollout,
            "get",
        )
    ):
        present.append("precision.rollout")
    if present:
        names = ", ".join(present)
        raise ValueError(
            f"legacy precision key(s) are no longer supported: {names}. Use "
            "precision.training.dtype, precision.training.outer_autocast, "
            "precision.rollout.dtype, precision.rollout.outer_autocast, "
            "precision.rollout.quantization, precision.diffusion_math.dtype, "
            "precision.rollout.prompt_encoders.dtype, and "
            "precision.float32_precision.",
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
    "Float32Precision",
    "PrecisionConfig",
    "PrecisionPolicy",
    "QuantizationPolicy",
    "RolePrecision",
    "normalize_precision",
    "resolve_precision_policy",
]
