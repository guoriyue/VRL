"""Tests for the unified precision policy resolver (P0)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from tests.config.test_load_all_experiments import (
    _experiment_names,
    _load_experiment_for_static_validation,
)
from vrl.config.loading import load_config
from vrl.config.precision import (
    PrecisionConfig,
    PrecisionPolicy,
    QuantizationPolicy,
    RolePrecision,
    normalize_precision,
)
from vrl.config.schema import parse_config
from vrl.models.dtypes import (
    dtype_to_precision_token,
    dtype_to_wire_name,
    resolve_torch_dtype,
)


def _section(precision: Any = None) -> PrecisionConfig | None:
    """Parse one ``precision`` block the way every production caller does.

    ``parse_config`` is the single gate: shape, vocabulary, and unknown keys are
    all rejected there with the repo's message format, and the resolver only
    ever sees a typed section (or ``None`` when the block is absent).
    """

    top: dict[str, Any] = {} if precision is None else {"precision": precision}
    return parse_config(OmegaConf.create(top)).precision


# -- normalize / convert ----------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("bf16", "bf16"),
        ("fp32", "fp32"),
        ("fp16", "fp16"),
        ("fp8", "fp8"),
        ("nvfp4", "nvfp4"),
        ("no", "fp32"),  # the one legacy spelling we still accept
        (None, "fp32"),
        ("", "fp32"),
    ],
)
def test_normalize_precision(value, expected):
    """The accepted vocabulary: five plain tokens, plus `no`/empty/None meaning fp32."""
    assert normalize_precision(value) == expected


def test_normalize_rejects_unknown():
    """Neither a quantization-only token nor a torch dtype spelling is a policy token."""
    with pytest.raises(ValueError):
        normalize_precision("int8")
    with pytest.raises(ValueError):
        normalize_precision("bfloat16")  # no longer an accepted alias


def test_resolve_torch_dtype():
    """The plain policy tokens map onto torch dtypes without an alias table."""
    assert resolve_torch_dtype("fp32") is torch.float32
    assert resolve_torch_dtype("bf16") is torch.bfloat16
    assert resolve_torch_dtype("fp16") is torch.float16


def test_dtype_to_wire_name_is_not_a_public_config_token():
    """Wire serialization uses stable torch names, not precision-policy spellings."""
    assert dtype_to_wire_name(torch.bfloat16) == "bfloat16"
    assert dtype_to_wire_name("fp16") == "float16"
    assert dtype_to_wire_name("torch.float32") == "float32"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (torch.float32, "fp32"),
        ("bfloat16", "bf16"),
        ("half", "fp16"),
    ],
)
def test_dtype_to_precision_token_uses_the_public_plain_vocabulary(value, expected):
    assert dtype_to_precision_token(value) == expected


@pytest.mark.parametrize("value", ["fp8", "fp4"])
def test_dtype_to_precision_token_rejects_rollout_quantization(value):
    with pytest.raises(ValueError, match="not a plain public precision token"):
        dtype_to_precision_token(value)


# -- nested public policy ---------------------------------------------


def _plain_precision(dtype: str = "bf16") -> dict:
    return {
        "float32_precision": "tf32",
        "training": {"dtype": dtype},
        "rollout": {"dtype": dtype},
    }


def test_plain_block_resolves_to_the_same_policy_for_both_roles():
    policy = PrecisionPolicy.from_section(_section(_plain_precision()))

    assert policy == PrecisionPolicy(
        training=RolePrecision(dtype="bf16", float32_precision="tf32"),
        rollout=RolePrecision(dtype="bf16", float32_precision="tf32"),
        diffusion_math="fp32",
        prompt_encoder_dtype="bf16",
    )


def test_nested_bf16_resolves_role_dtypes_and_protected_defaults():
    p = PrecisionPolicy.from_section(
        _section(
            {
                "float32_precision": "tf32",
                "training": {"dtype": "bf16"},
            },
        ),
    )
    assert p == PrecisionPolicy(
        training=RolePrecision(dtype="bf16", float32_precision="tf32"),
        rollout=RolePrecision(dtype="bf16", float32_precision="tf32"),
        diffusion_math="fp32",
        prompt_encoder_dtype="bf16",
    )
    assert p.stages_match is True


def test_rollout_quantization_inherits_training_base_dtype():
    p = PrecisionPolicy.from_section(
        _section(
            {
                "float32_precision": "tf32",
                "training": {"dtype": "bf16"},
                "rollout": {"quantization": {"format": "fp8"}},
            },
        ),
    )

    assert p.rollout.dtype == "bf16"
    assert p.rollout.label == "bf16+fp8"


def test_rollout_inherits_training_outer_autocast() -> None:
    p = PrecisionPolicy.from_section(
        _section(
            {
                "float32_precision": "ieee",
                "training": {"dtype": "fp16", "outer_autocast": False},
            },
        ),
    )

    expected = RolePrecision(
        dtype="fp16",
        float32_precision="ieee",
        outer_autocast=False,
    )
    assert p.training == expected
    assert p.rollout == expected
    assert p.training.label == "fp16+no-autocast"
    assert p.stages_match is True


def test_rollout_can_override_training_outer_autocast() -> None:
    block = _plain_precision("fp16")
    block["rollout"]["outer_autocast"] = False

    p = PrecisionPolicy.from_section(_section(block))

    assert p.training.outer_autocast is True
    assert p.rollout.outer_autocast is False
    assert p.rollout.label == "fp16+no-autocast"
    assert p.stages_match is False


def test_outer_autocast_rejects_non_boolean_values() -> None:
    block = _plain_precision()
    block["training"]["outer_autocast"] = 1

    with pytest.raises(
        ValueError,
        match=r"precision\.training\.outer_autocast: Input should be a valid boolean",
    ):
        _section(block)


def test_prompt_encoders_default_to_rollout_dtype_even_for_fp32():
    p = PrecisionPolicy.from_section(_section(_plain_precision("fp32")))
    assert p.prompt_encoder_dtype == "fp32"


def test_diffusion_math_and_prompt_encoders_can_be_explicit():
    block = _plain_precision()
    block["diffusion_math"] = {"dtype": "bf16"}
    block["rollout"]["prompt_encoders"] = {"dtype": "fp16"}
    p = PrecisionPolicy.from_section(_section(block))
    assert p.diffusion_math == "bf16"
    assert p.prompt_encoder_dtype == "fp16"


def test_base_preset_keeps_prompt_encoders_aligned_with_rollout():
    cfg = load_config("experiment/sd3_5/online_grpo_ocr")
    p = PrecisionPolicy.from_section(parse_config(cfg).precision)
    assert p.rollout.dtype == "bf16"
    assert p.prompt_encoder_dtype == "bf16"
    assert p.training.float32_precision == "tf32"
    assert p.rollout.float32_precision == "tf32"


@pytest.mark.parametrize("mode", ["ieee", "tf32"])
def test_float32_precision_is_explicit_and_resolved(mode):
    block = _plain_precision()
    block["float32_precision"] = mode

    policy = PrecisionPolicy.from_section(_section(block))
    assert policy.training.float32_precision == mode
    assert policy.rollout.float32_precision == mode


def test_float32_precision_is_required():
    block = _plain_precision()
    del block["float32_precision"]

    with pytest.raises(
        ValueError,
        match=r"config missing required field: precision\.float32_precision",
    ):
        _section(block)


def test_float32_precision_rejects_unknown_modes():
    block = _plain_precision()
    block["float32_precision"] = "fp32"

    with pytest.raises(ValueError, match=r"precision\.float32_precision must be one of"):
        PrecisionPolicy.from_section(_section(block))


def test_scalar_precision_is_rejected_with_migration_path():
    with pytest.raises(ValueError, match="scalar `precision` is no longer supported"):
        _section("bf16")


# -- removed legacy config --------------------------------------------


def test_top_level_precision_is_required():
    """There is no implicit default: a config with no precision block must not load."""
    with pytest.raises(ValueError, match="top-level `precision` is required"):
        PrecisionPolicy.from_section(_section())


# -- role dtype versus selective quantization -------------------------


def test_fp8_rollout_split_keeps_bf16_base_and_prompt_default():
    block = _plain_precision()
    block["rollout"]["quantization"] = {"format": "fp8", "recipe": "rowwise"}
    p = PrecisionPolicy.from_section(_section(block))
    assert p.training == RolePrecision(dtype="bf16", float32_precision="tf32")
    assert p.rollout == RolePrecision(
        dtype="bf16",
        float32_precision="tf32",
        quantization=QuantizationPolicy(format="fp8", recipe="rowwise"),
    )
    assert p.prompt_encoder_dtype == "bf16"
    assert p.diffusion_math == "fp32"
    assert p.rollout.label == "bf16+fp8"
    assert p.stages_match is False


def test_quantized_role_label_preserves_its_base_dtype():
    block = _plain_precision()
    block["rollout"] = {
        "dtype": "fp16",
        "quantization": {"format": "fp8"},
    }

    p = PrecisionPolicy.from_section(_section(block))

    assert p.rollout.label == "fp16+fp8"


def test_nvfp4_rollout_is_a_recipe_free_selective_policy():
    block = _plain_precision()
    block["rollout"]["quantization"] = {"format": "nvfp4"}
    p = PrecisionPolicy.from_section(_section(block))
    assert p.rollout.quantization == QuantizationPolicy(format="nvfp4")
    assert p.rollout.quantization.recipe is None
    assert p.rollout.label == "bf16+nvfp4"
    assert p.stages_match is False


def test_legacy_generic_fp4_format_has_a_migration_error():
    block = _plain_precision()
    block["rollout"]["quantization"] = {"format": "fp4"}
    with pytest.raises(ValueError, match="use `format: nvfp4`"):
        PrecisionPolicy.from_section(_section(block))


def test_nvfp4_rejects_every_recipe():
    block = _plain_precision()
    block["rollout"]["quantization"] = {"format": "nvfp4", "recipe": "rowwise"}
    with pytest.raises(ValueError, match="does not accept a recipe"):
        PrecisionPolicy.from_section(_section(block))


@pytest.mark.parametrize("recipe", ["rowwise", "tensorwise", "blockwise"])
def test_fp8_accepts_only_declared_recipes(recipe):
    block = _plain_precision()
    block["rollout"]["quantization"] = {"format": "fp8", "recipe": recipe}
    p = PrecisionPolicy.from_section(_section(block))
    assert p.rollout.quantization == QuantizationPolicy(format="fp8", recipe=recipe)


def test_fp8_omitted_recipe_resolves_to_rowwise():
    block = _plain_precision()
    block["rollout"]["quantization"] = {"format": "fp8"}
    p = PrecisionPolicy.from_section(_section(block))
    assert p.rollout.quantization == QuantizationPolicy(format="fp8", recipe="rowwise")


def test_fp8_rejects_unknown_recipe_during_config_resolution():
    block = _plain_precision()
    block["rollout"]["quantization"] = {"format": "fp8", "recipe": "nvfp4"}
    with pytest.raises(ValueError, match="invalid for format 'fp8'"):
        PrecisionPolicy.from_section(_section(block))


def test_quantization_format_is_rejected_in_dtype_position():
    block = _plain_precision()
    block["rollout"]["dtype"] = "fp8"
    with pytest.raises(ValueError, match=r"belongs under a `quantization\.format` key"):
        PrecisionPolicy.from_section(_section(block))


def test_training_quantization_parses_but_fails_without_runtime():
    block = _plain_precision()
    block["training"]["quantization"] = {"format": "fp8"}
    with pytest.raises(ValueError, match=r"training\.quantization is unavailable"):
        PrecisionPolicy.from_section(_section(block))


def test_quantization_requires_format():
    block = _plain_precision()
    block["rollout"]["quantization"] = {"recipe": "rowwise"}
    with pytest.raises(
        ValueError,
        match=r"config missing required field: precision\.rollout\.quantization\.format",
    ):
        _section(block)


@pytest.mark.parametrize(
    ("spelling", "torch_name"),
    [
        ("fp8", "float8_e4m3fn"),
        ("e4m3", "float8_e4m3fn"),
        ("e5m2", "float8_e5m2"),
        ("fp4", "float4_e2m1fn_x2"),
    ],
)
def test_resolve_torch_dtype_subbyte(spelling, torch_name):
    """Wire dtype parsing is independent from FP4 policy availability."""
    assert resolve_torch_dtype(spelling) is getattr(torch, torch_name)


def test_shipped_online_recipes_keep_training_and_rollout_precision_aligned():
    """No bundled online recipe ships a rollout precision split by accident."""

    for name in _experiment_names():
        if Path(name).name.startswith("offline_"):
            continue
        policy = PrecisionPolicy.from_section(
            parse_config(_load_experiment_for_static_validation(name)).precision,
        )
        assert policy.training == policy.rollout, name
        assert policy.diffusion_math == "fp32", name
