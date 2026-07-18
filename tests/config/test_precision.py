"""Tests for the unified precision policy resolver (P0)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.config.test_load_all_experiments import _experiment_names
from vrl.config.loading import load_config
from vrl.config.precision import (
    PrecisionPolicy,
    QuantizationPolicy,
    RolePrecision,
    normalize_precision,
    resolve_precision_policy,
)
from vrl.models.dtypes import (
    dtype_to_precision_token,
    dtype_to_wire_name,
    resolve_torch_dtype,
)


def _cfg(precision=None, mixed_precision=None, bf16=None, allow_tf32=None):
    actor = {}
    if mixed_precision is not None:
        actor["mixed_precision"] = mixed_precision
    if bf16 is not None:
        actor["bf16"] = bf16
    if allow_tf32 is not None:
        actor["optim"] = {"allow_tf32": allow_tf32}
    top = {"actor": actor}
    if precision is not None:
        top["precision"] = precision
    return SimpleNamespace(
        precision=top.get("precision"),
        actor=SimpleNamespace(
            **{
                key: SimpleNamespace(**value) if isinstance(value, dict) else value
                for key, value in actor.items()
            },
        ),
    )


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
    """Checks normalize precision."""
    assert normalize_precision(value) == expected


def test_normalize_rejects_unknown():
    """Checks normalize rejects unknown."""
    with pytest.raises(ValueError):
        normalize_precision("int8")
    with pytest.raises(ValueError):
        normalize_precision("bfloat16")  # no longer an accepted alias


def test_resolve_torch_dtype():
    """Checks resolve torch dtype."""
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


def test_nested_bf16_resolves_role_dtypes_and_protected_defaults():
    p = resolve_precision_policy(
        _cfg(
            precision={
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
    p = resolve_precision_policy(
        _cfg(
            precision={
                "float32_precision": "tf32",
                "training": {"dtype": "bf16"},
                "rollout": {"quantization": {"format": "fp8"}},
            },
        ),
    )

    assert p.rollout.dtype == "bf16"
    assert p.rollout.label == "bf16+fp8"


def test_prompt_encoders_default_to_rollout_dtype_even_for_fp32():
    p = resolve_precision_policy(_cfg(precision=_plain_precision("fp32")))
    assert p.prompt_encoder_dtype == "fp32"


def test_diffusion_math_and_prompt_encoders_can_be_explicit():
    block = _plain_precision()
    block["diffusion_math"] = {"dtype": "bf16"}
    block["rollout"]["prompt_encoders"] = {"dtype": "fp16"}
    p = resolve_precision_policy(_cfg(precision=block))
    assert p.diffusion_math == "bf16"
    assert p.prompt_encoder_dtype == "fp16"


def test_base_preset_keeps_prompt_encoders_aligned_with_rollout():
    cfg = load_config("experiment/diffusion/sd3_5/online_grpo_ocr")
    p = resolve_precision_policy(cfg)
    assert p.rollout.dtype == "bf16"
    assert p.prompt_encoder_dtype == "bf16"
    assert p.training.float32_precision == "tf32"
    assert p.rollout.float32_precision == "tf32"


@pytest.mark.parametrize("mode", ["ieee", "tf32"])
def test_float32_precision_is_explicit_and_resolved(mode):
    block = _plain_precision()
    block["float32_precision"] = mode

    policy = resolve_precision_policy(_cfg(precision=block))
    assert policy.training.float32_precision == mode
    assert policy.rollout.float32_precision == mode


def test_float32_precision_is_required():
    block = _plain_precision()
    del block["float32_precision"]

    with pytest.raises(ValueError, match=r"precision\.float32_precision is required"):
        resolve_precision_policy(_cfg(precision=block))


@pytest.mark.parametrize("mode", ["", "fp32", "true"])
def test_float32_precision_rejects_unknown_modes(mode):
    block = _plain_precision()
    block["float32_precision"] = mode

    with pytest.raises(ValueError, match=r"precision\.float32_precision must be one of"):
        resolve_precision_policy(_cfg(precision=block))


@pytest.mark.parametrize("allow_tf32", [False, True])
def test_legacy_optimizer_tf32_key_has_migration_error(allow_tf32):
    with pytest.raises(
        ValueError,
        match=r"actor\.optim\.allow_tf32.*precision\.float32_precision",
    ):
        resolve_precision_policy(
            _cfg(precision=_plain_precision(), allow_tf32=allow_tf32),
        )


@pytest.mark.parametrize("scalar", ["bf16", "fp32", "fp8", False])
def test_scalar_precision_is_rejected_with_migration_path(scalar):
    with pytest.raises(ValueError, match="scalar `precision` is no longer supported"):
        resolve_precision_policy(_cfg(precision=scalar))


@pytest.mark.parametrize(
    "block",
    [
        {"train": "bf16", "rollout": "fp8"},
        {"training": {"dtype": "bf16"}, "rollout": "fp16"},
        {"train": "bf16", "math": "fp32", "frozen": "fp16"},
        {"training": {"dtype": "bf16"}, "frozen_components": "fp16"},
        {
            "training": {"dtype": "bf16"},
            "rollout": {"frozen_components": {"dtype": "fp16"}},
        },
        {"training": {"dtype": "bf16"}, "rollout_recipe": "rowwise"},
    ],
)
def test_legacy_precision_keys_are_rejected(block):
    with pytest.raises(ValueError, match="legacy precision key"):
        resolve_precision_policy(_cfg(precision=block))


def test_nested_unknown_keys_are_reported_at_full_path():
    from omegaconf import OmegaConf

    from vrl.config.unknown_keys import find_unknown_keys

    block = _plain_precision()
    block["rollout"]["quantization"] = {
        "format": "fp8",
        "recipe": "rowwise",
        "typo": True,
    }
    block["forward"] = "fp16"
    unknown = find_unknown_keys(OmegaConf.create({"precision": block}))
    assert unknown == [
        "precision.forward",
        "precision.rollout.quantization.typo",
    ]


def test_prompt_encoder_keys_are_derived_from_precision_config_dataclasses():
    from omegaconf import OmegaConf

    from vrl.config.unknown_keys import find_unknown_keys

    block = _plain_precision()
    block["rollout"]["prompt_encoders"] = {"dtype": "fp16"}
    assert find_unknown_keys(OmegaConf.create({"precision": block})) == []


# -- removed legacy config --------------------------------------------


def test_top_level_precision_is_required():
    """Checks missing top-level precision is rejected."""
    with pytest.raises(ValueError, match="top-level `precision` is required"):
        resolve_precision_policy(_cfg())


# -- role dtype versus selective quantization -------------------------


def test_fp8_rollout_split_keeps_bf16_base_and_prompt_default():
    block = _plain_precision()
    block["rollout"]["quantization"] = {"format": "fp8", "recipe": "rowwise"}
    p = resolve_precision_policy(_cfg(precision=block))
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

    p = resolve_precision_policy(_cfg(precision=block))

    assert p.rollout.label == "fp16+fp8"


def test_nvfp4_rollout_is_a_recipe_free_selective_policy():
    block = _plain_precision()
    block["rollout"]["quantization"] = {"format": "nvfp4"}
    p = resolve_precision_policy(_cfg(precision=block))
    assert p.rollout.quantization == QuantizationPolicy(format="nvfp4")
    assert p.rollout.quantization.recipe is None
    assert p.rollout.label == "bf16+nvfp4"
    assert p.stages_match is False


def test_legacy_generic_fp4_format_has_a_migration_error():
    block = _plain_precision()
    block["rollout"]["quantization"] = {"format": "fp4"}
    with pytest.raises(ValueError, match="use `format: nvfp4`"):
        resolve_precision_policy(_cfg(precision=block))


@pytest.mark.parametrize("recipe", ["nvfp4", "rowwise", "tensorwise", "blockwise"])
def test_nvfp4_rejects_every_recipe(recipe):
    block = _plain_precision()
    block["rollout"]["quantization"] = {
        "format": "nvfp4",
        "recipe": recipe,
    }
    with pytest.raises(ValueError, match="does not accept a recipe"):
        resolve_precision_policy(_cfg(precision=block))


@pytest.mark.parametrize("recipe", ["rowwise", "tensorwise", "blockwise"])
def test_fp8_accepts_only_declared_recipes(recipe):
    block = _plain_precision()
    block["rollout"]["quantization"] = {"format": "fp8", "recipe": recipe}
    p = resolve_precision_policy(_cfg(precision=block))
    assert p.rollout.quantization == QuantizationPolicy(format="fp8", recipe=recipe)


def test_fp8_omitted_recipe_resolves_to_rowwise():
    block = _plain_precision()
    block["rollout"]["quantization"] = {"format": "fp8"}
    p = resolve_precision_policy(_cfg(precision=block))
    assert p.rollout.quantization == QuantizationPolicy(format="fp8", recipe="rowwise")


def test_fp8_rejects_unknown_recipe_during_config_resolution():
    block = _plain_precision()
    block["rollout"]["quantization"] = {"format": "fp8", "recipe": "nvfp4"}
    with pytest.raises(ValueError, match="invalid for format 'fp8'"):
        resolve_precision_policy(_cfg(precision=block))


@pytest.mark.parametrize("format_name", ["fp8", "nvfp4"])
@pytest.mark.parametrize("role", ["training", "rollout"])
def test_quantization_format_is_rejected_in_dtype_position(role, format_name):
    block = _plain_precision()
    block[role]["dtype"] = format_name
    with pytest.raises(ValueError, match=r"belongs under a `quantization\.format` key"):
        resolve_precision_policy(_cfg(precision=block))


@pytest.mark.parametrize(
    "quantization",
    [
        {"format": "fp8", "recipe": "rowwise"},
        {"format": "nvfp4"},
    ],
)
def test_training_quantization_parses_but_fails_without_runtime(quantization):
    block = _plain_precision()
    block["training"]["quantization"] = quantization
    with pytest.raises(ValueError, match=r"training\.quantization is unavailable"):
        resolve_precision_policy(_cfg(precision=block))


def test_quantization_requires_format():
    block = _plain_precision()
    block["rollout"]["quantization"] = {"recipe": "rowwise"}
    with pytest.raises(
        ValueError,
        match=r"precision\.rollout\.quantization\.format is required",
    ):
        resolve_precision_policy(_cfg(precision=block))


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


def test_legacy_actor_precision_keys_warn_via_schema(caplog):
    """actor.mixed_precision/bf16 are plain unknown keys now: warn, still load."""
    import logging

    from vrl.config.loading import load_config
    from vrl.config.validation import validate_training_config

    cfg = load_config(
        "experiment/diffusion/sd3_5/online_grpo_ocr",
        overrides=["actor.mixed_precision=bf16"],
    )
    with caplog.at_level(logging.WARNING):
        validate_training_config(cfg)
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "mixed_precision" in joined


# Every online GRPO recipe must keep rollout/replay role precision aligned.
# Derive the list from the experiment glob (the single source of truth in
# test_load_all_experiments) instead of hand-maintaining it — a hand list
# silently drops new recipes (it had already drifted, missing
# cosmos_predict2_5/online_nft_motion_physics). "online" == every experiment
# whose final path component is not an `offline_*` recipe.
def _online_recipes() -> list[str]:
    return [name for name in _experiment_names() if not Path(name).name.startswith("offline_")]


_ONLINE_RECIPES = _online_recipes()


@pytest.mark.parametrize("experiment", _ONLINE_RECIPES)
def test_online_recipe_equivalence(experiment):
    """Checks online recipes use aligned public precision."""
    cfg = load_config(f"experiment/{experiment}")
    policy = resolve_precision_policy(cfg)

    assert policy.training == policy.rollout
    assert resolve_torch_dtype(policy.training.dtype) is resolve_torch_dtype(
        policy.rollout.dtype,
    )
    assert "mixed_precision" not in cfg.actor
    assert "bf16" not in cfg.actor
    # math is always protected at fp32.
    assert policy.diffusion_math == "fp32"
