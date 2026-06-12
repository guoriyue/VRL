"""Tests for the unified precision policy resolver (P0)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vrl.config.loading import load_config
from vrl.config.precision import (
    PrecisionPolicy,
    normalize_precision,
    resolve_precision_policy,
)
from vrl.models.dtypes import resolve_torch_dtype


def _cfg(precision=None, mixed_precision=None, bf16=None):
    actor = {}
    if mixed_precision is not None:
        actor["mixed_precision"] = mixed_precision
    if bf16 is not None:
        actor["bf16"] = bf16
    top = {"actor": actor}
    if precision is not None:
        top["precision"] = precision
    return SimpleNamespace(
        precision=top.get("precision"),
        actor=SimpleNamespace(**actor),
    )


# -- normalize / convert ----------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("bf16", "bf16"),
        ("fp32", "fp32"),
        ("fp16", "fp16"),
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


# -- scalar / dict block ----------------------------------------------


def test_scalar_bf16_expands_all_compute_axes():
    """Checks scalar bf16 expands all compute axes."""
    p = resolve_precision_policy(_cfg(precision="bf16"))
    assert p == PrecisionPolicy(compute="bf16", rollout="bf16", math="fp32", frozen="bf16")


def test_scalar_fp32_keeps_frozen_fp16():
    """Checks scalar FP32 keeps frozen FP16."""
    p = resolve_precision_policy(_cfg(precision="fp32"))
    assert p == PrecisionPolicy(compute="fp32", rollout="fp32", math="fp32", frozen="fp16")


def test_unknown_precision_keys_are_flagged_and_cannot_split():
    """Unknown precision keys (incl. old compute/rollout) are reported by the
    whole-tree walker and ignored by the parser, so rollout/replay forward
    precision stays coupled via `forward` only."""
    from omegaconf import OmegaConf

    from vrl.config.unknown_keys import find_unknown_keys

    block = {"forward": "bf16", "rollout": "fp32", "compute": "fp16"}
    p = resolve_precision_policy(_cfg(precision=block))
    assert p.compute == "bf16"
    assert p.rollout == "bf16"  # still coupled to forward — split is impossible
    unknown = find_unknown_keys(OmegaConf.create({"precision": block}))
    assert "precision.compute" in unknown
    assert "precision.rollout" in unknown


def test_math_protected_unless_explicit():
    """Checks math protected unless explicit."""
    p = resolve_precision_policy(_cfg(precision={"forward": "bf16", "math": "bf16"}))
    assert p.math == "bf16"  # only when explicitly asked
    assert p.compute == "bf16"
    assert p.rollout == "bf16"


# -- removed legacy config --------------------------------------------


def test_top_level_precision_is_required():
    """Checks missing top-level precision is rejected."""
    with pytest.raises(ValueError, match="top-level `precision` is required"):
        resolve_precision_policy(_cfg())


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


# Every online GRPO recipe must keep rollout/replay forward precision aligned.
_ONLINE_RECIPES = [
    "ar/janus_pro/online_grpo_ocr",
    "ar/janus_pro/online_r1_grpo_ocr",
    "ar/nextstep_1/online_grpo_ocr",
    "diffusion/anima_preview3/online_grpo_aesthetic",
    "diffusion/anima_preview3/online_grpo_aesthetic_nsfw_safety",
    "diffusion/cosmos_predict2_5/online_nft_kling_video_reward",
    "diffusion/cosmos_predict2/online_grpo_kling_video_reward",
    "diffusion/cosmos_predict2/online_grpo_v2w_reference",
    "diffusion/sd3_5/online_grpo_geneval",
    "diffusion/sd3_5/online_grpo_ocr",
    "diffusion/sd3_5/online_grpo_ocr_crossnode_debug",
    "diffusion/sd3_5/online_grpo_pickscore",
    "diffusion/wan_2_1/online_grpo_kling_video_reward",
    "diffusion/wan_2_1/online_grpo_ocr",
    "diffusion/wan_2_1/online_grpo_physics",
    "diffusion/wan_2_1/online_grpo_physics_i2v",
]


@pytest.mark.parametrize("experiment", _ONLINE_RECIPES)
def test_online_recipe_equivalence(experiment):
    """Checks online recipes use aligned public precision."""
    cfg = load_config(f"experiment/{experiment}")
    policy = resolve_precision_policy(cfg)

    assert policy.compute == policy.rollout
    assert resolve_torch_dtype(policy.compute) is resolve_torch_dtype(policy.rollout)
    assert "mixed_precision" not in cfg.actor
    assert "bf16" not in cfg.actor
    # math is always protected at fp32.
    assert policy.math == "fp32"
