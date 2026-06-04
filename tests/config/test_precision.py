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
from vrl.trainers.precision import (
    torch_dtype_for_trainer_precision,
    trainer_mixed_precision,
)


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
    assert normalize_precision(value) == expected


def test_normalize_rejects_unknown():
    with pytest.raises(ValueError):
        normalize_precision("int8")
    with pytest.raises(ValueError):
        normalize_precision("bfloat16")  # no longer an accepted alias


def test_resolve_torch_dtype():
    assert resolve_torch_dtype("fp32") is torch.float32
    assert resolve_torch_dtype("bf16") is torch.bfloat16
    assert resolve_torch_dtype("fp16") is torch.float16


# -- scalar / dict block ----------------------------------------------


def test_scalar_bf16_expands_all_compute_axes():
    p = resolve_precision_policy(_cfg(precision="bf16"))
    assert p == PrecisionPolicy(compute="bf16", rollout="bf16", math="fp32", frozen="bf16")


def test_scalar_fp32_keeps_frozen_fp16():
    p = resolve_precision_policy(_cfg(precision="fp32"))
    assert p == PrecisionPolicy(compute="fp32", rollout="fp32", math="fp32", frozen="fp16")


def test_dict_decoupled_rollout():
    # The validated "bf16 replay / fp32 rollout" split.
    p = resolve_precision_policy(_cfg(precision={"compute": "bf16", "rollout": "fp32"}))
    assert p.compute == "bf16"
    assert p.rollout == "fp32"
    assert p.math == "fp32"  # protected default
    assert p.frozen == "fp16"  # derived from fp32 rollout


def test_math_protected_unless_explicit():
    p = resolve_precision_policy(_cfg(precision={"compute": "bf16", "math": "bf16"}))
    assert p.math == "bf16"  # only when explicitly asked


# -- back-compat (no precision block) ---------------------------------


def test_legacy_fp32():
    p = resolve_precision_policy(_cfg(mixed_precision="no", bf16=False))
    assert p == PrecisionPolicy(compute="fp32", rollout="fp32", math="fp32", frozen="fp16")


def test_legacy_bf16_explicit():
    p = resolve_precision_policy(_cfg(mixed_precision="bf16", bf16=True))
    assert p == PrecisionPolicy(compute="bf16", rollout="bf16", math="fp32", frozen="bf16")


def test_legacy_empty_defaults_to_bf16_flag():
    # mixed_precision unset -> follow bf16 flag (TrainerConfig default True).
    assert resolve_precision_policy(_cfg(bf16=True)).compute == "bf16"
    assert resolve_precision_policy(_cfg(bf16=False)).compute == "fp32"


# -- equivalence vs existing derivation (the zero-behavior-change proof) --


@pytest.mark.parametrize("mp,bf16", [("no", False), ("bf16", True), ("fp16", False), ("", True), ("", False)])
def test_compute_matches_legacy_weight_dtype(mp, bf16):
    cfg = _cfg(mixed_precision=mp, bf16=bf16)
    policy = resolve_precision_policy(cfg)
    legacy = torch_dtype_for_trainer_precision(cfg.actor, torch)
    # compute and rollout both map to the single legacy weight/compute dtype today.
    assert resolve_torch_dtype(policy.compute) is legacy
    assert resolve_torch_dtype(policy.rollout) is legacy


# Every online GRPO recipe — the resolver (back-compat path) must reproduce the
# current dtype derivation on every axis, byte-for-byte, before P1 wiring.
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
    "diffusion/sd3_5/online_grpo_ocr_prompt_alignment",
    "diffusion/sd3_5/online_grpo_pickscore",
    "diffusion/wan_2_1/online_grpo_kling_video_reward",
    "diffusion/wan_2_1/online_grpo_ocr",
    "diffusion/wan_2_1/online_grpo_physics",
    "diffusion/wan_2_1/online_grpo_physics_i2v",
]


@pytest.mark.parametrize("experiment", _ONLINE_RECIPES)
def test_online_recipe_equivalence(experiment):
    cfg = load_config(f"experiment/{experiment}")
    policy = resolve_precision_policy(cfg)

    weight_dtype = torch_dtype_for_trainer_precision(cfg.actor, torch)  # legacy storage
    autocast = trainer_mixed_precision(cfg.actor)                       # legacy "no/bf16/fp16"
    frozen = torch.float16 if weight_dtype is torch.float32 else weight_dtype

    # compute/rollout both map to the single legacy weight/compute dtype today.
    assert resolve_torch_dtype(policy.compute) is weight_dtype
    assert resolve_torch_dtype(policy.rollout) is weight_dtype
    # autocast on/off boundary is preserved (fp32 <-> "no").
    assert (policy.compute == "fp32") == (autocast == "no")
    # frozen reproduces the sd3 "fp32 -> fp16, else follow" derivation.
    assert resolve_torch_dtype(policy.frozen) is frozen
    # math is always protected at fp32.
    assert policy.math == "fp32"
