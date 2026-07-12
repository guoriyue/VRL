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
    assert p == PrecisionPolicy(train="bf16", rollout="bf16", math="fp32", frozen="bf16")


def test_scalar_fp32_keeps_frozen_fp16():
    """Checks scalar FP32 keeps frozen FP16."""
    p = resolve_precision_policy(_cfg(precision="fp32"))
    assert p == PrecisionPolicy(train="fp32", rollout="fp32", math="fp32", frozen="fp16")


def test_rollout_split_is_honored_legacy_forward_still_flagged():
    """`rollout` is the experimental split key: it overrides the training-forward
    dtype on the rollout side (train/replay stays on `train`). The dropped legacy
    `forward` key is still reported by the whole-tree walker; `rollout` is not."""
    from omegaconf import OmegaConf

    from vrl.config.unknown_keys import find_unknown_keys

    block = {"train": "bf16", "rollout": "fp32", "forward": "fp16"}
    p = resolve_precision_policy(_cfg(precision=block))
    assert p.train == "bf16"  # replay/training forward follows `train`
    assert p.rollout == "fp32"  # explicit rollout override is honored (the split)
    unknown = find_unknown_keys(OmegaConf.create({"precision": block}))
    assert "precision.forward" in unknown  # dropped legacy key still flagged
    assert "precision.rollout" not in unknown  # now a valid override


def test_math_protected_unless_explicit():
    """Checks math protected unless explicit."""
    p = resolve_precision_policy(_cfg(precision={"train": "bf16", "math": "bf16"}))
    assert p.math == "bf16"  # only when explicitly asked
    assert p.train == "bf16"
    assert p.rollout == "bf16"


# -- removed legacy config --------------------------------------------


def test_top_level_precision_is_required():
    """Checks missing top-level precision is rejected."""
    with pytest.raises(ValueError, match="top-level `precision` is required"):
        resolve_precision_policy(_cfg())


# -- fp8/fp4 rollout split (the quantized-rollout precision axis) ------------


def test_fp8_rollout_split_keeps_replay_bf16_and_frozen_fp16():
    """{train: bf16, rollout: fp8} → bf16 replay, fp8 rollout, fp16 frozen, fp32 math."""
    p = resolve_precision_policy(_cfg(precision={"train": "bf16", "rollout": "fp8"}))
    assert p == PrecisionPolicy(train="bf16", rollout="fp8", math="fp32", frozen="fp16")


def test_fp4_rollout_split_allowed():
    """fp4 is a valid rollout-axis token (Blackwell)."""
    p = resolve_precision_policy(_cfg(precision={"train": "bf16", "rollout": "fp4"}))
    assert p.rollout == "fp4"
    assert p.train == "bf16"
    assert p.rollout_quantization == "fp4"
    assert p.rollout_storage_precision == "bf16"


@pytest.mark.parametrize("token", ["fp8", "fp4"])
def test_scalar_subbyte_precision_rejected(token):
    """A scalar fp8/fp4 would set the replay forward sub-byte — rejected."""
    with pytest.raises(ValueError, match="rollout-only"):
        resolve_precision_policy(_cfg(precision=token))


@pytest.mark.parametrize("token", ["fp8", "fp4"])
@pytest.mark.parametrize("axis", ["train", "math", "frozen"])
def test_subbyte_on_non_rollout_axis_rejected(axis, token):
    """fp8/fp4 is only valid on the rollout axis, never a storage/math axis."""
    with pytest.raises(ValueError, match="invalid"):
        resolve_precision_policy(_cfg(precision={axis: token, "rollout": token}))


def test_rollout_recipe_parsed_with_quantized_rollout():
    """rollout_recipe rides along with an fp8 rollout; absent → None (scheme default)."""
    p = resolve_precision_policy(
        _cfg(precision={"train": "bf16", "rollout": "fp8", "rollout_recipe": "blockwise"}),
    )
    assert p.rollout_recipe == "blockwise"
    p = resolve_precision_policy(_cfg(precision={"train": "bf16", "rollout": "fp8"}))
    assert p.rollout_recipe is None
    p = resolve_precision_policy(_cfg(precision="bf16"))
    assert p.rollout_recipe is None


def test_rollout_recipe_without_quantized_rollout_rejected():
    """A recipe on a plain-dtype rollout would be a silent no-op knob — rejected."""
    with pytest.raises(ValueError, match="rollout_recipe"):
        resolve_precision_policy(
            _cfg(precision={"train": "bf16", "rollout_recipe": "blockwise"}),
        )


def test_rollout_recipe_key_known_to_walker():
    """precision.rollout_recipe is a declared block key, not an unknown-key warning."""
    from omegaconf import OmegaConf

    from vrl.config.unknown_keys import find_unknown_keys

    block = {"train": "bf16", "rollout": "fp8", "rollout_recipe": "blockwise"}
    assert "precision.rollout_recipe" not in find_unknown_keys(
        OmegaConf.create({"precision": block})
    )


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
    """Sub-byte spellings resolve to the matching torch dtype (torch 2.11 has them)."""
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


# Every online GRPO recipe must keep rollout/replay forward precision aligned.
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

    assert policy.train == policy.rollout
    assert resolve_torch_dtype(policy.train) is resolve_torch_dtype(policy.rollout)
    assert "mixed_precision" not in cfg.actor
    assert "bf16" not in cfg.actor
    # math is always protected at fp32.
    assert policy.math == "fp32"
