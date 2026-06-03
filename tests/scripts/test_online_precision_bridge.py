"""Integration: the unified precision policy drives the online trainer (P1)."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from vrl.config.builders import build_configs
from vrl.config.loading import load_config
from vrl.config.precision import resolve_precision_policy
from vrl.scripts.common.online import _apply_precision_policy
from vrl.trainers.precision import torch_dtype_for_trainer_precision

_RECIPES = [
    "diffusion/sd3_5/online_grpo_ocr",          # fp32
    "diffusion/sd3_5/online_grpo_geneval",      # fp32
    "diffusion/wan_2_1/online_grpo_ocr",        # bf16
    "ar/janus_pro/online_grpo_ocr",             # bf16
]


@pytest.mark.parametrize("experiment", _RECIPES)
def test_bridge_preserves_legacy_dtype(experiment):
    cfg = load_config(f"experiment/{experiment}")
    trainer_config = build_configs(cfg)["trainer"]
    legacy = torch_dtype_for_trainer_precision(trainer_config, torch)
    _apply_precision_policy(cfg, trainer_config)
    assert torch_dtype_for_trainer_precision(trainer_config, torch) is legacy


def test_precision_block_drives_trainer():
    # An fp32 recipe + a top-level `precision: bf16` must flip the trainer to bf16.
    cfg = load_config("experiment/diffusion/sd3_5/online_grpo_ocr")
    assert torch_dtype_for_trainer_precision(build_configs(cfg)["trainer"], torch) is torch.float32

    cfg = OmegaConf.merge(cfg, OmegaConf.create({"precision": "bf16"}))
    trainer_config = build_configs(cfg)["trainer"]
    _apply_precision_policy(cfg, trainer_config)
    assert trainer_config.mixed_precision == "bf16"
    assert trainer_config.bf16 is True
    assert torch_dtype_for_trainer_precision(trainer_config, torch) is torch.bfloat16


def _with_precision(experiment, block):
    cfg = load_config(f"experiment/{experiment}")
    return OmegaConf.merge(cfg, OmegaConf.create({"precision": block}))


def test_decoupled_rollout_compute():
    # P3: bf16 replay / fp32 rollout — compute drives the trainer, rollout is
    # a separate generation dtype (no longer refused).
    cfg = _with_precision(
        "diffusion/sd3_5/online_grpo_ocr", {"compute": "bf16", "rollout": "fp32"},
    )
    trainer_config = build_configs(cfg)["trainer"]
    _apply_precision_policy(cfg, trainer_config)  # must not raise
    assert trainer_config.mixed_precision == "bf16"
    assert resolve_precision_policy(cfg).rollout == "fp32"


@pytest.mark.parametrize("math,expected", [("fp32", torch.float32), ("bf16", torch.bfloat16)])
def test_math_axis_threaded_to_evaluator(math, expected):
    # P2: the evaluator's log-prob math dtype follows the `math` axis, resolved
    # through the shared `resolve_axis_dtype` util.
    from vrl.config.precision import resolve_axis_dtype

    cfg = _with_precision("diffusion/sd3_5/online_grpo_ocr", {"compute": "fp32", "math": math})
    assert resolve_axis_dtype(cfg, "math") is expected


@pytest.mark.parametrize(
    "frozen,expected", [("fp16", torch.float16), ("bf16", torch.bfloat16)],
)
def test_frozen_axis_in_runtime_spec(frozen, expected):
    # P1b: the frozen axis rides the spec as a real torch.dtype (like `dtype`).
    from vrl.models.diffusion.sd3_5.runtime import extract_sd3_5_runtime_spec

    cfg = _with_precision("diffusion/sd3_5/online_grpo_ocr", {"compute": "fp32", "frozen": frozen})
    spec = extract_sd3_5_runtime_spec(cfg, torch.device("cpu"), torch.float32)
    assert spec.frozen_dtype is expected
