"""Integration: the unified precision policy drives the online trainer (P1)."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from vrl.algorithms.logprob_mismatch import PrecisionCorrectionConfig
from vrl.config.builders import BuiltConfigs
from vrl.config.loading import load_config
from vrl.config.precision import PrecisionPolicy
from vrl.config.schema import parse_config
from vrl.models.dtypes import resolve_torch_dtype


def test_fp16_precision_block_drives_trainer_and_rollout():
    """Checks scalar fp16 drives both replay compute and rollout precision."""
    cfg = load_config("experiment/sd3_5/online_grpo_ocr")
    cfg = OmegaConf.merge(cfg, OmegaConf.create({"precision": _plain_policy("fp16")}))

    built = BuiltConfigs.from_cfg(cfg)

    assert built.precision.training.label == "fp16"
    assert built.precision.rollout.label == "fp16"
    assert built.precision.denoise_math == "fp32"


@pytest.mark.parametrize(
    ("format_name", "expected_label"),
    [("fp8", "bf16+fp8"), ("nvfp4", "bf16+nvfp4")],
)
def test_rollout_precision_split_auto_derives_correction_policy(
    format_name: str,
    expected_label: str,
):
    """A low-precision rollout split is a user intent; correction is derived."""
    cfg = _with_precision(
        "sd3_5/online_grpo_ocr",
        _rollout_quantization_policy(format_name),
    )

    built = BuiltConfigs.from_cfg(cfg)
    trainer_config = built.trainer

    assert built.precision.training.label == "bf16"
    assert built.precision.rollout.label == expected_label
    # The auto split-precision policy is whatever the builder helper installs;
    # assert the whole struct equals that single source, not a per-field copy of
    # its constants (which would falsely fail on any retune of the policy).
    assert trainer_config.precision_correction == PrecisionCorrectionConfig(
        tis_mode="truncate",
        rs_mode="seq_mean_k1",
    )


def test_outer_autocast_split_is_preserved_in_trainer_role_labels() -> None:
    block = _plain_policy("bf16")
    block["rollout"]["outer_autocast"] = False
    cfg = _with_precision("sd3_5/online_grpo_ocr", block)

    built = BuiltConfigs.from_cfg(cfg)

    assert built.precision.training.label == "bf16"
    assert built.precision.rollout.label == "bf16+no-autocast"
    assert built.trainer.precision_correction.tis_mode == "truncate"


def test_explicit_precision_correction_is_respected_on_rollout_split():
    """Expert correction blocks override the auto split-precision defaults."""
    cfg = _with_precision(
        "sd3_5/online_grpo_ocr",
        _rollout_fp8_policy(),
    )
    cfg = OmegaConf.merge(
        cfg,
        OmegaConf.create(
            {
                "trainer": {
                    "precision_correction": {"tis_mode": "off", "rs_mode": "off"},
                },
            },
        ),
    )

    trainer_config = BuiltConfigs.from_cfg(cfg).trainer

    assert trainer_config.precision_correction.tis_mode == "off"
    assert trainer_config.precision_correction.rs_mode == "off"


def _with_precision(experiment, block):
    cfg = load_config(f"experiment/{experiment}")
    return OmegaConf.merge(cfg, OmegaConf.create({"precision": block}))


def _plain_policy(dtype: str) -> dict:
    return {
        "float32_precision": "tf32",
        "training": {"dtype": dtype},
        "rollout": {"dtype": dtype},
    }


def _rollout_fp8_policy() -> dict:
    return _rollout_quantization_policy("fp8")


def _rollout_quantization_policy(format_name: str) -> dict:
    block = _plain_policy("bf16")
    block["rollout"]["quantization"] = {"format": format_name}
    return block


@pytest.mark.parametrize("math,expected", [("fp32", torch.float32), ("bf16", torch.bfloat16)])
def test_math_axis_resolves_to_dtype(math, expected):
    # P2: the `math` axis resolves to the evaluator's log-prob math dtype.

    block = _plain_policy("fp32")
    block["denoise_math"] = {"dtype": math}
    cfg = _with_precision("sd3_5/online_grpo_ocr", block)
    assert (
        resolve_torch_dtype(PrecisionPolicy.from_section(parse_config(cfg).precision).denoise_math)
        is expected
    )
    assert BuiltConfigs.from_cfg(cfg).precision.denoise_math == math


@pytest.mark.parametrize(
    "prompt_encoder,expected",
    [("fp16", torch.float16), ("bf16", torch.bfloat16)],
)
def test_prompt_encoder_axis_in_model_build(prompt_encoder, expected):
    """Prompt-encoder precision reaches the runtime as a real torch dtype."""
    # sd3_5 is a registry-descriptor family: its build comes from the generic
    # resolver (family resolved from cfg.model.family).
    from vrl.models.families.registry import get_model_family_entry

    block = _plain_policy("fp32")
    block["rollout"]["prompt_encoders"] = {"dtype": prompt_encoder}
    cfg = _with_precision("sd3_5/online_grpo_ocr", block)
    built = BuiltConfigs.from_cfg(cfg)
    build = get_model_family_entry("sd3_5").resolve_model_build(
        built.root,
        torch.device("cpu"),
        precision=built.precision,
    )
    assert build.family == "sd3_5"
    assert build.rollout is not None
    assert build.rollout.prompt_encoder_dtype is expected


def test_family_parameter_dtype_is_derived_from_runtime_role() -> None:
    """Replay follows train while rollout follows the plain rollout precision."""
    from vrl.models.families.registry import get_model_family_entry

    cfg = _with_precision(
        "sd3_5/online_grpo_ocr",
        {
            "training": {"dtype": "bf16"},
            "rollout": {"dtype": "fp16"},
        },
    )

    built = BuiltConfigs.from_cfg(cfg)
    entry = get_model_family_entry("sd3_5")
    rollout = entry.resolve_model_build(
        built.root,
        torch.device("cpu"),
        precision=built.precision,
    )
    replay = entry.resolve_model_build(
        built.root,
        torch.device("cpu"),
        precision=built.precision,
        for_rollout=False,
    )

    assert rollout.parameter_dtype is torch.float16
    assert replay.parameter_dtype is torch.bfloat16
