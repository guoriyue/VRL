from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from vrl.models.families.registry import get_model_family_entry
from vrl.scripts.generation import full_sequence_probe as generate


@pytest.mark.parametrize("value", ["nan", "inf", "-1"])
def test_replay_tolerance_rejects_invalid_values(value):
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        generate._nonnegative_finite(value)


@pytest.mark.parametrize("pred,lp", [(float("nan"), 0), (0, float("inf")), (0, 0.002)])
def test_replay_guard_fails_closed(pred, lp):
    with pytest.raises(SystemExit, match="FAIL"):
        generate._check_replay_errors(pred, lp, pred_atol=1e-3, lp_atol=1e-3)


def test_replay_guard_accepts_threshold_boundary():
    generate._check_replay_errors(1e-3, 0, pred_atol=1e-3, lp_atol=1e-3)


def test_probe_uses_existing_cosmos_lora_preset():
    preset = (
        Path(__file__).resolve().parents[2] / "vrl/config/presets/model/cosmos/predict2_5_2b.yaml"
    )
    args = generate._build_arg_parser().parse_args(
        [
            "--family",
            "cosmos-predict2.5",
            "--path",
            "/local/pinned/cosmos",
            "--dtype",
            "bf16",
            "--float32-precision",
            "ieee",
            "--outer-autocast",
            "--model-preset",
            str(preset),
        ]
    )
    build = generate._resolve_probe_model_build(
        args, get_model_family_entry("cosmos-predict2.5"), torch.device("cpu")
    )
    assert build.use_lora
    assert build.lora.rank == 32
    assert build.model_name_or_path == "/local/pinned/cosmos"
    assert build.model_config["skip_text_encoder"] is False


def test_generate_rejects_non_full_sequence_family_before_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate",
            "--family",
            "magi_1",
            "--path",
            "unused",
            "--dtype",
            "bf16",
            "--float32-precision",
            "tf32",
            "--outer-autocast",
        ],
    )

    with pytest.raises(SystemExit, match="does not expose a full-sequence denoise policy"):
        generate.main()


def test_probe_model_build_uses_family_parameter_and_public_precision_policy() -> None:
    args = generate._build_arg_parser().parse_args(
        [
            "--family",
            "sana",
            "--path",
            "unused",
            "--dtype",
            "fp16",
            "--float32-precision",
            "ieee",
            "--no-outer-autocast",
        ],
    )

    build = generate._resolve_probe_model_build(
        args,
        get_model_family_entry("sana"),
        torch.device("cpu"),
    )

    assert build.parameter_dtype is torch.float16
    rollout = build.require_rollout()
    assert build.precision.dtype == "fp16"
    assert build.precision.float32_precision == "ieee"
    assert build.precision.quantization is None
    assert build.precision.outer_autocast is False
    assert rollout.prompt_encoder_dtype is torch.float16
    assert rollout.base_weight_sync is False


def test_probe_model_build_uses_explicit_precision_without_family_override() -> None:
    args = generate._build_arg_parser().parse_args(
        [
            "--family",
            "sana",
            "--path",
            "unused",
            "--dtype",
            "fp16",
            "--float32-precision",
            "tf32",
            "--outer-autocast",
        ],
    )

    build = generate._resolve_probe_model_build(
        args,
        get_model_family_entry("sana"),
        torch.device("cpu"),
    )

    assert build.precision.float32_precision == "tf32"
    assert build.precision.outer_autocast is True


@pytest.mark.parametrize("quantization_format", ["fp8", "nvfp4"])
def test_probe_model_build_derives_quantization_from_role_precision(
    quantization_format: str,
) -> None:
    args = generate._build_arg_parser().parse_args(
        [
            "--family",
            "sana",
            "--path",
            "unused",
            "--dtype",
            "fp16",
            "--float32-precision",
            "ieee",
            "--no-outer-autocast",
            "--quantize",
            quantization_format,
        ],
    )

    build = generate._resolve_probe_model_build(
        args,
        get_model_family_entry("sana"),
        torch.device("cpu"),
    )

    assert build.parameter_dtype is torch.float16
    rollout = build.require_rollout()
    assert build.precision.dtype == "fp16"
    assert build.precision.float32_precision == "ieee"
    assert build.precision.quantization is not None
    assert build.precision.quantization.format == quantization_format
    assert build.precision.outer_autocast is False
    assert rollout.prompt_encoder_dtype is torch.float16
    assert rollout.base_weight_sync is False
