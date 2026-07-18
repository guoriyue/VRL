from __future__ import annotations

import sys

import pytest
import torch

from vrl.families.registry import get_model_family_entry
from vrl.scripts.diffusion import generate


def test_generate_rejects_ar_family_before_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate",
            "--family",
            "emu3",
            "--path",
            "unused",
            "--dtype",
            "bf16",
            "--float32-precision",
            "tf32",
            "--outer-autocast",
        ],
    )

    with pytest.raises(SystemExit, match="emu3 is an AR family"):
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
    assert not hasattr(rollout, "quantization")
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
    assert not hasattr(rollout, "quantization")
    assert rollout.base_weight_sync is False
