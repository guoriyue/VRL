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
        ["generate", "--family", "emu3", "--path", "unused"],
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
            "float32",
        ],
    )

    build = generate._resolve_probe_model_build(
        args,
        get_model_family_entry("sana"),
        torch.device("cpu"),
    )

    assert build.parameter_dtype is torch.float16
    rollout = build.require_rollout()
    assert rollout.autocast_dtype is torch.float32
    assert rollout.prompt_encoder_dtype is torch.float32
    assert rollout.quantization_format is None
    assert rollout.base_weight_sync is False


@pytest.mark.parametrize("quantization_format", ["fp8", "nvfp4"])
def test_probe_model_build_derives_quantization_from_autocast_precision(
    quantization_format: str,
) -> None:
    args = generate._build_arg_parser().parse_args(
        [
            "--family",
            "sana",
            "--path",
            "unused",
            "--dtype",
            "bfloat16",
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
    assert rollout.autocast_dtype is torch.bfloat16
    assert rollout.prompt_encoder_dtype is torch.bfloat16
    assert rollout.quantization_format == quantization_format
    assert rollout.base_weight_sync is False
