"""Emu3 runtime spec / executor wiring tests."""

from __future__ import annotations

from omegaconf import OmegaConf

from vrl.models.ar.build import extract_family_ar_runtime_spec
from vrl.models.ar.emu3.runner import Emu3TokenRunner
from vrl.models.ar.emu3.runtime import (
    EMU3_FAMILY_CAPABILITY,
    Emu3ChunkExecutor,
    emu3_config_from_runtime_spec,
)


def test_extract_runtime_spec_defaults_to_gen_hf_checkpoint() -> None:
    cfg = OmegaConf.create(
        {"model": {"family": "emu3"}, "precision": "fp32", "sampling": {}},
    )

    spec = extract_family_ar_runtime_spec(cfg, device="cpu", weight_dtype="float32")

    assert spec.ar_task == "ar_t2i"
    assert spec.model_name_or_path == "BAAI/Emu3-Gen-hf"


def test_extract_runtime_spec_carries_sampling_and_lora_overrides() -> None:
    cfg = OmegaConf.create(
        {
            "model": {
                "family": "emu3",
                "path": "/ckpt/emu3-gen",
                "use_lora": True,
                "lora": {"rank": 8},
            },
            "precision": "fp32",
            "sampling": {
                "guidance_scale": 4.0,
                "temperature": 0.9,
                "image_area": 262144,
                "ratio": "1:1",
                "max_text_length": 128,
            },
        }
    )

    spec = extract_family_ar_runtime_spec(cfg, device="cpu", weight_dtype="float32")
    config = emu3_config_from_runtime_spec(spec)

    assert spec.model_name_or_path == "/ckpt/emu3-gen"
    assert config["guidance_scale"] == 4.0
    assert config["temperature"] == 0.9
    assert config["image_area"] == 262144
    assert config["ratio"] == "1:1"
    # Carried lora block overrides the family defaults, rest stays default.
    assert config["lora_rank"] == 8
    assert config["lora_alpha"] == 64
    # image_token_num/image_size are grid-derived, never config knobs.
    assert "image_token_num" not in config
    assert "image_size" not in config


def test_executor_declares_family_capability_and_runner() -> None:
    executor = Emu3ChunkExecutor(model=object())

    assert executor.family == "emu3"
    assert executor.task == "ar_t2i"
    assert executor.capability() is EMU3_FAMILY_CAPABILITY
    assert EMU3_FAMILY_CAPABILITY.family == "emu3"
    assert EMU3_FAMILY_CAPABILITY.task == "ar_t2i"
    assert EMU3_FAMILY_CAPABILITY.trajectory_kind == "ar_discrete"
    assert executor._runner_cls is Emu3TokenRunner
    assert executor._runner_attention_family == "emu3"
