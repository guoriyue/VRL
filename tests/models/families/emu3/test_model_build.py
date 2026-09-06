"""Emu3 model-build and executor wiring tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from vrl.config.precision import PrecisionPolicy, RolePrecision
from vrl.config.schema import parse_config
from vrl.generation import GenerationRequest
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.models.families.emu3.config import Emu3Config
from vrl.models.families.emu3.model import Emu3Config as ModelEmu3Config
from vrl.models.families.emu3.runner import Emu3TokenRunner
from vrl.models.families.emu3.runtime import Emu3BatchExecutor, emu3_config_from_build
from vrl.models.families.registry import get_model_family_entry


@pytest.mark.parametrize("use_lora", [True, False])
def test_runtime_config_defaults_and_model_compatibility_export(use_lora: bool) -> None:
    config = Emu3Config(use_lora=use_lora)
    entry = get_model_family_entry("emu3")

    assert config.model_path == entry.family_build.default_model_path
    assert config.use_lora is use_lora
    assert ModelEmu3Config is Emu3Config


def test_resolve_model_build_defaults_to_gen_hf_checkpoint() -> None:
    cfg = OmegaConf.create(
        {
            "model": {"family": "emu3"},
            "precision": {
                "float32_precision": "tf32",
                "training": {"dtype": "fp32"},
                "rollout": {"dtype": "fp32"},
            },
            "sampling": {},
        },
    )

    root = parse_config(cfg)
    precision = PrecisionPolicy.from_section(root.precision)
    build = get_model_family_entry("emu3").resolve_model_build(
        root,
        device="cpu",
        precision=precision,
    )

    assert build.model_name_or_path == "BAAI/Emu3-Gen-hf"
    assert build.precision == RolePrecision("fp32", "tf32")


def test_resolve_model_build_carries_sampling_and_lora_overrides() -> None:
    cfg = OmegaConf.create(
        {
            "model": {
                "family": "emu3",
                "path": "/ckpt/emu3-gen",
                "use_lora": True,
                "lora": {"rank": 8},
            },
            "precision": {
                "float32_precision": "tf32",
                "training": {"dtype": "fp32"},
                "rollout": {"dtype": "fp32"},
            },
            "sampling": {
                "guidance_scale": 4.0,
                "temperature": 0.9,
                "image_area": 262144,
                "ratio": "1:1",
                "max_text_length": 128,
            },
        }
    )

    root = parse_config(cfg)
    precision = PrecisionPolicy.from_section(root.precision)
    build = get_model_family_entry("emu3").resolve_model_build(
        root,
        device="cpu",
        precision=precision,
    )
    config = emu3_config_from_build(build)
    resolved = Emu3Config(**config)

    assert build.model_name_or_path == "/ckpt/emu3-gen"
    assert config["guidance_scale"] == 4.0
    assert config["temperature"] == 0.9
    assert config["image_area"] == 262144
    assert config["ratio"] == "1:1"
    # Only the explicit override is projected; the config owns all defaults.
    assert config["lora_rank"] == 8
    assert "lora_alpha" not in config
    assert resolved.lora_alpha == 64
    assert resolved.lora_target_modules == (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )
    # image_token_num/image_size are grid-derived, never config knobs.
    assert "image_token_num" not in config
    assert "image_size" not in config


def test_executor_declares_identity_and_runner() -> None:
    executor = Emu3BatchExecutor(model=object())

    assert executor.family == "emu3"
    assert executor.task == "ar_t2i"
    assert executor._runner_cls is Emu3TokenRunner
    assert executor._runner_attention_family == "emu3"


@pytest.mark.parametrize(
    ("sampling_overrides", "expected_guidance", "expected_temperature"),
    [
        ({"temperature": 0.8}, 6.25, 0.8),
        ({"guidance_scale": 4.0}, 4.0, 0.45),
    ],
)
def test_chunk_sampling_uses_request_overrides_then_model_defaults(
    monkeypatch: pytest.MonkeyPatch,
    sampling_overrides: dict[str, float],
    expected_guidance: float,
    expected_temperature: float,
) -> None:
    ids = torch.tensor([[1, 2]], dtype=torch.long)
    mask = torch.ones_like(ids)
    model = SimpleNamespace(
        config=SimpleNamespace(
            guidance_scale=6.25,
            temperature=0.45,
        ),
        processor=SimpleNamespace(
            tokenizer=SimpleNamespace(pad_token_id=0),
        ),
    )

    def encode_generation_prompts(
        prompts: list[str],
        *,
        max_text_length: int,
        image_area: int | None,
        ratio: str | None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        assert max_text_length == 8
        assert image_area == 24
        assert ratio == "3:2"
        assert prompts in (["draw text"], [""])
        return ids, mask, (2, 3)

    model.encode_generation_prompts = encode_generation_prompts
    executor = Emu3BatchExecutor(model)
    monkeypatch.setattr(executor, "_embed", lambda token_ids: token_ids.unsqueeze(-1).float())
    request = GenerationRequest(
        request_id="req",
        family="emu3",
        task="ar_t2i",
        inputs=["draw text"],
        samples_per_prompt=1,
        sampling={
            "image_area": 24,
            "ratio": "3:2",
            "max_text_length": 8,
            **sampling_overrides,
        },
    )

    prepared = executor.prepare_batch_inputs(
        request,
        GenerationSampleBatch(
            prompt_index=0,
            sample_start=0,
            sample_count=1,
        ),
    )

    assert prepared.context == {
        "temperature": expected_temperature,
        "image_height": 2,
        "image_width": 3,
        "guidance_scale": expected_guidance,
    }
