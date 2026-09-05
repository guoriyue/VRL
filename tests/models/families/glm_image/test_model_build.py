"""GLM-Image model-build and executor wiring tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from vrl.config.precision import PrecisionPolicy
from vrl.config.schema import parse_config
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.types import GenerationRequest
from vrl.models.families.glm_image.config import GlmImageConfig
from vrl.models.families.glm_image.runner import GlmImageTokenRunner
from vrl.models.families.glm_image.runtime import (
    GlmImageBatchExecutor,
    glm_image_config_from_build,
)
from vrl.models.families.registry import get_model_family_entry


def test_resolve_model_build_defaults_to_glm_image_checkpoint() -> None:
    cfg = OmegaConf.create(
        {
            "model": {"family": "glm_image"},
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
    build = get_model_family_entry("glm_image").resolve_model_build(
        root,
        device="cpu",
        precision=precision,
    )

    assert build.model_name_or_path == "zai-org/GLM-Image"


def test_resolve_model_build_carries_sampling_and_lora_overrides() -> None:
    cfg = OmegaConf.create(
        {
            "model": {
                "family": "glm_image",
                "path": "/ckpt/glm-image",
                "use_lora": True,
                "lora": {"rank": 8},
            },
            "precision": {
                "float32_precision": "tf32",
                "training": {"dtype": "fp32"},
                "rollout": {"dtype": "fp32"},
            },
            "sampling": {
                "temperature": 0.8,
                "top_p": 0.9,
                "image_height": 768,
                "image_width": 1152,
                "decode_num_inference_steps": 50,
                "decode_guidance_scale": 2.0,
                "max_text_length": 128,
            },
        }
    )

    root = parse_config(cfg)
    precision = PrecisionPolicy.from_section(root.precision)
    build = get_model_family_entry("glm_image").resolve_model_build(
        root,
        device="cpu",
        precision=precision,
    )
    config = glm_image_config_from_build(build)
    resolved = GlmImageConfig(**config)

    assert build.model_name_or_path == "/ckpt/glm-image"
    assert config["temperature"] == 0.8
    assert config["top_p"] == 0.9
    assert config["image_height"] == 768
    assert config["image_width"] == 1152
    assert config["decode_num_inference_steps"] == 50
    assert config["decode_guidance_scale"] == 2.0
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
    # image_token_num/image_size are grid-derived; guidance_scale does not
    # exist for the AR (no CFG) — never config knobs.
    assert "image_token_num" not in config
    assert "image_size" not in config
    assert "guidance_scale" not in config


def test_executor_rejects_explicit_attention_backend() -> None:
    executor = GlmImageBatchExecutor(model=object())
    request = GenerationRequest(
        request_id="req",
        family="glm_image",
        task="ar_t2i",
        inputs=["p"],
        samples_per_prompt=1,
        sampling={"attention_backend": "vllm_paged"},
    )
    with pytest.raises(ValueError, match="attention_backend"):
        executor._ar_runner(request)

    request_native = GenerationRequest(
        request_id="req",
        family="glm_image",
        task="ar_t2i",
        inputs=["p"],
        samples_per_prompt=1,
        sampling={},
    )
    runner = executor._ar_runner(request_native)
    assert isinstance(runner, GlmImageTokenRunner)


def test_batch_context_keeps_replay_shape_and_sampling_provenance_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = torch.tensor([[1, 2]], dtype=torch.long)
    mask = torch.ones_like(ids)
    model = SimpleNamespace(
        config=SimpleNamespace(
            temperature=0.9,
            top_p=0.75,
            image_height=64,
            image_width=96,
        ),
    )

    def encode_generation_prompts(
        prompts: list[str],
        *,
        max_text_length: int,
        image_height: int,
        image_width: int,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int, int]]:
        assert prompts == ["draw text"]
        assert max_text_length == 8
        assert (image_height, image_width) == (64, 96)
        return ids, mask, (4, 6, 2, 3)

    model.encode_generation_prompts = encode_generation_prompts
    executor = GlmImageBatchExecutor(model)
    monkeypatch.setattr(executor, "_embed", lambda token_ids: token_ids.unsqueeze(-1).float())
    request = GenerationRequest(
        request_id="req",
        family="glm_image",
        task="ar_t2i",
        inputs=["draw text"],
        samples_per_prompt=1,
        sampling={
            "temperature": 0.8,
            "top_p": 0.6,
            "image_height": 64,
            "image_width": 96,
            "max_text_length": 8,
            "decode_guidance_scale": 2.0,
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
        "temperature": 0.8,
        "image_height": 64,
        "image_width": 96,
        "top_p": 0.6,
    }
    assert prepared.image_decode_kwargs["guidance_scale"] == 2.0
