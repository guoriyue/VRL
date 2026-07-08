"""GLM-Image runtime spec / executor wiring tests."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from vrl.generation.types import GenerationRequest
from vrl.models.ar.glm_image.runner import GlmImageTokenRunner
from vrl.models.ar.glm_image.runtime import (
    GLM_IMAGE_FAMILY_CAPABILITY,
    GlmImageChunkExecutor,
    _glm_image_config_from_runtime_spec,
    extract_glm_image_runtime_spec,
)


def test_extract_runtime_spec_defaults_to_glm_image_checkpoint() -> None:
    cfg = OmegaConf.create({"model": {}, "precision": "fp32", "sampling": {}})

    spec = extract_glm_image_runtime_spec(cfg, device="cpu", weight_dtype="float32")

    assert spec.ar_task == "ar_t2i"
    assert spec.model_name_or_path == "zai-org/GLM-Image"


def test_extract_runtime_spec_carries_sampling_and_lora_overrides() -> None:
    cfg = OmegaConf.create(
        {
            "model": {
                "path": "/ckpt/glm-image",
                "use_lora": True,
                "lora": {"rank": 8},
            },
            "precision": "fp32",
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

    spec = extract_glm_image_runtime_spec(cfg, device="cpu", weight_dtype="float32")
    config = _glm_image_config_from_runtime_spec(spec)

    assert spec.model_name_or_path == "/ckpt/glm-image"
    assert config["temperature"] == 0.8
    assert config["top_p"] == 0.9
    assert config["image_height"] == 768
    assert config["image_width"] == 1152
    assert config["decode_num_inference_steps"] == 50
    assert config["decode_guidance_scale"] == 2.0
    # Carried lora block overrides the family defaults, rest stays default.
    assert config["lora_rank"] == 8
    assert config["lora_alpha"] == 64
    # image_token_num/image_size are grid-derived; guidance_scale does not
    # exist for the AR (no CFG) — never config knobs.
    assert "image_token_num" not in config
    assert "image_size" not in config
    assert "guidance_scale" not in config


def test_executor_declares_family_capability_and_runner() -> None:
    executor = GlmImageChunkExecutor(model=object())

    assert executor.family == "glm_image"
    assert executor.task == "ar_t2i"
    assert executor.capability() is GLM_IMAGE_FAMILY_CAPABILITY
    assert GLM_IMAGE_FAMILY_CAPABILITY.family == "glm_image"
    assert GLM_IMAGE_FAMILY_CAPABILITY.task == "ar_t2i"
    assert GLM_IMAGE_FAMILY_CAPABILITY.trajectory_kind == "ar_discrete"
    assert executor._runner_cls is GlmImageTokenRunner
    assert executor._runner_attention_family == "glm_image"


def test_executor_rejects_explicit_attention_backend() -> None:
    executor = GlmImageChunkExecutor(model=object())
    request = GenerationRequest(
        request_id="req",
        family="glm_image",
        task="ar_t2i",
        prompts=["p"],
        samples_per_prompt=1,
        sampling={"attention_backend": "vllm_paged"},
    )
    with pytest.raises(ValueError, match="attention_backend"):
        executor._ar_runner(request)

    request_native = GenerationRequest(
        request_id="req",
        family="glm_image",
        task="ar_t2i",
        prompts=["p"],
        samples_per_prompt=1,
        sampling={},
    )
    runner = executor._ar_runner(request_native)
    assert isinstance(runner, GlmImageTokenRunner)
