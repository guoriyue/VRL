"""GLM-Image model-build and executor wiring tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from vrl.families.registry import get_model_family_entry
from vrl.generation.execution.chunks import SampleChunk
from vrl.generation.types import GenerationRequest
from vrl.models.families.glm_image.runner import GlmImageTokenRunner
from vrl.models.families.glm_image.runtime import (
    GlmImageChunkExecutor,
    glm_image_config_from_build,
)


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

    build = get_model_family_entry("glm_image").resolve_model_build(cfg, device="cpu")

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

    build = get_model_family_entry("glm_image").resolve_model_build(cfg, device="cpu")
    config = glm_image_config_from_build(build)

    assert build.model_name_or_path == "/ckpt/glm-image"
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


def test_executor_declares_family_identity() -> None:
    executor = GlmImageChunkExecutor(model=object())

    assert executor.family == "glm_image"
    assert executor.task == "ar_t2i"


def test_executor_rejects_explicit_attention_backend() -> None:
    executor = GlmImageChunkExecutor(model=object())
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


def test_chunk_context_keeps_replay_shape_and_sampling_provenance_only(
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
    executor = GlmImageChunkExecutor(model)
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

    prepared = executor.prepare_chunk_inputs(
        request,
        SampleChunk(
            prompt_index=0,
            prompt="draw text",
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
