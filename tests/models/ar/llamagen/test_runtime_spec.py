"""LlamaGen runtime spec extraction and executor request-parsing tests."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from vrl.generation import GenerationRequest
from vrl.models.ar.build import extract_family_ar_runtime_spec
from vrl.models.ar.llamagen.runtime import (
    LlamaGenChunkExecutor,
    llamagen_config_from_runtime_spec,
)


def _cfg():
    return OmegaConf.create(
        {
            "model": {
                "family": "llamagen",
                "path": "peizesun/llamagen_t2i",
                "use_lora": True,
            },
            "precision": "fp32",
            "sampling": {
                "guidance_scale": 7.5,
                "temperature": 1.0,
                "top_k": 1000,
                "image_token_num": 256,
            },
        }
    )


def test_extract_runtime_spec_defaults() -> None:
    """Checks default checkpoint path and AR task selection."""
    cfg = OmegaConf.create(
        {"model": {"family": "llamagen", "use_lora": False}, "precision": "fp32"},
    )
    spec = extract_family_ar_runtime_spec(cfg, device="cpu", weight_dtype="float32")
    assert spec.model_name_or_path == "peizesun/llamagen_t2i"
    assert spec.ar_task == "ar_t2i"


def test_config_from_runtime_spec_uses_fused_projection_lora_targets() -> None:
    """The vendored GPT has wqkv/wo, not q_proj/k_proj/v_proj."""
    spec = extract_family_ar_runtime_spec(_cfg(), device="cpu", weight_dtype="float32")
    config = llamagen_config_from_runtime_spec(spec)
    assert config["lora_target_modules"] == ("wqkv", "wo")
    assert config["guidance_scale"] == 7.5
    assert config["top_k"] == 1000
    assert config["image_token_num"] == 256
    assert config["model_path"] == "peizesun/llamagen_t2i"


def test_executor_layout_defaults_match_xl_stage1_256() -> None:
    """256 tokens (16x16), 256 px, fixed 120-token caption prefix."""
    request = GenerationRequest(
        request_id="req",
        family="llamagen",
        task="ar_t2i",
        prompts=["draw text"],
        samples_per_prompt=1,
        sampling={},
    )
    params = LlamaGenChunkExecutor(model=object()).layout.parse_sampling_params(request)
    assert params.image_token_num == 256
    assert params.image_size == 256
    assert params.max_text_length == 120


def test_executor_rejects_shared_attention_backend_selection() -> None:
    """The vendored static KV cache cannot serve the shared backends."""
    request = GenerationRequest(
        request_id="req",
        family="llamagen",
        task="ar_t2i",
        prompts=["draw text"],
        samples_per_prompt=1,
        sampling={"attention_backend": "vllm_paged"},
    )
    with pytest.raises(ValueError, match="attention_backend"):
        LlamaGenChunkExecutor(model=object())._ar_runner(request)
