"""LlamaGen model-build resolution and executor request-parsing tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from vrl.families.registry import get_model_family_entry
from vrl.generation import GenerationRequest
from vrl.generation.execution.chunks import SampleChunk
from vrl.models.families.llamagen.runtime import (
    LlamaGenChunkExecutor,
    llamagen_config_from_build,
)


def _cfg():
    return OmegaConf.create(
        {
            "model": {
                "family": "llamagen",
                "path": "peizesun/llamagen_t2i",
                "use_lora": True,
            },
            "precision": {
                "float32_precision": "tf32",
                "training": {"dtype": "fp32"},
                "rollout": {"dtype": "fp32"},
            },
            "sampling": {
                "guidance_scale": 7.5,
                "temperature": 1.0,
                "top_k": 1000,
                "image_token_num": 256,
            },
        }
    )


def test_resolve_model_build_defaults() -> None:
    """Checks default checkpoint path and AR task selection."""
    cfg = OmegaConf.create(
        {
            "model": {"family": "llamagen", "use_lora": False},
            "precision": {
                "float32_precision": "tf32",
                "training": {"dtype": "fp32"},
                "rollout": {"dtype": "fp32"},
            },
        },
    )
    build = get_model_family_entry("llamagen").resolve_model_build(cfg, device="cpu")
    assert build.model_name_or_path == "peizesun/llamagen_t2i"


def test_config_from_build_uses_fused_projection_lora_targets() -> None:
    """The vendored GPT has wqkv/wo, not q_proj/k_proj/v_proj."""
    build = get_model_family_entry("llamagen").resolve_model_build(
        _cfg(),
        device="cpu",
    )
    config = llamagen_config_from_build(build)
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
        inputs=["draw text"],
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
        inputs=["draw text"],
        samples_per_prompt=1,
        sampling={"attention_backend": "vllm_paged"},
    )
    with pytest.raises(ValueError, match="attention_backend"):
        LlamaGenChunkExecutor(model=object())._ar_runner(request)


def test_chunk_context_keeps_temperature_and_sampling_provenance_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = torch.tensor([[1, 2]], dtype=torch.long)
    mask = torch.ones_like(ids)
    model = SimpleNamespace(
        config=SimpleNamespace(
            cls_token_num=120,
            top_k=0,
            top_p=1.0,
        ),
        encode_caption=lambda token_ids, token_mask: (
            token_ids.unsqueeze(-1).float(),
            token_mask,
        ),
        uncond_caption_embeds=lambda count: torch.zeros(count, 2, 1),
    )
    executor = LlamaGenChunkExecutor(model)
    monkeypatch.setattr(
        executor,
        "_tokenize_prompts",
        lambda prompts, *, max_text_length: (ids, mask),
    )
    request = GenerationRequest(
        request_id="req",
        family="llamagen",
        task="ar_t2i",
        inputs=["draw text"],
        samples_per_prompt=1,
        sampling={
            "guidance_scale": 6.0,
            "temperature": 0.7,
            "top_k": 32,
            "top_p": 0.9,
            "image_token_num": 4,
            "image_size": 32,
            "max_text_length": 120,
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
        "temperature": 0.7,
        "guidance_scale": 6.0,
        "top_k": 32,
        "top_p": 0.9,
    }
