"""Janus runtime AR request parsing tests."""

from __future__ import annotations

from omegaconf import OmegaConf

from vrl.generation import GenerationRequest
from vrl.models.ar.build import extract_family_ar_runtime_spec
from vrl.models.ar.janus_pro.runtime import (
    JanusProChunkExecutor,
)


def test_janus_runtime_spec_does_not_expose_decode_strategy() -> None:
    """Checks Janus runtime spec does not expose decode strategy."""
    cfg = OmegaConf.create(
        {
            "model": {
                "family": "janus_pro",
                "path": "deepseek-ai/Janus-Pro-1B",
                "use_lora": False,
            },
            "precision": "fp32",
            "sampling": {
                "guidance_scale": 5.0,
                "temperature": 1.0,
                "image_token_num": 4,
                "ar_scheduler_batch_size": 2,
            },
        }
    )

    spec = extract_family_ar_runtime_spec(cfg, device="cpu", weight_dtype="float32")

    assert spec.sampling_config is not None
    assert "ar_decode_strategy" not in spec.sampling_config
    assert spec.sampling_config["ar_scheduler_batch_size"] == 2


def test_janus_executor_parse_sampling_params_reads_scheduler_batch_size() -> None:
    """Checks Janus executor parse sampling params reads scheduler batch size."""
    request = GenerationRequest(
        request_id="req",
        family="janus_pro",
        task="ar_t2i",
        prompts=["draw text"],
        samples_per_prompt=1,
        sampling={
            "image_token_num": 4,
            "image_size": 384,
            "max_text_length": 16,
            "ar_scheduler_batch_size": 8,
        },
    )

    params = JanusProChunkExecutor(model=object()).layout.parse_sampling_params(request)

    assert params.ar_scheduler_batch_size == 8
