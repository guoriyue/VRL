"""Janus runtime AR request parsing tests."""

from __future__ import annotations

from omegaconf import OmegaConf

from vrl.engine import GenerationRequest
from vrl.models.ar.janus_pro.runtime import (
    JanusProPipelineExecutor,
    extract_janus_pro_runtime_spec,
)


def test_janus_runtime_spec_does_not_expose_decode_strategy() -> None:
    cfg = OmegaConf.create(
        {
            "model": {
                "path": "deepseek-ai/Janus-Pro-1B",
                "use_lora": False,
            },
            "sampling": {
                "cfg_weight": 5.0,
                "temperature": 1.0,
                "image_token_num": 4,
                "ar_scheduler_batch_size": 2,
            },
        }
    )

    spec = extract_janus_pro_runtime_spec(cfg, device="cpu", weight_dtype="float32")

    assert "ar_decode_strategy" not in spec.scheduler_config
    assert spec.scheduler_config["ar_scheduler_batch_size"] == 2


def test_janus_executor_parse_sampling_params_reads_scheduler_batch_size() -> None:
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

    params = JanusProPipelineExecutor(model=object()).parse_sampling_params(request)

    assert params.ar_scheduler_batch_size == 8
