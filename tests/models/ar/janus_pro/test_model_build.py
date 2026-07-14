"""Janus runtime AR request parsing tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from vrl.config.loading import load_config
from vrl.families.registry import get_model_family_entry
from vrl.generation import GenerationRequest
from vrl.generation.execution.chunks import SampleChunk
from vrl.models.ar.janus_pro.runtime import (
    JanusProChunkExecutor,
)


def test_janus_r1_replay_build_keeps_the_explicit_runtime_family() -> None:
    cfg = load_config("experiment/ar/janus_pro/online_r1_grpo_ocr")

    build = get_model_family_entry("janus_pro_r1").resolve_model_build(
        cfg,
        device="cpu",
        for_rollout=False,
    )

    assert cfg.model.family == "janus_pro_r1"
    assert build.family == cfg.model.family


def test_janus_model_build_does_not_expose_decode_strategy() -> None:
    """Checks the Janus model build does not expose decode strategy."""
    cfg = OmegaConf.create(
        {
            "model": {
                "family": "janus_pro",
                "path": "deepseek-ai/Janus-Pro-1B",
                "use_lora": False,
            },
            "precision": {"training": {"dtype": "fp32"}, "rollout": {"dtype": "fp32"}},
            "sampling": {
                "guidance_scale": 5.0,
                "temperature": 1.0,
                "image_token_num": 4,
                "ar_scheduler_batch_size": 2,
            },
        }
    )

    build = get_model_family_entry("janus_pro").resolve_model_build(cfg, device="cpu")

    assert build.sampling_config is not None
    assert "ar_decode_strategy" not in build.sampling_config
    assert build.sampling_config["ar_scheduler_batch_size"] == 2


def test_ar_runtime_rejects_duplicate_model_dtype() -> None:
    cfg = OmegaConf.create(
        {
            "model": {
                "family": "janus_pro",
                "path": "deepseek-ai/Janus-Pro-1B",
                "dtype": "bfloat16",
            },
            "precision": {"training": {"dtype": "fp16"}, "rollout": {"dtype": "fp16"}},
            "sampling": {},
        },
    )

    with pytest.raises(ValueError, match=r"model\.dtype.*top-level precision"):
        get_model_family_entry("janus_pro").resolve_model_build(cfg, device="cpu")


def test_janus_executor_parse_sampling_params_reads_scheduler_batch_size() -> None:
    """Checks Janus executor parse sampling params reads scheduler batch size."""
    request = GenerationRequest(
        request_id="req",
        family="janus_pro",
        task="ar_t2i",
        inputs=["draw text"],
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


def test_janus_chunk_context_records_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay must receive the same temperature used by rollout sampling."""
    executor = JanusProChunkExecutor(
        model=SimpleNamespace(
            processor=SimpleNamespace(
                tokenizer=SimpleNamespace(pad_token_id=0),
            ),
        ),
    )
    token_ids = torch.ones(1, 2, dtype=torch.long)
    mask = torch.ones_like(token_ids)

    tokenized_prompts: list[list[str]] = []

    def fake_tokenize(
        prompts: list[str],
        *,
        max_text_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokenized_prompts.append(prompts)
        assert max_text_length == 16
        return token_ids, mask

    monkeypatch.setattr(executor, "_tokenize_prompts", fake_tokenize)
    monkeypatch.setattr(
        executor,
        "_embed",
        lambda ids: torch.zeros(*ids.shape, 4),
    )
    request = GenerationRequest(
        request_id="req",
        family="janus_pro",
        task="ar_t2i",
        inputs=["draw text"],
        samples_per_prompt=1,
        sampling={
            "temperature": 0.7,
            "image_token_num": 4,
            "image_size": 384,
            "max_text_length": 16,
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

    assert prepared.context["temperature"] == 0.7
    assert tokenized_prompts == [["draw text"], [""]]
