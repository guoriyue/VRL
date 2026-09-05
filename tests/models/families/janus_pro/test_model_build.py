"""Janus runtime AR request parsing tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from vrl.config.loading import load_config
from vrl.config.precision import PrecisionPolicy
from vrl.config.schema import parse_config
from vrl.generation import GenerationRequest
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.models.families.janus_pro.config import (
    JANUS_IMAGE_TOKEN_NUM,
    JanusProConfig,
)
from vrl.models.families.janus_pro.runtime import (
    JanusProBatchExecutor,
    janus_config_from_build,
)
from vrl.models.families.registry import get_model_family_entry


def _build_cfg(*, family: str, trust_remote_code: bool):
    return OmegaConf.create(
        {
            "model": {
                "family": family,
                "use_lora": True,
                "trust_remote_code": trust_remote_code,
            },
            "precision": {
                "float32_precision": "tf32",
                "training": {"dtype": "fp32"},
                "rollout": {"dtype": "fp32"},
            },
            "sampling": {
                "guidance_scale": 5.0,
                "temperature": 1.0,
                "image_token_num": JANUS_IMAGE_TOKEN_NUM,
            },
        },
    )


@pytest.mark.parametrize(
    ("family", "trust_remote_code"),
    [("janus_pro", True), ("janus_pro_r1", False)],
)
def test_config_projection_preserves_trust_remote_code_boolean(
    family: str,
    trust_remote_code: bool,
) -> None:
    entry = get_model_family_entry(family)
    cfg = _build_cfg(family=family, trust_remote_code=trust_remote_code)
    root = parse_config(cfg)
    precision = PrecisionPolicy.from_section(root.precision)
    build = entry.resolve_model_build(
        root,
        device="cpu",
        precision=precision,
    )

    projected = janus_config_from_build(build)
    resolved = JanusProConfig(**projected)

    assert projected["trust_remote_code"] is trust_remote_code
    assert "lora_target_modules" not in projected
    assert "lora_init" not in projected
    assert resolved.lora_target_modules == (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )
    assert resolved.lora_init == "gaussian"


def test_janus_r1_replay_build_keeps_the_explicit_runtime_family() -> None:
    cfg = load_config("experiment/janus_pro/online_r1_grpo_ocr")
    root = parse_config(cfg)
    precision = PrecisionPolicy.from_section(root.precision)

    build = get_model_family_entry("janus_pro_r1").resolve_model_build(
        root,
        device="cpu",
        precision=precision,
        for_rollout=False,
    )

    assert cfg.model.family == "janus_pro_r1"
    assert build.family == root.model.family


def test_janus_model_build_carries_scheduler_batch_size() -> None:
    cfg = OmegaConf.create(
        {
            "model": {
                "family": "janus_pro",
                "path": "deepseek-ai/Janus-Pro-1B",
                "use_lora": False,
            },
            "precision": {
                "float32_precision": "tf32",
                "training": {"dtype": "fp32"},
                "rollout": {"dtype": "fp32"},
            },
            "sampling": {
                "guidance_scale": 5.0,
                "temperature": 1.0,
                "image_token_num": 4,
                "ar_scheduler_batch_size": 2,
            },
        }
    )

    root = parse_config(cfg)
    precision = PrecisionPolicy.from_section(root.precision)
    build = get_model_family_entry("janus_pro").resolve_model_build(
        root,
        device="cpu",
        precision=precision,
    )

    assert build.sampling_config is not None
    assert build.sampling_config["ar_scheduler_batch_size"] == 2


def test_schema_rejects_duplicate_model_dtype() -> None:
    cfg = OmegaConf.create(
        {
            "model": {
                "family": "janus_pro",
                "path": "deepseek-ai/Janus-Pro-1B",
                "dtype": "bfloat16",
            },
            "precision": {
                "float32_precision": "tf32",
                "training": {"dtype": "fp16"},
                "rollout": {"dtype": "fp16"},
            },
            "sampling": {},
        },
    )

    with pytest.raises(ValueError, match=r"unknown model\.dtype"):
        parse_config(cfg)


def test_janus_executor_layout_resolves_scheduler_batch_size() -> None:
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

    layout = JanusProBatchExecutor(model=object()).layout

    assert layout.resolve_scheduler_batch_size(request) == 8


@pytest.mark.parametrize(
    ("sampling_overrides", "expected_guidance", "expected_temperature"),
    [
        ({"temperature": 0.7}, 6.25, 0.7),
        ({"guidance_scale": 4.0}, 4.0, 0.45),
    ],
)
def test_janus_chunk_sampling_uses_request_overrides_then_model_defaults(
    monkeypatch: pytest.MonkeyPatch,
    sampling_overrides: dict[str, float],
    expected_guidance: float,
    expected_temperature: float,
) -> None:
    """Replay keeps behavior sampling without duplicating model defaults."""
    executor = JanusProBatchExecutor(
        model=SimpleNamespace(
            config=SimpleNamespace(
                guidance_scale=6.25,
                temperature=0.45,
            ),
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
            "image_token_num": 4,
            "image_size": 384,
            "max_text_length": 16,
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
        "guidance_scale": expected_guidance,
    }
    assert tokenized_prompts == [["draw text"], [""]]
