"""Janus runtime AR request parsing tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from vrl.config.loading import load_config
from vrl.config.precision import resolve_precision_policy
from vrl.config.schema import parse_config
from vrl.families.registry import get_model_family_entry
from vrl.generation import GenerationRequest
from vrl.generation.execution.chunks import SampleChunk
from vrl.models.families.janus_pro.config import (
    JANUS_IMAGE_TOKEN_NUM,
    JanusProConfig,
    JanusProModelSection,
)
from vrl.models.families.janus_pro.model import (
    JANUS_IMAGE_TOKEN_NUM as ModelImageTokenNum,
)
from vrl.models.families.janus_pro.model import (
    JanusProConfig as ModelJanusProConfig,
)
from vrl.models.families.janus_pro.model import (
    JanusProModel,
    image_token_logits_from_hidden,
)
from vrl.models.families.janus_pro.runtime import (
    JanusProChunkExecutor,
    JanusProR1ChunkExecutor,
    JanusProR1ChunkGatherer,
    janus_config_from_build,
)


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


def test_runtime_config_defaults_and_model_compatibility_export() -> None:
    config = JanusProConfig()
    base_entry = get_model_family_entry("janus_pro")
    r1_entry = get_model_family_entry("janus_pro_r1")

    assert (
        config.model_path
        == base_entry.family_build.default_model_path
        == r1_entry.family_build.default_model_path
    )
    assert config.image_token_num == JANUS_IMAGE_TOKEN_NUM
    assert config.lora_target_modules == (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )
    assert config.lora_init == "gaussian"
    assert ModelImageTokenNum == JANUS_IMAGE_TOKEN_NUM
    assert ModelJanusProConfig is JanusProConfig


@pytest.mark.parametrize("family", ["janus_pro", "janus_pro_r1"])
@pytest.mark.parametrize("trust_remote_code", [True, False])
def test_config_projection_preserves_trust_remote_code_boolean(
    family: str,
    trust_remote_code: bool,
) -> None:
    entry = get_model_family_entry(family)
    cfg = _build_cfg(family=family, trust_remote_code=trust_remote_code)
    root = parse_config(cfg)
    precision = resolve_precision_policy(root)
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


def test_package_facade_preserves_transition_table_and_public_symbols() -> None:
    import vrl.models.families.janus_pro as family

    assert family.JANUS_R1_SEGMENTS == (
        "initial_image",
        "selfcheck_text",
        "final_image",
    )
    assert family.JanusProChunkExecutor is JanusProChunkExecutor
    assert family.JanusProConfig is JanusProConfig
    assert family.JanusProModel is JanusProModel
    assert family.JanusProModelSection is JanusProModelSection
    assert family.JanusProR1ChunkExecutor is JanusProR1ChunkExecutor
    assert family.JanusProR1ChunkGatherer is JanusProR1ChunkGatherer
    assert family.image_token_logits_from_hidden is image_token_logits_from_hidden


def test_janus_r1_replay_build_keeps_the_explicit_runtime_family() -> None:
    cfg = load_config("experiment/janus_pro/online_r1_grpo_ocr")
    root = parse_config(cfg)
    precision = resolve_precision_policy(root)

    build = get_model_family_entry("janus_pro_r1").resolve_model_build(
        root,
        device="cpu",
        precision=precision,
        for_rollout=False,
    )

    assert cfg.model.family == "janus_pro_r1"
    assert build.family == root.model.family


def test_janus_model_build_does_not_expose_decode_strategy() -> None:
    """Checks the Janus model build does not expose decode strategy."""
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
    precision = resolve_precision_policy(root)
    build = get_model_family_entry("janus_pro").resolve_model_build(
        root,
        device="cpu",
        precision=precision,
    )

    assert build.sampling_config is not None
    assert "ar_decode_strategy" not in build.sampling_config
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

    layout = JanusProChunkExecutor(model=object()).layout

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
    executor = JanusProChunkExecutor(
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
        "temperature": expected_temperature,
        "guidance_scale": expected_guidance,
    }
    assert tokenized_prompts == [["draw text"], [""]]
