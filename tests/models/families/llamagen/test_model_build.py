"""LlamaGen model-build resolution and executor request-parsing tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from vrl.families.registry import get_model_family_entry
from vrl.generation import GenerationRequest
from vrl.generation.execution.chunks import SampleChunk
from vrl.models.families.llamagen.config import (
    LLAMAGEN_CAPTION_DIM,
    LLAMAGEN_CAPTION_TOKEN_NUM,
    LLAMAGEN_DOWNSAMPLE_SIZE,
    LLAMAGEN_GPT_CKPT,
    LLAMAGEN_HF_REPO,
    LLAMAGEN_IMAGE_TOKEN_NUM,
    LLAMAGEN_T5_PATH,
    LLAMAGEN_VQ_CKPT,
    LlamaGenConfig,
)
from vrl.models.families.llamagen.model import (
    LLAMAGEN_IMAGE_VOCAB_SIZE,
    LlamaGenModel,
    LlamaGenReplayModel,
)
from vrl.models.families.llamagen.model import (
    LlamaGenConfig as ModelLlamaGenConfig,
)
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


class _TinyGpt(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.wqkv = torch.nn.Linear(2, 2, bias=False)
        self.wo = torch.nn.Linear(2, 2, bias=False)


@pytest.mark.parametrize("use_lora", [True, False])
def test_runtime_config_defaults_and_model_compatibility_export(use_lora: bool) -> None:
    config = LlamaGenConfig(use_lora=use_lora)
    entry = get_model_family_entry("llamagen")

    assert config.model_path == LLAMAGEN_HF_REPO == entry.family_build.default_model_path
    assert config.gpt_ckpt == LLAMAGEN_GPT_CKPT
    assert config.vq_ckpt == LLAMAGEN_VQ_CKPT
    assert config.t5_path == LLAMAGEN_T5_PATH
    assert config.image_token_num == LLAMAGEN_IMAGE_TOKEN_NUM
    assert config.cls_token_num == LLAMAGEN_CAPTION_TOKEN_NUM
    assert config.caption_dim == LLAMAGEN_CAPTION_DIM
    assert config.downsample_size == LLAMAGEN_DOWNSAMPLE_SIZE
    assert config.lora_target_modules == ("wqkv", "wo")
    assert config.use_lora is use_lora
    assert ModelLlamaGenConfig is LlamaGenConfig


def test_package_facade_preserves_existing_public_symbols() -> None:
    import vrl.models.families.llamagen as family

    assert family.LLAMAGEN_CAPTION_TOKEN_NUM == LLAMAGEN_CAPTION_TOKEN_NUM
    assert family.LLAMAGEN_IMAGE_TOKEN_NUM == LLAMAGEN_IMAGE_TOKEN_NUM
    assert family.LLAMAGEN_IMAGE_VOCAB_SIZE == LLAMAGEN_IMAGE_VOCAB_SIZE
    assert family.LlamaGenChunkExecutor is LlamaGenChunkExecutor
    assert family.LlamaGenConfig is LlamaGenConfig
    assert family.LlamaGenModel is LlamaGenModel
    assert family.LlamaGenReplayModel is LlamaGenReplayModel


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
    resolved = LlamaGenConfig(**config)
    assert "lora_target_modules" not in config
    assert resolved.lora_target_modules == ("wqkv", "wo")
    assert config["guidance_scale"] == 7.5
    assert config["top_k"] == 1000
    assert config["image_token_num"] == 256
    assert config["model_path"] == "peizesun/llamagen_t2i"


def test_false_lora_initialization_reaches_peft_as_bool() -> None:
    pytest.importorskip("peft")

    cfg = _cfg()
    cfg.model.lora = {"init_lora_weights": False}
    build = get_model_family_entry("llamagen").resolve_model_build(cfg, device="cpu")
    config = LlamaGenConfig(**llamagen_config_from_build(build))

    wrapped = LlamaGenModel._apply_lora(
        SimpleNamespace(config=config),
        _TinyGpt(),
    )

    assert wrapped.peft_config["default"].init_lora_weights is False


def test_lora_checkpoint_roundtrip_loads_trainable_adapter_weights(tmp_path) -> None:
    pytest.importorskip("peft")

    fresh = LlamaGenModel._apply_lora(
        SimpleNamespace(
            config=LlamaGenConfig(
                lora_rank=2,
                lora_alpha=4,
                lora_init=False,
            ),
        ),
        _TinyGpt(),
    )
    with torch.no_grad():
        expected = {}
        for index, (name, parameter) in enumerate(fresh.named_parameters(), start=1):
            if "lora_" not in name:
                continue
            parameter.fill_(index + 0.25)
            expected[name] = parameter.detach().clone()
    assert expected
    adapter_path = tmp_path / "adapter"
    fresh.save_pretrained(adapter_path)

    cfg = _cfg()
    cfg.model.lora = {"path": str(adapter_path)}
    build = get_model_family_entry("llamagen").resolve_model_build(cfg, device="cpu")
    config = LlamaGenConfig(**llamagen_config_from_build(build))
    loaded = LlamaGenModel._apply_lora(
        SimpleNamespace(config=config),
        _TinyGpt(),
    )
    actual = {
        name: parameter.detach()
        for name, parameter in loaded.named_parameters()
        if "lora_" in name
    }

    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[name], expected[name]) for name in expected)
    assert all(
        parameter.requires_grad for name, parameter in loaded.named_parameters() if "lora_" in name
    )


def test_lora_checkpoint_error_names_the_bad_path(tmp_path) -> None:
    pytest.importorskip("peft")

    missing_path = tmp_path / "missing-adapter"
    config = LlamaGenConfig(lora_path=str(missing_path))

    with pytest.raises(
        RuntimeError,
        match="failed to load trainable token LoRA adapter",
    ) as error:
        LlamaGenModel._apply_lora(
            SimpleNamespace(config=config),
            _TinyGpt(),
        )

    assert str(missing_path) in str(error.value)


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


@pytest.mark.parametrize(
    ("batch_size", "expected"),
    [(None, None), (2, 2), (3, 3)],
)
def test_executor_allows_scheduler_bounds_covering_full_static_kv_chunk(
    batch_size: int | None,
    expected: int | None,
) -> None:
    request = GenerationRequest(
        request_id="req",
        family="llamagen",
        task="ar_t2i",
        inputs=["draw text"],
        samples_per_prompt=2,
        sampling={"ar_scheduler_batch_size": batch_size},
    )

    resolved = LlamaGenChunkExecutor(model=object()).resolve_scheduler_batch_size(
        request,
        row_count=2,
    )

    assert resolved == expected


def test_executor_rejects_partial_scheduler_before_static_kv_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = LlamaGenChunkExecutor(model=object())
    prepare_calls = 0

    def prepare_chunk_inputs(*_args, **_kwargs):
        nonlocal prepare_calls
        prepare_calls += 1
        raise AssertionError("preparation must not run")

    monkeypatch.setattr(executor, "prepare_chunk_inputs", prepare_chunk_inputs)
    request = GenerationRequest(
        request_id="req",
        family="llamagen",
        task="ar_t2i",
        inputs=["draw text"],
        samples_per_prompt=2,
        sampling={"ar_scheduler_batch_size": 1},
    )
    chunk = SampleChunk(
        prompt_index=0,
        prompt="draw text",
        sample_start=0,
        sample_count=2,
    )

    with pytest.raises(ValueError, match="null or >= chunk sample count"):
        executor.forward_chunk_plan(request, chunk)

    assert prepare_calls == 0


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
