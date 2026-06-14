from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from vrl.models.interfaces.runtime import RuntimeBuildSpec
from vrl.models.replay_loading import (
    MINIMAL_REPLAY_RUNTIME_ROLE,
    module_loading_profile_from_metadata,
    require_minimal_replay_bundle,
)


class _TinyTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    @property
    def dtype(self) -> torch.dtype:
        return self.weight.dtype


class _TinyScheduler:
    def __init__(self) -> None:
        self.timesteps = torch.tensor([1.0])
        self.sigmas = torch.tensor([1.0])

    def set_timesteps(self, n: int, device: Any = None) -> None:
        self.timesteps = torch.arange(n, device=device, dtype=torch.float32)
        self.sigmas = torch.ones(n, device=device, dtype=torch.float32)


class _TinyRuntimeModel(nn.Module):
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.language_model = nn.Linear(1, 1)

    def replay_forward(self, *_args: Any, **_kwargs: Any) -> Any:
        return None

    def disable_adapter(self) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    def load_trainable_state(self, state_dict: dict[str, Any]) -> Any:
        return self.load_state_dict(state_dict, strict=False)


def _spec(**overrides: Any) -> RuntimeBuildSpec:
    """Build a RuntimeBuildSpec from friendly overrides.

    Translates the legacy ``use_lora`` / ``lora_config`` / ``scheduler_config``
    test kwargs into the carried ``model_config`` / ``sampling_config`` blocks
    so tests exercise the same read helpers the families use.
    """

    use_lora = bool(overrides.pop("use_lora", False))
    lora_config = overrides.pop("lora_config", None)
    scheduler_config = overrides.pop("scheduler_config", {"num_steps": 2})
    extra = overrides.pop("extra", None)

    model_config: dict[str, Any] = {"path": "fake/repo", "use_lora": use_lora}
    if lora_config is not None:
        model_config["lora"] = dict(lora_config)
    if extra is not None:
        # Legacy ``extra`` test fields (anima artifact paths, scheduler_shift)
        # now ride directly in the carried model block.
        model_config.update(dict(extra))

    values: dict[str, Any] = {
        "model_name_or_path": "fake/repo",
        "device": "cpu",
        "dtype": torch.float32,
        "task_variant": "t2i",
        "model_config": model_config,
        "sampling_config": dict(scheduler_config),
    }
    values.update(overrides)
    return RuntimeBuildSpec(**values)


@pytest.mark.parametrize(
    ("module_path", "builder_name"),
    [
        (
            "vrl.models.diffusion.sd3_5.runtime",
            "build_sd3_5_replay_runtime_bundle",
        ),
        (
            "vrl.models.diffusion.wan_2_1.runtime",
            "build_wan_2_1_replay_runtime_bundle",
        ),
        (
            "vrl.models.diffusion.cosmos.predict2.runtime",
            "build_cosmos_predict2_replay_runtime_bundle",
        ),
    ],
)
def test_diffusion_replay_builders_return_minimal_bundles(
    monkeypatch: pytest.MonkeyPatch,
    module_path: str,
    builder_name: str,
) -> None:
    """Checks diffusion replay builders return minimal bundles."""
    module = __import__(module_path, fromlist=[builder_name])
    monkeypatch.setattr(
        module,
        "load_diffusers_transformer",
        lambda *_args, **_kwargs: _TinyTransformer(),
    )
    monkeypatch.setattr(
        module,
        "load_flow_match_scheduler",
        lambda *_args, **_kwargs: _TinyScheduler(),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "load_diffusers_scheduler",
        lambda *_args, **_kwargs: _TinyScheduler(),
        raising=False,
    )

    bundle = getattr(module, builder_name)(_spec())

    require_minimal_replay_bundle(bundle)
    profile = module_loading_profile_from_metadata(bundle.metadata)
    assert profile.runtime_role == MINIMAL_REPLAY_RUNTIME_ROLE
    # The replay bundle declares the generation-only modules it deliberately
    # never loads (text encoders, VAE) — this IS the trainer-side memory
    # boundary: nothing to offload because nothing was loaded.
    assert "text_encoder" in profile.generation_only_modules
    assert "vae" in profile.generation_only_modules
    assert bundle.backend_handle is None
    assert set(bundle.trainable_modules) == {"transformer"}
    assert "pipeline" not in vars(bundle.model)
    with pytest.raises(RuntimeError, match="pipeline"):
        _ = bundle.model.pipeline


def test_wan_replay_builder_uses_wan_pipeline_scheduler_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks Wan replay builder uses Wan pipeline scheduler class."""
    from vrl.models.diffusion.wan_2_1 import runtime

    scheduler_classes: list[str] = []

    def fake_scheduler_loader(_spec: Any, class_name: str, **_kwargs: Any) -> _TinyScheduler:
        scheduler_classes.append(class_name)
        return _TinyScheduler()

    monkeypatch.setattr(
        runtime,
        "load_diffusers_transformer",
        lambda *_args, **_kwargs: _TinyTransformer(),
    )
    monkeypatch.setattr(runtime, "load_diffusers_scheduler", fake_scheduler_loader)

    bundle = runtime.build_wan_2_1_replay_runtime_bundle(_spec())

    assert scheduler_classes == ["UniPCMultistepScheduler"]
    assert bundle.scheduler.timesteps.tolist() == [1.0]


def test_wan_i2v_replay_builder_uses_i2v_replay_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks Wan I2V replay builder uses I2V replay model."""
    from vrl.models.diffusion.wan_2_1 import runtime
    from vrl.models.diffusion.wan_2_1.model import WanI2VReplayModel

    monkeypatch.setattr(
        runtime,
        "load_diffusers_transformer",
        lambda *_args, **_kwargs: _TinyTransformer(),
    )
    monkeypatch.setattr(
        runtime,
        "load_diffusers_scheduler",
        lambda *_args, **_kwargs: _TinyScheduler(),
    )

    bundle = runtime.build_wan_2_1_replay_runtime_bundle(
        _spec(task_variant="i2v"),
    )

    require_minimal_replay_bundle(bundle)
    assert isinstance(bundle.model, WanI2VReplayModel)
    assert bundle.runtime_caps["supports_reference_conditioning"] is True
    assert "image_encoder" in bundle.metadata["generation_only_modules"]


def test_cosmos_predict25_replay_builder_keeps_diffusion_nft_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks Cosmos predict25 replay builder keeps diffusion NFT surface."""
    from vrl.models.diffusion.cosmos import predict2_5
    from vrl.models.diffusion.cosmos.predict2_5 import runtime

    monkeypatch.setattr(
        runtime,
        "load_diffusers_transformer",
        lambda *_args, **_kwargs: _TinyTransformer(),
    )
    monkeypatch.setattr(
        runtime,
        "load_diffusers_scheduler",
        lambda *_args, **_kwargs: _TinyScheduler(),
    )
    monkeypatch.setattr(
        predict2_5.model.CosmosPredict25ReplayModel,
        "apply_lora",
        lambda self, _spec: self.transformer.requires_grad_(True),
    )

    bundle = runtime.build_cosmos_predict25_replay_runtime_bundle(
        _spec(
            task_variant="text2world",
            use_lora=True,
            lora_config={"rank": 1, "alpha": 1, "target_modules": ["to_q"]},
        ),
    )

    require_minimal_replay_bundle(bundle)
    assert bundle.backend_handle is None
    assert callable(bundle.model.diffusion_nft_prepare_transformer_input)
    with pytest.raises(RuntimeError, match="pipeline"):
        _ = bundle.model.pipeline


def test_anima_replay_builder_uses_only_transformer_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks Anima replay builder uses only transformer checkpoint."""
    from vrl.models.diffusion.cosmos.anima import runtime

    monkeypatch.setattr(
        runtime,
        "load_anima_transformer_component",
        lambda _spec: _TinyTransformer(),
    )

    bundle = runtime.build_anima_replay_runtime_bundle(
        _spec(
            task_variant="text_to_image",
            extra={
                "transformer_path": "/tmp/anima-preview3-base.safetensors",
                "scheduler_shift": 3.0,
            },
        ),
    )

    require_minimal_replay_bundle(bundle)
    profile = module_loading_profile_from_metadata(bundle.metadata)
    assert profile.runtime_role == MINIMAL_REPLAY_RUNTIME_ROLE
    assert profile.generation_only_modules == (
        "text_encoder",
        "llm_adapter",
        "vae",
        "tokenizers",
    )
    assert bundle.backend_handle is None
    assert set(bundle.trainable_modules) == {"transformer"}
    assert not hasattr(bundle.model, "text_encoder")
    assert not hasattr(bundle.model, "vae")
    with pytest.raises(RuntimeError, match="pipeline"):
        _ = bundle.model.pipeline
    with pytest.raises(RuntimeError, match="encode prompts"):
        bundle.model.encode_prompt("prompt")


def test_anima_empty_prompts_are_replaced_before_tokenization() -> None:
    """Checks Anima empty prompts are replaced before tokenization."""
    from vrl.models.diffusion.cosmos.anima.model import _non_empty_prompts

    assert _non_empty_prompts(["", "  ", "anime"]) == [".", ".", "anime"]


def test_anima_runtime_spec_uses_explicit_local_paths() -> None:
    """Checks Anima runtime spec uses explicit local paths."""
    from vrl.config.loading import load_config
    from vrl.models.diffusion.cosmos.anima.runtime import (
        extract_anima_replay_runtime_spec,
        extract_anima_runtime_spec,
    )

    cfg = load_config(
        "experiment/diffusion/anima_preview3/online_grpo_aesthetic",
        overrides=[
            "model.path=/models/anima",
            "model.transformer_path=/models/anima/transformer.safetensors",
            "model.text_encoder_path=/models/anima/text_encoder.safetensors",
            "model.vae_path=/models/anima/vae.safetensors",
            "model.qwen_tokenizer_path=/tokenizers/qwen",
            "model.t5_tokenizer_path=/tokenizers/t5",
            "sampling.num_steps=1",
            "model.use_lora=false",
        ],
    )

    full = extract_anima_runtime_spec(cfg, "cpu", torch.float32)
    replay = extract_anima_replay_runtime_spec(cfg, "cpu", torch.float32)

    assert full.model_config["transformer_path"] == "/models/anima/transformer.safetensors"
    assert full.model_config["text_encoder_path"] == "/models/anima/text_encoder.safetensors"
    assert full.model_config["vae_path"] == "/models/anima/vae.safetensors"
    assert full.model_config["qwen_tokenizer_path"] == "/tokenizers/qwen"
    assert full.model_config["t5_tokenizer_path"] == "/tokenizers/t5"
    assert replay.model_config["transformer_path"] == "/models/anima/transformer.safetensors"


def test_anima_runtime_spec_rejects_hf_repo_id_without_cached_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Checks Anima runtime spec rejects HF repo ID without cached artifacts."""
    from vrl.config.loading import load_config
    from vrl.models.diffusion.cosmos.anima.runtime import extract_anima_replay_runtime_spec

    cfg = load_config(
        "experiment/diffusion/anima_preview3/online_grpo_aesthetic",
        overrides=[
            "sampling.num_steps=1",
            "model.use_lora=false",
        ],
    )
    spec = extract_anima_replay_runtime_spec(cfg, "cpu", torch.float32)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty_hf_cache"))
    monkeypatch.delenv("HF_HOME", raising=False)

    with pytest.raises(ValueError, match=r"model\.path='circlestone-labs/Anima'"):
        spec.model_config["transformer_path"] = ""
        from vrl.models.diffusion.cosmos.anima.runtime import _resolve_artifact

        _resolve_artifact(
            spec.model_name_or_path,
            explicit_path="",
            relative_file=spec.model_config["transformer_file"],
            field_name="transformer_path",
        )


@pytest.mark.parametrize(
    ("module_path", "builder_name", "model_attr", "spec_kwargs"),
    [
        (
            "vrl.models.ar.janus_pro.runtime",
            "build_janus_pro_replay_runtime_bundle",
            "JanusProReplayModel",
            {"task_variant": "ar_t2i"},
        ),
        (
            "vrl.models.ar.nextstep_1.runtime",
            "build_nextstep_1_replay_runtime_bundle",
            "NextStep1ReplayModel",
            {"task_variant": "ar_t2i"},
        ),
    ],
)
def test_ar_replay_builders_return_minimal_bundles(
    monkeypatch: pytest.MonkeyPatch,
    module_path: str,
    builder_name: str,
    model_attr: str,
    spec_kwargs: dict[str, Any],
) -> None:
    """Checks AR replay builders return minimal bundles."""
    module = __import__(module_path, fromlist=[builder_name])
    monkeypatch.setattr(module, model_attr, _TinyRuntimeModel)

    bundle = getattr(module, builder_name)(_spec(**spec_kwargs))

    require_minimal_replay_bundle(bundle)
    assert bundle.backend_handle is None
    assert set(bundle.trainable_modules) == {"model"}
