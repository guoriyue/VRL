from __future__ import annotations

import contextlib
from typing import Any

import pytest
import torch
import torch.nn as nn

from vrl.models.interfaces.runtime import RuntimeBuildSpec
from vrl.models.replay_loading import bundle_loads_full_generation_modules


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
            "vrl.models.diffusion.wan_2_1.runtime",
            "build_wan_2_1_replay_runtime_bundle",
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
    # Single-transformer families (sd3_5) delegate loading to the shared
    # ``vrl.models.diffusion.build`` module, so the loaders must be patched
    # there; families not yet migrated (wan, cosmos) still bind the loaders in
    # their own runtime namespace. Patch both with ``raising=False`` so one set
    # of monkeypatches covers the mixed migration state.
    from vrl.models.diffusion import build as _shared_build

    for target in (module, _shared_build):
        monkeypatch.setattr(
            target,
            "load_diffusers_transformer",
            lambda *_args, **_kwargs: _TinyTransformer(),
            raising=False,
        )
        monkeypatch.setattr(
            target,
            "load_flow_match_scheduler",
            lambda *_args, **_kwargs: _TinyScheduler(),
            raising=False,
        )
        monkeypatch.setattr(
            target,
            "load_diffusers_scheduler",
            lambda *_args, **_kwargs: _TinyScheduler(),
            raising=False,
        )

    bundle = getattr(module, builder_name)(_spec())

    # The replay bundle is the trainer-side memory boundary: it does not own the
    # full generation modules (text encoders, VAE), so there is nothing to offload.
    assert bundle_loads_full_generation_modules(bundle) is False
    assert bundle.raw_handle is None
    assert set(bundle.trainable_modules) == {"transformer"}
    assert "pipeline" not in vars(bundle.model)
    with pytest.raises(RuntimeError, match="pipeline"):
        _ = bundle.model.pipeline


@pytest.mark.parametrize("family", ["sd3_5", "qwen_image", "flux", "cosmos-predict2", "sana", "lumina2", "hunyuan_video", "mochi", "cogvideox", "pixart_sigma", "hunyuan_image"])
def test_registry_descriptor_replay_builder_returns_minimal_bundle(
    monkeypatch: pytest.MonkeyPatch,
    family: str,
) -> None:
    """The descriptor-driven generic replay builder (descriptor families).

    These families ship no builder functions: the registry entry's
    ``DiffusionFamilyBuild`` recipe drives the generic builder, keyed by
    ``spec.family``. Behavioral contract matches the per-family builders above.
    """
    from vrl.models.diffusion import build as _shared_build

    monkeypatch.setattr(
        _shared_build,
        "load_diffusers_transformer",
        lambda *_args, **_kwargs: _TinyTransformer(),
    )
    monkeypatch.setattr(
        _shared_build,
        "load_flow_match_scheduler",
        lambda *_args, **_kwargs: _TinyScheduler(),
    )
    # Families with a scheduler_classname (cogvideox) load through the
    # classname path instead of the flow-match default.
    monkeypatch.setattr(
        _shared_build,
        "load_diffusers_scheduler",
        lambda *_args, **_kwargs: _TinyScheduler(),
    )

    bundle = _shared_build.build_family_replay_runtime_bundle(
        _spec(family=family),
    )

    assert bundle_loads_full_generation_modules(bundle) is False
    assert bundle.raw_handle is None
    assert set(bundle.trainable_modules) == {"transformer"}
    assert bundle.metadata["family"] == family
    with pytest.raises(RuntimeError, match="pipeline"):
        _ = bundle.model.pipeline

    # A spec without family fails loud instead of guessing.
    with pytest.raises(ValueError, match="spec.family"):
        _shared_build.build_family_replay_runtime_bundle(_spec())


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

    assert bundle_loads_full_generation_modules(bundle) is False
    assert isinstance(bundle.model, WanI2VReplayModel)
    assert bundle.runtime_caps["supports_reference_conditioning"] is True


def test_wan_dual_stage_replay_builder_loads_low_noise_transformer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks Wan 2.2 replay loads transformer_2 and trains it by default."""
    from vrl.models.diffusion.wan_2_1 import runtime

    loaded_subfolders: list[str] = []

    def fake_transformer_loader(
        _spec: Any,
        _class_name: str,
        *,
        subfolder: str = "transformer",
    ) -> _TinyTransformer:
        loaded_subfolders.append(subfolder)
        return _TinyTransformer()

    monkeypatch.setattr(runtime, "load_diffusers_transformer", fake_transformer_loader)
    monkeypatch.setattr(
        runtime,
        "load_diffusers_scheduler",
        lambda *_args, **_kwargs: _TinyScheduler(),
    )

    bundle = runtime.build_wan_2_1_replay_runtime_bundle(
        _spec(
            task_variant="i2v",
            extra={
                "boundary_ratio": 0.9,
                "trainable_transformers": ["transformer_2"],
            },
        ),
    )

    assert bundle_loads_full_generation_modules(bundle) is False
    assert loaded_subfolders == ["transformer_2", "transformer"]
    assert set(bundle.trainable_modules) == {"transformer_2"}
    # boundary_ratio is behavior-consumed on the model (dual-stage transformer
    # routing), not bundle metadata — assert the consumed surface.
    assert bundle.model.boundary_ratio == 0.9


def test_cosmos_predict25_replay_builder_keeps_diffusion_nft_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks Cosmos predict25 replay builder keeps diffusion NFT surface."""
    from vrl.models.diffusion import build as _shared_build
    from vrl.models.diffusion.cosmos import predict2_5

    # predict2_5 is a registry-descriptor family: the generic replay builder
    # constructs it, so the loaders are patched on the shared build module.
    monkeypatch.setattr(
        _shared_build,
        "load_diffusers_transformer",
        lambda *_args, **_kwargs: _TinyTransformer(),
    )
    monkeypatch.setattr(
        _shared_build,
        "load_diffusers_scheduler",
        lambda *_args, **_kwargs: _TinyScheduler(),
    )
    monkeypatch.setattr(
        predict2_5.model.CosmosPredict25ReplayModel,
        "apply_lora",
        lambda self, _spec: self.transformer.requires_grad_(True),
    )

    bundle = _shared_build.build_family_replay_runtime_bundle(
        _spec(
            family="cosmos-predict2.5",
            task_variant="text2world",
            use_lora=True,
            lora_config={"rank": 1, "alpha": 1, "target_modules": ["to_q"]},
        ),
    )

    assert bundle_loads_full_generation_modules(bundle) is False
    assert bundle.raw_handle is None
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
        "load_anima_transformer",
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

    assert bundle_loads_full_generation_modules(bundle) is False
    assert bundle.raw_handle is None
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
    from vrl.models.diffusion.build import extract_family_runtime_spec
    from vrl.models.diffusion.cosmos.anima.runtime import (
        extract_anima_replay_runtime_spec,
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

    full = extract_family_runtime_spec(cfg, "cpu", torch.float32)
    replay = extract_anima_replay_runtime_spec(cfg, "cpu", torch.float32)

    assert full.model_config["transformer_path"] == "/models/anima/transformer.safetensors"
    assert full.model_config["text_encoder_path"] == "/models/anima/text_encoder.safetensors"
    assert full.model_config["vae_path"] == "/models/anima/vae.safetensors"
    assert full.model_config["qwen_tokenizer_path"] == "/tokenizers/qwen"
    assert full.model_config["t5_tokenizer_path"] == "/tokenizers/t5"
    assert replay.model_config["transformer_path"] == "/models/anima/transformer.safetensors"


def test_anima_artifact_resolution_fails_loud_when_hub_fetch_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hub-fetch failure surfaces the config knob, not a raw download error."""
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

    # Resolution delegates to hf_hub_download (auto-fetch, same contract as
    # from_pretrained); when the hub fetch fails the error names the config
    # knob to set instead of leaking a bare download traceback.
    import huggingface_hub

    def _refuse(*_args: Any, **_kwargs: Any) -> str:
        raise OSError("offline")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _refuse)

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
            {"ar_task": "ar_t2i"},
        ),
        (
            "vrl.models.ar.nextstep_1.runtime",
            "build_nextstep_1_replay_runtime_bundle",
            "NextStep1ReplayModel",
            {"ar_task": "ar_t2i"},
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

    assert bundle_loads_full_generation_modules(bundle) is False
    assert bundle.raw_handle is None
    assert set(bundle.trainable_modules) == {"model"}
