from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from vrl.models.interfaces.runtime import (
    ForwardPrecision,
    ModelBuild,
    RolloutBuildOptions,
    bundle_loads_full_generation_modules,
    full_generation_bundle_metadata,
    minimal_replay_bundle_metadata,
)


def test_forward_precision_rejects_quantized_outer_autocast() -> None:
    """Quantization formats are selective GEMM policies, not autocast dtypes."""

    with pytest.raises(ValueError, match="forward autocast must be one of"):
        ForwardPrecision(
            autocast="fp8",  # type: ignore[arg-type]
            float32_precision="tf32",
        )


def test_forward_precision_rejects_unknown_float32_backend() -> None:
    with pytest.raises(ValueError, match="forward float32_precision must be one of"):
        ForwardPrecision(
            autocast="off",
            float32_precision="fast",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("dtype", ["fp8", "fp4"])
def test_model_build_rejects_subbyte_parameter_storage(dtype: str) -> None:
    with pytest.raises(ValueError, match="neither is parameter storage"):
        ModelBuild(
            model_name_or_path="fake/repo",
            device="cpu",
            parameter_dtype=dtype,
            family="sd3_5",
            forward_precision=ForwardPrecision("off", "ieee"),
        )


@pytest.mark.parametrize(
    ("quantization_format", "message"),
    [("fp4", "replaced.*nvfp4"), ("int8", "quantization.format")],
)
def test_rollout_build_options_rejects_unknown_or_ambiguous_quantization(
    quantization_format: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RolloutBuildOptions(
            prompt_encoder_dtype="fp16",
            quantization_format=quantization_format,
        )


def test_rollout_build_options_normalizes_fp8_default_recipe() -> None:
    options = RolloutBuildOptions(
        prompt_encoder_dtype="bf16",
        quantization_format="FP8",
    )

    assert options.quantization_format == "fp8"
    assert options.quantization_recipe == "rowwise"


def test_rollout_build_options_accepts_nvfp4_without_recipe() -> None:
    options = RolloutBuildOptions(
        prompt_encoder_dtype="bf16",
        quantization_format="NVFP4",
    )

    assert options.quantization_format == "nvfp4"
    assert options.quantization_recipe is None


def test_rollout_build_options_rejects_nvfp4_recipe() -> None:
    with pytest.raises(ValueError, match=r"nvfp4.*does not accept.*recipe"):
        RolloutBuildOptions(
            prompt_encoder_dtype="bf16",
            quantization_format="nvfp4",
            quantization_recipe="rowwise",
        )


def test_model_build_reconstructs_nested_rollout_payload() -> None:
    """The primitive Ray mapping becomes the typed one-layer rollout contract."""

    build = ModelBuild(
        model_name_or_path="fake/repo",
        device="cpu",
        parameter_dtype="fp16",
        family="sd3_5",
        forward_precision={
            "autocast": "bf16",
            "float32_precision": "tf32",
        },
        rollout={
            "prompt_encoder_dtype": "fp32",
            "quantization_format": "fp8",
            "quantization_recipe": "rowwise",
            "base_weight_sync": False,
        },
    )

    assert build.parameter_dtype is torch.float16
    assert build.forward_precision == ForwardPrecision("bf16", "tf32")
    assert isinstance(build.rollout, RolloutBuildOptions)
    assert build.rollout.prompt_encoder_dtype is torch.float32
    assert build.rollout.quantization_format == "fp8"
    assert build.rollout.base_weight_sync is False


def test_model_build_resolver_projects_nvfp4_over_the_rollout_base_dtype() -> None:
    from omegaconf import OmegaConf

    from vrl.families.registry import get_model_family_entry

    cfg = OmegaConf.create(
        {
            "model": {"path": "fake/repo"},
            "precision": {
                "float32_precision": "tf32",
                "training": {"dtype": "bf16"},
                "rollout": {
                    "dtype": "bf16",
                    "quantization": {"format": "nvfp4"},
                    "prompt_encoders": {"dtype": "fp16"},
                },
            },
        },
    )

    build = get_model_family_entry("sd3_5").resolve_model_build(
        cfg,
        "cuda",
        for_rollout=True,
    )
    rollout = build.require_rollout()

    assert build.parameter_dtype is torch.bfloat16
    assert build.forward_precision == ForwardPrecision("bf16", "tf32")
    assert rollout.prompt_encoder_dtype is torch.float16
    assert rollout.quantization_format == "nvfp4"
    assert rollout.quantization_recipe is None


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


def _build(**overrides: Any) -> ModelBuild:
    """Build a ModelBuild from friendly overrides.

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
        "parameter_dtype": torch.float32,
        "family": "sd3_5",
        "forward_precision": ForwardPrecision("off", "tf32"),
        "model_config": model_config,
        "sampling_config": dict(scheduler_config),
    }
    values.update(overrides)
    return ModelBuild(**values)


@pytest.mark.parametrize(
    "family",
    [
        "sd3_5",
        "qwen_image",
        "flux",
        "cosmos-predict2",
        "sana",
        "lumina2",
        "hunyuan_video",
        "mochi",
        "cogvideox",
        "pixart_sigma",
        "hunyuan_image",
        "wan_2_1",
    ],
)
def test_registry_descriptor_replay_builder_returns_minimal_bundle(
    monkeypatch: pytest.MonkeyPatch,
    family: str,
) -> None:
    """The descriptor-driven generic replay builder (descriptor families).

    These families ship no builder functions: the registry entry's
    ``DiffusionFamilyBuild`` recipe drives the generic builder, keyed by
    ``build.family``. Behavioral contract matches the per-family builders above.
    """
    from vrl.families.registry import get_model_family_entry
    from vrl.models.diffusion import build as _shared_build

    loaded_builds: list[ModelBuild] = []

    def fake_transformer_loader(build: ModelBuild, *_args: Any, **_kwargs: Any):
        loaded_builds.append(build)
        return _TinyTransformer()

    monkeypatch.setattr(
        _shared_build,
        "load_diffusers_transformer",
        fake_transformer_loader,
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

    entry = get_model_family_entry(family)
    bundle = entry.build_replay(
        _build(
            family=family,
            parameter_dtype=torch.float16 if family == "sana" else torch.float32,
        ),
    )

    assert bundle_loads_full_generation_modules(bundle) is False
    assert loaded_builds
    if family == "sana":
        assert loaded_builds[-1].parameter_dtype is torch.float16
    assert bundle.raw_handle is None
    assert set(bundle.trainable_modules) == {"transformer"}
    with pytest.raises(RuntimeError, match="pipeline"):
        _ = bundle.model.pipeline

    # ModelBuild rejects a missing identity before any registry dispatch.
    with pytest.raises(ValueError, match=r"ModelBuild\.family"):
        _build(family="")


@pytest.mark.parametrize(
    "entry_method",
    ["build_rollout", "build_replay"],
)
@pytest.mark.parametrize("parameter_dtype", [torch.bfloat16, torch.float32])
def test_sana_builders_enforce_family_parameter_dtype(
    entry_method: str,
    parameter_dtype: torch.dtype,
) -> None:
    """Manual builds cannot bypass SANA's native-FP16 boundary."""
    from vrl.families.registry import get_model_family_entry

    with pytest.raises(ValueError, match=r"sana.*supports base parameter dtypes.*fp16"):
        getattr(get_model_family_entry("sana"), entry_method)(
            _build(family="sana", parameter_dtype=parameter_dtype),
        )


def test_wan_replay_builder_uses_wan_pipeline_scheduler_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wan descriptor's scheduler_classname drives the generic replay loader."""
    from vrl.families.registry import get_model_family_entry
    from vrl.models.diffusion import build as _shared_build

    scheduler_classes: list[str] = []

    def fake_scheduler_loader(_build: Any, class_name: str, **_kwargs: Any) -> _TinyScheduler:
        scheduler_classes.append(class_name)
        return _TinyScheduler()

    monkeypatch.setattr(
        _shared_build,
        "load_diffusers_transformer",
        lambda *_args, **_kwargs: _TinyTransformer(),
    )
    monkeypatch.setattr(_shared_build, "load_diffusers_scheduler", fake_scheduler_loader)

    bundle = get_model_family_entry("wan_2_1").build_replay(
        _build(family="wan_2_1"),
    )

    assert scheduler_classes == ["UniPCMultistepScheduler"]
    assert bundle.scheduler.timesteps.tolist() == [1.0]


def test_wan_i2v_replay_builder_uses_i2v_replay_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The i2v registry entry's replay_cls selects the I2V replay model."""
    from vrl.families.registry import get_model_family_entry
    from vrl.models.diffusion import build as _shared_build
    from vrl.models.diffusion.wan_2_1.model import WanI2VReplayModel

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

    bundle = get_model_family_entry("wan_2_1_i2v").build_replay(
        _build(family="wan_2_1_i2v"),
    )

    assert bundle_loads_full_generation_modules(bundle) is False
    assert isinstance(bundle.model, WanI2VReplayModel)


def test_wan_dual_stage_replay_builder_loads_low_noise_transformer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wan 2.2 dual-stage: prepare_replay late-loads transformer_2 and trains it."""
    import vrl.models.loader as _loader
    from vrl.families.registry import get_model_family_entry
    from vrl.models.diffusion import build as _shared_build

    loaded_subfolders: list[str] = []

    def fake_transformer_loader(
        _build: Any,
        _class_name: str,
        *,
        subfolder: str = "transformer",
    ) -> _TinyTransformer:
        loaded_subfolders.append(subfolder)
        return _TinyTransformer()

    # The generic builder loads the primary transformer; prepare_replay
    # late-loads transformer_2 through vrl.models.loader — patch both.
    monkeypatch.setattr(_shared_build, "load_diffusers_transformer", fake_transformer_loader)
    monkeypatch.setattr(_loader, "load_diffusers_transformer", fake_transformer_loader)
    monkeypatch.setattr(
        _shared_build,
        "load_diffusers_scheduler",
        lambda *_args, **_kwargs: _TinyScheduler(),
    )

    bundle = get_model_family_entry("wan_2_1_i2v").build_replay(
        _build(
            family="wan_2_1_i2v",
            extra={
                "boundary_ratio": 0.9,
                "trainable_transformers": ["transformer_2"],
            },
        ),
    )

    assert bundle_loads_full_generation_modules(bundle) is False
    # Primary first (generic ctor), then the prepare_replay late-load.
    assert loaded_subfolders == ["transformer", "transformer_2"]
    assert set(bundle.trainable_modules) == {"transformer_2"}
    # boundary_ratio is behavior-consumed on the model (dual-stage transformer
    # routing), not bundle metadata — assert the consumed surface.
    assert bundle.model.boundary_ratio == 0.9


def test_cosmos_predict25_replay_builder_keeps_diffusion_nft_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks Cosmos predict25 replay builder keeps diffusion NFT surface."""
    from vrl.families.registry import get_model_family_entry
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
        lambda self, _build: self.transformer.requires_grad_(True),
    )

    bundle = get_model_family_entry("cosmos-predict2.5").build_replay(
        _build(
            family="cosmos-predict2.5",
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
        lambda _build: _TinyTransformer(),
    )

    bundle = runtime.build_anima_replay_runtime_bundle(
        _build(
            family="cosmos-predict2-anima",
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


def test_anima_model_build_uses_explicit_local_paths() -> None:
    """Checks the Anima model build uses explicit local paths."""
    from vrl.config.loading import load_config
    from vrl.families.registry import get_model_family_entry

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

    entry = get_model_family_entry("cosmos-predict2-anima")
    full = entry.resolve_model_build(cfg, "cpu")
    replay = entry.resolve_model_build(cfg, "cpu", for_rollout=False)

    assert isinstance(full.rollout, RolloutBuildOptions)
    assert replay.rollout is None
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
    from vrl.families.registry import get_model_family_entry

    cfg = load_config(
        "experiment/diffusion/anima_preview3/online_grpo_aesthetic",
        overrides=[
            "sampling.num_steps=1",
            "model.use_lora=false",
        ],
    )
    build = get_model_family_entry("cosmos-predict2-anima").resolve_model_build(
        cfg,
        "cpu",
        for_rollout=False,
    )

    # Resolution delegates to hf_hub_download (auto-fetch, same contract as
    # from_pretrained); when the hub fetch fails the error names the config
    # knob to set instead of leaking a bare download traceback.
    import huggingface_hub

    def _refuse(*_args: Any, **_kwargs: Any) -> str:
        raise OSError("offline")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _refuse)

    with pytest.raises(ValueError, match=r"model\.path='circlestone-labs/Anima'"):
        build.model_config["transformer_path"] = ""
        from vrl.models.diffusion.cosmos.anima.runtime import _resolve_artifact

        _resolve_artifact(
            build.model_name_or_path,
            explicit_path="",
            relative_file=build.model_config["transformer_file"],
            field_name="transformer_path",
        )


@pytest.mark.parametrize(
    ("family", "model_module_path", "model_attr"),
    [
        (
            "janus_pro",
            "vrl.models.ar.janus_pro.model",
            "JanusProReplayModel",
        ),
        (
            "janus_pro_r1",
            "vrl.models.ar.janus_pro.model",
            "JanusProReplayModel",
        ),
        (
            "nextstep_1",
            "vrl.models.ar.nextstep_1.model",
            "NextStep1ReplayModel",
        ),
        (
            "emu3",
            "vrl.models.ar.emu3.model",
            "Emu3ReplayModel",
        ),
        (
            "glm_image",
            "vrl.models.ar.glm_image.model",
            "GlmImageReplayModel",
        ),
        (
            "llamagen",
            "vrl.models.ar.llamagen.model",
            "LlamaGenReplayModel",
        ),
    ],
)
def test_ar_replay_builders_return_minimal_bundles(
    monkeypatch: pytest.MonkeyPatch,
    family: str,
    model_module_path: str,
    model_attr: str,
) -> None:
    """Checks AR replay builders return minimal bundles."""
    from vrl.families.registry import get_model_family_entry

    model_module = __import__(model_module_path, fromlist=[model_attr])
    monkeypatch.setattr(model_module, model_attr, _TinyRuntimeModel)

    bundle = get_model_family_entry(family).build_replay(
        _build(family=family),
    )

    assert bundle_loads_full_generation_modules(bundle) is False
    assert bundle.raw_handle is None
    assert set(bundle.trainable_modules) == {"model"}


@pytest.mark.parametrize(
    ("family", "model_module_path", "model_attr"),
    [
        ("janus_pro", "vrl.models.ar.janus_pro.model", "JanusProModel"),
        (
            "janus_pro_r1",
            "vrl.models.ar.janus_pro.model",
            "JanusProModel",
        ),
        (
            "nextstep_1",
            "vrl.models.ar.nextstep_1.model",
            "NextStep1Model",
        ),
        ("emu3", "vrl.models.ar.emu3.model", "Emu3Model"),
        ("glm_image", "vrl.models.ar.glm_image.model", "GlmImageModel"),
        ("llamagen", "vrl.models.ar.llamagen.model", "LlamaGenModel"),
    ],
)
def test_ar_rollout_builders_follow_registry_descriptors(
    monkeypatch: pytest.MonkeyPatch,
    family: str,
    model_module_path: str,
    model_attr: str,
) -> None:
    from vrl.families.registry import get_model_family_entry

    model_module = __import__(model_module_path, fromlist=[model_attr])
    monkeypatch.setattr(model_module, model_attr, _TinyRuntimeModel)

    bundle = get_model_family_entry(family).build_rollout(
        _build(
            family=family,
            rollout=RolloutBuildOptions(
                prompt_encoder_dtype=torch.float16,
            ),
        ),
    )

    assert bundle_loads_full_generation_modules(bundle) is True
    assert bundle.raw_handle is bundle.model


def test_bundle_metadata_drives_consumer_down_opposite_branches() -> None:
    """The two bundle builders are consumed into opposite ownership decisions.

    Asserts the behavior the metadata exists for — ``bundle_loads_full_generation_modules``
    returning True for a full-generation bundle and False for a minimal one — instead
    of mirroring each builder's literal ``{KEY: bool}`` return value.
    """
    full = SimpleNamespace(metadata=full_generation_bundle_metadata())
    minimal = SimpleNamespace(metadata=minimal_replay_bundle_metadata())

    assert bundle_loads_full_generation_modules(full) is True
    assert bundle_loads_full_generation_modules(minimal) is False


def test_bundle_loads_full_generation_modules_defaults_false() -> None:
    """Missing flag, missing metadata, or null metadata all read as not-owning."""
    assert bundle_loads_full_generation_modules(SimpleNamespace(metadata={})) is False
    assert bundle_loads_full_generation_modules(SimpleNamespace()) is False
    assert bundle_loads_full_generation_modules(SimpleNamespace(metadata=None)) is False
