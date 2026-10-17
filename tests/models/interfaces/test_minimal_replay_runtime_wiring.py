from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn

from vrl.config.precision import QuantizationPolicy, RolePrecision
from vrl.models.interfaces.runtime import ModelBuild, RolloutBuildOptions


def test_model_build_rejects_subbyte_parameter_storage() -> None:
    with pytest.raises(ValueError, match="neither is parameter storage"):
        ModelBuild(
            model_name_or_path="fake/repo",
            revision=None,
            device="cpu",
            parameter_dtype="fp8",
            family="sd3_5",
            precision=RolePrecision("fp32", "ieee", outer_autocast=False),
        )


def test_quantization_policy_rejects_unknown_format() -> None:
    with pytest.raises(ValueError, match=r"quantization\.format must be one of"):
        QuantizationPolicy(format="int8")


def _compile_scope_build(scope: str | None, *, rollout: bool) -> ModelBuild:
    torch_compile: dict[str, Any] = {"enable": True, "mode": "default"}
    if scope is not None:
        torch_compile["scope"] = scope
    return ModelBuild(
        model_name_or_path="fake/repo",
        revision=None,
        device="cpu",
        parameter_dtype=torch.float32,
        family="sd3_5",
        precision=RolePrecision("fp32", "tf32", outer_autocast=False),
        model_config={"torch_compile": torch_compile},
        rollout={"prompt_encoder_dtype": "fp32"} if rollout else None,
    )


@pytest.mark.parametrize(
    ("scope", "rollout_compiles", "replay_compiles"),
    [
        (None, True, True),  # absent scope means "all": today's behavior
        ("all", True, True),
        ("rollout", True, False),
        ("replay", False, True),
    ],
)
def test_torch_compile_property_resolves_scope_per_role(
    scope: str | None,
    rollout_compiles: bool,
    replay_compiles: bool,
) -> None:
    """One knob, two builds: each role reads only its own compile decision."""

    rollout_build = _compile_scope_build(scope, rollout=True)
    replay_build = _compile_scope_build(scope, rollout=False)

    expected = {"enable": True, "mode": "default", "regional": False}
    assert rollout_build.torch_compile == (expected if rollout_compiles else None)
    assert replay_build.torch_compile == (expected if replay_compiles else None)


def test_torch_compile_property_refuses_unknown_scope() -> None:
    build = _compile_scope_build("trainer", rollout=False)

    with pytest.raises(ValueError, match=r"torch_compile\.scope must be one of"):
        _ = build.torch_compile


def test_model_build_reconstructs_nested_rollout_payload() -> None:
    """The primitive Ray mapping becomes the typed one-layer rollout contract."""

    build = ModelBuild(
        model_name_or_path="fake/repo",
        revision=None,
        device="cpu",
        parameter_dtype="fp16",
        family="sd3_5",
        precision={
            "dtype": "bf16",
            "float32_precision": "tf32",
            "quantization": {"format": "fp8", "recipe": "rowwise"},
            "outer_autocast": True,
        },
        rollout={
            "prompt_encoder_dtype": "fp32",
            "base_weight_sync": False,
        },
    )

    assert build.parameter_dtype is torch.float16
    assert build.precision == RolePrecision(
        "bf16",
        "tf32",
        QuantizationPolicy(format="fp8", recipe="rowwise"),
    )
    assert build.precision.outer_autocast is True
    assert isinstance(build.rollout, RolloutBuildOptions)
    assert build.rollout.prompt_encoder_dtype is torch.float32
    assert build.precision.quantization is not None
    assert build.precision.quantization.format == "fp8"
    assert build.rollout.base_weight_sync is False


def test_model_build_resolver_projects_nvfp4_over_the_rollout_base_dtype() -> None:
    from omegaconf import OmegaConf

    from vrl.config.precision import PrecisionPolicy
    from vrl.config.schema import parse_config
    from vrl.models.families.registry import get_model_family_entry

    cfg = OmegaConf.create(
        {
            "model": {"family": "sd3_5", "path": "fake/repo"},
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
    root = parse_config(cfg)
    precision = PrecisionPolicy.from_section(root.precision)

    build = get_model_family_entry("sd3_5").resolve_model_build(
        root,
        "cuda",
        precision=precision,
        for_rollout=True,
    )
    rollout = build.require_rollout()

    assert build.parameter_dtype is torch.bfloat16
    assert build.precision == RolePrecision(
        "bf16",
        "tf32",
        QuantizationPolicy(format="nvfp4"),
    )
    assert build.precision.outer_autocast is True
    assert rollout.prompt_encoder_dtype is torch.float16
    assert build.precision.quantization is not None
    assert build.precision.quantization.format == "nvfp4"
    assert build.precision.quantization.recipe is None


def test_full_generation_build_with_training_role_excludes_rollout_quantization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Offline DPO may load generation modules without adopting rollout kernels."""

    from omegaconf import OmegaConf

    from vrl.config.precision import PrecisionPolicy
    from vrl.config.schema import parse_config
    from vrl.models.families.registry import get_model_family_entry

    monkeypatch.setattr(
        "diffusers.DiffusionPipeline.load_config",
        lambda *_args, **_kwargs: {
            "boundary_ratio": None,
            "expand_timesteps": False,
        },
    )
    cfg = OmegaConf.create(
        {
            "model": {
                "family": "wan_2_1",
                "path": "fake/repo",
                "revision": "a" * 40,
            },
            "precision": {
                "float32_precision": "tf32",
                "training": {"dtype": "bf16"},
                "rollout": {
                    "dtype": "bf16",
                    "quantization": {"format": "fp8"},
                },
            },
        },
    )
    root = parse_config(cfg)
    precision = PrecisionPolicy.from_section(root.precision)

    build = get_model_family_entry("wan_2_1").resolve_model_build(
        root,
        "cpu",
        precision=precision,
        for_rollout=True,
        precision_role="training",
    )

    assert isinstance(build.require_rollout(), RolloutBuildOptions)
    assert build.precision == RolePrecision("bf16", "tf32")
    assert build.precision.quantization is None


class _TinyTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    @property
    def dtype(self) -> torch.dtype:
        return self.weight.dtype


def _flow_match_scheduler(*_args: Any, **_kwargs: Any) -> Any:
    """Stand-in for ``load_flow_match_scheduler``: the real class from its defaults."""

    from diffusers import FlowMatchEulerDiscreteScheduler

    return FlowMatchEulerDiscreteScheduler()


def _diffusers_scheduler(_build: Any, class_name: str, **_kwargs: Any) -> Any:
    """Stand-in for ``load_diffusers_scheduler``.

    The registry's ``scheduler_classname`` is its own source of truth, so build
    exactly that real class from its defaults instead of echoing a double.
    """

    import diffusers

    return getattr(diffusers, class_name)()


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
        # Legacy ``extra`` test fields (artifact paths, scheduler_shift)
        # now ride directly in the carried model block.
        model_config.update(dict(extra))

    values: dict[str, Any] = {
        "model_name_or_path": "fake/repo",
        "revision": None,
        "device": "cpu",
        "parameter_dtype": torch.float32,
        "family": "sd3_5",
        "precision": RolePrecision("fp32", "tf32", outer_autocast=False),
        "model_config": model_config,
        "sampling_config": dict(scheduler_config),
    }
    values.update(overrides)
    if str(values["family"]).startswith("wan_2_1"):
        model_config.setdefault("boundary_ratio", None)
        model_config.setdefault(
            "trainable_transformers",
            ["transformer_2"] if model_config["boundary_ratio"] is not None else ["transformer"],
        )
    return ModelBuild(**values)


@pytest.mark.parametrize(
    "family",
    ["sd3_5", "sana", "wan_2_1"],
)
def test_registry_descriptor_replay_builder_returns_minimal_bundle(
    monkeypatch: pytest.MonkeyPatch,
    family: str,
) -> None:
    """The descriptor-driven generic replay builder (descriptor families).

    These families ship no builder functions: the registry entry's
    ``DenoiseFamilyBuild`` recipe drives the generic builder, keyed by
    ``build.family``. Behavioral contract matches the per-family builders above.
    """
    from vrl.models.families.registry import get_model_family_entry
    from vrl.models.steps.denoise import build as _shared_build

    loaded_builds: list[ModelBuild] = []

    def fake_transformer_loader(build: ModelBuild, *_args: Any, **_kwargs: Any):
        loaded_builds.append(build)
        return _TinyTransformer()

    monkeypatch.setattr(
        _shared_build,
        "load_diffusers_transformer",
        fake_transformer_loader,
    )
    monkeypatch.setattr(_shared_build, "load_flow_match_scheduler", _flow_match_scheduler)
    # Families with a scheduler_classname (wan_2_1) load through the
    # classname path instead of the flow-match default.
    monkeypatch.setattr(_shared_build, "load_diffusers_scheduler", _diffusers_scheduler)

    entry = get_model_family_entry(family)
    bundle = entry.build_replay(
        _build(
            family=family,
            parameter_dtype=torch.float16 if family == "sana" else torch.float32,
        ),
        expected_identity=None,
    )

    assert loaded_builds
    if family == "sana":
        assert loaded_builds[-1].parameter_dtype is torch.float16
    assert set(bundle.trainable_modules) == {"transformer"}
    assert not hasattr(bundle.model, "pipeline")


def test_model_build_rejects_missing_family() -> None:
    """ModelBuild rejects a missing identity before any registry dispatch."""
    with pytest.raises(ValueError, match=r"ModelBuild\.family"):
        _build(family="")


def test_wan_replay_builder_uses_wan_pipeline_scheduler_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wan descriptor's scheduler_classname drives the generic replay loader."""
    from vrl.models.families.registry import get_model_family_entry
    from vrl.models.steps.denoise import build as _shared_build

    scheduler_classes: list[str] = []

    def fake_scheduler_loader(build: Any, class_name: str, **kwargs: Any) -> Any:
        scheduler_classes.append(class_name)
        return _diffusers_scheduler(build, class_name, **kwargs)

    monkeypatch.setattr(
        _shared_build,
        "load_diffusers_transformer",
        lambda *_args, **_kwargs: _TinyTransformer(),
    )
    monkeypatch.setattr(_shared_build, "load_diffusers_scheduler", fake_scheduler_loader)

    bundle = get_model_family_entry("wan_2_1").build_replay(
        _build(family="wan_2_1"), expected_identity=None
    )

    from diffusers import UniPCMultistepScheduler

    assert scheduler_classes == ["UniPCMultistepScheduler"]
    assert isinstance(bundle.scheduler, UniPCMultistepScheduler)


def test_wan_i2v_replay_builder_uses_i2v_replay_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The i2v registry entry's replay_cls selects the I2V replay model."""
    from vrl.models.families.registry import get_model_family_entry
    from vrl.models.families.wan_2_1.model import WanI2VModel
    from vrl.models.steps.denoise import build as _shared_build

    monkeypatch.setattr(
        _shared_build,
        "load_diffusers_transformer",
        lambda *_args, **_kwargs: _TinyTransformer(),
    )
    monkeypatch.setattr(_shared_build, "load_diffusers_scheduler", _diffusers_scheduler)

    bundle = get_model_family_entry("wan_2_1_i2v").build_replay(
        _build(family="wan_2_1_i2v"), expected_identity=None
    )

    assert isinstance(bundle.model, WanI2VModel)


def test_wan_dual_stage_replay_builder_loads_low_noise_transformer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wan 2.2 dual-stage: prepare_replay late-loads transformer_2 and trains it."""
    import vrl.models.loader as _loader
    from vrl.models.families.registry import get_model_family_entry
    from vrl.models.steps.denoise import build as _shared_build

    loaded_subfolders: list[str] = []

    def fake_transformer_loader(
        _build: Any,
        _class_name: str,
        *,
        subfolder: str = "transformer",
        materialize_weights: bool = True,
    ) -> _TinyTransformer:
        del materialize_weights
        loaded_subfolders.append(subfolder)
        return _TinyTransformer()

    # The generic builder loads the primary transformer; prepare_replay
    # late-loads transformer_2 through vrl.models.loader — patch both.
    monkeypatch.setattr(_shared_build, "load_diffusers_transformer", fake_transformer_loader)
    monkeypatch.setattr(_loader, "load_diffusers_transformer", fake_transformer_loader)
    monkeypatch.setattr(_shared_build, "load_diffusers_scheduler", _diffusers_scheduler)

    bundle = get_model_family_entry("wan_2_1_i2v").build_replay(
        _build(
            family="wan_2_1_i2v",
            extra={
                "boundary_ratio": 0.9,
                "trainable_transformers": ["transformer_2"],
            },
        ),
        expected_identity=None,
    )

    # Primary first (generic ctor), then the prepare_replay late-load.
    assert loaded_subfolders == ["transformer", "transformer_2"]
    assert set(bundle.trainable_modules) == {"transformer_2"}
    # boundary_ratio is behavior-consumed on the model (dual-stage transformer
    # routing), not bundle metadata — assert the consumed surface.
    assert bundle.model.boundary_ratio == 0.9


def test_cosmos_predict25_replay_builder_is_pipeline_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The predict2.5 replay bundle is minimal: no pipeline, no raw handle, and the
    shared replay forward the objectives call is still reachable on the model."""
    from vrl.models.families.cosmos.predict2_5 import model as predict25_model
    from vrl.models.families.registry import get_model_family_entry
    from vrl.models.steps.denoise import build as _shared_build

    # predict2_5 is a registry-descriptor family: the generic replay builder
    # constructs it, so the loaders are patched on the shared build module.
    monkeypatch.setattr(
        _shared_build,
        "load_diffusers_transformer",
        lambda *_args, **_kwargs: _TinyTransformer(),
    )
    monkeypatch.setattr(_shared_build, "load_diffusers_scheduler", _diffusers_scheduler)
    monkeypatch.setattr(
        predict25_model.CosmosPredict25Model,
        "apply_lora",
        lambda self, _build: self.transformer.requires_grad_(True),
    )

    bundle = get_model_family_entry("cosmos-predict2.5").build_replay(
        _build(
            family="cosmos-predict2.5",
            use_lora=True,
            lora_config={"rank": 1, "alpha": 1, "target_modules": ["to_q"]},
        ),
        expected_identity=None,
    )

    assert callable(bundle.model.replay_forward_with_latents)
    assert not hasattr(bundle.model, "pipeline")
