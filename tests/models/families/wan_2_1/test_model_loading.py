"""Download-free ``from_build`` loading test for Wan 2.1 Image-to-Video.

Mirrors the shared ``from_build`` tests in
``tests/models/steps/denoise/common/test_model_base.py``: monkeypatch the
diffusers pipeline ``from_pretrained`` to return a fake pipeline (no Hub fetch,
no real weights) and assert the loader-constructed state, NOT literal YAML
config values: the vae is staged in fp32 and the text/image encoders at the
build dtype on the build device (``WanI2VPipelineModel.from_build`` in
``vrl/models/wan_2_1/model.py``), the generation-only modules (vae /
text_encoder / image_encoder) are frozen and the progress bar is disabled;
``model.offload`` reaches the rollout build only. Wan 2.2 A14B dual-stage pipelines
are accepted when they expose ``transformer_2``; expand-timesteps pipelines
remain unsupported.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.models.steps.denoise.fixtures import RecordingModule
from vrl.config.precision import RolePrecision
from vrl.models.interfaces.runtime import ModelBuild, RolloutBuildOptions
from vrl.nn.offload import Offload


def _rollout_build_options() -> RolloutBuildOptions:
    return RolloutBuildOptions(prompt_encoder_dtype=torch.bfloat16)


class _FakePipeline:
    def __init__(self) -> None:
        self.transformer = RecordingModule()
        self.transformer_2 = None
        self.vae = RecordingModule()
        self.text_encoder = RecordingModule()
        self.image_encoder = RecordingModule()
        self.device = "cpu"
        # Single-transformer Wan 2.1 I2V: no boundary_ratio / expand_timesteps.
        self.config = SimpleNamespace(boundary_ratio=None, expand_timesteps=False)
        self.progress_bar_disabled: bool | None = None

    def set_progress_bar_config(self, *, disable: bool) -> None:
        self.progress_bar_disabled = disable


def _canonical_model_config(**overrides: Any) -> dict[str, Any]:
    return {
        "boundary_ratio": None,
        "trainable_transformers": ["transformer"],
        **overrides,
    }


def _i2v_build(
    *,
    model_name_or_path: str = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
    device: torch.device | None = None,
    rollout: RolloutBuildOptions | None = None,
    **model_config: Any,
) -> ModelBuild:
    """A Wan I2V build; keyword overrides ride the canonical ``model_config`` block."""
    return ModelBuild(
        model_name_or_path=model_name_or_path,
        revision=None,
        device=device if device is not None else torch.device("cuda:0"),
        parameter_dtype=torch.bfloat16,
        family="wan_2_1_i2v",
        precision=RolePrecision("bf16", "tf32"),
        model_config=_canonical_model_config(**model_config),
        rollout=rollout if rollout is not None else _rollout_build_options(),
    )


def _patch_from_pretrained(monkeypatch) -> tuple[_FakePipeline, list[dict[str, Any]]]:
    from diffusers import WanImageToVideoPipeline

    calls: list[dict[str, Any]] = []
    pipeline = _FakePipeline()

    def fake_from_pretrained(model_name_or_path: str, **kwargs: Any) -> _FakePipeline:
        calls.append({"model_name_or_path": model_name_or_path, **kwargs})
        return pipeline

    monkeypatch.setattr(
        WanImageToVideoPipeline,
        "from_pretrained",
        staticmethod(fake_from_pretrained),
    )
    return pipeline, calls


def _assert_frozen_and_loaded(pipeline: _FakePipeline, calls: list[dict[str, Any]]) -> None:
    """Shared ``from_build`` assertions: the load call and the freezing."""
    assert calls == [
        {
            "model_name_or_path": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
            "torch_dtype": {
                "default": torch.bfloat16,
                "vae": torch.float32,
                "text_encoder": torch.bfloat16,
                "image_encoder": torch.bfloat16,
            },
        },
    ]
    assert pipeline.progress_bar_disabled is True
    for module in (pipeline.vae, pipeline.text_encoder, pipeline.image_encoder):
        assert module.requires_grad_enabled is False


def test_wan_resolver_projects_offload_to_rollout_only(monkeypatch) -> None:
    from diffusers import DiffusionPipeline
    from omegaconf import OmegaConf

    from vrl.models.families.registry import get_model_family_entry

    monkeypatch.setattr(
        DiffusionPipeline,
        "load_config",
        staticmethod(lambda *a, **k: {"boundary_ratio": None}),
    )
    cfg = OmegaConf.create(
        {
            "model": {
                "family": "wan_2_1",
                "path": "fake/repo",
                "revision": "a" * 40,
                "offload": {},
                "use_lora": False,
            },
            "precision": {
                "float32_precision": "tf32",
                "training": {"dtype": "bf16"},
                "rollout": {"dtype": "bf16"},
            },
            "distributed": {"training": {"strategy": "single_process"}},
        },
    )
    entry = get_model_family_entry("wan_2_1")
    from vrl.config.precision import PrecisionPolicy
    from vrl.config.schema import parse_config

    root = parse_config(cfg)
    precision = PrecisionPolicy.from_section(root.precision)
    rollout = entry.resolve_model_build(root, "cpu", precision=precision, for_rollout=True)
    replay = entry.resolve_model_build(root, "cpu", precision=precision, for_rollout=False)

    assert rollout.require_rollout().offload == Offload()
    assert replay.rollout is None
    assert "offload" not in (rollout.model_config or {})
    assert "offload" not in (replay.model_config or {})


def test_wan_full_finetune_normalizes_rollout_parameter_dtype() -> None:
    """Rollout storage must accept the exact BF16 payload exported by FSDP."""
    from torch import nn

    from vrl.models.families.wan_2_1.model import WanT2VPipelineModel

    class _MixedTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bf16_weight = nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
            self.fp32_table = nn.Parameter(torch.ones(2, dtype=torch.float32))

    pipeline = SimpleNamespace(
        transformer=_MixedTransformer(),
        transformer_2=None,
        config=SimpleNamespace(boundary_ratio=None),
    )
    model = WanT2VPipelineModel(pipeline=pipeline, device=torch.device("cpu"))
    build = SimpleNamespace(
        parameter_dtype=torch.bfloat16,
        model_config=_canonical_model_config(),
        rollout=_rollout_build_options(),
    )

    model.apply_full_finetune(build)
    payload = {
        f"transformer.{name}": torch.zeros_like(parameter)
        for name, parameter in pipeline.transformer.named_parameters()
    }
    model.validate_trainable_state(payload)

    assert {parameter.dtype for parameter in pipeline.transformer.parameters()} == {
        torch.bfloat16,
    }
    assert all(parameter.requires_grad for parameter in pipeline.transformer.parameters())


@pytest.mark.parametrize("dual", [False, True])
def test_wan_replay_loads_trainable_state_without_pipeline(dual) -> None:
    from vrl.models.families.wan_2_1.model import WanT2VModel

    model = WanT2VModel(
        transformer=torch.nn.Linear(2, 2),
        transformer_2=torch.nn.Linear(2, 2) if dual else None,
        scheduler=object(),
        device=torch.device("cpu"),
        boundary_ratio=0.875 if dual else None,
        trainable_transformers=("transformer", "transformer_2") if dual else ("transformer",),
    )
    payload = {}
    for root, module in model.trainable_modules.items():
        module.requires_grad_(True)
        payload.update(
            {
                f"{root}.{name}": torch.full_like(value, 0.25)
                for name, value in module.named_parameters()
            }
        )
    model.load_trainable_state(payload)
    for module in model.trainable_modules.values():
        for parameter in module.parameters():
            assert torch.equal(parameter.detach(), torch.full_like(parameter, 0.25))


def test_wan_replay_full_finetune_leaves_placement_to_the_strategy() -> None:
    """A replay model never moves or casts its experts at attach.

    ``prepare_model`` owns placement, and FSDP normalizes the storage dtype
    right before sharding.
    """
    from vrl.models.families.wan_2_1.model import WanT2VModel

    transformer = RecordingModule()
    model = WanT2VModel(
        transformer=transformer,
        scheduler=object(),
        device=torch.device("cuda:1"),
    )
    build = SimpleNamespace(
        parameter_dtype=torch.bfloat16,
        model_config=_canonical_model_config(),
        rollout=None,
    )

    model.apply_full_finetune(build)

    assert transformer.to_calls == []


def test_wan_i2v_from_build_stages_frozen_modules(monkeypatch) -> None:
    """``from_build`` stages the VAE in fp32 and the encoders at the build dtype."""
    from vrl.models.families.wan_2_1.model import WanI2VPipelineModel

    pipeline, calls = _patch_from_pretrained(monkeypatch)

    build = _i2v_build()

    model = WanI2VPipelineModel.from_build(build)

    assert model.pipeline is pipeline
    _assert_frozen_and_loaded(pipeline, calls)
    # vae stays fp32; text/image encoders ride the build dtype.
    assert pipeline.vae.to_calls == [(build.device, torch.float32)]
    assert pipeline.text_encoder.to_calls == [(build.device, torch.bfloat16)]
    assert pipeline.image_encoder.to_calls == [(build.device, torch.bfloat16)]


def test_wan_i2v_from_build_honors_local_files_only(monkeypatch) -> None:
    from vrl.models.families.wan_2_1.model import WanI2VPipelineModel

    _, calls = _patch_from_pretrained(monkeypatch)
    build = _i2v_build(local_files_only=True)

    WanI2VPipelineModel.from_build(build)

    assert calls[0]["local_files_only"] is True


def test_wan_i2v_from_build_accepts_dual_stage_pipeline(monkeypatch) -> None:
    """Wan 2.2 A14B dual-stage pipelines train the low-noise transformer by default."""
    from vrl.models.families.wan_2_1.model import WanI2VPipelineModel

    pipeline, _ = _patch_from_pretrained(monkeypatch)
    pipeline.config = SimpleNamespace(boundary_ratio=0.5, expand_timesteps=False)
    pipeline.transformer_2 = RecordingModule()

    build = _i2v_build(
        boundary_ratio=0.5,
        trainable_transformers=["transformer_2"],
    )

    model = WanI2VPipelineModel.from_build(build)

    assert model.boundary_ratio == 0.5
    assert model.transformer_2 is pipeline.transformer_2
    assert model.trainable_modules == {"transformer_2": pipeline.transformer_2}


def test_wan_model_build_normalization_is_shared_by_replay_and_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from diffusers import DiffusionPipeline
    from omegaconf import OmegaConf

    from vrl.config.precision import PrecisionPolicy
    from vrl.config.schema import parse_config
    from vrl.models.families.registry import get_model_family_entry

    revision = "a" * 40
    calls: list[dict[str, Any]] = []

    def fake_load_config(model_name_or_path: str, **kwargs: Any) -> dict[str, Any]:
        calls.append({"model_name_or_path": model_name_or_path, **kwargs})
        return {"boundary_ratio": 0.9, "expand_timesteps": False}

    monkeypatch.setattr(
        DiffusionPipeline,
        "load_config",
        staticmethod(fake_load_config),
    )
    root = parse_config(
        OmegaConf.create(
            {
                "model": {
                    "family": "wan_2_1_i2v",
                    "path": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                    "revision": revision,
                    "trainable_transformers": "both",
                    "use_lora": False,
                },
                "precision": {
                    "float32_precision": "ieee",
                    "training": {"dtype": "fp32"},
                    "rollout": {"dtype": "fp32"},
                },
            },
        ),
    )
    precision = PrecisionPolicy.from_section(root.precision)
    entry = get_model_family_entry("wan_2_1_i2v")

    replay = entry.resolve_model_build(
        root,
        "cpu",
        precision=precision,
        for_rollout=False,
    )
    rollout = entry.resolve_model_build(
        root,
        "cpu",
        precision=precision,
        for_rollout=True,
    )

    expected_topology = {
        "boundary_ratio": 0.9,
        "trainable_transformers": ["transformer", "transformer_2"],
    }
    assert {key: replay.model_config[key] for key in expected_topology} == expected_topology
    assert {key: rollout.model_config[key] for key in expected_topology} == expected_topology
    assert calls == [
        {
            "model_name_or_path": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            "revision": revision,
        },
        {
            "model_name_or_path": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            "revision": revision,
        },
    ]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, ("transformer_2",)),
        (
            ["transformer_2", "transformer", "transformer_2"],
            ("transformer", "transformer_2"),
        ),
    ],
)
def test_wan_trainable_transformer_spellings_share_one_canonical_order(
    value: object,
    expected: tuple[str, ...],
) -> None:
    from vrl.models.families.wan_2_1.config import (
        normalize_wan_trainable_transformers,
    )

    assert normalize_wan_trainable_transformers(value, dual_stage=True) == expected


def test_wan_single_stage_rejects_low_noise_transformer_selection() -> None:
    from vrl.models.families.wan_2_1.config import (
        normalize_wan_trainable_transformers,
    )

    with pytest.raises(ValueError, match=r"transformer_2.*allowed=.*transformer"):
        normalize_wan_trainable_transformers(
            ["transformer_2"],
            dual_stage=False,
        )


def test_wan_build_normalization_rejects_expand_timesteps_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Wan 2.2 5B expand-timesteps pipelines still need a separate runner contract."""
    from diffusers import DiffusionPipeline

    from vrl.models.families.wan_2_1.config import normalize_wan_model_build

    monkeypatch.setattr(
        DiffusionPipeline,
        "load_config",
        staticmethod(lambda *_args, **_kwargs: {"boundary_ratio": 0.5, "expand_timesteps": True}),
    )
    build = SimpleNamespace(
        family="wan_2_1_i2v",
        model_name_or_path=str(tmp_path),
        revision=None,
        revision_kwargs={},
        model_config={},
    )

    with pytest.raises(NotImplementedError, match="expand_timesteps"):
        normalize_wan_model_build(build)


def test_wan_build_normalization_rejects_unpinned_remote_before_config_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from diffusers import DiffusionPipeline

    from vrl.models.families.wan_2_1.config import normalize_wan_model_build

    calls: list[str] = []

    def fake_load_config(model_name_or_path: str, **_kwargs: Any) -> dict[str, Any]:
        calls.append(model_name_or_path)
        return {"boundary_ratio": 0.9, "expand_timesteps": False}

    monkeypatch.setattr(
        DiffusionPipeline,
        "load_config",
        staticmethod(fake_load_config),
    )
    build = SimpleNamespace(
        family="wan_2_1_i2v",
        model_name_or_path="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        revision=None,
        model_config={"trainable_transformers": ["transformer_2"]},
    )

    with pytest.raises(ValueError, match=r"40-character commit"):
        normalize_wan_model_build(build)
    assert calls == []


def test_wan_t2v_loads_vae_in_fp32_and_stages_it(monkeypatch) -> None:
    from diffusers import WanPipeline

    from vrl.models.families.wan_2_1.model import WanT2VPipelineModel

    pipeline = _FakePipeline()
    calls = []

    def load(path, **kwargs):
        calls.append(kwargs)
        return pipeline

    monkeypatch.setattr(WanPipeline, "from_pretrained", staticmethod(load))
    build = _i2v_build()
    WanT2VPipelineModel.from_build(build)
    assert calls[0]["torch_dtype"] == {
        "default": torch.bfloat16,
        "vae": torch.float32,
        "text_encoder": torch.bfloat16,
    }
    assert pipeline.vae.to_calls == [(build.device, torch.float32)]
