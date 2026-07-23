"""Download-free ``from_build`` loading test for Wan 2.1 Image-to-Video.

Mirrors ``tests/models/sd3_5/test_model_loading.py``: monkeypatch the
diffusers pipeline ``from_pretrained`` to return a fake pipeline (no Hub fetch,
no real weights) and assert the loader-constructed state, NOT literal YAML
config values. The Wan I2V wrapper has a load-time branch the sd3 test does not
exercise: single-GPU offload mode is selected from the ``model.offload_mode`` config
key
(``WanI2VDiffusersModel.from_build`` in ``vrl/models/wan_2_1/model.py``):

  * ``offload_mode: sequential`` -> ``pipeline.enable_sequential_cpu_offload(gpu_id=...)``
    and frozen modules are NOT eagerly staged to the device (accelerate streams
    them per layer);
  * ``offload_mode: model`` -> ``pipeline.enable_model_cpu_offload(gpu_id=...)``,
    likewise no eager staging;
  * ``offload_mode: none`` -> vae fp32 + text_encoder/image_encoder build dtype staged to
    the build device.

In every branch the generation-only modules (vae / text_encoder / image_encoder)
are frozen and the progress bar is disabled. Wan 2.2 A14B dual-stage pipelines
are accepted when they expose ``transformer_2``; expand-timesteps pipelines
remain unsupported.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch


class _FakeModule:
    def __init__(self) -> None:
        self.dtype: torch.dtype | None = None
        self.requires_grad_enabled: bool | None = None
        self.to_calls: list[tuple[Any, torch.dtype | None]] = []

    def requires_grad_(self, enabled: bool) -> None:
        self.requires_grad_enabled = enabled

    def to(self, device: Any, dtype: torch.dtype | None = None) -> _FakeModule:
        self.to_calls.append((device, dtype))
        if dtype is not None:
            self.dtype = dtype
        return self


class _FakePipeline:
    def __init__(self) -> None:
        self.transformer = _FakeModule()
        self.transformer_2 = None
        self.vae = _FakeModule()
        self.text_encoder = _FakeModule()
        self.image_encoder = _FakeModule()
        self.device = "cpu"
        # Single-transformer Wan 2.1 I2V: no boundary_ratio / expand_timesteps.
        self.config = SimpleNamespace(boundary_ratio=None, expand_timesteps=False)
        self.progress_bar_disabled: bool | None = None
        self.sequential_offload_gpu: int | None = None
        self.model_offload_gpu: int | None = None

    def set_progress_bar_config(self, *, disable: bool) -> None:
        self.progress_bar_disabled = disable

    def enable_sequential_cpu_offload(self, *, gpu_id: int) -> None:
        self.sequential_offload_gpu = gpu_id

    def enable_model_cpu_offload(self, *, gpu_id: int) -> None:
        self.model_offload_gpu = gpu_id


def _canonical_model_config(**overrides: Any) -> dict[str, Any]:
    return {
        "boundary_ratio": None,
        "trainable_transformers": ["transformer"],
        **overrides,
    }


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
    """Shared assertions for every offload branch (load call + freezing)."""
    assert calls == [
        {
            "model_name_or_path": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
            "torch_dtype": torch.bfloat16,
        },
    ]
    assert pipeline.progress_bar_disabled is True
    for module in (pipeline.vae, pipeline.text_encoder, pipeline.image_encoder):
        assert module.requires_grad_enabled is False


def test_wan_i2v_from_build_sequential_cpu_offload(monkeypatch) -> None:
    """Sequential-offload branch uses the build GPU and skips eager staging."""
    from vrl.models.families.wan_2_1.model import WanI2VDiffusersModel

    pipeline, calls = _patch_from_pretrained(monkeypatch)

    build = SimpleNamespace(
        model_name_or_path="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        parameter_dtype=torch.bfloat16,
        device=torch.device("cuda:3"),
        model_config=_canonical_model_config(offload_mode="sequential"),
    )

    model = WanI2VDiffusersModel.from_build(build)

    assert model.pipeline is pipeline
    _assert_frozen_and_loaded(pipeline, calls)
    # gpu_id is taken from the build device index.
    assert pipeline.sequential_offload_gpu == 3
    assert pipeline.model_offload_gpu is None
    # accelerate streams the frozen modules per layer -> no eager .to(device).
    assert pipeline.vae.to_calls == []
    assert pipeline.text_encoder.to_calls == []
    assert pipeline.image_encoder.to_calls == []


def test_wan_i2v_from_build_model_cpu_offload(monkeypatch) -> None:
    """Model-offload branch uses the build GPU and skips eager staging."""
    from vrl.models.families.wan_2_1.model import WanI2VDiffusersModel

    pipeline, calls = _patch_from_pretrained(monkeypatch)

    build = SimpleNamespace(
        model_name_or_path="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        parameter_dtype=torch.bfloat16,
        device=torch.device("cuda:2"),
        model_config=_canonical_model_config(offload_mode="model"),
    )

    model = WanI2VDiffusersModel.from_build(build)

    assert model.pipeline is pipeline
    _assert_frozen_and_loaded(pipeline, calls)
    assert pipeline.model_offload_gpu == 2
    assert pipeline.sequential_offload_gpu is None
    assert pipeline.vae.to_calls == []
    assert pipeline.text_encoder.to_calls == []
    assert pipeline.image_encoder.to_calls == []


def test_wan_i2v_from_build_no_offload_stages_frozen_modules(monkeypatch) -> None:
    """No-offload stages VAE fp32 plus encoders at the build dtype, without hooks."""
    from vrl.models.families.wan_2_1.model import WanI2VDiffusersModel

    pipeline, calls = _patch_from_pretrained(monkeypatch)

    build = SimpleNamespace(
        model_name_or_path="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        parameter_dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
        model_config=_canonical_model_config(),
    )

    model = WanI2VDiffusersModel.from_build(build)

    assert model.pipeline is pipeline
    _assert_frozen_and_loaded(pipeline, calls)
    assert pipeline.sequential_offload_gpu is None
    assert pipeline.model_offload_gpu is None
    # vae stays fp32; text/image encoders ride the build dtype.
    assert pipeline.vae.to_calls == [(build.device, torch.float32)]
    assert pipeline.text_encoder.to_calls == [(build.device, torch.bfloat16)]
    assert pipeline.image_encoder.to_calls == [(build.device, torch.bfloat16)]


def test_wan_i2v_from_build_rejects_legacy_offload_keys(monkeypatch) -> None:
    """Legacy offload bools fail loud instead of becoming no-op runtime keys."""
    from vrl.models.families.wan_2_1.model import WanI2VDiffusersModel

    _patch_from_pretrained(monkeypatch)

    build = SimpleNamespace(
        model_name_or_path="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        parameter_dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
        model_config=_canonical_model_config(enable_model_cpu_offload=True),
    )

    with pytest.raises(ValueError, match=r"model\.enable_model_cpu_offload"):
        WanI2VDiffusersModel.from_build(build)


def test_wan_i2v_from_build_accepts_dual_stage_pipeline(monkeypatch) -> None:
    """Wan 2.2 A14B dual-stage pipelines train the low-noise transformer by default."""
    from vrl.models.families.wan_2_1.model import WanI2VDiffusersModel

    pipeline, _ = _patch_from_pretrained(monkeypatch)
    pipeline.config = SimpleNamespace(boundary_ratio=0.5, expand_timesteps=False)
    pipeline.transformer_2 = _FakeModule()

    build = SimpleNamespace(
        model_name_or_path="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        parameter_dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
        model_config=_canonical_model_config(
            boundary_ratio=0.5,
            trainable_transformers=["transformer_2"],
        ),
    )

    model = WanI2VDiffusersModel.from_build(build)

    assert model.boundary_ratio == 0.5
    assert model.transformer_2 is pipeline.transformer_2
    assert model.trainable_modules == {"transformer_2": pipeline.transformer_2}


def test_wan_i2v_from_build_rejects_expand_timesteps_pipeline(monkeypatch) -> None:
    """Wan 2.2 5B expand-timesteps pipelines still need a separate runner contract."""
    from vrl.models.families.wan_2_1.model import WanI2VDiffusersModel

    pipeline, _ = _patch_from_pretrained(monkeypatch)
    pipeline.config = SimpleNamespace(boundary_ratio=0.5, expand_timesteps=True)
    pipeline.transformer_2 = _FakeModule()

    build = SimpleNamespace(
        model_name_or_path="Wan-AI/Wan2.2-I2V-5B-Diffusers",
        parameter_dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
        model_config=_canonical_model_config(
            boundary_ratio=0.5,
            trainable_transformers=["transformer_2"],
        ),
    )

    with pytest.raises(NotImplementedError, match="expand_timesteps"):
        WanI2VDiffusersModel.from_build(build)


def test_wan_model_build_normalization_is_shared_by_replay_and_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from diffusers import DiffusionPipeline
    from omegaconf import OmegaConf

    from vrl.config.precision import resolve_precision_policy
    from vrl.config.schema import parse_config
    from vrl.families.registry import get_model_family_entry

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
    precision = resolve_precision_policy(root)
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
        ("all", ("transformer", "transformer_2")),
        ("both", ("transformer", "transformer_2")),
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


def test_wan_rollout_rejects_source_change_after_build_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vrl.models.families.wan_2_1.model import WanI2VDiffusersModel

    pipeline, _ = _patch_from_pretrained(monkeypatch)
    pipeline.config = SimpleNamespace(boundary_ratio=0.5, expand_timesteps=False)
    pipeline.transformer_2 = _FakeModule()
    build = SimpleNamespace(
        model_name_or_path="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        parameter_dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
        model_config=_canonical_model_config(
            boundary_ratio=0.9,
            trainable_transformers=["transformer_2"],
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"pipeline boundary_ratio disagrees.*pipeline=0\.5.*build=0\.9",
    ):
        WanI2VDiffusersModel.from_build(build)
