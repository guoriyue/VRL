"""Download-free ``from_spec`` loading test for Wan 2.1 Image-to-Video.

Mirrors ``tests/models/diffusion/sd3_5/test_model_loading.py``: monkeypatch the
diffusers pipeline ``from_pretrained`` to return a fake pipeline (no Hub fetch,
no real weights) and assert the loader-constructed state, NOT literal YAML
config values. The Wan I2V wrapper has a load-time branch the sd3 test does not
exercise: single-GPU offload mode is selected from the ``model.offload_mode`` config
key
(``WanI2VDiffusersModel.from_spec`` in ``vrl/models/diffusion/wan_2_1/model.py``):

  * ``offload_mode: sequential`` -> ``pipeline.enable_sequential_cpu_offload(gpu_id=...)``
    and frozen modules are NOT eagerly staged to the device (accelerate streams
    them per layer);
  * ``offload_mode: model`` -> ``pipeline.enable_model_cpu_offload(gpu_id=...)``,
    likewise no eager staging;
  * ``offload_mode: none`` -> vae fp32 + text_encoder/image_encoder spec-dtype staged to
    the spec device.

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


def test_wan_i2v_from_spec_sequential_cpu_offload(monkeypatch) -> None:
    """Sequential-offload branch calls enable_sequential_cpu_offload with the spec gpu and skips eager staging."""
    from vrl.models.diffusion.wan_2_1.model import WanI2VDiffusersModel

    pipeline, calls = _patch_from_pretrained(monkeypatch)

    spec = SimpleNamespace(
        model_name_or_path="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        dtype=torch.bfloat16,
        device=torch.device("cuda:3"),
        model_config={"offload_mode": "sequential"},
    )

    model = WanI2VDiffusersModel.from_spec(spec)

    assert model.pipeline is pipeline
    _assert_frozen_and_loaded(pipeline, calls)
    # gpu_id is taken from the spec device index.
    assert pipeline.sequential_offload_gpu == 3
    assert pipeline.model_offload_gpu is None
    # accelerate streams the frozen modules per layer -> no eager .to(device).
    assert pipeline.vae.to_calls == []
    assert pipeline.text_encoder.to_calls == []
    assert pipeline.image_encoder.to_calls == []


def test_wan_i2v_from_spec_model_cpu_offload(monkeypatch) -> None:
    """Model-offload branch calls enable_model_cpu_offload with the spec gpu and skips eager staging."""
    from vrl.models.diffusion.wan_2_1.model import WanI2VDiffusersModel

    pipeline, calls = _patch_from_pretrained(monkeypatch)

    spec = SimpleNamespace(
        model_name_or_path="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        dtype=torch.bfloat16,
        device=torch.device("cuda:2"),
        model_config={"offload_mode": "model"},
    )

    model = WanI2VDiffusersModel.from_spec(spec)

    assert model.pipeline is pipeline
    _assert_frozen_and_loaded(pipeline, calls)
    assert pipeline.model_offload_gpu == 2
    assert pipeline.sequential_offload_gpu is None
    assert pipeline.vae.to_calls == []
    assert pipeline.text_encoder.to_calls == []
    assert pipeline.image_encoder.to_calls == []


def test_wan_i2v_from_spec_no_offload_stages_frozen_modules(monkeypatch) -> None:
    """No-offload branch stages vae fp32 + text/image encoders at spec dtype, no offload hooks."""
    from vrl.models.diffusion.wan_2_1.model import WanI2VDiffusersModel

    pipeline, calls = _patch_from_pretrained(monkeypatch)

    spec = SimpleNamespace(
        model_name_or_path="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
        model_config={},
    )

    model = WanI2VDiffusersModel.from_spec(spec)

    assert model.pipeline is pipeline
    _assert_frozen_and_loaded(pipeline, calls)
    assert pipeline.sequential_offload_gpu is None
    assert pipeline.model_offload_gpu is None
    # vae stays fp32; text/image encoders ride the spec dtype.
    assert pipeline.vae.to_calls == [(spec.device, torch.float32)]
    assert pipeline.text_encoder.to_calls == [(spec.device, torch.bfloat16)]
    assert pipeline.image_encoder.to_calls == [(spec.device, torch.bfloat16)]


def test_wan_i2v_from_spec_rejects_legacy_offload_keys(monkeypatch) -> None:
    """Legacy offload bools fail loud instead of becoming no-op runtime keys."""
    from vrl.models.diffusion.wan_2_1.model import WanI2VDiffusersModel

    _patch_from_pretrained(monkeypatch)

    spec = SimpleNamespace(
        model_name_or_path="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
        model_config={"enable_model_cpu_offload": True},
    )

    with pytest.raises(ValueError, match=r"model\.enable_model_cpu_offload"):
        WanI2VDiffusersModel.from_spec(spec)


def test_wan_i2v_from_spec_accepts_dual_stage_pipeline(monkeypatch) -> None:
    """Wan 2.2 A14B dual-stage pipelines train the low-noise transformer by default."""
    from vrl.models.diffusion.wan_2_1.model import WanI2VDiffusersModel

    pipeline, _ = _patch_from_pretrained(monkeypatch)
    pipeline.config = SimpleNamespace(boundary_ratio=0.5, expand_timesteps=False)
    pipeline.transformer_2 = _FakeModule()

    spec = SimpleNamespace(
        model_name_or_path="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
        model_config={},
    )

    model = WanI2VDiffusersModel.from_spec(spec)

    assert model.boundary_ratio == 0.5
    assert model.transformer_2 is pipeline.transformer_2
    assert model.trainable_modules == {"transformer_2": pipeline.transformer_2}


def test_wan_i2v_from_spec_rejects_expand_timesteps_pipeline(monkeypatch) -> None:
    """Wan 2.2 5B expand-timesteps pipelines still need a separate runner contract."""
    from vrl.models.diffusion.wan_2_1.model import WanI2VDiffusersModel

    pipeline, _ = _patch_from_pretrained(monkeypatch)
    pipeline.config = SimpleNamespace(boundary_ratio=0.5, expand_timesteps=True)
    pipeline.transformer_2 = _FakeModule()

    spec = SimpleNamespace(
        model_name_or_path="Wan-AI/Wan2.2-I2V-5B-Diffusers",
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
        model_config={},
    )

    with pytest.raises(NotImplementedError, match="expand_timesteps"):
        WanI2VDiffusersModel.from_spec(spec)
