from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from vrl.models.diffusion.sd3_5.model import SD3_5Model


class _FakeModule:
    def __init__(self) -> None:
        self.dtype: torch.dtype | None = None
        self.requires_grad_enabled: bool | None = None
        self.to_calls: list[tuple[Any, torch.dtype | None]] = []

    def requires_grad_(self, enabled: bool) -> None:
        self.requires_grad_enabled = enabled

    def to(self, device: Any, dtype: torch.dtype | None = None) -> None:
        self.to_calls.append((device, dtype))
        if dtype is not None:
            self.dtype = dtype


class _FakePipeline:
    def __init__(self) -> None:
        self.transformer = _FakeModule()
        self.vae = _FakeModule()
        self.text_encoder = _FakeModule()
        self.text_encoder_2 = _FakeModule()
        self.text_encoder_3 = _FakeModule()
        self.device = "cpu"


def test_sd3_fp32_runtime_loads_frozen_components_without_fp32_peak(monkeypatch) -> None:
    from diffusers import StableDiffusion3Pipeline

    calls: list[dict[str, Any]] = []
    pipeline = _FakePipeline()

    def fake_from_pretrained(model_name_or_path: str, **kwargs: Any) -> _FakePipeline:
        calls.append({"model_name_or_path": model_name_or_path, **kwargs})
        return pipeline

    monkeypatch.setattr(
        StableDiffusion3Pipeline,
        "from_pretrained",
        staticmethod(fake_from_pretrained),
    )

    spec = SimpleNamespace(
        model_name_or_path="stabilityai/stable-diffusion-3.5-medium",
        dtype=torch.float32,
        extra={"frozen_dtype": "float16"},
        device="cuda:0",
    )

    model = SD3_5Model.from_spec(spec)

    assert model.pipeline is pipeline
    assert calls == [
        {
            "model_name_or_path": "stabilityai/stable-diffusion-3.5-medium",
            "torch_dtype": {
                "transformer": torch.float32,
                "vae": torch.float32,
                "default": torch.float16,
            },
        },
    ]
    for encoder in (
        pipeline.text_encoder,
        pipeline.text_encoder_2,
        pipeline.text_encoder_3,
    ):
        assert encoder.requires_grad_enabled is False
        assert encoder.to_calls == [("cuda:0", torch.float16)]
    assert pipeline.vae.requires_grad_enabled is False
    assert pipeline.vae.to_calls == [("cuda:0", torch.float32)]
