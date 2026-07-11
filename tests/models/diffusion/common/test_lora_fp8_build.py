"""LoRA + quantized rollout construction order.

The 17B single-card path depends on attaching LoRA and dropping fp8 masters on
CPU before the compact rollout policy is moved to CUDA. These tests pin both
halves of that contract without loading a real checkpoint.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import torch
from torch import nn

from vrl.models.diffusion.build import build_diffusion_runtime_bundle
from vrl.models.diffusion.common.lora import LoraModelMixin
from vrl.models.interfaces.runtime import RuntimeBuildSpec


class _TrackingTransformer(nn.Module):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 2)
        self.events = events

    def to(self, *args: Any, **kwargs: Any) -> _TrackingTransformer:
        self.events.append("move")
        return super().to(*args, **kwargs)


class _LoraPolicy(LoraModelMixin):
    def __init__(self, events: list[str]) -> None:
        self.transformer = _TrackingTransformer(events)
        self.device = "cpu"

    def _set_transformer(self, transformer: Any) -> None:
        self.transformer = transformer


def _spec(*, quantized: bool) -> SimpleNamespace:
    return SimpleNamespace(
        rollout_quantization="fp8" if quantized else None,
        dtype=torch.float16,
        lora_path=None,
        lora={"rank": 2, "alpha": 2, "target_modules": ["proj"]},
    )


def test_quantized_lora_attach_defers_device_move(monkeypatch) -> None:
    events: list[str] = []
    fake_peft = ModuleType("peft")
    fake_peft.LoraConfig = lambda **_kwargs: object()
    fake_peft.PeftModel = object
    fake_peft.get_peft_model = lambda transformer, _cfg: transformer
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    _LoraPolicy(events).apply_lora(_spec(quantized=True))
    assert events == []


def test_plain_lora_attach_keeps_direct_device_move(monkeypatch) -> None:
    events: list[str] = []
    fake_peft = ModuleType("peft")
    fake_peft.LoraConfig = lambda **_kwargs: object()
    fake_peft.PeftModel = object
    fake_peft.get_peft_model = lambda transformer, _cfg: transformer
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    _LoraPolicy(events).apply_lora(_spec(quantized=False))
    assert events == ["move"]


def test_shared_builder_drops_master_before_quantized_lora_gpu_move(monkeypatch) -> None:
    events: list[str] = []

    class _Policy:
        def __init__(self) -> None:
            self.transformer = _TrackingTransformer(events)
            self.device = "cpu"
            self.scheduler = object()
            self.raw_handle = object()
            self.trainable_modules = {"transformer": self.transformer}

        @classmethod
        def from_spec(cls, _spec: Any) -> _Policy:
            return cls()

        def apply_lora(self, _spec: Any) -> None:
            events.append("attach")

        def quantize_rollout_fp8(self, recipe: str = "rowwise") -> list[str]:
            assert recipe == "rowwise"
            events.append("quantize")
            return ["proj"]

        def generation_memory_targets(self) -> dict[str, Any]:
            return {}

        def set_num_steps(self, _steps: int) -> None:
            return None

    import vrl.nn.quantization as quantization

    monkeypatch.setattr(
        quantization,
        "drop_fp8_masters",
        lambda _model: events.append("drop") or 4,
    )
    spec = RuntimeBuildSpec(
        model_name_or_path="fake",
        device="cpu",
        dtype=torch.float16,
        rollout_quantization="fp8",
        model_config={
            "use_lora": True,
            "lora": {"rank": 2, "alpha": 2, "target_modules": ["proj"]},
        },
        sampling_config={"num_steps": 1},
    )

    build_diffusion_runtime_bundle(
        spec,
        model_cls=_Policy,
        capability=SimpleNamespace(family="fake"),
        memory_owner="fake VAE",
    )

    assert events == ["attach", "quantize", "drop", "move"]
