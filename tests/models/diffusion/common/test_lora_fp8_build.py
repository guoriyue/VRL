"""LoRA + quantized rollout construction order.

The 17B single-card path depends on attaching LoRA and dropping fp8 masters on
CPU before the compact rollout policy is moved to CUDA. These tests pin both
halves of that contract without loading a real checkpoint.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
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
        "drop_quantized_masters",
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


@pytest.mark.parametrize("scheme", ["fp8", "fp4"])
def test_full_finetune_dtype_move_preserves_quantized_cache(scheme: str, monkeypatch) -> None:
    """The shared full path swaps on CPU before a model-owned dtype move."""

    from vrl.nn.quantization import Fp4Linear, Fp8Linear

    if scheme == "fp4":
        # This structural test intentionally keeps the fake policy on CPU so it
        # can isolate Module._apply cache handling. Production hardware rejection
        # is covered separately in test_fp4_loader_rejects_unsupported_target.
        monkeypatch.setattr("vrl.nn.quantization.fp4.nvfp4_available", lambda _device: True)

    class _Policy:
        def __init__(self) -> None:
            self.transformer = nn.Sequential(
                nn.Linear(64, 64, bias=False).to(torch.bfloat16),
            )
            self.device = "cpu"
            self.scheduler = object()
            self.raw_handle = object()

        @classmethod
        def from_spec(cls, _spec: Any) -> _Policy:
            return cls()

        @property
        def trainable_modules(self) -> dict[str, Any]:
            return {"transformer": self.transformer}

        def quantize_rollout_fp8(self, recipe: str = "rowwise") -> list[str]:
            assert recipe == "rowwise"
            self.transformer[0] = Fp8Linear(self.transformer[0])
            return ["0"]

        def quantize_rollout_fp4(self) -> list[str]:
            self.transformer[0] = Fp4Linear(self.transformer[0])
            return ["0"]

        def apply_full_finetune(self) -> None:
            self.transformer.to(self.device, dtype=torch.bfloat16)

        def generation_memory_targets(self) -> dict[str, Any]:
            return {}

    spec = RuntimeBuildSpec(
        model_name_or_path="fake",
        device="cpu",
        dtype=torch.bfloat16,
        rollout_quantization=scheme,
        model_config={"use_lora": False},
    )

    bundle = build_diffusion_runtime_bundle(
        spec,
        model_cls=_Policy,
        capability=SimpleNamespace(family="fake"),
        memory_owner="fake VAE",
    )

    quantized = bundle.model.transformer[0]
    if scheme == "fp8":
        assert quantized.weight_fp8.dtype is torch.float8_e4m3fn
    else:
        assert quantized.weight_fp4.dtype is torch.float4_e2m1fn_x2
        assert quantized.weight_scale.dtype is torch.float8_e4m3fn
