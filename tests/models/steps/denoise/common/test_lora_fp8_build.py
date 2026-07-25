"""LoRA + quantized rollout construction order.

The 17B single-card path depends on attaching LoRA and dropping source masters
on CPU before the compact rollout policy is moved to CUDA. These tests pin both
FP8 and NVFP4 lifecycle contracts without loading a real checkpoint.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from vrl.config.precision import QuantizationPolicy, RolePrecision
from vrl.models.interfaces.runtime import ModelBuild, RolloutBuildOptions
from vrl.models.steps.denoise.build import build_denoise_runtime_bundle
from vrl.models.steps.denoise.common.lora import LoraModelMixin


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


def _build(
    *,
    quantization_format: str | None,
    defer_trainable_device_move: bool = False,
) -> SimpleNamespace:
    quantization = (
        None if quantization_format is None else QuantizationPolicy(format=quantization_format)
    )
    return SimpleNamespace(
        precision=RolePrecision("fp16", "tf32", quantization),
        rollout=SimpleNamespace(),
        defer_trainable_device_move=defer_trainable_device_move,
        parameter_dtype=torch.float16,
        lora_path=None,
        lora={"rank": 2, "alpha": 2, "target_modules": ["proj"]},
    )


def _build_sd35_rollout(
    monkeypatch: pytest.MonkeyPatch,
    build: ModelBuild,
    model_cls: type[Any],
) -> Any:
    """Build through the production registry while replacing checkpoint I/O."""

    from vrl.families.registry import get_model_family_entry
    from vrl.models.families.sd3_5 import model as sd35_model

    monkeypatch.setattr(sd35_model, "SD3_5Model", model_cls)
    return get_model_family_entry("sd3_5").build_rollout(build)


@pytest.mark.parametrize("quantization_format", ["fp8", "nvfp4"])
def test_quantized_lora_attach_defers_device_move(
    quantization_format: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    fake_peft = ModuleType("peft")
    fake_peft.LoraConfig = lambda **_kwargs: object()
    fake_peft.PeftModel = object
    fake_peft.get_peft_model = lambda transformer, _cfg: transformer
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    _LoraPolicy(events).apply_lora(
        _build(quantization_format=quantization_format),
    )
    assert events == []


def test_plain_lora_attach_keeps_direct_device_move(monkeypatch) -> None:
    events: list[str] = []
    fake_peft = ModuleType("peft")
    fake_peft.LoraConfig = lambda **_kwargs: object()
    fake_peft.PeftModel = object
    fake_peft.get_peft_model = lambda transformer, _cfg: transformer
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    policy = _LoraPolicy(events)
    policy.apply_lora(_build(quantization_format=None))
    assert events == ["move"]
    assert policy.transformer.proj.weight.dtype is torch.float16


def test_fsdp_replay_lora_attach_defers_device_move(monkeypatch) -> None:
    events: list[str] = []
    fake_peft = ModuleType("peft")
    fake_peft.LoraConfig = lambda **_kwargs: object()
    fake_peft.PeftModel = object
    fake_peft.get_peft_model = lambda transformer, _cfg: transformer
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    _LoraPolicy(events).apply_lora(
        _build(
            quantization_format=None,
            defer_trainable_device_move=True,
        ),
    )

    assert events == []


def test_fp8_config_replay_build_does_not_defer_device_move(monkeypatch) -> None:
    """Replay owns no rollout options even when collection uses fp8."""
    from omegaconf import OmegaConf

    from vrl.config.precision import resolve_precision_policy
    from vrl.config.schema import parse_config
    from vrl.families.registry import get_model_family_entry

    events: list[str] = []
    fake_peft = ModuleType("peft")
    fake_peft.LoraConfig = lambda **_kwargs: object()
    fake_peft.PeftModel = object
    fake_peft.get_peft_model = lambda transformer, _cfg: transformer
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    cfg = OmegaConf.create(
        {
            "model": {
                "family": "sd3_5",
                "path": "fake",
                "use_lora": True,
                "lora": {"rank": 2, "alpha": 2, "target_modules": ["proj"]},
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
    precision = resolve_precision_policy(root)
    build = get_model_family_entry("sd3_5").resolve_model_build(
        root,
        "cpu",
        precision=precision,
        for_rollout=False,
    )

    assert build.rollout is None
    _LoraPolicy(events).apply_lora(build)
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
            # No save_pretrained on the tracking transformer: nothing to publish.
            self.adapter_roots: dict[str, Any] = {}

        @classmethod
        def from_build(cls, _build: Any) -> _Policy:
            return cls()

        def apply_lora(self, _build: Any) -> None:
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
    build = ModelBuild(
        model_name_or_path="fake",
        revision=None,
        device="cpu",
        parameter_dtype=torch.float16,
        family="sd3_5",
        precision=RolePrecision("bf16", "tf32", QuantizationPolicy(format="fp8")),
        rollout=RolloutBuildOptions(
            prompt_encoder_dtype=torch.bfloat16,
            base_weight_sync=False,
        ),
        model_config={
            "use_lora": True,
            "lora": {"rank": 2, "alpha": 2, "target_modules": ["proj"]},
        },
        sampling_config={"num_steps": 1},
    )

    _build_sd35_rollout(monkeypatch, build, _Policy)

    assert events == ["attach", "quantize", "drop", "move"]


def test_shared_builder_installs_pipeline_offload_after_final_cpu_module_tree(
    monkeypatch,
) -> None:
    """Hooks follow wrappers without first staging quantized weights on GPU."""

    events: list[str] = []

    class _Policy:
        def __init__(self) -> None:
            self.transformer = _TrackingTransformer(events)
            self.device = "cpu"
            self.scheduler = object()
            self.raw_handle = object()
            self.trainable_modules = {"transformer": self.transformer}
            # No save_pretrained on the tracking transformer: nothing to publish.
            self.adapter_roots: dict[str, Any] = {}
            self.uses_pipeline_cpu_offload = False
            self.pipeline_cpu_offload_healthy = True

        @classmethod
        def from_build(cls, _build: Any) -> _Policy:
            return cls()

        def apply_lora(self, _build: Any) -> None:
            events.append("attach_lora")

        def quantize_rollout_fp8(self, recipe: str = "rowwise") -> list[str]:
            assert recipe == "rowwise"
            events.append("quantize")
            return ["proj"]

        def torch_compile_transformer(self, mode: str) -> None:
            assert mode == "default"
            events.append("compile")

        def apply_generation_offload(self, _build: Any) -> None:
            events.append("install_offload_hooks")
            self.uses_pipeline_cpu_offload = True

        def generation_memory_targets(self) -> dict[str, Any]:
            return {}

    import vrl.nn.quantization as quantization

    monkeypatch.setattr(
        quantization,
        "drop_quantized_masters",
        lambda _model: events.append("drop") or 4,
    )
    build = ModelBuild(
        model_name_or_path="fake",
        revision=None,
        device="cpu",
        parameter_dtype=torch.bfloat16,
        family="sd3_5",
        precision=RolePrecision(
            "bf16",
            "tf32",
            QuantizationPolicy(format="fp8"),
        ),
        rollout=RolloutBuildOptions(
            prompt_encoder_dtype=torch.bfloat16,
            base_weight_sync=False,
            pipeline_offload_mode="sequential",
        ),
        model_config={
            "use_lora": True,
            "lora": {"rank": 2, "alpha": 2, "target_modules": ["proj"]},
            "torch_compile": {"enable": True, "mode": "default"},
        },
    )

    _build_sd35_rollout(monkeypatch, build, _Policy)

    assert events == [
        "attach_lora",
        "quantize",
        "drop",
        "compile",
        "install_offload_hooks",
    ]


def test_shared_builder_preserves_resolved_role_precision() -> None:
    """The builder carries the resolved contract without model introspection."""

    class _Policy:
        def __init__(self) -> None:
            self.transformer = nn.Linear(2, 2)
            self.device = "cpu"
            self.scheduler = object()
            self.raw_handle = object()

        @classmethod
        def from_build(cls, _build: Any) -> _Policy:
            return cls()

        @property
        def trainable_modules(self) -> dict[str, Any]:
            return {"transformer": self.transformer}

        @property
        def adapter_roots(self) -> dict[str, Any]:
            # No save_pretrained on a plain nn module: nothing to publish.
            return {}

        def apply_full_finetune(self, _build: Any) -> None:
            self.transformer.requires_grad_(True)

        def generation_memory_targets(self) -> dict[str, Any]:
            return {}

    build = ModelBuild(
        model_name_or_path="fake",
        revision=None,
        device="cpu",
        parameter_dtype=torch.float16,
        family="sd3_5",
        precision=RolePrecision("fp32", "ieee", outer_autocast=False),
        rollout=RolloutBuildOptions(
            prompt_encoder_dtype=torch.bfloat16,
        ),
        model_config={"use_lora": False},
    )

    bundle = build_denoise_runtime_bundle(build, model_cls=_Policy)

    assert bundle.precision is build.precision
    assert bundle.precision.outer_autocast is False


def test_nvfp4_hardware_guard_runs_before_quantization_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vrl.models.loader import apply_rollout_quantization

    class _Policy:
        def __init__(self) -> None:
            self.quantize_calls = 0

        def quantize_rollout_nvfp4(self) -> list[str]:
            self.quantize_calls += 1
            return ["proj"]

    monkeypatch.setattr("vrl.nn.quantization.nvfp4_available", lambda _device: False)
    model = _Policy()
    build = ModelBuild(
        model_name_or_path="fake",
        revision=None,
        device="cpu",
        parameter_dtype=torch.bfloat16,
        family="sd3_5",
        precision=RolePrecision("bf16", "tf32", QuantizationPolicy(format="nvfp4")),
        rollout=RolloutBuildOptions(
            prompt_encoder_dtype=torch.bfloat16,
        ),
    )

    with pytest.raises(RuntimeError, match="NVFP4-capable CUDA target"):
        apply_rollout_quantization(model, build)

    assert model.quantize_calls == 0


@pytest.mark.parametrize("quantization_format", ["fp8", "nvfp4"])
def test_full_finetune_dtype_move_preserves_quantized_cache(
    quantization_format: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared full path swaps on CPU before a model-owned dtype move."""
    from vrl.nn.quantization import Fp4Linear, Fp8Linear

    if quantization_format == "nvfp4":
        monkeypatch.setattr("vrl.nn.quantization.nvfp4_available", lambda _device: True)

    class _Policy:
        def __init__(self) -> None:
            self.transformer = nn.Sequential(
                nn.Linear(64, 64, bias=False).to(torch.bfloat16),
            )
            self.device = "cpu"
            self.scheduler = object()
            self.raw_handle = object()

        @classmethod
        def from_build(cls, _build: Any) -> _Policy:
            return cls()

        @property
        def trainable_modules(self) -> dict[str, Any]:
            return {"transformer": self.transformer}

        @property
        def adapter_roots(self) -> dict[str, Any]:
            # No save_pretrained on a plain nn module: nothing to publish.
            return {}

        def quantize_rollout_fp8(self, recipe: str = "rowwise") -> list[str]:
            assert recipe == "rowwise"
            self.transformer[0] = Fp8Linear(self.transformer[0])
            return ["0"]

        def quantize_rollout_nvfp4(self) -> list[str]:
            self.transformer[0] = Fp4Linear(self.transformer[0])
            return ["0"]

        def apply_full_finetune(self, _build: Any) -> None:
            self.transformer.to(self.device, dtype=torch.bfloat16)

        def generation_memory_targets(self) -> dict[str, Any]:
            return {}

    build = ModelBuild(
        model_name_or_path="fake",
        revision=None,
        device="cpu",
        parameter_dtype=torch.bfloat16,
        family="sd3_5",
        precision=RolePrecision(
            "bf16",
            "tf32",
            QuantizationPolicy(format=quantization_format),
        ),
        rollout=RolloutBuildOptions(
            prompt_encoder_dtype=torch.bfloat16,
        ),
        model_config={"use_lora": False},
    )

    bundle = _build_sd35_rollout(monkeypatch, build, _Policy)

    quantized = bundle.model.transformer[0]
    if quantization_format == "fp8":
        assert quantized.weight_fp8.dtype is torch.float8_e4m3fn
    else:
        assert quantized.weight_fp4.dtype is torch.float4_e2m1fn_x2
        assert quantized.weight_scale.dtype is torch.float8_e4m3fn
