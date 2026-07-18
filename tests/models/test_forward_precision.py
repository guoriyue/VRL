from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vrl.models import forward_precision
from vrl.models.interfaces.runtime import AutocastMode, ForwardPrecision


@pytest.mark.parametrize("mode", ["ieee", "tf32"])
def test_apply_float32_precision_uses_string_api_exclusively(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    matmul = SimpleNamespace(fp32_precision="none", allow_tf32="untouched")
    cudnn = SimpleNamespace(fp32_precision="none", allow_tf32="untouched")
    monkeypatch.setattr(
        forward_precision,
        "torch",
        SimpleNamespace(
            backends=SimpleNamespace(
                cuda=SimpleNamespace(matmul=matmul),
                cudnn=cudnn,
            ),
        ),
    )

    forward_precision.apply_float32_precision(mode)

    assert matmul.fp32_precision == mode
    assert cudnn.fp32_precision == mode
    assert matmul.allow_tf32 == "untouched"
    assert cudnn.allow_tf32 == "untouched"
    assert forward_precision.float32_precision_state() == {
        "matmul": mode,
        "cudnn": mode,
    }


@pytest.mark.parametrize(("mode", "enabled"), [("ieee", False), ("tf32", True)])
def test_apply_float32_precision_uses_legacy_bool_fallback(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    enabled: bool,
) -> None:
    matmul = SimpleNamespace(allow_tf32=not enabled)
    cudnn = SimpleNamespace(allow_tf32=not enabled)
    monkeypatch.setattr(
        forward_precision,
        "torch",
        SimpleNamespace(
            backends=SimpleNamespace(
                cuda=SimpleNamespace(matmul=matmul),
                cudnn=cudnn,
            ),
        ),
    )

    forward_precision.apply_float32_precision(mode)

    assert matmul.allow_tf32 is enabled
    assert cudnn.allow_tf32 is enabled
    assert forward_precision.float32_precision_state() == {
        "matmul": mode,
        "cudnn": mode,
    }


@pytest.mark.parametrize(
    ("autocast", "expected_enabled"),
    [("off", False), ("fp16", False), ("bf16", True)],
)
def test_forward_autocast_applies_only_supported_cpu_modes(
    autocast: AutocastMode,
    expected_enabled: bool,
) -> None:
    precision = ForwardPrecision(
        autocast=autocast,
        float32_precision="ieee",
    )

    with forward_precision.forward_autocast(precision, torch.device("cpu")):
        assert torch.is_autocast_enabled("cpu") is expected_enabled
