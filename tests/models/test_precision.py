from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

from vrl.config.precision import RolePrecision
from vrl.models import precision


@pytest.mark.parametrize("mode", ["ieee", "tf32"])
def test_apply_float32_precision_uses_string_api_exclusively(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    matmul = SimpleNamespace(fp32_precision="none", allow_tf32="untouched")
    cudnn = SimpleNamespace(fp32_precision="none", allow_tf32="untouched")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            backends=SimpleNamespace(
                cuda=SimpleNamespace(matmul=matmul),
                cudnn=cudnn,
            ),
        ),
    )

    precision.apply_float32_precision(mode)

    assert matmul.fp32_precision == mode
    assert cudnn.fp32_precision == mode
    assert matmul.allow_tf32 == "untouched"
    assert cudnn.allow_tf32 == "untouched"
    assert precision.float32_precision_state() == {
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
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            backends=SimpleNamespace(
                cuda=SimpleNamespace(matmul=matmul),
                cudnn=cudnn,
            ),
        ),
    )

    precision.apply_float32_precision(mode)

    assert matmul.allow_tf32 is enabled
    assert cudnn.allow_tf32 is enabled
    assert precision.float32_precision_state() == {
        "matmul": mode,
        "cudnn": mode,
    }


@pytest.mark.parametrize(
    ("dtype", "enabled", "expected_enabled"),
    [
        ("fp32", True, False),
        ("fp16", True, False),
        ("bf16", True, True),
        ("bf16", False, False),
    ],
)
def test_forward_autocast_applies_only_supported_cpu_modes(
    dtype: str,
    enabled: bool,
    expected_enabled: bool,
) -> None:
    with precision.forward_autocast(dtype, torch.device("cpu"), enabled=enabled):
        assert torch.is_autocast_enabled("cpu") is expected_enabled


def test_model_precision_reads_stamped_role_precision() -> None:
    role_precision = RolePrecision("bf16", "ieee")
    model = SimpleNamespace(precision=role_precision)

    assert precision.model_precision(model) is role_precision


def test_model_precision_rejects_unstamped_model() -> None:
    with pytest.raises(TypeError, match="precision is missing or not a RolePrecision"):
        precision.model_precision(SimpleNamespace())
