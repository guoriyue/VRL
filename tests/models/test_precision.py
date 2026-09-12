from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

from vrl.config.precision import RolePrecision
from vrl.models import precision


@pytest.mark.parametrize("mode", ["ieee", "tf32"])
def test_apply_float32_precision_uses_string_api_exclusively(mode: str) -> None:
    # Real torch roundtrip: the state reader is the production diagnostic, so a
    # mismatch between writer and reader fails here. The legacy bool getter is
    # deliberately NOT read: this torch raises RuntimeError on mixed-API access,
    # which is itself the guarantee that the two mechanisms cannot mix silently.
    matmul = torch.backends.cuda.matmul
    cudnn = torch.backends.cudnn
    if not (hasattr(matmul, "fp32_precision") and hasattr(cudnn, "fp32_precision")):
        pytest.skip("this torch predates the string fp32_precision API")
    original = precision.float32_precision_state()
    try:
        precision.apply_float32_precision(mode)
        assert precision.float32_precision_state() == {"matmul": mode, "cudnn": mode}
    finally:
        # Restore by attribute: the process default "none" is a valid torch
        # state but not an apply_float32_precision input.
        matmul.fp32_precision = original["matmul"]
        cudnn.fp32_precision = original["cudnn"]


@pytest.mark.parametrize(("mode", "enabled"), [("ieee", False), ("tf32", True)])
def test_apply_float32_precision_uses_legacy_bool_fallback(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    enabled: bool,
) -> None:
    # Simulates a pre-2.9 torch (no fp32_precision attribute) — a dependency
    # version boundary the installed torch cannot express, hence the fake.
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
def test_model_autocast_applies_only_supported_cpu_modes(
    dtype: str,
    enabled: bool,
    expected_enabled: bool,
) -> None:
    model = SimpleNamespace(
        precision=RolePrecision(dtype, "ieee", outer_autocast=enabled),
    )
    with precision.model_autocast(model, torch.device("cpu")):
        assert torch.is_autocast_enabled("cpu") is expected_enabled


def test_precision_trace_observes_real_recompute_and_leaves_no_hooks():
    import json

    import torch
    from torch.utils.checkpoint import checkpoint

    from vrl.models.precision import ModulePrecisionTrace

    net = torch.nn.Sequential(torch.nn.LayerNorm(4), torch.nn.Linear(4, 4).bfloat16())
    value = torch.randn(2, 4, dtype=torch.bfloat16, requires_grad=True)
    with ModulePrecisionTrace({"norm": net[0], "projection": net[1]}) as trace:
        trace.snapshot("loaded")
        trace.phase = "forward"
        output = checkpoint(net, value, use_reentrant=False)
        trace.phase = "backward"
        output.float().square().sum().backward()
    entries = [event for event in trace.events if event["kind"] == "forward_enter"]
    assert {event["phase"] for event in entries} == {"forward", "backward"}
    assert all(
        event["state"]["parameters"][0]["dtype"] == "torch.float32"
        for event in entries
        if event["module"] == "norm"
    )
    assert all(
        event["state"]["parameters"][0]["dtype"] == "torch.bfloat16"
        for event in entries
        if event["module"] == "projection"
    )
    assert all(event["inputs"][0]["dtype"] == "torch.bfloat16" for event in entries)
    assert value.grad is not None and torch.isfinite(value.grad).all()
    json.dumps(trace.events, allow_nan=False)
    count = len(trace.events)
    net(value)
    assert len(trace.events) == count
    assert not any(module._forward_pre_hooks or module._forward_hooks for module in net)


def test_precision_trace_exposes_actual_fsdp_normalization_changes():
    import torch

    from vrl.models.precision import ModulePrecisionTrace
    from vrl.trainers.fsdp import normalize_fsdp_parameter_dtype

    net = torch.nn.Linear(2, 2).bfloat16()
    net.bias.data = net.bias.data.float()
    net.register_buffer("scale", torch.ones(1))
    trace = ModulePrecisionTrace({"root": net})
    before = trace.snapshot("loader")
    normalize_fsdp_parameter_dtype(net, torch.bfloat16, allow_cast=True)
    after = trace.snapshot("before-fsdp-wrap")
    assert [p["dtype"] for p in before["modules"]["root"]["parameters"]] == [
        "torch.bfloat16",
        "torch.float32",
    ]
    assert {p["dtype"] for p in after["modules"]["root"]["parameters"]} == {"torch.bfloat16"}
    assert after["modules"]["root"]["buffers"][0]["dtype"] == "torch.float32"


def test_precision_trace_removes_hooks_on_forward_failure():
    import pytest
    import torch

    from vrl.models.precision import ModulePrecisionTrace

    module = torch.nn.Linear(2, 2)
    with pytest.raises(RuntimeError), ModulePrecisionTrace({"linear": module}) as trace:
        module(torch.ones(3))
    assert [event["kind"] for event in trace.events] == ["forward_enter"]
    assert not module._forward_pre_hooks and not module._forward_hooks
