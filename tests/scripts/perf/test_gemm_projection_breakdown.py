"""Numerical reporting boundaries for projection profiling."""

import pytest

from vrl.scripts.perf.gemm_projection_breakdown import PROJECTION_ORDER, Breakdown


@pytest.mark.parametrize("device_kind", ["cpu", "cuda"])
@pytest.mark.parametrize("duration", [0.0, 4.0])
def test_report_preserves_measured_total_when_zero(device_kind, duration):
    times = dict.fromkeys(PROJECTION_ORDER, 0.0)
    times["qkv"] = duration
    calls = dict.fromkeys(PROJECTION_ORDER, 0)
    calls["qkv"] = 1
    report = Breakdown(device_us=times, cpu_us=times, calls=calls, device_kind=device_kind)
    lines = report.to_text().splitlines()
    total = next(line for line in lines if line.startswith("TOTAL")).split()
    category = next(line for line in lines if line.startswith("qkv")).split()
    assert float(total[1]) == duration
    expected_percent = "100.0%" if duration else "0.0%"
    assert total[2] == expected_percent
    assert category[2] == expected_percent


@pytest.mark.parametrize(
    "field, value",
    [
        ("warmup", -1),
        ("warmup", True),
        ("warmup", 1.5),
        ("active", 0),
        ("active", -1),
        ("active", True),
        ("active", 1.5),
    ],
)
def test_profile_rejects_invalid_counts_before_forward(field, value):
    from vrl.scripts.perf.gemm_projection_breakdown import profile_projection_gemms

    def forward():
        pytest.fail("invalid profiling counts must fail before model execution")

    with pytest.raises(ValueError, match=field):
        profile_projection_gemms(None, forward, **{field: value})


def test_profile_executes_exact_warmup_and_active_counts():
    import torch

    from vrl.scripts.perf.gemm_projection_breakdown import profile_projection_gemms

    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    inputs = torch.ones(1, 2)
    calls = []

    def forward():
        calls.append(None)
        return model(inputs)

    report = profile_projection_gemms(model, forward, warmup=0, active=2)
    assert len(calls) == 2
    assert report.calls["other"] == 2
