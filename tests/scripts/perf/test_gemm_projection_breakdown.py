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
