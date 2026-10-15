"""Hardware gating of the quantized linear benchmark."""

from __future__ import annotations

import pytest

import vrl.scripts.perf.quantized_linear_benchmark as benchmark


def test_benchmark_fails_when_nvfp4_hardware_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(benchmark.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(benchmark, "nvfp4_available", lambda: False)

    with pytest.raises(SystemExit) as exc_info:
        benchmark.main()

    assert exc_info.value.code != 0
    assert "NVFP4-capable" in str(exc_info.value)
