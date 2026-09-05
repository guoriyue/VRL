"""Scheme validation tests for the quantized linear benchmark."""

from __future__ import annotations

import pytest

import vrl.scripts.perf.quantized_linear_benchmark as benchmark


@pytest.mark.parametrize("schemes", [(), ("int8",), ("fp8", "bogus")])
def test_canonical_benchmark_rejects_invalid_scheme_sets(schemes) -> None:
    with pytest.raises(ValueError, match="non-empty fp8/nvfp4 subset"):
        benchmark.main(schemes=schemes)


def test_explicit_nvfp4_request_fails_when_hardware_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(benchmark.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(benchmark, "nvfp4_available", lambda: False)

    with pytest.raises(SystemExit) as exc_info:
        benchmark.main(schemes=("nvfp4",))

    assert exc_info.value.code != 0
    assert "NVFP4-capable" in str(exc_info.value)
