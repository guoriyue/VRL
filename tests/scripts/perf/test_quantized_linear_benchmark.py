"""Compatibility tests for the scheme-neutral linear benchmark."""

from __future__ import annotations

import pytest

import vrl.scripts.perf.fp8_linear_benchmark as fp8_facade
import vrl.scripts.perf.fp8_rollout_drift_probe as fp8_drift_facade
import vrl.scripts.perf.quantized_linear_benchmark as benchmark


def test_legacy_fp8_entrypoint_runs_only_the_fp8_gate(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    monkeypatch.setattr(
        fp8_facade,
        "quantized_main",
        lambda *, schemes: calls.append(schemes),
    )

    fp8_facade.main()

    assert calls == [("fp8",)]


def test_legacy_fp8_drift_entrypoint_runs_only_the_fp8_probe(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        fp8_drift_facade,
        "quantized_main",
        lambda *, scheme: calls.append(scheme),
    )

    fp8_drift_facade.main()

    assert calls == ["fp8"]


@pytest.mark.parametrize("schemes", [(), ("int8",), ("fp8", "bogus")])
def test_canonical_benchmark_rejects_invalid_scheme_sets(schemes) -> None:
    with pytest.raises(ValueError, match="non-empty fp8/nvfp4 subset"):
        benchmark.main(schemes=schemes)


def test_canonical_benchmark_rejects_legacy_fp4_name() -> None:
    with pytest.raises(ValueError, match="non-empty fp8/nvfp4 subset"):
        benchmark.main(schemes=("fp4",))


def test_explicit_nvfp4_request_fails_when_hardware_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(benchmark.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(benchmark, "nvfp4_available", lambda: False)

    with pytest.raises(SystemExit) as exc_info:
        benchmark.main(schemes=("nvfp4",))

    assert exc_info.value.code != 0
    assert "NVFP4-capable" in str(exc_info.value)
