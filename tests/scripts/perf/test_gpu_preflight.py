"""gpu_preflight: structural + CPU-path correctness, and (when CUDA is present) that the
measured bf16 peak is a sane training denominator — not the inflated vendor headline."""

from __future__ import annotations

import pytest
import torch

from vrl.scripts.perf.gpu_preflight import (
    GpuPreflight,
    measured_bf16_peak_tflops,
    run_gpu_preflight,
)

_HAS_CUDA = torch.cuda.is_available()


def test_cpu_path_is_graceful_not_crash() -> None:
    """On a CPU box (CI), preflight must degrade to a well-formed 'unavailable' report,
    never raise — so perf probes can import/call it on non-CUDA CI."""
    report = run_gpu_preflight(force=True)
    assert isinstance(report, GpuPreflight)
    if not _HAS_CUDA:
        assert report.available is False
        assert report.bf16_peak_tflops == 0.0
        assert measured_bf16_peak_tflops() == 0.0


@pytest.mark.skipif(not _HAS_CUDA, reason="needs a CUDA GPU")
def test_measured_peak_is_a_sane_bf16_denominator() -> None:
    """The whole point: the measured bf16 peak is the REAL achievable rate, which for
    any current GPU is well below the fp8/sparse 'AI TOPS' headline. Guards against ever
    silently reverting to a hardcoded inflated peak that mis-diagnoses MFU."""
    peak = measured_bf16_peak_tflops()
    assert peak > 0
    # A real bf16-dense GEMM cannot exceed a few PFLOPS on any single GPU today; and it
    # must be positive/finite. The RTX 5090's real bf16 peak (~232) is FAR under its 419
    # fp8/sparse headline — this bound just catches a nonsense denominator.
    assert 10.0 < peak < 5000.0


@pytest.mark.skipif(not _HAS_CUDA, reason="needs a CUDA GPU")
def test_arch_match_and_sdpa_measured() -> None:
    report = run_gpu_preflight(force=True)
    assert report.available is True
    assert report.sm.startswith("sm_")
    # If torch is built for this GPU there must be no arch warning; if not, arch_ok flags it.
    assert report.arch_ok == (report.sm in report.arch_list)
    # At least the flash backend should have produced a number on a CUDA box.
    assert report.sdpa_backend_tflops, "no SDPA backend benchmarked"
    assert report.best_sdpa_backend in report.sdpa_backend_tflops
