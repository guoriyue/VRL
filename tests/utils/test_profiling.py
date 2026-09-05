"""Contract + smoke tests for the trustworthy profiling API.

P0/P1 prove the API contract without a GPU; P2 runs a real CPU torch trace and
asserts the trace/summary/manifest trust triad. The nvtx tests carry a
``real_cover`` label for the CUDA emission no in-process test can observe.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from vrl.scripts.perf.profile_smoke import run_smoke
from vrl.utils.profiling import (
    TorchProfilerConfig,
    _resolve_activities,
    _safe_label,
    _safe_worker_name,
    capture_torch_trace,
    profile_range,
)

CPU = torch.profiler.ProfilerActivity.CPU
CUDA = torch.profiler.ProfilerActivity.CUDA

# The three push/pop pairing tests below share one blocker, so the label is a
# module constant instead of the same string three times.
_NVTX_DEPTH_IS_UNOBSERVABLE = pytest.mark.real_cover(
    None,
    why=(
        "nvtx range depth is unobservable from inside the process: with no profiler "
        "attached torch.cuda.nvtx.range_push/range_pop both return -2 even when CUDA is "
        "available, so counting the calls is the only way to assert push/pop stay paired. "
        "The real CUDA emission is driven by vrl/scripts/perf/profile_smoke.py under nsys "
        "on a GPU box, which is a script, not a test"
    ),
    tracked_in="docs/sprints/done/SPRINT_tier-policy-and-real-cover-labels.md",
)


# ---------------------------------------------------------------------------
# P0 — API contract, no GPU required
# ---------------------------------------------------------------------------


def test_profile_range_is_transparent_to_return_value() -> None:
    with profile_range("test.noop"):
        result = 1 + 1
    assert result == 2


@_NVTX_DEPTH_IS_UNOBSERVABLE
def test_profile_range_no_nvtx_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"push": 0, "pop": 0}
    monkeypatch.setattr(
        torch.cuda.nvtx, "range_push", lambda name: calls.__setitem__("push", calls["push"] + 1)
    )
    monkeypatch.setattr(
        torch.cuda.nvtx, "range_pop", lambda: calls.__setitem__("pop", calls["pop"] + 1)
    )
    monkeypatch.delenv("VRL_PROFILE", raising=False)
    with profile_range("test.no_nvtx", emit_nvtx=False):
        pass
    assert calls == {"push": 0, "pop": 0}


@_NVTX_DEPTH_IS_UNOBSERVABLE
def test_profile_range_pops_nvtx_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"push": 0, "pop": 0}
    monkeypatch.setattr(
        torch.cuda.nvtx, "range_push", lambda name: calls.__setitem__("push", calls["push"] + 1)
    )
    monkeypatch.setattr(
        torch.cuda.nvtx, "range_pop", lambda: calls.__setitem__("pop", calls["pop"] + 1)
    )
    with pytest.raises(ValueError), profile_range("test.boom", emit_nvtx=True):
        raise ValueError("boom")
    # push and pop must be strictly paired even when the body raises.
    assert calls == {"push": 1, "pop": 1}


@_NVTX_DEPTH_IS_UNOBSERVABLE
def test_profile_range_nvtx_follows_env(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"push": 0, "pop": 0}
    monkeypatch.setattr(
        torch.cuda.nvtx, "range_push", lambda name: calls.__setitem__("push", calls["push"] + 1)
    )
    monkeypatch.setattr(
        torch.cuda.nvtx, "range_pop", lambda: calls.__setitem__("pop", calls["pop"] + 1)
    )
    monkeypatch.setenv("VRL_PROFILE", "1")
    with profile_range("test.env_nvtx"):  # emit_nvtx defaults to env signal
        pass
    assert calls == {"push": 1, "pop": 1}


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("", "unknown"),
        ("   ", "unknown"),
        ("a/b\\c", "a_b_c"),
        ("worker 0", "worker_0"),
        ("café", "café"),
        ("__edge__", "edge"),
    ],
)
def test_safe_label_stable(raw: str, expected: str) -> None:
    assert _safe_label(raw) == expected


def test_safe_worker_name_includes_step() -> None:
    name = _safe_worker_name("trainer", 3)
    assert name.endswith("_trainer_step3")


# ---------------------------------------------------------------------------
# P1 — activity resolution is trustworthy
# ---------------------------------------------------------------------------


def test_resolve_unknown_activity_fails_fast() -> None:
    cfg = TorchProfilerConfig(activities=("cpu", "gpu"))
    with pytest.raises(ValueError, match="Unknown torch profiler activities"):
        _resolve_activities(cfg, supported={CPU, CUDA})


def test_resolve_cpu_only_reports_missing_cuda() -> None:
    cfg = TorchProfilerConfig(activities=("cpu", "cuda"))
    resolved = _resolve_activities(cfg, supported={CPU})
    assert resolved.requested == ("cpu", "cuda")
    assert resolved.effective == ("cpu",)
    assert resolved.missing == ("cuda",)
    assert resolved.torch_activities == (CPU,)


def test_resolve_all_supported() -> None:
    cfg = TorchProfilerConfig(activities=("cpu", "cuda"))
    resolved = _resolve_activities(cfg, supported={CPU, CUDA})
    assert resolved.missing == ()
    assert resolved.effective == ("cpu", "cuda")


def test_resolve_deduplicates_requested() -> None:
    cfg = TorchProfilerConfig(activities=("cpu", "cpu", "cuda"))
    resolved = _resolve_activities(cfg, supported={CPU, CUDA})
    assert resolved.requested == ("cpu", "cuda")


def test_capture_fails_fast_on_missing_activity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force a CPU-only machine view so requesting cuda is genuinely unsupported.
    monkeypatch.setattr(torch.profiler, "supported_activities", lambda: {CPU})
    cfg = TorchProfilerConfig(enabled=True, activities=("cuda",))
    with (
        pytest.raises(RuntimeError, match="unsupported activities"),
        capture_torch_trace(
            cfg,
            output_dir=str(tmp_path),
            step=0,
            device="cpu",
            worker_name="t",
        ),
    ):
        pass


def test_capture_disabled_is_passthrough(tmp_path: Path) -> None:
    cfg = TorchProfilerConfig(enabled=False)
    with capture_torch_trace(cfg, output_dir=str(tmp_path), step=0, device="cpu", worker_name="t"):
        ran = True
    assert ran
    assert not (tmp_path / "torch_profiler").exists()


# ---------------------------------------------------------------------------
# P2 — real CPU torch trace produces the full trust triad
# ---------------------------------------------------------------------------


def test_cpu_trace_writes_trace_summary_manifest(tmp_path: Path) -> None:
    manifest = run_smoke(("cpu",), tmp_path)
    assert manifest["effective_activities"] == ["cpu"]
    assert manifest["missing_activities"] == []
    assert manifest["trace_files"]
    assert manifest["summary_file"]

    trace_dir = tmp_path / "torch_profiler" / "profile_smoke"
    assert (trace_dir / "profile_manifest.json").is_file()
    summary_text = (trace_dir / manifest["summary_file"]).read_text(encoding="utf-8")
    # Summary must warn against summing nested-range percentages.
    assert "not additive wall-time" in summary_text


def test_manifest_is_json_with_required_fields(tmp_path: Path) -> None:
    run_smoke(("cpu",), tmp_path)
    manifest_path = tmp_path / "torch_profiler" / "profile_smoke" / "profile_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "requested_activities",
        "effective_activities",
        "missing_activities",
        "trace_files",
        "summary_file",
        "torch_version",
        "cuda_available",
        "hostname",
    }
    assert required <= set(manifest)
