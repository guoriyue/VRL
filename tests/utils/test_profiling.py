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
    ProfilerActivitySelection,
    TimeIntervals,
    TorchProfilerConfig,
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
        ProfilerActivitySelection.from_config(cfg, supported={CPU, CUDA})


def test_resolve_cpu_only_reports_missing_cuda() -> None:
    cfg = TorchProfilerConfig(activities=("cpu", "cuda"))
    selection = ProfilerActivitySelection.from_config(cfg, supported={CPU})
    assert selection.requested == ("cpu", "cuda")
    assert selection.effective == ("cpu",)
    assert selection.missing == ("cuda",)
    assert selection.torch_activities == (CPU,)


def test_resolve_all_supported() -> None:
    cfg = TorchProfilerConfig(activities=("cpu", "cuda"))
    selection = ProfilerActivitySelection.from_config(cfg, supported={CPU, CUDA})
    assert selection.missing == ()
    assert selection.effective == ("cpu", "cuda")


def test_resolve_deduplicates_requested() -> None:
    cfg = TorchProfilerConfig(activities=("cpu", "cpu", "cuda"))
    selection = ProfilerActivitySelection.from_config(cfg, supported={CPU, CUDA})
    assert selection.requested == ("cpu", "cuda")


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

    # run_smoke returns the manifest parsed from the file it just produced.
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


@pytest.mark.parametrize(
    ("enabled", "skip_first", "max_steps", "expected"),
    [
        (False, 0, 1, []),
        (True, 0, 1, [0]),
        (True, 2, 2, [2, 3]),
        (True, 2, 0, [2, 3, 4]),
        (True, 2, -1, [2, 3, 4]),
    ],
)
def test_config_selects_capture_window(enabled, skip_first, max_steps, expected) -> None:
    config = TorchProfilerConfig(enabled=enabled, skip_first=skip_first, max_steps=max_steps)
    assert [step for step in range(5) if config.should_capture(step)] == expected


def test_capture_manifest_does_not_include_another_step_trace(tmp_path: Path) -> None:
    cfg = TorchProfilerConfig(enabled=True, activities=("cpu",), max_steps=0)
    for step in (10, 1):
        with capture_torch_trace(
            cfg,
            output_dir=str(tmp_path),
            step=step,
            device="cpu",
            worker_name="trainer",
        ):
            torch.ones(2).sum()
    manifest_path = tmp_path / "torch_profiler" / "trainer" / "profile_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["step"] == 1
    assert manifest["trace_files"]
    assert all("_step1." in name for name in manifest["trace_files"])


def test_trace_discovery_excludes_temporary_and_backup_files(tmp_path: Path) -> None:
    from vrl.utils.profiling import _discover_trace_files

    names = [
        "worker.1.pt.trace.json",
        "worker.2.pt.trace.json.gz",
        "worker.3.pt.trace.json.tmp",
        "worker.4.pt.trace.json.bak",
        "worker.5.pt.trace.json.gz.tmp",
        "another.1.pt.trace.json",
    ]
    for name in names:
        (tmp_path / name).touch()
    (tmp_path / "worker.6.pt.trace.json").mkdir()
    assert _discover_trace_files(tmp_path, "worker") == names[:2]


@pytest.mark.parametrize(
    ("left", "right", "duration", "overlap"),
    [
        ([], [(0, 9)], 0, 0),
        ([(0, 6)], [(4, 9)], 6, 2),
        ([(0, 2), (5, 7)], [(2, 5)], 4, 0),
        ([(5, 9), (0, 6), (1, 2)], [(3, 7), (4, 8)], 9, 5),
    ],
)
def test_time_intervals_measure_coverage_and_overlap(left, right, duration, overlap):
    first = TimeIntervals(left)
    second = TimeIntervals(right)
    assert first.duration == duration
    assert first.overlap(second) == overlap
    assert second.overlap(first) == overlap


def test_time_intervals_preserve_integer_nanoseconds():
    start = 10**18
    first = TimeIntervals([(start, start + 3), (start + 2, start + 5)])
    second = TimeIntervals([(start + 1, start + 4)])
    assert first.duration == 5
    assert first.overlap(second) == 3
    assert first.intervals == ((start, start + 5),)
