"""Tests for the append-only perf baseline record."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vrl.scripts.perf.common.baseline import (
    SCHEMA_VERSION,
    BaselineRecord,
    Provenance,
    append_baseline,
    collect_provenance,
    read_baseline,
)


def _provenance(**overrides: object) -> Provenance:
    base = {
        "timestamp": "2026-08-16T00:00:00+00:00",
        "hostname": "testhost",
        "gpu_name": "NVIDIA GeForce RTX 5090",
        "gpu_count": 1,
        "torch_version": "2.9.0",
        "cuda_version": "12.8",
        "python_version": "3.12.0",
        "commit": "abc123",
        "dirty": False,
    }
    base.update(overrides)
    return Provenance(**base)  # type: ignore[arg-type]


def test_append_then_read_roundtrips(tmp_path: Path) -> None:
    path = tmp_path / "BASELINE.jsonl"
    append_baseline(
        BaselineRecord(probe="compile_benchmark", metrics={"fusion_ms": 29.37}),
        path=path,
        provenance=_provenance(),
    )

    rows = read_baseline(path)
    assert len(rows) == 1
    assert rows[0]["probe"] == "compile_benchmark"
    assert rows[0]["metrics"]["fusion_ms"] == pytest.approx(29.37)
    assert rows[0]["gpu_name"] == "NVIDIA GeForce RTX 5090"
    assert rows[0]["schema_version"] == SCHEMA_VERSION


def test_append_is_additive_not_overwriting(tmp_path: Path) -> None:
    """The record is history; a second run must never replace the first."""
    path = tmp_path / "BASELINE.jsonl"
    append_baseline(
        BaselineRecord(probe="p", metrics={"ms": 1.0}),
        path=path,
        provenance=_provenance(timestamp="2026-06-01T00:00:00+00:00"),
    )
    append_baseline(
        BaselineRecord(probe="p", metrics={"ms": 2.0}),
        path=path,
        provenance=_provenance(timestamp="2026-08-16T00:00:00+00:00"),
    )

    rows = read_baseline(path)
    assert [row["metrics"]["ms"] for row in rows] == [1.0, 2.0]
    assert [row["timestamp"] for row in rows] == [
        "2026-06-01T00:00:00+00:00",
        "2026-08-16T00:00:00+00:00",
    ]


def test_each_record_is_one_standalone_json_line(tmp_path: Path) -> None:
    """JSONL so concurrent probes on different days never collide on a line."""
    path = tmp_path / "BASELINE.jsonl"
    for value in (1.0, 2.0):
        append_baseline(
            BaselineRecord(probe="p", metrics={"ms": value}),
            path=path,
            provenance=_provenance(),
        )

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        json.loads(line)  # each line parses independently


def test_dirty_tree_is_recorded(tmp_path: Path) -> None:
    """A number measured on a dirty tree is not reproducible from the commit."""
    path = tmp_path / "BASELINE.jsonl"
    append_baseline(
        BaselineRecord(probe="p", metrics={"ms": 1.0}),
        path=path,
        provenance=_provenance(dirty=True),
    )
    assert read_baseline(path)[0]["dirty"] is True


def test_unknown_git_state_is_not_reported_as_clean(tmp_path: Path) -> None:
    """Outside a repo, dirty must stay None rather than defaulting to False."""
    path = tmp_path / "BASELINE.jsonl"
    append_baseline(
        BaselineRecord(probe="p", metrics={"ms": 1.0}),
        path=path,
        provenance=_provenance(commit=None, dirty=None),
    )
    row = read_baseline(path)[0]
    assert row["commit"] is None
    assert row["dirty"] is None


def test_non_numeric_metric_fails_fast() -> None:
    record = BaselineRecord(probe="p", metrics={"ms": "fast"})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="must be numbers"):
        record.to_json(_provenance())


def test_empty_metrics_fails_fast() -> None:
    with pytest.raises(ValueError, match="no metrics"):
        BaselineRecord(probe="p", metrics={}).to_json(_provenance())


def test_missing_probe_name_fails_fast() -> None:
    with pytest.raises(ValueError, match="probe name"):
        BaselineRecord(probe="", metrics={"ms": 1.0}).to_json(_provenance())


def test_context_carries_structured_run_shape(tmp_path: Path) -> None:
    """Metrics stay flat numbers; run shape belongs in context."""
    path = tmp_path / "BASELINE.jsonl"
    append_baseline(
        BaselineRecord(
            probe="compile_benchmark",
            metrics={"fusion_ms": 29.37, "cudagraph_ms": 28.35},
            context={"family": "sd3_5", "layers": 24, "arm": "compiled"},
            notes="re-measure of the 2026-06 record",
        ),
        path=path,
        provenance=_provenance(),
    )
    row = read_baseline(path)[0]
    assert row["context"]["family"] == "sd3_5"
    assert row["notes"].startswith("re-measure")


def test_read_missing_file_is_empty(tmp_path: Path) -> None:
    assert read_baseline(tmp_path / "nope.jsonl") == []


def test_malformed_line_does_not_poison_the_record(tmp_path: Path) -> None:
    """One bad write must not make the whole append-only history unreadable."""
    path = tmp_path / "BASELINE.jsonl"
    append_baseline(
        BaselineRecord(probe="good", metrics={"ms": 1.0}),
        path=path,
        provenance=_provenance(),
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{ this is not json\n")
    append_baseline(
        BaselineRecord(probe="also_good", metrics={"ms": 2.0}),
        path=path,
        provenance=_provenance(),
    )

    rows = read_baseline(path)
    assert [row["probe"] for row in rows] == ["good", "also_good"]


def test_collect_provenance_never_raises() -> None:
    """Metadata collection must not be able to fail a measurement run."""
    prov = collect_provenance()
    assert prov.timestamp
    assert prov.hostname
    assert isinstance(prov.gpu_count, int)
