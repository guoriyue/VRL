"""Contract tests for the kernel-union GPU-busy analysis API.

The whole point of this module is that the load-bearing metric (kernel-interval
union over wall, per device) is provable WITHOUT a GPU: the interval algebra is
pure, and the sqlite layer is exercised against a synthetic DB built with the
exact CUPTI/NVTX schema nsys 2025.3 exports (verified empirically). The real-GPU
path is the nsys runbook in ``vrl/scripts/perf/nsys_gpu_busy.py``.
"""

from __future__ import annotations

import sqlite3

import pytest

from vrl.utils.nsys_report import (
    analyze,
    clip_intervals,
    format_report,
    merge_intervals,
    overlap_length,
    report_to_dict,
    union_length,
)


# ---------------------------------------------------------------------------
# Pure interval algebra — no sqlite, no GPU
# ---------------------------------------------------------------------------
def test_merge_unions_overlapping_and_touching() -> None:
    # [0,10] and [5,15] overlap -> [0,15]; [20,25] disjoint; [25,30] touches -> [20,30]
    merged = merge_intervals([(0, 10), (5, 15), (20, 25), (25, 30)])
    assert merged == [(0, 15), (20, 30)]


def test_union_length_counts_overlap_once() -> None:
    # naive sum would be 10+10+5 = 25; the union is [0,15]+[20,25] = 20
    assert union_length([(0, 10), (5, 15), (20, 25)]) == 20


def test_union_drops_empty_intervals() -> None:
    assert union_length([(5, 5), (10, 9)]) == 0


def test_clip_keeps_only_the_in_window_part() -> None:
    # a kernel [8,20] crossing the window edge [0,10] must contribute only [8,10].
    # This is the projection-vs-union distinction: projection would credit all 12.
    assert clip_intervals([(8, 20)], 0, 10) == [(8, 10)]


def test_overlap_length_disjoint_is_zero() -> None:
    assert overlap_length(0, 5, 10, 20) == 0
    assert overlap_length(0, 15, 10, 20) == 5


# ---------------------------------------------------------------------------
# Synthetic sqlite with the verified nsys 2025.3 schema (CPU-only)
# ---------------------------------------------------------------------------
def _make_db(path) -> str:
    """Build a minimal capture: one device, a window of 1000 ns, two kernels with
    a 300 ns idle gap between them, an NVTX 'rollout' range covering both, and a
    memcpy + cudaLaunchKernel sitting inside the gap.

    Timeline (ns):  [0 kernel 200] ... gap 200..500 ... [500 kernel 800] ... 1000
    Kernel-union busy = 200 + 300 = 500 over a 800 wall (last_end) = 62.5%.
    """

    db = sqlite3.connect(str(path))
    db.executescript(
        """
        create table StringIds (id integer, value text);
        create table CUPTI_ACTIVITY_KIND_KERNEL (start integer, end integer, deviceId integer, demangledName integer, shortName integer, streamId integer);
        create table CUPTI_ACTIVITY_KIND_RUNTIME (start integer, end integer, nameId integer, correlationId integer, globalTid integer);
        create table CUPTI_ACTIVITY_KIND_MEMCPY (start integer, end integer, deviceId integer, bytes integer, streamId integer);
        create table NVTX_EVENTS (text text, start integer, end integer, eventType integer);
        create table TARGET_INFO_GPU (id integer, name text, smCount integer);
        create table META_DATA_EXPORT (name text, value text);
        """
    )
    db.executemany("insert into StringIds values (?,?)", [
        (1, "matmul_kernel"),
        (2, "cudaLaunchKernel_v7000"),
        (3, "cudaMemcpyAsync_v3020"),
    ])
    # two kernels with a 200..500 idle gap
    db.executemany(
        "insert into CUPTI_ACTIVITY_KIND_KERNEL values (?,?,?,?,?,?)",
        [(0, 200, 0, 1, 1, 7), (500, 800, 0, 1, 1, 7)],
    )
    # host API inside the gap: a launch (50ns) and the host side of a memcpy (120ns)
    db.executemany(
        "insert into CUPTI_ACTIVITY_KIND_RUNTIME values (?,?,?,?,?)",
        [(250, 300, 2, 11, 99), (300, 420, 3, 12, 99)],
    )
    # copy-engine memcpy inside the gap (overlaps compute-idle but is copy-engine work)
    db.execute("insert into CUPTI_ACTIVITY_KIND_MEMCPY values (?,?,?,?,?)", (310, 410, 0, 1_000_000, 8))
    db.execute("insert into NVTX_EVENTS values (?,?,?,?)", ("rollout", 0, 800, 59))
    db.execute("insert into TARGET_INFO_GPU values (?,?,?)", (0, "Synthetic GPU", 170))
    db.execute("insert into META_DATA_EXPORT values (?,?)", ("Export Version", "2025.3.2"))
    db.commit()
    db.close()
    return str(path)


def test_default_window_busy_fraction(tmp_path) -> None:
    path = _make_db(tmp_path / "cap.sqlite")
    rep = analyze(path)
    assert rep.window == (0, 800)  # kernel-span default
    (dev,) = rep.per_device
    assert dev.kernel_count == 2
    assert dev.busy_ns == 500  # 200 + 300, the idle 200..500 excluded
    assert dev.busy_fraction == pytest.approx(500 / 800)
    assert dev.name == "Synthetic GPU"


def test_idle_gap_is_decomposed(tmp_path) -> None:
    path = _make_db(tmp_path / "cap.sqlite")
    rep = analyze(path, min_gap_ns=100)
    (gap,) = rep.idle_gaps
    assert (gap.start, gap.end) == (200, 500)
    assert gap.duration_ns == 300
    # the copy-engine memcpy (310..410) shows up apart from compute
    assert gap.memcpy_ns == 100
    assert gap.memcpy_bytes == 1_000_000
    # host API is version-suffix-stripped and summed by overlap with the gap
    names = {a.name: a.total_ns for a in gap.api_breakdown}
    assert names["cudaLaunchKernel"] == 50
    assert names["cudaMemcpyAsync"] == 120


def test_nvtx_attribution_uses_union_not_projection(tmp_path) -> None:
    path = _make_db(tmp_path / "cap.sqlite")
    rep = analyze(path)
    (stage,) = rep.nvtx
    assert stage.name == "rollout"
    assert stage.occurrences == 1
    assert stage.summed_wall_ns == 800
    assert stage.union_busy_ns == 500  # kernels clipped to the range, unioned
    assert stage.busy_fraction == pytest.approx(500 / 800)


def test_window_nvtx_selects_named_range(tmp_path) -> None:
    path = _make_db(tmp_path / "cap.sqlite")
    rep = analyze(path, window_nvtx="rollout")
    assert rep.window == (0, 800)
    assert "nvtx" in rep.provenance.window_source


def test_unknown_nvtx_window_fails_loud(tmp_path) -> None:
    path = _make_db(tmp_path / "cap.sqlite")
    with pytest.raises(RuntimeError, match="no NVTX range matching"):
        analyze(path, window_nvtx="does-not-exist")


def test_explicit_window_clips_busy(tmp_path) -> None:
    path = _make_db(tmp_path / "cap.sqlite")
    # window 100..600 clips kernel1 to [100,200]=100 and kernel2 to [500,600]=100
    rep = analyze(path, window=(100, 600))
    (dev,) = rep.per_device
    assert dev.busy_ns == 200
    assert dev.wall_ns == 500


def test_multi_device_busy_is_per_device(tmp_path) -> None:
    path = _make_db(tmp_path / "cap2.sqlite")
    db = sqlite3.connect(path)
    # add a second device whose kernel overlaps device 0 in wall time
    db.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?,?,?,?,?,?)", (0, 800, 1, 1, 1, 9))
    db.execute("insert into TARGET_INFO_GPU values (?,?,?)", (1, "Synthetic GPU 2", 170))
    db.commit()
    db.close()
    rep = analyze(path)
    busy = {d.device_id: d.busy_ns for d in rep.per_device}
    # dev0 union = 500 (the gap stays idle); dev1 = 800 (one continuous kernel).
    # They are NOT merged across devices even though they overlap in wall time.
    assert busy == {0: 500, 1: 800}


def test_report_renders_and_serialises(tmp_path) -> None:
    path = _make_db(tmp_path / "cap.sqlite")
    rep = analyze(path, min_gap_ns=100)
    text = format_report(rep)
    assert "kernel-interval UNION" in text
    assert "NOT nsys nvtx_gpu_proj_sum" in text
    payload = report_to_dict(rep)
    assert payload["per_device"][0]["busy_ns"] == 500
    assert payload["nvtx"][0]["union_busy_ns"] == 500


def test_capture_without_nvtx_or_memcpy_still_reports(tmp_path) -> None:
    """A narrower capture (no nvtx, no memcpy, no gpu-info tables) must still yield
    a valid per-device busy report, not crash on the missing optional tables."""

    path = tmp_path / "bare.sqlite"
    db = sqlite3.connect(str(path))
    db.executescript(
        """
        create table StringIds (id integer, value text);
        create table CUPTI_ACTIVITY_KIND_KERNEL (start integer, end integer, deviceId integer, demangledName integer, shortName integer, streamId integer);
        """
    )
    db.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?,?,?,?,?,?)", (0, 100, 0, 1, 1, 0))
    db.commit()
    db.close()
    rep = analyze(str(path))
    assert rep.per_device[0].busy_ns == 100
    assert rep.per_device[0].name == "device0"  # no TARGET_INFO_GPU -> fallback label
    assert rep.nvtx == ()
    assert rep.idle_gaps == ()
