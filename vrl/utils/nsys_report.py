"""Trustworthy GPU-busy attribution from a Nsight Systems (nsys) capture.

This is the ANALYSIS half of the trustworthy profiling API; the
collection/annotation half lives in :mod:`vrl.utils.profiling`
(``profile_range`` emits the NVTX ranges this module reads back). nsys stays the
external collector — this module never launches or wraps a profiling *run*; it
only reads the sqlite that ``nsys export`` produces from an already-captured
``.nsys-rep``. See ``docs/sprints/done/SPRINT_trustworthy_profiling_api.md``
Part B.

THE LOAD-BEARING CORRECTION (cost three wrong reads to get right; see
``project_real_run_profiling`` memory):

    GPU-busy = the UNION of kernel [start, end] intervals over the wall window,
    computed PER DEVICE (merge overlaps, one deviceId at a time).

    It is NOT nsys' ``nvtx_gpu_proj_sum`` "Proj Time". Projection sums the GPU
    time of kernels *launched* inside a range; because launches are async the
    kernels spill past the range wall, so projection reports GPU-time > wall-time
    (e.g. denoise_forward 78.2s "GPU" vs 65.7s wall) and misleads anyone trying
    to read it as a busy fraction. Only the kernel-interval union over wall
    answers "is the GPU idle".

Two more honest-by-construction rules this module enforces:

* Union is per device. Two GPUs running kernels at the same wall instant are both
  busy; their intervals must not be merged into one. A run with N devices yields N
  busy fractions, never one cross-device number.
* NVTX range attribution is reported as union-busy clipped to each range, plus the
  summed per-occurrence wall with an explicit "ranges may nest/overlap; do not sum
  to a wall fraction" caveat — never as additive projection percentages.
"""

from __future__ import annotations

import re
import shutil
import sqlite3
import subprocess
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from vrl.utils.logging import init_logger

logger = init_logger(__name__)

# nsys names CUDA runtime API calls with a binding-version suffix
# (``cudaLaunchKernel_v7000``, ``cudaMemcpyAsync_v3020``). Strip it so the idle
# breakdown groups all versions of one API under a single readable name.
_API_VERSION_SUFFIX = re.compile(r"_v\d+$")

# Interval = (start_ns, end_ns) on the nsys session clock (nanoseconds, monotonic
# across the whole capture; kernels, runtime API, NVTX and memcpy share it).
Interval = tuple[int, int]


# ---------------------------------------------------------------------------
# Interval algebra — the core of the kernel-union metric, kept pure so it is
# unit-testable on any machine with no GPU and no sqlite.
# ---------------------------------------------------------------------------
def merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    """Merge overlapping/touching ``(start, end)`` intervals into disjoint runs.

    Empty (``end <= start``) intervals are dropped. Touching intervals
    (``next.start == prev.end``) merge — back-to-back kernels are continuous GPU
    busy, not two separate busy spans.
    """

    items = sorted((s, e) for s, e in intervals if e > s)
    if not items:
        return []
    merged: list[list[int]] = [list(items[0])]
    for start, end in items[1:]:
        last = merged[-1]
        if start <= last[1]:
            if end > last[1]:
                last[1] = end
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def union_length(intervals: Iterable[Interval]) -> int:
    """Total length covered by the union of ``intervals`` (overlaps counted once)."""

    return sum(end - start for start, end in merge_intervals(intervals))


def clip_intervals(intervals: Iterable[Interval], lo: int, hi: int) -> list[Interval]:
    """Clip each interval to ``[lo, hi]``, dropping anything outside the window.

    This is what keeps union-busy honest at window edges: a kernel that starts
    inside a window but ends after it contributes only the part that actually
    falls inside — unlike projection, which would credit the whole kernel.
    """

    out: list[Interval] = []
    for start, end in intervals:
        s2 = max(start, lo)
        e2 = min(end, hi)
        if e2 > s2:
            out.append((s2, e2))
    return out


def overlap_length(start: int, end: int, lo: int, hi: int) -> int:
    """Length of ``[start, end]`` ∩ ``[lo, hi]`` (0 if disjoint)."""

    return max(0, min(end, hi) - max(start, lo))


# ---------------------------------------------------------------------------
# Result structs — derived/resolved; every field has a non-logging consumer
# (CLI control-flow, JSON export, or a test assertion).
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DeviceBusy:
    """Kernel-union GPU-busy for one device over the window."""

    device_id: int = field()
    name: str = field()
    kernel_count: int = field()
    busy_ns: int = field()
    wall_ns: int = field()

    @property
    def busy_fraction(self) -> float:
        return self.busy_ns / self.wall_ns if self.wall_ns > 0 else 0.0


@dataclass(frozen=True, slots=True)
class ApiSpan:
    """One host-side CUDA API (or copy-engine) cost, summed over a window/gap.

    ``name`` is version-suffix-stripped (``cudaLaunchKernel``, ``cudaMemcpyAsync``,
    ``cudaStreamSynchronize``). ``total_ns`` is summed overlap with the window —
    these CAN overlap each other (different threads), so it is a sum, never a
    union, and may exceed the window wall.
    """

    name: str = field()
    count: int = field()
    total_ns: int = field()


@dataclass(frozen=True, slots=True)
class IdleGap:
    """One stretch where the chosen device ran no kernel — the reclaimable idle.

    ``api_breakdown`` decomposes what the host/copy-engine were doing during the
    gap, which is how a between-sample orchestration bubble gets pinned to
    memcpy / launch overhead / sync / pure-Python (a gap with little API time is
    host Python).
    """

    start: int = field()
    end: int = field()
    api_breakdown: tuple[ApiSpan, ...] = field()
    memcpy_ns: int = field()
    memcpy_bytes: int = field()

    @property
    def duration_ns(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class NvtxBusy:
    """Kernel-union busy attributed to all occurrences of one NVTX range name.

    ``summed_wall_ns`` adds each occurrence's wall span; occurrences of different
    names may NEST (denoise_forward inside a per-sample range), so this column is
    NOT additive to a wall fraction — read ``union_busy_ns`` for "did the GPU work
    here", and treat ``summed_wall_ns`` only as this name's own footprint.
    """

    name: str = field()
    occurrences: int = field()
    summed_wall_ns: int = field()
    union_busy_ns: int = field()

    @property
    def busy_fraction(self) -> float:
        return self.union_busy_ns / self.summed_wall_ns if self.summed_wall_ns > 0 else 0.0


@dataclass(frozen=True, slots=True)
class ReportProvenance:
    """Trust anchor: where the numbers came from and what tool produced them.

    Mirrors the manifest discipline of :mod:`vrl.utils.profiling` — a later claim
    cites this (source, nsys version, devices, window source), not just a busy
    percentage that could come from an empty or truncated capture.
    """

    source_path: str = field()
    sqlite_path: str = field()
    nsys_version: str = field()
    total_kernels: int = field()
    devices: tuple[tuple[int, str], ...] = field()
    window_source: str = field()


@dataclass(frozen=True, slots=True)
class GpuBusyReport:
    """The trustworthy GPU-busy attribution for one capture window."""

    window: Interval = field()
    wall_ns: int = field()
    per_device: tuple[DeviceBusy, ...] = field()
    gap_device: int = field()
    idle_gaps: tuple[IdleGap, ...] = field()
    nvtx: tuple[NvtxBusy, ...] = field()
    provenance: ReportProvenance = field()


# ---------------------------------------------------------------------------
# sqlite access. All optional tables are probed for existence first so a capture
# taken without nvtx/memcpy/gpu-info still yields a valid (narrower) report.
# ---------------------------------------------------------------------------
def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "select 1 from sqlite_master where type='table' and name=?", (name,)
    ).fetchone()
    return row is not None


def _strip_api_version(name: str) -> str:
    return _API_VERSION_SUFFIX.sub("", name)


def open_report(path: str | Path) -> tuple[sqlite3.Connection, str]:
    """Open a capture as a sqlite connection, exporting from ``.nsys-rep`` if needed.

    Accepts a ``.sqlite`` (used directly) or a ``.nsys-rep`` (exported to a temp
    sqlite via ``nsys export --type sqlite`` — a post-collection step, not a
    profiling run). Returns ``(connection, sqlite_path)``.
    """

    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"nsys capture not found: {src}")
    if src.suffix == ".sqlite":
        return sqlite3.connect(str(src)), str(src)
    if src.suffix in (".nsys-rep", ".qdrep"):
        nsys = shutil.which("nsys")
        if nsys is None:
            raise RuntimeError(
                f"{src.suffix} given but `nsys` not on PATH to export sqlite; "
                "run `nsys export --type sqlite` yourself and pass the .sqlite",
            )
        out = Path(tempfile.mkdtemp(prefix="nsys_report_")) / (src.stem + ".sqlite")
        logger.info("Exporting %s -> %s via nsys export", src.name, out)
        subprocess.run(
            [nsys, "export", "--type", "sqlite", "--force-overwrite=true",
             "--output", str(out), str(src)],
            check=True,
            capture_output=True,
        )
        return sqlite3.connect(str(out)), str(out)
    raise ValueError(f"unsupported capture extension {src.suffix!r} (want .sqlite/.nsys-rep)")


def _device_names(conn: sqlite3.Connection) -> dict[int, str]:
    if not _has_table(conn, "TARGET_INFO_GPU"):
        return {}
    return {int(i): str(n) for i, n in conn.execute("select id, name from TARGET_INFO_GPU")}


def _kernels_by_device(
    conn: sqlite3.Connection, window: Interval | None
) -> dict[int, list[Interval]]:
    """All kernel intervals bucketed by deviceId, optionally clipped to ``window``."""

    rows = conn.execute(
        "select start, end, deviceId from CUPTI_ACTIVITY_KIND_KERNEL"
    ).fetchall()
    by_dev: dict[int, list[Interval]] = {}
    lo, hi = window if window else (None, None)
    for start, end, dev in rows:
        if end <= start:
            continue
        if window is not None:
            start = max(start, lo)
            end = min(end, hi)
            if end <= start:
                continue
        by_dev.setdefault(int(dev), []).append((int(start), int(end)))
    return by_dev


def capture_window(conn: sqlite3.Connection) -> Interval:
    """Default window = first kernel start → last kernel end (the GPU-active span).

    Falls back to the capture's ANALYSIS_DETAILS span if no kernels were recorded
    (so an empty/CUPTI-less capture is visibly empty, not silently zero-wall).
    """

    row = conn.execute(
        "select min(start), max(end) from CUPTI_ACTIVITY_KIND_KERNEL"
    ).fetchone()
    if row and row[0] is not None:
        return int(row[0]), int(row[1])
    if _has_table(conn, "ANALYSIS_DETAILS"):
        row = conn.execute("select min(startTime), max(stopTime) from ANALYSIS_DETAILS").fetchone()
        if row and row[0] is not None:
            return int(row[0]), int(row[1])
    raise RuntimeError("capture has no kernels and no ANALYSIS_DETAILS span; nothing to window")


def nvtx_window(conn: sqlite3.Connection, name: str) -> Interval:
    """Window spanning every NVTX range whose text contains ``name``.

    Used for ``--window-nvtx rollout``: the window is min-start → max-end across
    all matching ranges, so "the rollout window" is exactly the wall the stage
    occupied. Raises if nothing matches (an empty window must never silently
    become a 0% busy result).
    """

    if not _has_table(conn, "NVTX_EVENTS"):
        raise RuntimeError("capture has no NVTX_EVENTS; rerun nsys with --trace=...,nvtx")
    row = conn.execute(
        "select min(start), max(end) from NVTX_EVENTS "
        "where text like ? and end is not null",
        (f"%{name}%",),
    ).fetchone()
    if not row or row[0] is None:
        raise RuntimeError(f"no NVTX range matching {name!r} in capture")
    return int(row[0]), int(row[1])


def _api_breakdown(conn: sqlite3.Connection, lo: int, hi: int) -> tuple[list[ApiSpan], int, int]:
    """Host CUDA-API time and copy-engine time overlapping ``[lo, hi]``.

    Returns ``(api_spans_sorted_desc, memcpy_ns, memcpy_bytes)``. API spans are
    summed overlap (they may concur across threads); memcpy is copy-engine GPU
    work, reported apart from compute because it runs on a separate engine and is
    exactly the data movement that CAN hide behind the next stage's compute.
    """

    spans: dict[str, list[int]] = {}
    if _has_table(conn, "CUPTI_ACTIVITY_KIND_RUNTIME") and _has_table(conn, "StringIds"):
        rows = conn.execute(
            "select r.start, r.end, s.value "
            "from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId = s.id "
            "where r.start < ? and r.end > ?",
            (hi, lo),
        ).fetchall()
        for start, end, raw in rows:
            ov = overlap_length(int(start), int(end), lo, hi)
            if ov <= 0:
                continue
            name = _strip_api_version(str(raw))
            agg = spans.setdefault(name, [0, 0])
            agg[0] += 1
            agg[1] += ov

    memcpy_ns = 0
    memcpy_bytes = 0
    if _has_table(conn, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        rows = conn.execute(
            "select start, end, bytes from CUPTI_ACTIVITY_KIND_MEMCPY "
            "where start < ? and end > ?",
            (hi, lo),
        ).fetchall()
        for start, end, nbytes in rows:
            ov = overlap_length(int(start), int(end), lo, hi)
            if ov <= 0:
                continue
            memcpy_ns += ov
            memcpy_bytes += int(nbytes or 0)

    api = [ApiSpan(name, c, t) for name, (c, t) in spans.items()]
    api.sort(key=lambda a: a.total_ns, reverse=True)
    return api, memcpy_ns, memcpy_bytes


def _idle_gaps(
    conn: sqlite3.Connection,
    busy_intervals: Sequence[Interval],
    window: Interval,
    *,
    top: int,
    min_gap_ns: int,
) -> list[IdleGap]:
    """Complement of the kernel-union within the window = device idle stretches.

    Returns the ``top`` longest gaps (>= ``min_gap_ns``), each decomposed by what
    the host/copy-engine were doing — this is the between-sample boundary
    breakdown done by hand in two prior sessions, now reproducible.
    """

    lo, hi = window
    merged = merge_intervals(busy_intervals)
    gaps: list[Interval] = []
    cursor = lo
    for start, end in merged:
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < hi:
        gaps.append((cursor, hi))

    gaps = [g for g in gaps if g[1] - g[0] >= min_gap_ns]
    gaps.sort(key=lambda g: g[1] - g[0], reverse=True)
    result: list[IdleGap] = []
    for start, end in gaps[:top]:
        api, memcpy_ns, memcpy_bytes = _api_breakdown(conn, start, end)
        result.append(
            IdleGap(
                start=start,
                end=end,
                api_breakdown=tuple(api),
                memcpy_ns=memcpy_ns,
                memcpy_bytes=memcpy_bytes,
            )
        )
    return result


def _nvtx_attribution(
    conn: sqlite3.Connection,
    kernels: Sequence[Interval],
    *,
    top: int,
) -> list[NvtxBusy]:
    """Per-NVTX-name kernel-union busy, for the chosen device.

    Groups every occurrence of a range *name* (e.g. ``denoise_dit_forward`` x T),
    sums each occurrence's wall, and unions the device kernels clipped to those
    occurrences. Returns the ``top`` names by summed wall.
    """

    if not _has_table(conn, "NVTX_EVENTS"):
        return []
    rows = conn.execute(
        "select text, start, end from NVTX_EVENTS where end is not null and text is not null"
    ).fetchall()
    by_name: dict[str, list[Interval]] = {}
    for text, start, end in rows:
        if end <= start:
            continue
        by_name.setdefault(str(text), []).append((int(start), int(end)))

    out: list[NvtxBusy] = []
    for name, spans in by_name.items():
        summed_wall = sum(e - s for s, e in spans)
        clipped: list[Interval] = []
        for s, e in spans:
            clipped.extend(clip_intervals(kernels, s, e))
        out.append(
            NvtxBusy(
                name=name,
                occurrences=len(spans),
                summed_wall_ns=summed_wall,
                union_busy_ns=union_length(clipped),
            )
        )
    out.sort(key=lambda n: n.summed_wall_ns, reverse=True)
    return out[:top]


def _nsys_version(conn: sqlite3.Connection) -> str:
    for table, col in (("META_DATA_EXPORT", "value"), ("META_DATA_CAPTURE", "value")):
        if not _has_table(conn, table):
            continue
        try:
            rows = conn.execute(f"select name, {col} from {table}").fetchall()
        except sqlite3.OperationalError:
            continue
        for key, value in rows:
            if "version" in str(key).lower():
                return str(value)
    return "unknown"


def analyze(
    path: str | Path,
    *,
    window_nvtx: str | None = None,
    window: Interval | None = None,
    device_id: int | None = None,
    top_gaps: int = 10,
    min_gap_ns: int = 100_000,
    top_nvtx: int = 20,
) -> GpuBusyReport:
    """Build the trustworthy GPU-busy report for one capture window.

    ``window`` (explicit ns) > ``window_nvtx`` (span of a named range) > default
    (the full kernel span). ``device_id`` selects the device for idle-gap and NVTX
    attribution; default is the device with the most kernels (the compute GPU).
    Per-device busy fractions are always reported for every device.
    """

    conn, sqlite_path = open_report(path)
    try:
        if window is not None:
            win = window
            window_source = f"explicit {win[0]}..{win[1]} ns"
        elif window_nvtx is not None:
            win = nvtx_window(conn, window_nvtx)
            window_source = f"nvtx~{window_nvtx!r}"
        else:
            win = capture_window(conn)
            window_source = "kernel-span (default)"
        lo, hi = win
        if hi <= lo:
            raise RuntimeError(f"empty window {win}")
        wall = hi - lo

        names = _device_names(conn)
        by_dev = _kernels_by_device(conn, win)
        total_kernels = sum(len(v) for v in by_dev.values())

        per_device: list[DeviceBusy] = []
        for dev in sorted(by_dev):
            ivals = by_dev[dev]
            per_device.append(
                DeviceBusy(
                    device_id=dev,
                    name=names.get(dev, f"device{dev}"),
                    kernel_count=len(ivals),
                    busy_ns=union_length(ivals),
                    wall_ns=wall,
                )
            )

        if device_id is None:
            gap_dev = max(by_dev, key=lambda d: len(by_dev[d])) if by_dev else 0
        else:
            gap_dev = device_id
        gap_kernels = by_dev.get(gap_dev, [])

        gaps = _idle_gaps(conn, gap_kernels, win, top=top_gaps, min_gap_ns=min_gap_ns)
        nvtx = _nvtx_attribution(conn, gap_kernels, top=top_nvtx)

        provenance = ReportProvenance(
            source_path=str(path),
            sqlite_path=sqlite_path,
            nsys_version=_nsys_version(conn),
            total_kernels=total_kernels,
            devices=tuple((d, names.get(d, f"device{d}")) for d in sorted(by_dev)),
            window_source=window_source,
        )
        return GpuBusyReport(
            window=win,
            wall_ns=wall,
            per_device=tuple(per_device),
            gap_device=gap_dev,
            idle_gaps=tuple(gaps),
            nvtx=tuple(nvtx),
            provenance=provenance,
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _ms(ns: int) -> float:
    return ns / 1e6


def _pct(part: int, whole: int) -> float:
    return 100.0 * part / whole if whole > 0 else 0.0


def format_report(report: GpuBusyReport) -> str:
    """Human-readable report. The header states the metric so a number copied out
    of it carries its own caveat (kernel-union over wall, NOT nsys projection)."""

    p = report.provenance
    lines: list[str] = [
        "GPU-busy report (kernel-interval UNION over wall, per device)",
        "  metric = merged [start,end] of CUPTI kernels / window wall.",
        "  This is NOT nsys nvtx_gpu_proj_sum 'Proj Time' (async launch makes",
        "  projection exceed wall — do not read it as a busy fraction).",
        "",
        f"source        : {p.source_path}",
        f"nsys version  : {p.nsys_version}",
        f"window        : {report.window[0]}..{report.window[1]} ns "
        f"({_ms(report.wall_ns):.1f} ms wall, {p.window_source})",
        f"kernels        : {p.total_kernels}",
        "",
        "Per-device GPU-busy:",
    ]
    for d in report.per_device:
        lines.append(
            f"  dev{d.device_id} {d.name:<28} {d.kernel_count:>7} kern  "
            f"busy {_ms(d.busy_ns):>10.1f} ms  = {d.busy_fraction * 100:5.1f}% busy "
            f"({(1 - d.busy_fraction) * 100:4.1f}% idle)"
        )

    if report.idle_gaps:
        lines += ["", f"Top idle gaps on dev{report.gap_device} (GPU ran no kernel):"]
        for g in report.idle_gaps:
            lines.append(
                f"  gap {_ms(g.duration_ns):>9.2f} ms @ +{_ms(g.start - report.window[0]):.1f} ms"
                f"  memcpy {_ms(g.memcpy_ns):.2f} ms / {g.memcpy_bytes / 1e6:.1f} MB"
            )
            for a in g.api_breakdown[:5]:
                lines.append(
                    f"      {a.name:<28} {a.count:>6}x  {_ms(a.total_ns):>9.2f} ms "
                    f"({_pct(a.total_ns, g.duration_ns):4.0f}% of gap)"
                )

    if report.nvtx:
        lines += [
            "",
            f"NVTX stage attribution on dev{report.gap_device} "
            "(union-busy clipped to range; summed wall may NEST — not additive):",
            f"  {'stage':<32}{'occ':>5}{'wall ms':>11}{'busy ms':>11}{'busy%':>7}",
        ]
        for n in report.nvtx:
            lines.append(
                f"  {n.name[:32]:<32}{n.occurrences:>5}"
                f"{_ms(n.summed_wall_ns):>11.1f}{_ms(n.union_busy_ns):>11.1f}"
                f"{n.busy_fraction * 100:>6.0f}%"
            )
    return "\n".join(lines)


def report_to_dict(report: GpuBusyReport) -> dict:
    """JSON-serialisable view (for --json export / programmatic consumers)."""

    return {
        "window_ns": list(report.window),
        "wall_ns": report.wall_ns,
        "gap_device": report.gap_device,
        "provenance": {
            "source_path": report.provenance.source_path,
            "sqlite_path": report.provenance.sqlite_path,
            "nsys_version": report.provenance.nsys_version,
            "total_kernels": report.provenance.total_kernels,
            "devices": [list(d) for d in report.provenance.devices],
            "window_source": report.provenance.window_source,
        },
        "per_device": [
            {
                "device_id": d.device_id,
                "name": d.name,
                "kernel_count": d.kernel_count,
                "busy_ns": d.busy_ns,
                "busy_fraction": d.busy_fraction,
            }
            for d in report.per_device
        ],
        "idle_gaps": [
            {
                "start": g.start,
                "end": g.end,
                "duration_ns": g.duration_ns,
                "memcpy_ns": g.memcpy_ns,
                "memcpy_bytes": g.memcpy_bytes,
                "api_breakdown": [
                    {"name": a.name, "count": a.count, "total_ns": a.total_ns}
                    for a in g.api_breakdown
                ],
            }
            for g in report.idle_gaps
        ],
        "nvtx": [
            {
                "name": n.name,
                "occurrences": n.occurrences,
                "summed_wall_ns": n.summed_wall_ns,
                "union_busy_ns": n.union_busy_ns,
                "busy_fraction": n.busy_fraction,
            }
            for n in report.nvtx
        ],
    }


__all__ = [
    "ApiSpan",
    "DeviceBusy",
    "GpuBusyReport",
    "IdleGap",
    "NvtxBusy",
    "ReportProvenance",
    "analyze",
    "capture_window",
    "clip_intervals",
    "format_report",
    "merge_intervals",
    "nvtx_window",
    "open_report",
    "overlap_length",
    "report_to_dict",
    "union_length",
]
