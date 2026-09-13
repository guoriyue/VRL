"""PyTorch profiler helpers for runtime and trainer steps.

Two layers, kept deliberately separate (see
``docs/sprints/done/SPRINT_trustworthy_profiling_api.md``):

* ``profile_range`` — stage annotation. Names a code block so it shows up as a
  range in the torch profiler trace and, when NVTX is on, in nsys/ncu. It does
  NOT collect anything by itself; it only labels.
* ``capture_torch_trace`` — trace collection. Opens ``torch.profiler.profile``
  around one step, exports the TensorBoard trace, writes a human summary, and
  writes ``profile_manifest.json`` recording what was actually captured.

nsys stays an external collector: Python only emits NVTX ranges, it never wraps
or manages the ``nsys`` CLI. The manifest is the trust anchor — any later claim
about a profile must cite ``requested``/``effective``/``missing`` activities
there, not just a summary table that may be silently incomplete.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vrl.utils.logging import init_logger

logger = init_logger(__name__)


class TimeIntervals[Number: (int, float)]:
    """Elapsed-time coverage of intervals in one unit on the same clock.

    Overlapping intervals are merged so concurrent work is not counted twice.
    This summarizes recorded timestamps; it does not start timers or synchronize GPUs.
    """

    def __init__(self, intervals: Iterable[tuple[Number, Number]]) -> None:
        merged: list[tuple[Number, Number]] = []
        for start, end in sorted(intervals):
            if end < start:
                raise ValueError("interval end must not precede its start")
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        self._intervals = tuple(merged)

    @property
    def intervals(self) -> tuple[tuple[Number, Number], ...]:
        """Merged intervals, preserving the input unit and numeric precision."""
        return self._intervals

    @property
    def duration(self) -> Number:
        """Total covered time, excluding gaps and counting overlaps once."""
        return sum(end - start for start, end in self._intervals)

    def overlap(self, other: TimeIntervals[Number]) -> Number:
        """Time covered by both timelines, measured on the same clock."""
        left = right = 0
        overlap = 0
        while left < len(self._intervals) and right < len(other._intervals):
            left_start, left_end = self._intervals[left]
            right_start, right_end = other._intervals[right]
            overlap += max(0, min(left_end, right_end) - max(left_start, right_start))
            if left_end <= right_end:
                left += 1
            else:
                right += 1
        return overlap


@dataclass(slots=True)
class TorchProfilerConfig:
    """PyTorch profiler settings for one captured step.

    Defaults are a low-overhead short-diagnostic preset: CUDA + CPU ranges, no
    shape/memory/stack/flops recording (those change overhead and tensor
    lifetimes, so they are explicit dotlist overrides for deep dives). See
    ``vrl/config/presets/profile/torch_profiler.yaml``.
    """

    enabled: bool = False
    output_dir: str = ""
    activities: tuple[str, ...] = ("cpu", "cuda")
    record_shapes: bool = False
    profile_memory: bool = False
    with_stack: bool = False
    with_flops: bool = False
    # skip_first/max_steps select WHICH outer step indices get a trace; each
    # qualifying step opens its own single-step profiler (not a torch schedule
    # window). See should_capture and capture_torch_trace.
    skip_first: int = 0
    max_steps: int = 1

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.output_dir = str(self.output_dir or "")
        self.activities = tuple(str(activity).lower() for activity in self.activities)
        self.record_shapes = bool(self.record_shapes)
        self.profile_memory = bool(self.profile_memory)
        self.with_stack = bool(self.with_stack)
        self.with_flops = bool(self.with_flops)
        self.skip_first = max(0, int(self.skip_first))
        self.max_steps = int(self.max_steps)

    def should_capture(self, step: int) -> bool:
        if not self.enabled:
            return False
        skip_first = max(0, int(self.skip_first))
        if step < skip_first:
            return False
        max_steps = int(self.max_steps)
        return max_steps <= 0 or step < skip_first + max_steps


@dataclass(frozen=True, slots=True)
class ProfilerActivitySelection:
    """What the profiler was asked for vs. what this machine can actually record.

    ``effective`` is the set the profiler runs with; ``missing`` is requested but
    unsupported (drives fail-fast). ``torch_activities`` is the concrete
    ``ProfilerActivity`` list handed to ``torch.profiler.profile``.
    """

    # display/provenance-only: recorded in the manifest so a trace can be audited.
    requested: tuple[str, ...]
    # display/provenance-only except for summary section selection.
    effective: tuple[str, ...]
    missing: tuple[str, ...]
    torch_activities: tuple[Any, ...]

    @classmethod
    def from_config(
        cls,
        config: TorchProfilerConfig,
        *,
        supported: Any = None,
    ) -> ProfilerActivitySelection:
        """Resolve requested activities against this machine's real capability.

        Unknown names fail fast (a typo'd activity is a config bug, not something to
        silently drop). ``supported`` defaults to ``supported_activities()`` and is
        injectable so CPU-only behaviour can be unit-tested on a CUDA box.
        """

        import torch

        # Derive legal names from Torch so new backends need no local allow-list.
        by_name = {
            name.lower(): member
            for name, member in torch.profiler.ProfilerActivity.__members__.items()
        }
        requested_names = tuple(dict.fromkeys(str(a).lower() for a in config.activities))

        unknown = [name for name in requested_names if name not in by_name]
        if unknown:
            raise ValueError(
                f"Unknown torch profiler activities {unknown}; valid names are {sorted(by_name)}",
            )

        if supported is None:
            supported = torch.profiler.supported_activities()
        supported_enums = set(supported)

        effective: list[str] = []
        missing: list[str] = []
        torch_activities: list[Any] = []
        for name in requested_names:
            member = by_name[name]
            if member in supported_enums:
                effective.append(name)
                torch_activities.append(member)
            else:
                missing.append(name)

        return cls(
            requested=requested_names,
            effective=tuple(effective),
            missing=tuple(missing),
            torch_activities=tuple(torch_activities),
        )


def nvtx_enabled() -> bool:
    """Process-wide NVTX signal.

    ``VRL_PROFILE=1`` is set once per profiling run (e.g. in
    ``vrl/scripts/common/online.py``) and inherited by Ray workers, so deep
    ``profile_range`` call sites that have no config object still know whether to
    emit NVTX. Explicit ``emit_nvtx=`` arguments override this.
    """

    return os.environ.get("VRL_PROFILE") == "1"


@contextlib.contextmanager
def profile_range(name: str, *, emit_nvtx: bool | None = None) -> Iterator[None]:
    """Annotate a code block as a named VRL stage range.

    Always opens a ``torch.profiler.record_function`` range (visible in the torch
    trace). Additionally pushes an NVTX range when ``emit_nvtx`` resolves true so
    nsys/ncu attribute GPU time to this stage. ``emit_nvtx=None`` defers to
    :func:`nvtx_enabled`. If torch is unavailable or the range cannot be opened,
    this is a transparent no-op that never changes the wrapped block's result.
    """

    try:
        import torch

        ctx = torch.profiler.record_function(name)
    except Exception:
        yield
        return
    if emit_nvtx is None:
        emit_nvtx = nvtx_enabled()
    pushed = False
    if emit_nvtx:
        try:
            torch.cuda.nvtx.range_push(name)
            pushed = True
        except Exception:
            pushed = False
    try:
        with ctx:
            yield
    finally:
        # Pair pop strictly with a successful push so an exception in the body
        # can never leave the NVTX stack unbalanced.
        if pushed:
            torch.cuda.nvtx.range_pop()


@contextlib.contextmanager
def capture_torch_trace(
    config: TorchProfilerConfig,
    *,
    output_dir: str,
    step: int,
    device: Any,
    worker_name: str,
    trace_subdir: str = "trainer",
) -> Iterator[None]:
    """Capture one step into a TensorBoard trace + summary + manifest.

    Single-step semantics: when ``step`` qualifies (see
    :meth:`TorchProfilerConfig.should_capture`) this opens a fresh profiler, captures exactly
    the wrapped block, and exports. Steps that do not qualify run untouched.
    """

    if not config.should_capture(step):
        yield
        return

    import torch

    selection = ProfilerActivitySelection.from_config(config)
    if selection.missing:
        raise RuntimeError(
            "Torch profiler requested unsupported activities "
            f"{list(selection.missing)} (requested={list(selection.requested)}, "
            f"supported={[a.name.lower() for a in torch.profiler.supported_activities()]})",
        )
    if not selection.torch_activities:
        logger.warning(
            "Torch profiler enabled but no requested activity is supported "
            "(requested=%s); skipping capture for step=%d",
            list(selection.requested),
            step,
        )
        yield
        return

    root = Path(config.output_dir) if config.output_dir else Path(output_dir) / "torch_profiler"
    trace_dir = root / trace_subdir if trace_subdir else root
    trace_dir.mkdir(parents=True, exist_ok=True)
    safe_worker_name = _safe_worker_name(worker_name, step)
    logger.info(
        "Starting torch profiler for step=%d (activities=%s); traces -> %s",
        step,
        list(selection.effective),
        trace_dir,
    )
    handler = torch.profiler.tensorboard_trace_handler(
        str(trace_dir),
        worker_name=safe_worker_name,
    )
    prof = None
    try:
        with torch.profiler.profile(
            activities=list(selection.torch_activities),
            record_shapes=bool(config.record_shapes),
            profile_memory=bool(config.profile_memory),
            with_stack=bool(config.with_stack),
            with_flops=bool(config.with_flops),
            on_trace_ready=handler,
        ) as active_prof:
            prof = active_prof
            try:
                yield
            finally:
                active_prof.step()
    finally:
        summary_path = trace_dir / f"{safe_worker_name}.summary.txt"
        if prof is not None:
            _write_summary(prof, summary_path, selection.effective)
        _write_manifest(
            trace_dir,
            config=config,
            selection=selection,
            step=step,
            worker_name=safe_worker_name,
            trace_subdir=trace_subdir,
            trace_files=_discover_trace_files(trace_dir, safe_worker_name),
            summary_file=summary_path.name if prof is not None else None,
            device=device,
        )
    logger.info("Finished torch profiler for step=%d", step)


def _safe_worker_name(worker_name: str, step: int) -> str:
    host = _safe_label(socket.gethostname())
    name = _safe_label(worker_name)
    return f"{host}_{name}_step{step}"


def _safe_label(value: str) -> str:
    safe = []
    for char in str(value):
        if char.isalnum() or char in {"-", "_", "."}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "unknown"


def _discover_trace_files(trace_dir: Path, safe_worker_name: str) -> list[str]:
    """Trace files the TensorBoard handler wrote for this worker.

    The handler names files ``{worker_name}.{ts}.pt.trace.json[.gz]``; an empty
    result means the trace did not materialise and downstream smoke/tests must
    fail rather than trust the summary alone.
    """

    return sorted(
        path.name
        for path in trace_dir.glob(f"{safe_worker_name}.*.pt.trace.json*")
        if path.name.endswith((".pt.trace.json", ".pt.trace.json.gz")) and path.is_file()
    )


def _write_summary(prof: Any, path: Path, effective: tuple[str, ...]) -> None:
    sections = [
        "PyTorch profiler summary",
        "",
        "NOTE: nested ranges (torch record_function / NVTX) are attribution aids,",
        "not additive wall-time percentages. A kernel inside two nested ranges is",
        "counted under both; do not sum nested-range percentages to a wall fraction.",
        "",
        "Top CPU ops",
        _table(prof, sort_by="cpu_time_total"),
    ]
    if "cuda" in effective:
        sections.extend(["", "Top CUDA ops", _table(prof, sort_by="cuda_time_total")])
    path.write_text("\n".join(sections), encoding="utf-8")


def _table(prof: Any, *, sort_by: str) -> str:
    try:
        return prof.key_averages().table(sort_by=sort_by, row_limit=40)
    except Exception as exc:
        return f"Unable to render profiler table sorted by {sort_by}: {exc}"


def _write_manifest(
    trace_dir: Path,
    *,
    config: TorchProfilerConfig,
    selection: ProfilerActivitySelection,
    step: int,
    worker_name: str,
    trace_subdir: str,
    trace_files: list[str],
    summary_file: str | None,
    device: Any,
) -> None:
    """Write the machine-readable trust record for this trace directory."""

    import torch

    cuda_available = bool(torch.cuda.is_available())
    cuda_device_name = None
    if cuda_available and "cuda" in selection.effective:
        try:
            cuda_device_name = torch.cuda.get_device_name()
        except Exception:
            cuda_device_name = None

    manifest = {
        "schema_version": 1,
        "worker_name": worker_name,
        "step": step,
        "trace_subdir": trace_subdir,
        "requested_activities": list(selection.requested),
        "effective_activities": list(selection.effective),
        "missing_activities": list(selection.missing),
        "record_shapes": bool(config.record_shapes),
        "profile_memory": bool(config.profile_memory),
        "with_stack": bool(config.with_stack),
        "with_flops": bool(config.with_flops),
        "emit_nvtx": nvtx_enabled(),
        "trace_files": trace_files,
        "summary_file": summary_file,
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_device_name": cuda_device_name,
        "device": str(getattr(device, "type", device)),
        "hostname": socket.gethostname(),
    }
    path = trace_dir / "profile_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    if not trace_files:
        logger.error(
            "Torch profiler produced no trace file in %s; manifest records an "
            "empty trace_files list (trace incomplete).",
            path.parent,
        )


__all__ = [
    "ProfilerActivitySelection",
    "TimeIntervals",
    "TorchProfilerConfig",
    "capture_torch_trace",
    "nvtx_enabled",
    "profile_range",
]
