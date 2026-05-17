"""Optional PyTorch profiler helpers for runtime and trainer steps."""

from __future__ import annotations

import contextlib
import logging
import socket
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TorchProfilerConfig:
    """PyTorch profiler settings for visual TensorBoard traces."""

    enabled: bool = False
    output_dir: str = ""
    activities: tuple[str, ...] = ("cpu", "cuda")
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = False
    with_flops: bool = False
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


@contextlib.contextmanager
def record_function(name: str) -> Iterator[None]:
    """Record a profiler range when torch profiler is available."""

    try:
        import torch

        ctx = torch.profiler.record_function(name)
    except Exception:
        ctx = contextlib.nullcontext()
    with ctx:
        yield


@contextlib.contextmanager
def torch_profiler_step(
    config: TorchProfilerConfig,
    *,
    output_dir: str,
    step: int,
    device: Any,
    worker_name: str,
    trace_subdir: str = "trainer",
) -> Iterator[None]:
    """Profile one trainer step and write a TensorBoard-compatible trace."""

    if not _should_profile_step(config, step):
        yield
        return

    import torch

    activities = _resolve_activities(config, device)
    if not activities:
        logger.warning("Torch profiler enabled but no supported activities were selected")
        yield
        return

    trace_dir = _trace_dir(config, output_dir, trace_subdir=trace_subdir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    safe_worker_name = _safe_worker_name(worker_name, step)
    logger.info(
        "Starting torch profiler for step=%d; traces will be written to %s",
        step,
        trace_dir,
    )
    handler = torch.profiler.tensorboard_trace_handler(
        str(trace_dir),
        worker_name=safe_worker_name,
    )
    prof = None
    try:
        with torch.profiler.profile(
            activities=activities,
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
        if prof is not None:
            _write_summary(prof, trace_dir / f"{safe_worker_name}.summary.txt", activities)
    logger.info("Finished torch profiler for step=%d", step)


def _should_profile_step(config: TorchProfilerConfig, step: int) -> bool:
    if not config.enabled:
        return False
    skip_first = max(0, int(config.skip_first))
    if step < skip_first:
        return False
    max_steps = int(config.max_steps)
    return max_steps <= 0 or step < skip_first + max_steps


def _resolve_activities(config: TorchProfilerConfig, device: Any) -> list[Any]:
    import torch

    requested = {str(activity).lower() for activity in config.activities}
    activities: list[Any] = []
    if "cpu" in requested:
        activities.append(torch.profiler.ProfilerActivity.CPU)
    device_type = getattr(device, "type", str(device)).lower()
    if "cuda" in requested and device_type.startswith("cuda") and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return activities


def _trace_dir(config: TorchProfilerConfig, output_dir: str, *, trace_subdir: str) -> Path:
    root = Path(config.output_dir) if config.output_dir else Path(output_dir) / "torch_profiler"
    return root / trace_subdir if trace_subdir else root


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


def _write_summary(prof: Any, path: Path, activities: list[Any]) -> None:
    import torch

    sections = [
        "PyTorch profiler summary",
        "",
        "Top CPU ops",
        _table(prof, sort_by="cpu_time_total"),
    ]
    if torch.profiler.ProfilerActivity.CUDA in activities:
        sections.extend(
            [
                "",
                "Top CUDA ops",
                _table(prof, sort_by="cuda_time_total"),
            ],
        )
    path.write_text("\n".join(sections), encoding="utf-8")


def _table(prof: Any, *, sort_by: str) -> str:
    try:
        return prof.key_averages().table(sort_by=sort_by, row_limit=40)
    except Exception as exc:
        return f"Unable to render profiler table sorted by {sort_by}: {exc}"


__all__ = [
    "TorchProfilerConfig",
    "record_function",
    "torch_profiler_step",
]
