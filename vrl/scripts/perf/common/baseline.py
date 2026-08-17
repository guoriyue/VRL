"""Append-only performance baseline record shared by perf probe CLIs.

A probe number is only comparable to a later number if you know what produced
it. The 2026-06 CUDA-graph note concluded "CUDA graph works but is worthless"
from one unlabelled data point and stood for two months until a re-measure
(commit ``f4edb3da``) showed the effect was neither zero nor uniform; the
correction had to reconstruct "same RTX 5090, same script" by hand. This module
exists so that provenance is recorded at measurement time instead.

Scope: this records *what was measured*. It does not run, schedule, or compare
measurements, and it is deliberately not a second profiler — stage attribution
stays with ``vrl.utils.profiling`` (see ``docs/PROFILE_PHASES.md``).

The record is JSONL because it is append-only and merge-friendly: two probes
writing on different days never touch the same line.
"""

from __future__ import annotations

import json
import platform
import socket
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_BASELINE_PATH = Path("docs/perf/BASELINE.jsonl")

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class Provenance:
    """Everything needed to decide whether two measurements are comparable.

    ``dirty`` matters as much as ``commit``: a number measured with uncommitted
    changes cannot be reproduced from the commit alone, so it must never be
    silently compared against a clean-tree number.
    """

    timestamp: str
    hostname: str
    gpu_name: str | None
    gpu_count: int
    torch_version: str | None
    cuda_version: str | None
    python_version: str
    commit: str | None
    dirty: bool | None


def _run_git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def _git_state() -> tuple[str | None, bool | None]:
    commit = _run_git("rev-parse", "HEAD")
    if commit is None:
        return None, None
    status = _run_git("status", "--porcelain")
    # status of "" means clean; None means the call failed, which must stay
    # unknown rather than being reported as clean.
    dirty = None if status is None else bool(status)
    return commit, dirty


def _gpu_state() -> tuple[str | None, int, str | None, str | None]:
    try:
        import torch
    except Exception:
        return None, 0, None, None
    torch_version = str(getattr(torch, "__version__", "")) or None
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    try:
        if not torch.cuda.is_available():
            return None, 0, torch_version, cuda_version
        return (
            torch.cuda.get_device_name(),
            int(torch.cuda.device_count()),
            torch_version,
            cuda_version,
        )
    except Exception:
        return None, 0, torch_version, cuda_version


def collect_provenance() -> Provenance:
    """Capture the run context. Never raises — a probe must not fail on metadata."""

    commit, dirty = _git_state()
    gpu_name, gpu_count, torch_version, cuda_version = _gpu_state()
    return Provenance(
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        hostname=socket.gethostname(),
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        torch_version=torch_version,
        cuda_version=cuda_version,
        python_version=platform.python_version(),
        commit=commit,
        dirty=dirty,
    )


@dataclass(slots=True)
class BaselineRecord:
    """One measurement: what was measured, on what, producing which numbers.

    ``metrics`` values must be plain numbers so the record stays diffable and
    comparable; nest anything structured under ``context`` instead.
    """

    probe: str
    metrics: dict[str, float]
    context: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_json(self, provenance: Provenance | None = None) -> dict[str, Any]:
        prov = provenance or collect_provenance()
        bad = sorted(k for k, v in self.metrics.items() if not isinstance(v, (int, float)))
        if bad:
            raise TypeError(
                f"baseline metrics must be numbers; {self.probe!r} has non-numeric {bad}",
            )
        if not self.probe:
            raise ValueError("baseline record requires a probe name")
        if not self.metrics:
            raise ValueError(f"baseline record {self.probe!r} has no metrics")
        return {
            "schema_version": SCHEMA_VERSION,
            "probe": self.probe,
            "metrics": {k: float(v) for k, v in self.metrics.items()},
            "context": dict(self.context),
            "notes": self.notes,
            **{
                "timestamp": prov.timestamp,
                "hostname": prov.hostname,
                "gpu_name": prov.gpu_name,
                "gpu_count": prov.gpu_count,
                "torch_version": prov.torch_version,
                "cuda_version": prov.cuda_version,
                "python_version": prov.python_version,
                "commit": prov.commit,
                "dirty": prov.dirty,
            },
        }


def append_baseline(
    record: BaselineRecord,
    *,
    path: Path | str | None = None,
    provenance: Provenance | None = None,
) -> Path:
    """Append one measurement to the baseline record and return the file path."""

    target = Path(path) if path is not None else DEFAULT_BASELINE_PATH
    payload = record.to_json(provenance)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return target


def read_baseline(path: Path | str | None = None) -> list[dict[str, Any]]:
    """Read every recorded measurement, oldest first.

    Malformed lines are skipped rather than fatal: the record is append-only
    history, and one bad write must not make the whole file unreadable.
    """

    target = Path(path) if path is not None else DEFAULT_BASELINE_PATH
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


__all__ = [
    "DEFAULT_BASELINE_PATH",
    "SCHEMA_VERSION",
    "BaselineRecord",
    "Provenance",
    "append_baseline",
    "collect_provenance",
    "read_baseline",
]
