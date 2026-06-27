"""End-to-end smoke for the trustworthy profiling API.

Runs a tiny tensor op through ``profile_range`` inside ``capture_torch_trace`` and
asserts the three trust artifacts exist and agree: TensorBoard trace JSON, the
human summary, and ``profile_manifest.json``. The point is not the kernels — it
is proving a profile run cannot look successful while silently producing a
half-true or missing trace.

Usage:
    python -m vrl.scripts.perf.profile_smoke --activities cpu \\
        --output-dir outputs/profile_smoke_cpu

    # on a CUDA box:
    python -m vrl.scripts.perf.profile_smoke --activities cpu,cuda \\
        --output-dir outputs/profile_smoke_cuda

    # NVTX runbook (external collector, nsys reads the NVTX range names):
    VRL_PROFILE=1 nsys profile --trace=cuda,nvtx \\
        --output outputs/profile_smoke/nsys_profile \\
        python -m vrl.scripts.perf.profile_smoke --activities cpu,cuda
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import torch

from vrl.utils.profiling import (
    TorchProfilerConfig,
    capture_torch_trace,
    profile_range,
)

_OUTER_RANGE = "test.outer"
_MATMUL_RANGE = "test.cuda_matmul"


def _read_trace_text(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return handle.read()
    return path.read_text(encoding="utf-8")


def run_smoke(activities: tuple[str, ...], output_dir: Path) -> dict:
    device = "cuda" if "cuda" in activities and torch.cuda.is_available() else "cpu"
    config = TorchProfilerConfig(
        enabled=True,
        activities=activities,
    )
    trace_subdir = "profile_smoke"
    with capture_torch_trace(
        config,
        output_dir=str(output_dir),
        step=0,
        device=device,
        worker_name="profile_smoke",
        trace_subdir=trace_subdir,
    ):
        with profile_range(_OUTER_RANGE):
            x = torch.ones(64, 64, device=device)
            y = x @ x
            if device == "cuda":
                with profile_range(_MATMUL_RANGE):
                    z = y @ y
                    torch.cuda.synchronize()
                y = z
        _ = y.sum().item()

    trace_dir = output_dir / "torch_profiler" / trace_subdir
    manifest_path = trace_dir / "profile_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"FAIL: manifest missing at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # 1. Trace files exist (empty list == trace never materialised).
    trace_files = manifest["trace_files"]
    if not trace_files:
        raise SystemExit(f"FAIL: no trace file written in {trace_dir}")
    trace_text = "".join(_read_trace_text(trace_dir / name) for name in trace_files)

    # 2. Summary exists.
    summary_name = manifest["summary_file"]
    if not summary_name or not (trace_dir / summary_name).is_file():
        raise SystemExit(f"FAIL: summary missing in {trace_dir}")

    # 3. Effective activities match what the machine could record.
    expected_effective = [a for a in activities if a == "cpu" or torch.cuda.is_available()]
    if manifest["effective_activities"] != expected_effective:
        raise SystemExit(
            "FAIL: effective_activities "
            f"{manifest['effective_activities']} != expected {expected_effective}",
        )

    # 4. The stage range names reached the trace.
    if _OUTER_RANGE not in trace_text:
        raise SystemExit(f"FAIL: trace does not contain range {_OUTER_RANGE!r}")
    if device == "cuda" and _MATMUL_RANGE not in trace_text:
        raise SystemExit(f"FAIL: trace does not contain range {_MATMUL_RANGE!r}")

    print(f"OK: trace+summary+manifest written under {trace_dir}")
    print(f"  effective_activities = {manifest['effective_activities']}")
    print(f"  missing_activities   = {manifest['missing_activities']}")
    print(f"  trace_files          = {trace_files}")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--activities",
        default="cpu",
        help="comma-separated activities, e.g. 'cpu' or 'cpu,cuda'",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/profile_smoke",
        help="root output dir; trace lands under <dir>/torch_profiler/profile_smoke",
    )
    args = parser.parse_args(argv)
    activities = tuple(a.strip().lower() for a in args.activities.split(",") if a.strip())
    run_smoke(activities, Path(args.output_dir))


if __name__ == "__main__":
    sys.exit(main())
