"""Isolated resident/disk AdamW memory and I/O comparison on synthetic FP32 tensors.

This measures optimizer residency, not a model forward/backward or recipe convergence.
Each arm runs in a fresh process. CUDA timings synchronize; state hashing and portable
checkpoint export happen after the measurement window. Linux I/O counters distinguish
logical reads from physical reads (which may be served by the filesystem page cache).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from vrl.trainers.disk_optimizer import DiskStreamingAdamW
from vrl.utils.artifacts import publish_evidence_record, sha256_file


def _io_counters():
    return {
        key: int(value)
        for line in Path("/proc/self/io").read_text().splitlines()
        for key, value in [line.split(":", 1)]
    }


def _measure(args):
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("only CPU and CUDA are supported")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    parameters = [
        torch.nn.Parameter(
            torch.full((args.elements,), (index + 1) / 100, dtype=torch.float32, device=device)
        )
        for index in range(args.parameters)
    ]
    for index, parameter in enumerate(parameters):
        parameter.grad = torch.full_like(parameter, (index + 1) / 1000)
    options = dict(lr=0.003, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    optimizer = (
        DiskStreamingAdamW(
            parameters, directory=args.directory, bucket_bytes=args.bucket_bytes, **options
        )
        if args.arm == "disk"
        else torch.optim.AdamW(parameters, foreach=False, fused=False, **options)
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device) if device.type == "cuda" else None
    before = _io_counters()
    timings = []
    io_deltas = []
    for _ in range(args.steps + 1):
        previous_io = _io_counters()
        started = time.perf_counter()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - started)
        after_step_io = _io_counters()
        io_deltas.append({key: after_step_io[key] - previous_io[key] for key in previous_io})
    after = _io_counters()
    measured = {
        "arm": args.arm,
        "measured_at_utc": datetime.now(UTC).isoformat(),
        "first_step_s": timings[0],
        "steady_step_s": timings[1:],
        "median_steady_step_s": statistics.median(timings[1:]),
        "per_step_io": io_deltas,
        "window_io": {key: after[key] - before[key] for key in before},
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "cuda_baseline_allocated_bytes": baseline,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
        if device.type == "cuda"
        else None,
        "cuda_end_allocated_bytes": torch.cuda.memory_allocated(device)
        if device.type == "cuda"
        else None,
        "cuda_device_uuid": str(torch.cuda.get_device_properties(device).uuid)
        if device.type == "cuda"
        else None,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }
    # Compare final parameter AND moment bytes, outside peak-memory/timing measurement.
    digest = hashlib.sha256()
    state = optimizer.state_dict()["state"]
    for index, parameter in enumerate(parameters):
        for value in (
            parameter,
            state[index]["step"],
            state[index]["exp_avg"],
            state[index]["exp_avg_sq"],
        ):
            digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy())
    measured["final_parameter_and_state_sha256"] = digest.hexdigest()
    return measured


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--parameters", type=int, default=16)
    parser.add_argument("--elements", type=int, default=1048576)
    parser.add_argument("--steps", type=int, default=3, help="measured steps after one warmup")
    parser.add_argument("--bucket-bytes", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--arm", choices=("resident", "disk"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if any(
        value <= 0
        for value in (
            args.parameters,
            args.elements,
            args.steps,
            args.bucket_bytes,
            args.timeout_s,
        )
    ):
        parser.error("counts, bucket size, and timeout must be positive")
    if args.elements * 8 + 4 > args.bucket_bytes:
        parser.error("one parameter must fit in the moment bucket")
    if sys.platform != "linux":
        parser.error("this probe uses Linux I/O counters and RSS units")
    if args.arm:
        print(json.dumps(_measure(args), allow_nan=False))
        return
    if args.report is None:
        parser.error("--report is required")
    if args.report.exists():
        raise FileExistsError(args.report)
    args.directory.mkdir(parents=True, exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory(prefix="optimizer-probe-", dir=args.directory) as scratch:
        for arm in ("resident", "disk"):
            command = [
                sys.executable,
                "-m",
                "vrl.scripts.perf.optimizer_streaming_probe",
                "--arm",
                arm,
                "--device",
                args.device,
                "--parameters",
                str(args.parameters),
                "--elements",
                str(args.elements),
                "--steps",
                str(args.steps),
                "--bucket-bytes",
                str(args.bucket_bytes),
                "--directory",
                scratch,
            ]
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=args.timeout_s, check=True
            )
            results.append(json.loads(completed.stdout))
    resident, disk = results
    exact = (
        resident["final_parameter_and_state_sha256"] == disk["final_parameter_and_state_sha256"]
    )
    root = Path(__file__).resolve().parents[3]
    record = {
        "schema": "optimizer-streaming-probe/v1",
        "workload": "synthetic FP32 parameters with constant preallocated gradients",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "arm"
        },
        "parameter_bytes": args.parameters * args.elements * 4,
        "moment_bytes": args.parameters * (args.elements * 8 + 4),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_sha256": {
            str(path.relative_to(root)): sha256_file(path)
            for path in (Path(__file__).resolve(), root / "vrl/trainers/disk_optimizer.py")
        },
        "arms": results,
        "exact_final_state_match": exact,
        "disk_to_resident_step_ratio": disk["median_steady_step_s"]
        / resident["median_steady_step_s"],
        "limitations": [
            "Synthetic optimizer-only workload; no forward/backward or FP32 master copy.",
            "Sequential arms on a shared device; no performance confidence interval.",
            "RSS includes process startup; checkpoint hashing/export excluded from metrics.",
            "Filesystem page cache is not flushed; physical reads can be zero.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    publish_evidence_record(args.report, record)
    print(json.dumps(record, indent=2))
    if not exact:
        raise SystemExit("resident/disk final parameter or moment bytes differ")


if __name__ == "__main__":
    main()
