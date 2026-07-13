"""Measure whether CO-RUNNING two processes on one GPU raises AGGREGATE throughput,
and whether CUDA MPS is actually in effect while doing so.

WHY: `sm_util` ("some kernel is resident") near 100% does not mean the GPU is full.
A kernel that occupies every SM but only ~20% of each SM's warp slots leaves the
machine half empty, and no batch-size increase fixes it — occupancy there is pinned
by kernel SHAPE, not by batch (measured across sd3_5/wan/anima/cosmos, see
docs/sprints/info/SPRINT_cross_model_performance.md §3). The only lever that fills a
GPU a single process cannot fill is running a SECOND process on it *concurrently* —
which on NVIDIA requires MPS. Without MPS, two processes on one GPU do not overlap at
all: the driver time-slices their contexts and each simply runs at half speed.

This probe brackets the answer with two regimes:

  under-filled kernel (small GEMM) -> one process cannot saturate the SMs
                                      => MPS co-run SHOULD raise aggregate TFLOPS
  saturated kernel   (large GEMM)  -> one process already at the bf16 tensor ceiling
                                      => MPS co-run should change NOTHING

If the saturated regime shows a gain, the "saturated" baseline was wrong. If the
under-filled regime shows no gain, MPS is not actually attached (see below) or this
GPU cannot co-schedule.

MPS ATTACHMENT IS THE #1 FOOTGUN: a client that fails to reach the control daemon
falls back to plain context time-slicing SILENTLY — you get a plausible-looking table
that measures the exact thing you meant to avoid. This probe therefore refuses to
label a run "MPS" unless it sees an `nvidia-cuda-mps-server` process holding the GPU
while the clients run, and prints the verdict either way. (Observed on this box: a
custom CUDA_MPS_PIPE_DIRECTORY under /tmp/claude-*/… makes the control daemon exit
immediately and every client silently time-slices; the DEFAULT pipe dir works.)

Run (measure BOTH, in this order — the no-MPS pass is the control):

    python -m vrl.scripts.perf.mps_colocation_probe                 # control: no MPS

    nvidia-cuda-mps-control -d                                      # start daemon
    python -m vrl.scripts.perf.mps_colocation_probe                 # MPS pass
    echo quit | nvidia-cuda-mps-control                             # ALWAYS tear down

The daemon is global to the user: while it is up, every CUDA process this user starts
attaches to it. Do not leave it running, and do not start it while an unrelated
training job is live.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import torch


def mps_server_pids() -> list[str]:
    """PIDs of MPS servers currently holding the GPU (empty == no MPS in effect)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [
        line.split(",")[0].strip()
        for line in out.splitlines()
        if "nvidia-cuda-mps-server" in line
    ]


def _run_worker(size: int, seconds: float, start_at: float) -> float:
    a = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
    for _ in range(10):  # warmup + cuBLAS autotune, so the window is steady-state
        torch.mm(a, b)
    torch.cuda.synchronize()

    while time.time() < start_at:  # all clients enter the window together
        time.sleep(0.001)

    iters = 0
    t0 = time.perf_counter()
    deadline = t0 + seconds
    while time.perf_counter() < deadline:
        for _ in range(20):
            torch.mm(a, b)
        iters += 20
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return 2.0 * size * size * size * iters / elapsed / 1e12


def _co_run(size: int, nproc: int, seconds: float, startup_s: float) -> tuple[list[float], bool]:
    """Launch `nproc` worker processes whose timed windows overlap; return their
    per-process TFLOPS and whether an MPS server was live during the window."""
    start_at = time.time() + startup_s
    procs = [
        subprocess.Popen(
            [sys.executable, "-m", "vrl.scripts.perf.mps_colocation_probe",
             "--worker", "--size", str(size), "--seconds", str(seconds),
             "--start-at", repr(start_at)],
            stdout=subprocess.PIPE, text=True,
        )
        for _ in range(nproc)
    ]
    time.sleep(startup_s + seconds / 2.0)  # sample while the clients are mid-window
    mps_live = bool(mps_server_pids())
    tflops = []
    for p in procs:
        out, _ = p.communicate(timeout=600)
        tflops.append(json.loads(out.strip().splitlines()[-1])["tflops"])
    return tflops, mps_live


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--size", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--start-at", type=float, default=0.0, help=argparse.SUPPRESS)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--startup-s", type=float, default=14.0,
                    help="grace period for every worker to import torch and warm up")
    ap.add_argument("--under-filled-size", type=int, default=1024)
    ap.add_argument("--saturated-size", type=int, default=8192)
    ap.add_argument("--nproc", type=int, default=2)
    args = ap.parse_args()

    if args.worker:
        print(json.dumps({"tflops": round(_run_worker(args.size, args.seconds, args.start_at), 1)}))
        return 0

    print(f"GPU: {torch.cuda.get_device_name(0)}   co-run clients: {args.nproc}")
    mps_pre = mps_server_pids()
    print(f"MPS server before start: {mps_pre or 'none (control pass — expect NO co-run gain)'}\n")

    verdicts = []
    for label, size in (("under-filled", args.under_filled_size), ("saturated", args.saturated_size)):
        solo, _ = _co_run(size, 1, args.seconds, args.startup_s)
        co, mps_live = _co_run(size, args.nproc, args.seconds, args.startup_s)
        aggregate = sum(co)
        speedup = aggregate / solo[0] if solo[0] else 0.0
        mode = "MPS" if mps_live else "no-MPS (context time-slicing)"
        print(f"--- {label} regime: {size}^3 bf16 GEMM   [{mode}] ---")
        print(f"  1 proc            : {solo[0]:6.1f} TFLOPS")
        print(f"  {args.nproc} proc co-run     : "
              f"{' + '.join(f'{t:.1f}' for t in co)} = {aggregate:6.1f} TFLOPS aggregate")
        print(f"  --> co-run speedup: {speedup:.2f}x\n")
        verdicts.append((label, mode, speedup))

    print("READ THIS AS:")
    for label, mode, speedup in verdicts:
        print(f"  {label:<12} [{mode}]  {speedup:.2f}x")
    print("\n  under-filled gain + saturated no-gain  -> co-location is a real lever HERE;")
    print("     the win is bounded by the fraction of GPU time your workload spends in")
    print("     under-filled kernels — measure that with gpm_sampler (sm_occupancy) first.")
    print("  no gain in EITHER regime under 'no-MPS' -> expected: without MPS the driver")
    print("     time-slices contexts, so two processes never overlap. That is the control.")
    print("  no gain in under-filled WITH 'MPS'      -> MPS is attached but this GPU is not")
    print("     co-scheduling; co-location is dead here, do not build on it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
