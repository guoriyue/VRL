"""Fixed-interval GPU duty/power/memory sampler for continuous-pipeline baselines.

SPRINT_continuous_stage_contracts_and_baseline T2/T3: the four-L4 baseline needs
GPU duty measured *outside* the production hot loop, so this probe runs as a
separate process next to a training run and polls ``nvidia-smi`` on a fixed
cadence. The CSV it writes is a one-shot validation artifact: record the
baseline conclusions in ``docs/sprints/info/`` and delete the CSV afterwards
(the probe itself stays, like the other ``vrl/scripts/perf`` tools).

Timestamps are this process's ``time.monotonic()`` (sprint invariant: never
compare unsynchronized wall clocks across processes); correlate with training
metrics via the wall column only as coarse provenance. Duty ratio in the
summary is the fraction of samples whose utilization exceeds ``--busy-util``,
which is a sampling approximation of kernel-union busy time — use
``vrl.scripts.perf.nsys_gpu_busy`` when exact attribution matters.

Usage (alongside a run, Ctrl-C or --duration-s ends it):

    python -m vrl.scripts.perf.gpu_duty_probe --output outputs/gpu_duty.csv \\
        --interval-s 0.5 --duration-s 3600
"""

from __future__ import annotations

import argparse
import csv
import datetime
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

_QUERY_FIELDS = (
    "index",
    "utilization.gpu",
    "power.draw",
    "memory.used",
    "memory.total",
)


def _sample() -> list[list[str]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(_QUERY_FIELDS)}",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10.0,
    )
    return [
        [cell.strip() for cell in line.split(",")]
        for line in result.stdout.strip().splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="CSV output path")
    parser.add_argument("--interval-s", type=float, default=0.5)
    parser.add_argument(
        "--duration-s",
        type=float,
        default=0.0,
        help="Stop after this many seconds; 0 = run until interrupted",
    )
    parser.add_argument(
        "--busy-util",
        type=float,
        default=5.0,
        help="Utilization %% above which a sample counts as busy in the summary",
    )
    args = parser.parse_args()
    if args.interval_s <= 0:
        raise SystemExit("--interval-s must be > 0")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    busy_samples: dict[str, int] = defaultdict(int)
    total_samples: dict[str, int] = defaultdict(int)

    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("monotonic_s", "wall_iso", *_QUERY_FIELDS))
        try:
            while True:
                elapsed = time.monotonic() - started
                if args.duration_s > 0 and elapsed >= args.duration_s:
                    break
                wall = datetime.datetime.now(datetime.UTC).isoformat()
                for row in _sample():
                    writer.writerow((f"{elapsed:.3f}", wall, *row))
                    gpu, util = row[0], float(row[1])
                    total_samples[gpu] += 1
                    if util > args.busy_util:
                        busy_samples[gpu] += 1
                handle.flush()
                time.sleep(args.interval_s)
        except KeyboardInterrupt:
            pass

    for gpu in sorted(total_samples):
        duty = busy_samples[gpu] / max(1, total_samples[gpu])
        print(f"gpu{gpu}: duty_ratio={duty:.3f} samples={total_samples[gpu]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
