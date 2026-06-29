"""Trustworthy GPU-busy attribution from an nsys capture (CLI).

Thin consumer of :mod:`vrl.utils.nsys_report`. Turns a captured ``.nsys-rep`` (or
exported ``.sqlite``) into the kernel-union GPU-busy fraction, the top idle gaps
with their host/copy-engine breakdown, and per-NVTX-stage attribution — the
analysis two prior sessions did by hand, now one command.

The collection step stays external (nsys is the collector). Capture a real run:

    VRL_PROFILE=1 nsys profile --trace=cuda,nvtx --trace-fork-before-exec=true \\
        --output outputs/run nsys python -m vrl.scripts.train --config ...

``--trace-fork-before-exec=true`` is load-bearing: Ray workers are fork+exec'd by
the raylet and are invisible to a launcher-attached profiler without it (verified:
0 kernels captured). Then analyse:

    python -m vrl.scripts.perf.nsys_gpu_busy outputs/run.nsys-rep
    python -m vrl.scripts.perf.nsys_gpu_busy outputs/run.sqlite --window-nvtx rollout
    python -m vrl.scripts.perf.nsys_gpu_busy outputs/run.sqlite --device 0 --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vrl.utils.nsys_report import analyze, format_report, report_to_dict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture", help="path to a .nsys-rep or exported .sqlite")
    parser.add_argument(
        "--window-nvtx",
        default=None,
        metavar="NAME",
        help="restrict the window to NVTX ranges whose text contains NAME "
        "(e.g. 'rollout'); default window = full kernel span",
    )
    parser.add_argument(
        "--window-ns",
        default=None,
        help="explicit window as START:END nanoseconds (overrides --window-nvtx)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="device id for idle-gap + NVTX attribution; default = busiest device",
    )
    parser.add_argument("--top-gaps", type=int, default=10, help="number of idle gaps to show")
    parser.add_argument(
        "--min-gap-ms",
        type=float,
        default=0.1,
        help="ignore idle gaps shorter than this (ms); the timeline is millions of "
        "sub-launch gaps, so a floor keeps the report about real bubbles",
    )
    parser.add_argument("--top-nvtx", type=int, default=20, help="number of NVTX stages to show")
    parser.add_argument("--json", default=None, metavar="PATH", help="also write the report as JSON")
    args = parser.parse_args(argv)

    window = None
    if args.window_ns is not None:
        try:
            lo, hi = (int(x) for x in args.window_ns.split(":"))
        except ValueError:
            parser.error("--window-ns must be START:END integer nanoseconds")
        window = (lo, hi)

    try:
        report = analyze(
            args.capture,
            window_nvtx=args.window_nvtx,
            window=window,
            device_id=args.device,
            top_gaps=args.top_gaps,
            min_gap_ns=int(args.min_gap_ms * 1e6),
            top_nvtx=args.top_nvtx,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print(format_report(report))
    if args.json:
        Path(args.json).write_text(json.dumps(report_to_dict(report), indent=2), encoding="utf-8")
        print(f"\nJSON written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
