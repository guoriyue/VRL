"""NVML GPM (GPU Performance Monitoring) SM-level metrics sampler.

Samples SM utilization/occupancy, tensor-core activity, FP16 math, and DRAM
bandwidth utilization via the NVML GPM API and appends one CSV row per
interval. GPM gives *SM-level* counters (what fraction of SMs were busy and
how occupied they were), unlike `nvidia-smi` utilization which only reports
"any kernel resident" time.

These counters explain execution; low occupancy alone does not prove unused or
recoverable throughput. Use a real-workload throughput A/B for that decision.

Usage:
    python -m vrl.scripts.perf.gpm_sampler OUT_CSV [INTERVAL_S=1.0]

Stops cleanly on SIGTERM/SIGINT (flushes the CSV before exit). Requires a
GPM-capable device (nvmlGpmQueryDeviceSupport.isSupportedDevice == 1, e.g.
Hopper/Blackwell-class GPUs).
"""

from __future__ import annotations

import signal
import sys
import time

import pynvml

# (metric id, csv column) — order defines the CSV column order after `ts`.
METRICS = (
    (pynvml.NVML_GPM_METRIC_SM_UTIL, "sm_util"),
    (pynvml.NVML_GPM_METRIC_SM_OCCUPANCY, "sm_occupancy"),
    (pynvml.NVML_GPM_METRIC_ANY_TENSOR_UTIL, "any_tensor_util"),
    (pynvml.NVML_GPM_METRIC_FP16_UTIL, "fp16_util"),
    (pynvml.NVML_GPM_METRIC_DRAM_BW_UTIL, "dram_bw_util"),
)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    out_csv = sys.argv[1]
    interval_s = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    stop = False

    def _handle(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    pynvml.nvmlInit()
    try:
        device = pynvml.nvmlDeviceGetHandleByIndex(0)
        support = pynvml.nvmlGpmQueryDeviceSupport(device)
        if not support.isSupportedDevice:
            print("GPM not supported on device 0", file=sys.stderr)
            return 1

        prev = pynvml.nvmlGpmSampleAlloc()
        curr = pynvml.nvmlGpmSampleAlloc()
        try:
            pynvml.nvmlGpmSampleGet(device, prev)
            with open(out_csv, "w") as fh:
                fh.write("ts," + ",".join(name for _, name in METRICS) + "\n")
                fh.flush()
                while not stop:
                    time.sleep(interval_s)
                    pynvml.nvmlGpmSampleGet(device, curr)

                    query = pynvml.c_nvmlGpmMetricsGet_t()
                    query.version = pynvml.NVML_GPM_METRICS_GET_VERSION
                    query.numMetrics = len(METRICS)
                    query.sample1 = prev
                    query.sample2 = curr
                    for i, (metric_id, _) in enumerate(METRICS):
                        query.metrics[i].metricId = metric_id
                    pynvml.nvmlGpmMetricsGet(query)

                    values = []
                    for i in range(len(METRICS)):
                        m = query.metrics[i]
                        values.append(
                            f"{m.value:.3f}" if m.nvmlReturn == pynvml.NVML_SUCCESS else "nan"
                        )
                    fh.write(f"{time.time():.3f}," + ",".join(values) + "\n")
                    fh.flush()
                    # Swap buffers: current sample becomes baseline for the next delta.
                    prev, curr = curr, prev
        finally:
            pynvml.nvmlGpmSampleFree(prev)
            pynvml.nvmlGpmSampleFree(curr)
    finally:
        pynvml.nvmlShutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
