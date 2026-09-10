# Parking evidence must attribute memory to the process

A real SD3.5/OCR attempt stopped at a parking residual check on the shared RTX
5090. The old helper used `torch.cuda.mem_get_info`, so it compared device-wide
usage before policy loading with device-wide usage after parking. That cannot
attribute allocations to the worker: another process can create a false failure,
or release memory and hide a worker leak. This establishes an ownership defect;
it does not by itself identify all residual memory in the failed SD3.5 attempt.

## Independent CUDA reproduction

On September 9, 2026 Pacific, a fresh process allocated a 64 MiB tensor inside
CuMem, while a second process subsequently allocated a 1 GiB CUDA tensor. NVML
selected the GPU by its CUDA UUID and selected the owner by PID. The measured
bytes were:

| Observation | Device used | Owner physical used | Owner Torch allocated |
|---|---:|---:|---:|
| Before pooled allocation | 759,889,920 | 608,174,080 | 0 |
| Pooled tensor resident | 826,998,784 | 675,282,944 | 67,108,864 |
| Pool asleep, second process active | 2,447,376,384 | 608,174,080 | 67,108,864 |

The owner's physical footprint returned exactly to its baseline. Device-wide
usage rose about 1.57 GiB, including the second process's context/runtime overhead.
Torch still counted the logically allocated CuMem tensor after its physical pages
were unmapped. Wake restored the original tensor contents exactly. Local raw
probe/log: `/tmp/vrl_parking_ownership_probe.py` and
`/tmp/vrl_parking_ownership_probe.log`.

## Change and preserved boundaries

`gpu_process_used_bytes` now supplies physical measurements to generation parking
and in-process reward parking. It synchronizes CUDA, resolves the actual CUDA GPU
UUID (rather than confusing CUDA-visible ordinals with physical NVML ordinals),
and requires one unambiguous NVML entry for the current PID. Missing, unavailable
or ambiguous accounting fails; it does not become zero or a device-wide fallback.
CPU-only paths return zero without importing NVML. Snapshots explicitly label the
measurement scope as `process`.

The 256 MiB runtime-residual allowance, CuMem backend requirement, strict cleanup,
quarantine behavior and phase leases are unchanged. `gpu_used_bytes` remains a
public device-wide diagnostic; it no longer supplies ownership evidence. The
shared process-measurement function is an actual driver/API boundary used by both
owners. Reward's existing method still binds that read to its configured device.
No new family vocabulary or memory-policy registry was introduced.

NVML's Python binding moved from the optional performance extra to core dependencies;
`perf` remains an accepted empty compatibility extra. Unsupported MPS, PID namespaces,
MIG accounting or driver restrictions fail closed if the current process cannot be
identified. This proves a process's physical release, not that the whole card has
sufficient free capacity, and it cannot distinguish multiple owners inside one
process. Existing phase ownership remains necessary.

Validation: 131 memory, worker sleep, Ray lease, reward and architecture checks
passed, one skipped. The added real CUDA subprocess test parks/wakes CuMem while
another process holds a 1 GiB allocation. CPU failure-injection checks cover foreign
PID changes, missing/duplicate entries, unavailable counters, query failures and
NVML cleanup. A full SD3.5/OCR retry is still needed to establish whether any genuine
worker residual exceeds the unchanged allowance.

## Real SD3.5/OCR retry after the fix

The unchanged two-epoch seed-17 attempt at
`outputs/repro/sd3_5_ocr_strict_d` ran from clean commit `a583d830c` and published
launch `39f2f4b878e64172be01dc0beac770e1`. The first two measured process parking
residuals were both 692,060,160 bytes (660 MiB), against a 522,190,848-byte baseline
(498 MiB): a stable 162 MiB excess, below the unchanged 256 MiB allowance.
The run passed the point where the old device-wide check had stopped attempt C.

It subsequently stopped at the first optimizer-update parity gate:
`max_abs_diff=0.02667667716741562`, with finite values and the unchanged `0.01`
limit. The persisted gate event records `trainer_step=0`, `global_step=0` and
`passed=false`. There are no completed epoch rows or successful final artifact
receipt. The memory ownership repair therefore has real-model evidence; the
separate rollout/replay numerical mismatch remains unresolved. No parking or
numerical threshold was widened. Full local log:
`/tmp/vrl-sd3-ocr-strict-d.log`.
