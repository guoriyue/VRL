# Opt-in reward reload handoff

Candidate `a6fc7356`, branch `feat/reward-reload-handoff`, follows the unchanged
review branch `review/all-mgpu-main-6b723075` at `4e2c1163`.

## Contract

Set `reward.kwargs.kling_video_reward.worker_config.memory_parking_mode: reload`
on a topology that already requires reward parking. Resource resolution still
injects `sleep_offload: true`; standalone scorer construction must provide it.
Default mode remains `cumem`. Reload mode is explicit, requires glibc
`malloc_trim`, and is rejected when parking is not enabled.

At `park_memory`, reload mode drops the model, performs existing strict CUDA
cleanup, then trims released host heap pages. A later activation reloads and
prepares the model inside the existing RNG-preserving build boundary. No CuMem
backup is retained. Failure propagates; cleanup can be retried. This trades
checkpoint load latency for lower cross-phase host residency. Physical release
proof remains owned by the existing caller/lifecycle; no budget is relaxed.

## Evidence

- Reward inference tests: 66 passed, 4 GPU tests deselected.
- Disk reward and online lifecycle tests: 49 passed.
- Scoped Ruff passed.
- Native four-device probe: all 16 complete score maps exactly equal after two
  loads and two scores per load on each device, using one scorer per worker.
  No manual trim is called by the probe. Four workers completed in 91.028s.
- Reload latency: 37.78-37.91s per worker. Second parked RSS:
  1,913,270,272; 1,918,697,472; 1,915,133,952; 1,909,043,200 bytes.
- Lowest sampled host available memory: 368,779,689,984 bytes. All workers
  exited successfully and the final GPU compute inventory was empty.

Raw evidence under `/mnt/nvme/outputs/wan22_i2v_cache/`:
`kling_native_reload_four_rebased/result.json`, per-rank logs and result files,
`kling_native_reload_probe.py`, and `kling_native_reload_four_probe.py`.
Weights are cached pinned Kling VideoReward weights; source MP4 SHA256 and
runtime configuration are embedded in every worker report. Torch 2.11.0+cu130
comes from the lock-synced review venv, not the older experiment environment.

## Next Combined Gate

`wan22_rebased_reload_prepare.py` generated new single/four-rank YAMLs and
`wan22_rebased_reload_preflight.json` without rewriting the prompt manifest.
Both arms retain two updates, eight global samples per update, matched prompt
seeds, model/optimizer/precision, and two samples per generation/replay batch.
Global request plans and partitioned group advantages match exactly. Online
batch fields now use main's `prompts_per_collection` and
`training_microbatch_size`.

These results prove the reward-only handoff, not combined Wan capacity,
throughput, quality, or long-run memory stability. Run the complete four-rank
pilot next, then the matching single-card arm; retain existing failed runs and
the 95% host-memory protection. Do not mix old-environment timings or scores
with the rebased comparison.

## Combined Four-Rank Attempt

Clean candidate `76714903` ran `wan22_rebased_reload_four` on all four GPUs,
using the locked venv and unchanged workload/thresholds. Supervisor exited 1
after 541.036s. All initial eight samples were generated and scored; the first
optimizer update completed and `checkpoint-1` was saved. The next rollout wake
failed because Ray had killed the generation workers for host-memory pressure.
No second update or accepted throughput comparison exists.

First-update full-precision metrics: pre-update maximum log-probability difference
0, pre-update clip fraction 0, gradient norm 0.030143787340297008. Per-rank
phase totals are 335.68-335.76s, which must not be reported as a successful
two-update benchmark or speedup. Independent `checkpoint_1_audit.json` confirms
1280 FP32 adapter tensors, complete Adam and EMA state, four-rank RNG, progress,
and eight global collected samples. It does not compare against initialization
or certify the failed subsequent rollout synchronization.

523 memory samples reached 7,638,007,808 bytes available; sampled maximum GPU
usage was 11,983,126,528 bytes. Raylet logs first cross the 95% threshold around
16:45:31-32 and kill workers around 16:45:39. Recorded backward ends at
16:45:31.378 and optimizer runs 16:45:31.384-32.575; rank phase summaries finish
around 16:45:35. The driver receives the stored failure on second rollout wake
around 16:45:52. Thus pressure begins at the optimizer/update boundary, not
necessarily at the later wake where the error becomes visible. Specific tensor
or allocator ownership at that peak is not yet proven.

At Ray's kill decision, generation workers each used approximately 55.6-56.4 GiB
and trainer processes approximately 29 GiB. Object-store occupancy was zero.
All processes exited and the fresh GPU compute inventory was empty. Preserve
the failed output, adjacent log/memory/process receipts, and the valid first
checkpoint. Next isolate optimizer/export/checkpoint host lifetimes at this
boundary, retaining samples, replay gates, and memory protection. The matching
single-card arm remains unrun.

## Optimizer-Boundary Trim Diagnostic

Candidate `ffbebe7e` ran `wan22_rebased_boundary_trim_four` with the same
two-update workload and unchanged 95% host-memory protection. An external entry
wrapper measured each trainer immediately before `_clip_and_step`, after GC,
after glibc `malloc_trim(0)`, and after the optimizer. It did not clear PyTorch's
pinned host allocator cache or change training math. The supervisor exited 1
after 532.850s; checkpoint-1 exists, but the second update failed.

This attempt corrects the location inferred from the previous attempt: Raylet
first reported threshold crossing at 16:58:27.701, during the fifth replay
backward (epoch timestamp 1789343899.643-1789343908.254). Worker killing began
around 16:58:35. The optimizer phase only started at 16:59:34.224. Therefore
optimizer-boundary cleanup is too late for this run. The much higher available
host memory measured at that boundary follows worker termination and must not
be interpreted as successful memory reclamation by the wrapper.

Rank 0 GC left RSS unchanged at 31,968,821,248 bytes; glibc trimming lowered it
to 31,800,492,032 bytes, only 168,329,216 bytes reclaimed. This does not identify
the earlier live-tensor or pinned-cache ownership responsible for pressure.

The completed CPU comparison `first_state_comparison.json` establishes exact
first-checkpoint equality against `wan22_rebased_reload_four`: 6,412 Torch
tensors, four NumPy arrays, and 8,962 scalar leaves, including model, Adam, EMA,
RNG, identity and progress. It does not establish second-update synchronization
or throughput. All experiment processes exited and GPU compute inventory was
empty. No numerical-work reduction or threshold increase was applied.

Next inspect host allocation ownership during replay, before the first threshold
crossing, including the distinction between active and cached pinned memory.
Do not promote optimizer-boundary trimming as a capacity fix or rerun this same
failed intervention. Keep the matching single-card performance arm pending a
complete four-card run.

## Pinned Allocator Observation Gate

The locked Torch 2.11.0+cu130 environment exposes
`torch.cuda.memory.host_memory_stats()` and the private
`torch._C._host_emptyCache()` used by its own CUDA graph implementation.
`vrl/trainers/activation_checkpointing.py:cpu_checkpoint_func` uses
`save_on_cpu(pin_memory=True)`, making this allocator relevant to replay.

The bounded real-CUDA `pinned_host_allocator_probe.py` exited 0 and preserved its
raw report in `pinned_host_allocator_probe.json`. After an asynchronous transfer
from a 256 MiB pinned buffer, dropping that buffer and synchronizing, emptying
the host cache reduced allocator-owned bytes from 269,484,040 to 1,048,576.
A separately retained 1 MiB pinned tensor still contained the expected values.
After releasing that tensor and emptying again, allocator-owned bytes were zero.
This proves the local API can reclaim this synthetic unused allocation while
preserving that live tensor, not that it fixes Wan or is a production contract.

Crucially, the raw active counters were inconsistent: after emptying, reported
active bytes exceeded allocator-owned bytes; the final active-byte count was
269,484,041 with a cumulative freed count of -1 despite zero owned allocations.
The probe's `passed` status concerns release and retained tensor contents only,
not validity of every statistic. Do not subtract active bytes from owned bytes
to diagnose a leak in this environment. Future replay traces must retain raw
counters and pair them with process RSS/PSS and measured release deltas. Do not
patch the installed Torch or assume its accounting anomaly explains Wan's OOM.
