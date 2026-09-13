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
