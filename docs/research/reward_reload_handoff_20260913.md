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
