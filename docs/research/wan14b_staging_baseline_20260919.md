# Wan 14B baseline for the next staging version

Saved evidence: [snapshot overview](../../outputs/wan14b_staging_baseline_20260919/20260919T190053Z/README.md).
Raw logs, resource samples, configs, reward results, code patch, replay preflight
receipts, hardware/software versions and SHA-256 manifest are in that snapshot.
`outputs/wan14b_staging_baseline_20260919/LATEST` identifies the newest export.

| Measurement | Batch 2, attempt 4 | Batch 1, attempt 5 |
|---|---:|---:|
| Generation wall, 32 videos, slowest rank | 1681.0 s | 1871.7 s |
| Aggregate generation throughput | 68.53 videos/hour | 61.55 videos/hour |
| Typical generation VRAM per GPU | ~39–40 GiB | ~32–33 GiB |
| Completed online optimizer updates at snapshot | 0 | 0 |
| Outcome at snapshot | GPU OOM in first replay forward | Live replay/backward |

The batch-1 run is about 11.3% longer per generation collection (10.2% lower
throughput). The memory changes enable progress beyond previous failures; they
are not yet a demonstrated end-to-end speedup or quality improvement.
Attempt 3 failed from host RAM exhaustion; attempt 4 reached reward scoring but
failed on GPU memory during replay. Attempt 5 adds matched microbatch 1,
expandable CUDA allocator segments, file-backed frozen trainer parking and
host allocator trimming. Synthetic full-shape, real-weight checks passed
exact-zero prediction/logprob parity and backward/optimizer step for both
experts on all four ranks. Full online acceptance remains pending.

## Comparison contract

Use attempt 5's archived `resolved_config.yaml` as the exact baseline:
four L40S; Wan2.2 A14B both experts; 480×832×33 frames, 20 denoise steps,
CFG 4.5; 32 videos/update; microbatch 1 on rollout and replay; rank-32 FP32
LoRA, alpha 64; full FSDP, native precision exceptions; no compile;
Kling `visual_quality`; lossless stored trajectories; exact-zero parity.
Keep prompts, seed, model/reward revisions, timestep selection and samples per
update unchanged when measuring the new staging implementation.

Report generation, reward, replay/backward, optimizer, weight sync, checkpoint,
and total update time. Separate cold startup from warm updates. Preserve
finite/nonzero gradients, reward variation, exact parity, all-rank receipts
and recoverable checkpoints. Evaluate model quality separately with the same
held-out prompts and seeds against the base model.

## Interpretation

`gpu_timeseries.csv` has physical VRAM, utilization, GPU power and host-available
RAM. `batch_timings.csv` contains rank-level queue/service timings. GPU energy
and utilization summaries are time-weighted sample estimates over the first
denoise event through generation completion, excluding startup. They do not
measure CPU/system energy. Sampled memory peaks can miss brief spikes.
Log timestamps use America/Los_Angeles; resource timestamps are Unix UTC.
The export converts them before aligning traces. The source patch was captured
at export time; it is not an exact historical code snapshot for attempts 3/4.

Refresh after a completed update or the overnight run:

```bash
cd /home/ubuntu/VRL-night-20260918
PYTHONPATH=.wan_runtime:. /home/ubuntu/VRL/.venv/bin/python \
  tools/overnight/export_staging_baseline.py \
  --out /home/ubuntu/VRL/outputs/wan14b_staging_baseline_20260919
```

## Replay timing observed during the live smoke

[Replay progress snapshot](../../outputs/wan14b_staging_baseline_20260919/live_replay_progress.json)
records completed model-forward events per rank and the median interval between
them. At 19:09 UTC, all four ranks had 12 forward events with approximately
54.9 seconds between completions. This interval includes the preceding backward,
the next forward, and orchestration; it is not a pure forward latency. No full
optimizer update or online parity acceptance had completed at that snapshot.
The run was alive. Refresh with `tools/overnight/replay_progress.py` in the active
worktree; the JSON embeds its observation time and live process check.

## Completed first update — 2026-09-19 14:18 PDT

[Completed-update snapshot](../../outputs/wan14b_staging_baseline_20260919/20260919T213314Z/README.md) supersedes the provisional zero-update observations above.

The first update finished with exact-zero maximum logprob difference and ratio
deviation, gradient norm 0.00037384292227216065, reward mean -0.7302333116531372
and reward standard deviation 0.5752043128013611. Both experts have 320 nonzero
LoRA B tensors. All four rank receipts report success. A final checkpoint was
saved on NVMe and copied to the persistent EBS checkpoint directory.

Launch-to-final-checkpoint time was approximately 178.2 minutes (11:20:23 to
14:18:33 PDT), including cold startup. This is one completed update, not a
warm-update average or proof of model quality improvement.

After checkpoint completion, Ray actor shutdown encountered host-memory
pressure and forced cleanup; sampled minimum host available fell to 14.1 GiB.
Thus the run is not an error-free lifecycle baseline, although training and
checkpoint acceptance passed. The dependent eight-hour continuation launched
at approximately 14:19:38 PDT and was generating on all four GPUs at 14:33 PDT.
