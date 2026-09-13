# Wan full-size CPU-checkpoint run: external signal interruption

Status: **interrupted before optimizer update; acceptance remains open**.
This attempt is neither a CUDA OOM result nor a successful training update.

## Run and completed work

Clean runtime `382d0825` in `/home/ubuntu/VRL-mgpu-integration`, unchanged
throughout execution. Output root:
`/mnt/nvme/outputs/wan_i2v_full_physics_cpu_default_l40s`.
The supervisor and saved launch configuration are in that directory; launcher
and verifier are under `/mnt/nvme/outputs/wan22_i2v_cache/`.

Original full-size geometry: 480x832, 81 frames, 20 denoise steps, CFG 5,
seed 7, six global samples across three FSDP policy ranks. Dedicated GPU 3
runs the original 0.3 Kling motion / 0.7 VideoCon physics HTTP objective.
The explicit memory mitigation is `actor.gradient_checkpointing=full_cpu`;
both CUDA allocator environment overrides are unset. Shared dependencies
and the original objective, LoRA targets and replay fraction are unchanged.

- All six videos generated. Per-rank generation walls: rank 0 1695.744 s,
  rank 1 1688.247 s, rank 2 1701.172 s.
- Both reward services returned six receipts. Their sample identity sets
  match; both services received byte-identical MP4s for each sample.
- All 12 retained reward artifacts decode as 81 frames, 480x832, **8 fps**.
  The 12 files represent six videos materialized separately by two rewards,
  not 12 generated samples. Sampled frames vary spatially and temporally;
  one middle frame was visually inspected. This is artifact validity, not
  an assessment of physical realism or learning quality. The earlier 16 fps
  standalone reward probe is a different input and is not a parity control.
- Read-only stack samples show real replay/backward progressing beyond the
  original first-forward OOM, through timestep index 13 of the first sample.
  At least indices 0-12 had completed backward. The update requires 19
  replay indices per sample and two samples per rank; it did not complete.
- Sampled policy device usage stabilizes at about 37,542 MiB with 100%
  utilization; dedicated reward usage is 21,114 MiB. These are sampled
  physical usage observations, not allocator peak counters. Host available
  memory remained about 37 GiB in the last checks; sampled memory PSI was zero.

## Interruption evidence

Times below are machine-local PDT on 2026-09-12.

- 21:54:13: all three Ray nodes receive SIGTERM. Rank 0 raylet reports
  `received SIGTERM. Existing local drain request = None`.
- GCS records node death as `EXPECTED_TERMINATION`, message `received SIGTERM`.
- 21:54:28: GCS servers themselves receive SIGTERM and shut down.
- 21:55:29: all three training ranks report inability to reconnect to GCS
  within 60 seconds and terminate.
- Torchrun exits 1. `supervisor_result.json` records return code 1, and the
  supervisor reaps its reward services. The originating exec session is terminal.

Ray evidence remains in the log directories for:

```text
/mnt/nvme/ray/ray/session_2026-09-12_20-25-59_948337_431156
/mnt/nvme/ray/ray/session_2026-09-12_20-26-00_174240_431155
/mnt/nvme/ray/ray/session_2026-09-12_20-26-00_367312_431157
```

The signal sender is not identified by these logs. A kernel-journal query
covering 21:50-21:57 returned no entries, so there is no recorded kernel OOM
evidence in that interval. Do not infer who stopped Ray or equate this with a
model numerical failure. The active agent issued no stop command for this run.
The user has been asked whether another session intentionally cleaned up Ray.

## Remaining gate

`metrics.full_precision.csv` has only its header. No final checkpoint or rank
success verdict exists. Full-update gradient norm, aggregate rollout/replay
parity, optimizer state and learning quality remain unverified. The prepared
verifier must fail on this run's nonzero training exit; no pass receipt exists.

Fresh compute-process inventory is empty, and no owned trainer/reward service
remains. GPUs 0-3 are released. Do not blindly rerun the expensive full update
until the termination/coordination issue is understood. Keep all failed and
interrupted roots, and never restart the disabled SD3 long queue.
