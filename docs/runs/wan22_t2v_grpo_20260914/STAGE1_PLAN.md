# Stage 1 plan (recorded before launch, 2026-09-14 ~14:00 PDT)

- Config: configs/stage1_grpo_4rank.yaml (4 ranks, GPUs 0-3). Commit: see logs/train_stage1_commit.txt.
- Data: 16 train_small prompts; 4 prompts/update (1 per rank) x 8 samples = 32 samples/update.
- Budget: 8 updates (= 2 passes over train_small), checkpoint every 2 updates. Estimated
  ~25-30 min/update at 20 steps (4-rank pilot: ~4.5 min/update for 8 samples at 10 steps),
  i.e. ~4 h. Hard stop: 6 h wall or any of the stop conditions below.
- Eval after stage 1: checkpoints 4 and 8 on train_small (16 prompts x 2 seeds) and val
  (24 x 2), native 20-step sampler, Kling VQ (objective) + MQ/TA/overall (diagnostics),
  prompt-level paired deltas vs the base arm already generated at eval_train_small/eval_val.
- Stop immediately if: non-finite loss/grad, replay-parity gate failure, reward_std -> 0 on
  every group (collapse), Ray memory kill, or a visible quality collapse in reward_artifacts.
- Decision at the end of stage 1: continue (extend to 16-24 updates) only if the paired
  train_small VQ delta at checkpoint-8 is positive with win rate > 0.5 and the val delta is
  not negative; otherwise diagnose (samples, lr, reward, geometry) and change ONE factor.

## Stage 2 (2026-09-14 21:42 PDT): strict resume stage1/checkpoint-8 -> 16 updates, same config
- Attempt 1 died at 21:51 in the first generation phase: Ray memory monitor killed a rollout
  worker at 354.11 GB / 372.73 GB (95.0%). Top users: 4 rollout workers 55.6-61.8 GB (both
  bf16 experts resident on host under sequential offload) + 4 trainer ranks 28.1 GB
  (FSDP CPU shards). Stage 1 had the same peak (min available 21.5 GB, 351 GB used) and
  survived by ~3 GB; the resumed process starts ~4 GB heavier (checkpoint payload) and a
  concurrent pytest from another agent used 0.55 GB.
- Attempt 2: environment-only change, no model/algorithm/data change:
  RAY_memory_usage_threshold 0.95 -> 0.98 (365 GB) and Ray object store 2 GB -> 1 GB.
  Justification: nine measured generation-phase peaks were bounded at 351-354 GB, 18 GB
  under physical RAM. Real fixes (sharing the frozen expert weights between the four
  rollout workers, or dropping the text encoder after encoding) are code changes deferred.
