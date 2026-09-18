# SD3.5 continuous: controlled follow-up

Status: four-arm short hardware acceptance completed on 2026-09-12; GPUs released.
See [results and remaining boundaries](../../research/sd35_global_std_short_acceptance_20260912.md).
The user subsequently
explicitly authorized stopping the old long queue and switching. Queue PID
284402 and continuous driver PID 318010 were stopped, their artifacts were
preserved, and GPU inventory was empty before the corrected launch. The old
dynamic stage must not restart automatically. The short-run claim is now
released; no further long experiment is authorized by this acceptance.

Corrected continuous, four-GPU strict and single-GPU strict completed from
`/home/ubuntu/VRL-mgpu-integration` at
`e11c04bc`, using the unchanged shared Python environment. Output root:
`/mnt/nvme/outputs/sd35_global_std_controlled`. Initial training arms are limited
to two updates each, preserving 512px, 10 denoising steps, 128 samples/update,
generation/replay batch 1, seed 1234, global_std and four-way accumulation.
One post-warmup update is preliminary timing evidence, not a statistically
established speedup. Startup logs confirm update-wide normalization is active.

## Purpose

SD3.5 + OCR already works on the user's single-GPU machine. Do not repeat a
long run merely to establish that fact. Check training semantics and matched
workload throughput before drawing conclusions about learning improvement.
Reward means from different training prompts are not a controlled comparison.

## Available evidence

- The original dedicated 3x1 strict run completed 40 epochs with a success
  verdict. Its pre-fix global_std streaming normalization is not an equivalent
  full-batch baseline. Preserve its artifacts separately from corrected runs.
- Candidate commit `e11c04bc` on `integration/multi-gpu-runtime` computes
  update-wide advantages before clipping/filtering, spools trajectories to
  disk, and corrects the loss denominator after zero-advantage filtering.
- Fixed-rollout CPU tests compare full-batch and streaming advantages,
  gradient norms, Adam state, and the weights passed to the sync hook.
  Regression: 529 passed, 4 skipped; log:
  `outputs/perf/global_std_full_regression.log`.
- These tests do not prove real worker weight receipt, arbitrary multi-rank
  DDP/FSDP equivalence, or hardware speedup. The target SD3.5 topology has one
  trainer and three rollout workers.

## Next steps after continuous

1. Preserve the stopped continuous config, metrics, logs and any existing
   checkpoints. Label it pre-fix and user-stopped, not successfully completed;
   do not invent a final checkpoint or success verdict for the interrupted run.
2. Review and integrate `e11c04bc` through the candidate's dependency chain,
   without changing source or dependencies under live jobs. Record the exact
   runtime revision and environment used by every comparison arm.
3. Recover the user's successful single-GPU resolved config and starting
   checkpoint. Hold prompts, seeds, samples per prompt, total optimizer batch,
   resolution, denoising steps, precision, OCR, optimizer, KL and EMA settings
   fixed. Record config differences explicitly; do not silently substitute a
   preset for the user's baseline.
4. Replay identical saved rollouts through full-batch and four-way streaming
   accumulation with global_std=true. Compare advantages after clipping and
   filtering, gradients, optimizer state, updated adapters and EMA. Retain
   zero-advantage groups and unequal reward scales in the test workload.
5. Verify real multi-GPU delivery: record each rollout worker's received weight
   version and digest at update boundaries, check expected continuous-policy
   lag, and recheck rollout/replay parity under the unchanged acceptance limit.
   The existing measured zero difference is prior evidence, not a substitute
   for testing the corrected path.
6. Run corrected single-GPU, dedicated strict and dedicated continuous arms
   with identical useful work. Report cold-start and steady-state wall time,
   samples/sec, seconds/update, and generation/OCR/replay/backward/sync/spool
   timings. Include disk I/O and actual worker weight versions; do not infer
   speedup from nvidia-smi utilization alone.
7. Only after semantic checks pass, compare learning on the same held-out
   prompts and seeds from matched starting weights and update budgets. Report
   paired OCR results; do not treat unrelated training reward means as proof
   of improvement or regression.

## Fourth arm: shared four-GPU phased execution

User-authorized short follow-up: the same four physical GPUs alternate four
rank-local rollout workers and four synchronous training ranks; CPU OCR runs
per rank. This is not eight GPUs or concurrent rollout/training on each card.
Use native-precision adapter-only FSDP (BF16 frozen base replicated, FP32 LoRA
sharded), since colocated DDP remains capability-gated. Global workload stays
8 groups x 16 samples, with 2 groups and 2 accumulation microsteps per rank.
The four rank-local prompt slices reproduce the single-rank global draw.

- Candidate `11837577`: four-rank fixed-rollout CPU equivalence passed for
  gradient norms, updated weights and Adam state. Unequal surviving group
  counts across ranks fail explicitly; arbitrary uneven filtering is not
  supported by this acceptance path.
- Preflight regression: 560 passed, 2 skipped. First hardware attempt failed
  during FSDP CUDA-to-CPU parking before any optimizer update, not SD3/OCR
  scoring or a numerical parity verdict. Preserve `colocated_fsdp.failed_parking`.
- A targeted parking correction moves local DTensor shards and refreshes FSDP
  padded-storage references. Four-GPU toy parking, live gradients, Adam updates
  and EMA tensor restoration passed. This is not an EMA refresh test.
- The second attempt completed rollout but failed on the first replay forward:
  FSDP leaves ignored frozen weights on the CPU when the model is CPU-staged.
  `75d69be2` explicitly places the replicated transformer before sharding.
  The GPU regression now starts from CPU-loaded weights and passed; 53 focused
  CPU tests passed. Preserve `colocated_fsdp.failed_frozen_placement` separately.
- Retry only the same two-update SD3.5 workload after regression checks. Require
  all four rank verdicts, strict replay parity, 128 global trained samples per
  update, checkpoint step 2 and full update-boundary timings before adding a
  fourth timing result. Include rank-local CPU OCR and torchrun thread defaults
  as topology differences; do not describe this as a pure GPU-kernel benchmark.

Completed retry from `75d69be2`: all four ranks succeeded and torchrun exited 0.
Both updates trained 128 global samples with zero replay mismatch; checkpoint
global_step=2 and finite FP32 model/optimizer state were verified. First update
243.825 s; second boundary-to-boundary 227.247 s including checkpoint-1 save.
Initial speedup is 2.77x over single and 2.11x over dedicated continuous.
`comparison_four_arm.json` and `summarize_four_arm.py` preserve/reproduce the
evidence. This completes short topology acceptance, not learning evaluation,
arbitrary uneven filtering, retained-slot weight activation or EMA refresh.
No further GPU job is queued; the original long queue stays stopped.

## Completion evidence

Keep per-arm configs, commands, revision/environment, verdicts, checkpoints,
parity and weight-receipt records, numerical equivalence results, and a matched
timing table. Mark semantics, delivery, throughput and learning separately;
passing one does not close the others. No new 40-epoch run is required merely
to confirm basic SD3.5 + OCR functionality.
