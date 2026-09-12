# SD3.5 continuous: controlled follow-up

Status: queued follow-up, requested by the user on 2026-09-12. Preserve the
running continuous experiment and existing GPU queue. This plan does not
authorize interruption, change the queue script, or claim a hardware slot.

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

1. Archive the current continuous verdict, resolved config, metrics and final
   checkpoint. Label it pre-fix. Coordinate a GPU slot with the queue owner;
   do not launch over the remaining queued work.
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

## Completion evidence

Keep per-arm configs, commands, revision/environment, verdicts, checkpoints,
parity and weight-receipt records, numerical equivalence results, and a matched
timing table. Mark semantics, delivery, throughput and learning separately;
passing one does not close the others. No new 40-epoch run is required merely
to confirm basic SD3.5 + OCR functionality.
