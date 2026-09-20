# Wan four-L40S parity investigation, 2026-09-19

The original `primary_fill_wan13_clean` run completed an update but **failed**
the user's exact-zero acceptance: global pre-update max error
`0.00230485200881958`. H3 remains blocked. This investigation does not change
that verdict or replace the original-recipe acceptance with a diagnostic.

## Gate configuration error

The recorded `training_debug.jsonl` says `max_abs_diff_limit: 0.01` and
`passed: true`. The external queue correctly applied zero and rejected the run,
but the trainer had already updated parameters. The four-L40S Wan preset now
explicitly sets `trainer.replay_parity.max_abs_logprob_diff: 0.0`, so future
mismatches abort before `optimizer.step`. The algorithm's PPO `clip_ratio` is
unrelated and remains unchanged. No old log-probabilities were rewritten.

Validation: 33 tests passed in `tests/trainers/online/test_diagnostics.py` and
`test_config.py`; added cases prove zero passes while both `2.980232238769531e-7`
and the actual `0.00230485200881958` drift fail. Resolved preset also checked
through the public config loader/parser: limit is 0.0.

## Controlled GPU evidence

`tools/overnight/wan_parity_probe.py` ran four independent GPU arms: eager/native
parameters, eager/actor-cast parameters, compiled/native, compiled/actor-cast.
All use the actual 1.3B checkpoint and a recorded four-sample prompt group at
480x832/81 frames, CFG 4.5, 20-step original scheduler, stochastic step 4.
Native means Diffusers' BF16 model with retained FP32 exceptions; actor-cast
means all parameters normalized to BF16 as the FSDP actor policy does.
LoRA B is initially zero. Same saved inputs and initial weights across arms.
These are forward diagnostics, not FSDP or online acceptance.

Results are under `outputs/overnight_20260918/wan_parity_diagnostic_v2/`.
`rank{0,1,2,3}.json` and `cross_arm_comparison.json` hold exact values;
`tensors*.pt` retain predictions and replay log-probabilities.
The v1 launch failed due to a missing precision stamp in the new probe, before
any model forward; v2 supplies the required role precision. The failed attempt
is retained and is not a model failure.

| Isolated change | Max absolute log-probability difference |
|---|---:|
| Same prediction, same full-precision sampled transition, repeated SDE scoring | **0** (all four arms) |
| Only round stored old log-probability to BF16 | 0.0018795132637023926 |
| Only round sampled action to BF16 | 0.0000661015510559082–0.0000711679458618164 |
| Round action and old log-probability to BF16 | 0.0019445419311523438–0.001950681209564209 |
| Native versus actor-cast parameters, both eager and batch 4 | 0.00018775463104248047 |
| Eager versus compiled, native parameters and batch 4 | 0.00019425153732299805 |
| Eager versus compiled, actor parameters and batch 4 | 0.00019174814224243164 |
| Batch 4 versus batch 1, eager/native | 0.0000002980232238769531 |
| Batch 4 versus batch 1, eager/actor | 0.00000035762786865234375 |
| Batch 4 versus batch 1, compiled/native | 0.00001519918441772461 |
| Batch 4 versus batch 1, compiled/actor | 0.000006794929504394531 |

Storage-only measurements generate a fresh one-step SDE transition using the
actual model prediction and fixed RNG seed, then change only the named stored
quantity. Thus they prove storage alone breaks exact parity, even with an
identical backbone result. They do not recover the original unrounded rollout
values, which the historical BF16 spool no longer contains. Errors in this
table are independent interventions, not additive contributions to the clean
run's maximum. The original full-run maximum is across 96 samples; this probe
uses four samples and cannot claim to account for every sample's error.

## Why this happens

- `DenoiseTrajectoryBuffers` initially holds FP32 latents and FP32 scores.
- `DiffusionBatchExecutorBase.apply_wire_storage_policy` casts latents **and
  log_probs**, followed by another idempotent collector-side conversion.
  The original recipe explicitly chooses `trajectory_storage.dtype: bfloat16`.
- Replay computes Gaussian log-density in FP32 using the rounded observation
  and action, then compares it with the independently rounded old score.
- FSDP `precision_policy: actor` normalizes native FP32 model parameters to
  BF16. The rollout loader retains Diffusers' FP32 exceptions. A nominal BF16
  role label therefore does not prove equal parameter bytes or computation.
- Rollout alone is compiled and processes four samples; replay is eager and
  processes one sample. Both differences independently produce nonzero drift.

The existing recipe comment attributing ~0.002 drift to compilation alone was
incomplete. BF16 score storage already reproduces that scale without changing
model computation. This is separate from rank-zero checkpoint loading: the
previous parent/candidate fixed-input full-FSDP control was bitwise equal.

## Required repair conditions

A zero-tolerance online acceptance needs lossless observation/action/old-score
storage (the existing `dtype: preserve` option), matched parameter precision,
matched forward kernels/compilation, and matched per-forward sample shape.
Merely preserving old scores, disabling compilation, increasing PPO clipping,
or relaxing the gate cannot establish exact parity. Recomputing old scores
with the replay backend would change the test and is not used here.

The original recipe's rollout/storage behavior is preserved in this debugging
change; only its internal gate is tightened. No new full online acceptance has
passed. A separate full-resolution four-rank FSDP diagnostic checks the matched
execution conditions with fresh SDE actions and an actual backward/update;
its result is recorded below when available. H3's prerequisite still refers
to the failed original online recipe, never to either diagnostic.

## Aligned four-rank FSDP result

`tools/overnight/wan_aligned_fsdp_probe.py` completed on all four L40S cards.
It uses the same full 480x832/81-frame Wan geometry and scheduler, one saved
prompt/sample per rank, and generates a fresh SDE action with the unsharded
policy. Both paths use eager execution, batch 1, actor-normalized BF16
parameters, BF16 outer autocast, and lossless FP32 SDE observations/actions/
old log-probabilities. The same model is then prepared as full FSDP with
`precision_policy: actor`, `reshard_after_forward: true`, checkpointing, and
no CPU offload. Replay runs with gradients enabled; it does not overwrite or
recompute the saved old score.

All four ranks measured **prediction max-abs = 0 and log-probability max-abs = 0**,
then completed backward and one AdamW step. Global gradient norm was
`0.00014550861540590145`; peak allocated memory was `15,384,906,752` bytes/card.
The timed forward/FSDP/backward/update section took 29.3–29.5 seconds.
Receipts: `outputs/overnight_20260918/wan_aligned_fsdp_diagnostic/rank*.json`.

This proves exact parity is attainable with the actual model and FSDP when
execution is aligned. It does not validate a 96-sample online GRPO update,
reward orchestration, random timestep coverage, or the original compiled,
batch-4, BF16-storage recipe. No H3 prerequisite was redirected to this probe.
Both completed diagnostics are marked passed in the queue only as diagnostics;
the original Wan acceptance remains failed and H3 remains blocked.

## User reprioritization after this investigation

The user subsequently requested Wan 14B overnight training. A separate
Wan2.2 A14B real online smoke now owns the long-run prerequisite: native FSDP
precision, matching batch size two, no compile, lossless trajectory storage,
and an exact-zero gate. H3 and the original failed Wan1.3B acceptance remain
held; no historical result is relabeled as passing. See
`outputs/wan14b_priority_20260919/README.md` for the active execution contract.
