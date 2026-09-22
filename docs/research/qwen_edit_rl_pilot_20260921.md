# Fixed-instruction localized editing RL pilot

This experiment follows the EditReward baseline, but updates the QwenImage2.1
LoRA parameters with online flow-matching GRPO. Prompt-feedback retries are not
training and are excluded from this experiment.

## Frozen protocol

- Train on the original `armchair_seat_blue` and `sweater_one_sleeve`
  instructions from `manifests/edit_locality/tasks.json`, with their original
  source photographs. No prompt rewriting or synthetic unchanged-image negatives.
- Use four stochastic candidates per instruction and compare EditReward scores
  within each group. The released EditReward mean is the sole optimization
  reward; its uncertainty output is logged, not optimized.
- QwenImage2.1 revision `b3179ad355be050328e483a9dfdd9e60cd62adfa`, LoRA
  rank 16 / alpha 32, learning rate 1e-4, 20 diffusion steps, SDE noise 0.7,
  576 x 384 training images, reference resolution 512. Full activation
  checkpointing; four sampled timesteps per trajectory for replay.
- Initial budget: 20 GRPO updates, eight candidates per update. Save every five
  updates. Check replay log-probability agreement every update at tolerance 1e-5.
- Evaluate the untrained base and final checkpoint with the same original
  instructions, source images, deterministic Euler sampler, and seeds 101/202.
  Require identical hashes of the actual initial latent tensors for every pair;
  seed labels alone are insufficient evidence of identical initial noise.
  Use reference resolution 512 and area-based output geometry. The six existing
  tasks from `two_chairs.jpg` and `dining.jpg` are excluded from training.
- Inspect both reward differences and the actual target/preserved regions.
  A higher optimization reward alone does not establish editing improvement.
  Sparse color witnesses are diagnostic only; the earlier region rules were
  developed with access to these photographs, so they are not an independent
  unseen benchmark. Human preference is not measured by assistant visual review.

## Reproduction

Start the released EditReward service in a separate compatible environment
(Transformers 4.57.0, PEFT 0.17.1), then the trainer:

```bash
OMP_NUM_THREADS=8 /tmp/vrl-edit-reward-env/bin/python -m vrl.rewards.service.server \
  --config vrl/config/reward_service/editreward_qwen25.yaml
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 /tmp/vrl-qwen21-rl-env/bin/python \
  -m vrl.scripts.train --config experiment/qwen_image_21/online_grpo_editreward
```

The HTTP judge uses the same physical GPU. All three role offload switches are
explicitly enabled: generation parks before scoring, the judge unloads before
replay, and the trainer parks before the next generation. An external HTTP
service reserves no Ray GPU bundle. Its reload parking mode avoids requiring
the trainer's vLLM/CuMem dependencies in the upstream reward environment.

The dedicated training environment starts from `uv sync --frozen --group test
--group lint --extra cosmos` and adds the optional vLLM 0.21.0 runtime described
in `pyproject.toml`, without upgrading the locked Torch 2.11.0. Tests run in a
separate lock-synced environment. The service config pins both reward and base
model revisions and identifies the local upstream checkout.

```bash
/tmp/vrl-qwen21-rl-env/bin/python -m vrl.scripts.eval.reference_image_checkpoint_eval \
  --config experiment/qwen_image_21/online_grpo_editreward \
  --label base --out outputs/qwen_image_21_edit_rl/eval/base
/tmp/vrl-qwen21-rl-env/bin/python -m vrl.scripts.eval.reference_image_checkpoint_eval \
  --config experiment/qwen_image_21/online_grpo_editreward \
  --checkpoint outputs/qwen_image_21_edit_rl/main/checkpoint-final \
  --label rl20 --out outputs/qwen_image_21_edit_rl/eval/rl20
```

Run evaluation only after the trainer releases the GPU. Evaluation restores the
training checkpoint with strict model identity checks; base evaluation explicitly
disables the adapter. The frozen manifests preserve instruction/reference pairs;
training archives retain candidate tensors and reward debug logs retain scores.

Score each evaluation directory with `vrl.scripts.eval.edit_reward_baseline`
(`--reward editreward` and `--reward editscore`) in the isolated reward environment.
Copy the existing version-3 `region_checks.json` into each directory and run
`vrl.scripts.eval.edit_color_constraints` for diagnostic color witnesses. These
previously developed rules remain unchanged throughout this RL experiment.
`vrl.scripts.eval.reference_image_checkpoint_report` checks pair identity and
renders the source/base/trained gallery plus machine-readable comparison metrics.
The report separates training instructions, new instructions on training photos,
and heldout photos. A material or whole-item task on a heldout photo is an
evaluation of transfer from the two color-training tasks, not direct training of
that capability.

## Training result

The 20-update run completed successfully on September 22, 2026, with 160 scored
training candidates. Mean online reward over the first five / last five updates
was 0.883447 / 1.006592. These are different stochastic samples, not a paired
estimate of policy improvement. Every update passed the replay gate; the maximum
absolute log-probability difference was 2.38419e-7. Gradient norms ranged from
0.0005254 to 0.0027116.

The first-update audit recorded both a successful optimizer step and changed
trainable-parameter hashes. All 128 zero-initialized LoRA B tensors were nonzero
in the final checkpoint, with joint L2 norm 1.343357. The adapter contains
16,777,216 trainable parameters. The final checkpoint is
`outputs/qwen_image_21_edit_rl/main/checkpoint-final`; its `checkpoint.pt` SHA-256
is `1ae891a6abe2e0124db2f44dc0a1494e3c117d78f69920ba7ca6cd3f4b30ac96`.

Startup fixes before the first optimizer update: explicitly park the shared-GPU
external judge/generator/trainer at phase transitions and raise the local HTTP
request limit from 16 MiB to 64 MiB for eight float32 image tensors. The reward
transport remains lossless. An attempted PNG archive setting was rejected by the
existing tensor/video archive schema and reverted before training. No failed
startup performed an optimizer update. The successful run's training result is
`{"schema_version": 1, "status": "success"}`.

Resource resolution, reward conversion/service, collector lifecycle, reload
parking, and Qwen reference replay tests: 200 passed in the lock-synced test
environment. Ruff checks apply only to changed files.

## Paired evaluation result

**This pilot did not demonstrate better localized editing.** All 20 base/LoRA
pairs have identical initial-latent SHA-256 hashes, original prompts, source
hashes, and sampling settings. Forty images received both released critics
(80 evaluations); eight additional EditScore evaluations used judge seed 43 to
investigate an unstable score. No prompt-feedback retries were included.

| Evaluation group | Pairs | EditReward base -> RL20 | EditScore base -> RL20 |
| --- | ---: | ---: | ---: |
| Training instructions, fresh seeds | 4 | 1.2329 -> 1.2983 | 9.4948 -> 9.4969 |
| New instructions on training photos | 4 | 2.2200 -> 2.2493 | 9.6990 -> 9.7495 |
| Heldout photos | 12 | 1.5619 -> 1.5351 | 9.6148 -> 9.3261 |

The frozen sparse color-witness score (minimum of target-color coverage and
protected-region color preservation) changed from 0.5241 to 0.5048 on the four
training-instruction pairs, and from 0.9728 to 0.9683 on the eight supported
heldout color pairs. It was **not** part of the optimization reward. This is a
diagnostic of specified patches, not full segmentation or a human quality score.

The decisive armchair comparison is a constraint tradeoff, not a successful edit:

- Both base seeds turn the seat blue but incorrectly recolor the separate brown
  magazine holder. The backrest remains grey.
- Both trained seeds keep the magazine holder brown but incorrectly recolor the
  protected backrest. Seed 101 recolors part of it; seed 202 recolors almost all
  of it. Neither satisfies the original instruction.
- EditReward increases from 0.7613 to 0.9458 for seed 101 but decreases from
  0.7380 to 0.4546 for seed 202. The judge is not universally blind to the error;
  it does penalize the more extensive backrest change. It still does not provide
  a reliable joint-success criterion.
- The training-instruction average reward increase is driven by the sleeve
  examples, whose visual localization already looked correct before training.
  Exact printed-letter preservation was not independently certified.

Assistant inspection of the 12 heldout pairs found the same coarse task outcome
before and after training: 11 apparent passes, and one leather edit that changes
the chair silhouette despite the instruction to preserve it. These are not
independent human labels. The set contains only two heldout photographs, so it
does not support a broad generalization claim.

EditScore also has material grading variance here. On the same
`far_chair_leather_s202` images, base/RL20 scores were 10.00/7.3321 at judge seed
42 and 9.7980/10.00 at judge seed 43. The comparison reverses without changing
either image. Consequently, the heldout EditScore mean drop alone is not
evidence of a corresponding perceptual regression.

## What this establishes

The full reference-conditioned RL loop works: original-instruction sampling,
released-model reward, group-relative advantages, LoRA optimization, weight
publication, strict checkpoint restore, and same-noise evaluation. The experiment
does **not** establish that this checkpoint improves the intended task or that
more steps of the same scalar reward would solve locality.

A subsequent reward experiment should separately verify target-part completion
and each protected object/part on natural model errors, then calibrate their
combination with the released quality reward. In particular, returning the
magazine holder to brown must not compensate for changing the backrest blue.
The current sparse witnesses could seed a development verifier, but training
against them would make them unsuitable as an independent success metric;
new photos and separately reviewed regions would be required for evaluation.
This is a proposed next experiment, not a tested improvement from this run.

## Artifacts

- Complete triptychs, prompts, scores, and review notes:
  `outputs/qwen_image_21_edit_rl/comparison/index.html`.
- Compact training-task comparison:
  `outputs/qwen_image_21_edit_rl/training_pairs_preview.jpg`.
- Training curves: `outputs/qwen_image_21_edit_rl/training_metrics.png`.
- Paired metrics: `outputs/qwen_image_21_edit_rl/comparison/comparison.json`.
- Raw training metrics, reward records, and parity/weight audits:
  `outputs/qwen_image_21_edit_rl/main/`.
- Environment snapshots: `training_environment.txt` and `reward_environment.txt`
  under `outputs/qwen_image_21_edit_rl/`.
- Tracked machine-readable evidence: `qwen_edit_rl_pilot_20260921.json` beside
  this document. Training integration commit: `90852130`.
