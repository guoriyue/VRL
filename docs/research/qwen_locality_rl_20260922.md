# Locality-constrained editing RL: active experiment

## Objective and completion gate

The previous 20-update EditReward-only run did not improve strict localized
editing. This experiment must improve the requested edit while preserving every
protected part. A checkpoint, a higher reward, or a passing color witness alone
does not complete the goal.

Start from the original model; preserve the previous checkpoint as a control.
Use the original two training instructions, eight candidates per instruction,
and initially 20 GRPO updates. Keep model identity, LoRA rank, learning rate,
sampling steps, replay checks, and memory scheduling from the prior pilot.

Compare base, the previous EditReward-only LoRA, and the new LoRA using identical
original prompts, source hashes, sampling configuration, and initial latent
hashes. Initial evaluation seeds are 303/404, not the 101/202 armchair examples
used in the verifier audit. A promising checkpoint requires additional fixed
training-task seeds 505/606/707/808/909/1001 before claiming consistency.

Full-image review must check the seat, entire backrest, both armrests, magazine
holder, and scene independently. The sparse regions optimized by training are
diagnostics, not independent success evidence. Evaluate the six existing tasks
on the two non-training photos without using their regions or outputs for reward
calibration. Record their whole-image task outcomes and any regressions. Exact
lettering preservation needs separate evidence; do not infer it from color scores.

If the new run merely moves errors or exploits sparse regions, retain the goal
as active and revise the verifier/data or training based on development evidence.
Do not redefine success as a completed optimization run.

## Reward and calibration

For each task, completion is the minimum requested-color membership across
target witnesses. Each protected witness gets its own preservation score based
on added target color relative to the source. Overall preservation is the worst
region, not an average. Locality is the minimum of completion and preservation.

The scalar used for GRPO is:

`locality + calibrated_weight * tanh(EditReward)`

Calibration uses natural outputs already generated for the armchair task.
The successful feedback-prompt images are verifier calibration positives only;
they never replace the original RL instruction and are not RL improvements.
Fit on baseline seeds 0/2. Audit on seeds 1/3 and the four failed base/LoRA images
from the previous 101/202 comparison. Labels come from the recorded assistant
visual reviews; they are not independent human annotations.

The weight is one quarter of the smallest successful-minus-failed locality gap
on the fit split. Since tanh quality spans [-1, 1], even its worst possible
reversal can consume at most half that observed fit margin. This bound applies
to those calibrated pairs, not arbitrary future outputs.

Observed weight: 0.21039109276716772. Fit ranking: 4/4 pairs correct, minimum
combined margin 0.901018. Audit ranking: 12/12 correct, minimum margin 0.933718.
Only training-photo rules enter `manifests/edit_locality/locality_reward.json`.
Task identity, exact instruction, and source-file hash are validated at scoring.

## Implementation boundaries

- Shared color membership moved from the evaluation CLI to
  `vrl/rewards/models/color_locality.py`, so online and offline scoring use the
  same pixel arithmetic. The existing evaluator still exposes its imported
  `membership` name for compatibility.
- The existing EditReward model adapter remains the lazy inference and HTTP
  integration boundary. Optional locality configuration adds separate evidence
  scores and the calibrated combined score; ordinary EditReward is unchanged.
- Prompt/region/color data stays in a named JSON experiment asset. No domain
  vocabulary or prompt table is embedded in workflow constants.
- No changes to family interfaces, replay mathematics, or unrelated sprint
  documents are part of this experiment.

## Execution and evidence

Calibration:

```bash
/tmp/vrl-qwen21-integration-venv/bin/python -m vrl.scripts.eval.calibrate_edit_locality
```

Service and trainer:

```bash
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 /tmp/vrl-edit-reward-env/bin/python \
  -m vrl.rewards.service.server --config vrl/config/reward_service/editreward_locality.yaml
OMP_NUM_THREADS=8 /tmp/vrl-qwen21-rl-env/bin/python -m vrl.scripts.train \
  --config experiment/qwen_image_21/online_grpo_locality
```

The service uses port 8316, a separate model identity, and a 128 MiB request
limit for the 16-candidate collection. Candidate PNGs and JSON records are
paired by artifact hash in `outputs/qwen_image_21_locality_rl/main/scored_candidates`;
each record includes the original instruction, task/policy metadata, all region
scores, quality score, and the exact locality configuration.

At launch preparation another checkout's evaluation (PID 593768) occupied the
GPU. `outputs/qwen_image_21_locality_rl/run_when_idle.sh` waits for that process
to exit and then for six consecutive low-memory checks before starting our
service and training. It does not interrupt other jobs. Queueing is not training
completion. The trainer log will be `outputs/qwen_image_21_locality_rl/train.log`.

Calibration report: `outputs/qwen_image_21_locality_rl/calibration.json`.
Unit/service tests: 82 passed before adding the candidate-audit assertions;
the subsequent focused five tests also passed. Training and quality evaluation
remain pending and must be verified from live processes and produced artifacts.

The separate `outputs/qwen_image_21_locality_rl/evaluate_after_training.sh`
queue waits for the training wrapper to exit, acquires the same GPU job lock,
and requires both a successful training result and the final checkpoint before
generating any comparisons. It evaluates base, the old 20-update checkpoint,
and the new checkpoint on seeds 303/404 at identical settings. Both comparison
galleries initially use `--images-only`, so visual review precedes aggregate
critic scores. These queued commands are not evidence of completed evaluation.
