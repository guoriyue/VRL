# Four-GPU Qwen-Image-2.1 aesthetic LoRA campaign

The user explicitly authorized ongoing four-GPU LoRA aesthetic RL and adapting
the pipeline based on results and online research. This supersedes the original
qualification document's conditional training hold; it does not turn that
inconclusive qualification into a pass. No aesthetic improvement is established
at launch.

## First experiment

- Fresh pinned Qwen-Image-2.1 base; the failed full-parameter run is not the base.
- Four L40S, FSDP, attention LoRA rank 32 / alpha 64, BF16.
- PickScore SFW repository training split: 15,486 prompts. Frozen validation and
  final test were drawn from the source test split, excluding exact train matches.
- SigLIP Aesthetic Predictor V2.5, pinned encoder revision; image-only objective.
- 1024px, 40 steps, Flow-GRPO SDE noise 0.2; GPU-resident text encoder.
- Learning rate 1e-4, adapter-off reference KL 0.04, two PPO epochs, four replay
  timesteps per trajectory, microbatch 1, EMA disabled.
- Streaming accumulation disabled because it releases samples and only permits
  one PPO epoch. Four samples per prompt during acceptance, eight thereafter.
- Following acceptance, generation and training microbatches both increase to
  two. Acceptance generation peaked near 33 GB and training used about 10 GB
  per card; the replay parity guard remains enabled for the batch-shape change.
- Host memory soft limit 260 GiB, hard limit 300 GiB, no automatic crash retry.
  Separate systemd services prevent a worker failure from killing the terminal.

Configuration: `experiment/qwen_image_21/online_grpo_aesthetic_lora_4gpu`.
Artifacts: `/mnt/nvme/outputs/qwen21_aesthetic_lora`.

## Execution and review

`qwen21-aesthetic-lora.service` runs two acceptance rounds. The separate
`qwen21-aesthetic-campaign.service` waits for its successful exit and four fresh
rank success records plus final checkpoint. It then runs native paired evaluation
on all four GPUs, followed by training/evaluation stages ending at rounds 5, 10,
and 20. The first three evaluations use a fixed 16-prompt subset; round 20 uses
all 64 validation prompts. Each GPU owns a stable 16-prompt shard and seed mapping.
Evaluation is 1024px/40 steps using the native deterministic scheduler, with base
and checkpoint using identical seeds. The 120-prompt final test remains reserved.

The campaign aggregates paired score differences with 10,000 bootstrap resamples.
A statistically clear mean decrease below -0.1 in the aesthetic proxy stops that
configuration for diagnosis. This is a regression guard, not an improvement gate.
Inspect blinded images for aesthetic appeal, prompt adherence, detail, artifacts,
and diversity before promoting a checkpoint. Proxy gains alone are insufficient.
After round 20 the bounded campaign awaits review and the active agent goal
continues experiment selection; it does not train indefinitely without assessment.

Generation, replay, and evaluation all use the four GPUs. Loading, checkpointing,
and role handoffs can leave them briefly idle; constant 100% utilization is not
guaranteed, and artificial load is not used.

Status: `campaign_status.json`, per-stage `service.log`, `metrics.csv`,
`training_debug.jsonl`, and `eval_step*/aggregate/summary.json` under the artifact
root. The campaign code is [lora_campaign.py](lora_campaign.py).

## Research informing subsequent changes

- [Official Flow-GRPO](https://github.com/yifan123/flow_grpo): supports LoRA and
  Qwen-Image, discusses partial-step training and proxy over-optimization. CPS
  noise levels are not interchangeable with the current SDE noise parameter.
- [Qwen-Image-2.0-RL report](https://arxiv.org/abs/2606.27608): adjacent-model
  evidence for GRPO and prompt/reward curation, not proof that this Qwen-2.1
  image-only reward experiment will improve human preference.

Potential next changes must follow actual evidence: group reward spread, replay
parity, clipping, KL, native validation, and independent visual comparison.

## First checkpoint evidence (2026-09-26 UTC)

Acceptance checkpoint 1 is complete: 403,092,417-byte training checkpoint plus
adapter export, completed_epoch=1 and global_step=2 (two PPO optimizer updates
per collection round). All 33,554,432 exported adapter parameters are finite;
16,777,216 LoRA-B elements are nonzero, maximum absolute value 0.000198952.
Replay parity passed at 2.384e-7 versus the 1e-5 limit. The first metrics row
reports gradient norm 0.000300, clip fraction 0.0312, and finite reward spread.
These establish real parameter updates, not aesthetic improvement.

The native evaluator dry-run loaded this checkpoint identity and built the paired
base/adapter evaluation plan successfully. The next-stage group-eight, microbatch-
two configuration passed the repository's config builder. GPU generation resumed
after checkpoint 1; second-round completion remains pending at this observation.

## Reduced-interruption continuation (latest user preference)

The user requested focusing on training, fewer image checks and messages, and
autonomous work while offline. `qwen21-aesthetic-long.service` now observes the
existing round-5 stage without interrupting it. Only after four successful rank
results and a final checkpoint will it stop the old supervisor and take over.
It then trains to rounds 20 and 50, retaining checkpoints every five rounds and
evaluating the 64-prompt validation set only at those stage boundaries. Gallery
generation is disabled for these evaluations. The stage plan is editable at
`/mnt/nvme/outputs/qwen21_aesthetic_lora/long_campaign_plan.json`; edits apply at
the next stage boundary. Numerical failures or substantial proxy regression
require diagnosis; there is no blind crash/retry loop. The original final test
remains untouched.

Round-2 native validation (16 prompts): base aesthetic mean 5.87445, LoRA 5.88166,
paired difference +0.00721, 95% interval [-0.02483, +0.03271]. This does not establish
improvement. The completed early comparison was published before the preference
change at https://qwen21-aesthetic-review.rigue.chatgpt.site/lora.html. No further
frequent page updates are scheduled.

The `distributed.training.fsdp.reshard_after_forward=false` throughput trial was
rejected on 2026-09-26 at 07:43 UTC: replay parity passed, but the first resumed
round failed exporting rollout weights with `FSDPPeftModel has no trainable
parameters`. Its first-step trainable digest was also empty. Host peak was 203.1
GiB; this was not an OOM. The failed run remains in `long_to_20` for diagnosis.
No checkpoint from that trial is used. The precise adapter/FSDP interaction has
not yet been isolated.

`qwen21-aesthetic-stable.service` resumes the successfully completed round-5
checkpoint with the previously working `reshard_after_forward=true`, retaining
group size eight and generation/training microbatch two. Outputs are separate
`stable_to_20` and `stable_to_50` directories. Evaluation remains scheduled only
at rounds 20 and 50, with no gallery generation. The supervisor accepts explicit
`resume_run` and `completed_round` plan fields for this recovery and requires all
four successful rank receipts before accepting the resume checkpoint.

## Round 20 native validation (2026-09-26)

All four ranks completed stable_to_20 successfully (40 optimizer updates).
Fixed 64-prompt native evaluation with paired seeds, 40 steps, and 1024-pixel
images gave aesthetic means 5.869667 (base) and 5.984771 (LoRA).
Paired improvement: +0.115104; bootstrap 95% CI [+0.067699, +0.164546];
49/64 pairs scored higher. Source:
/mnt/nvme/outputs/qwen21_aesthetic_lora/eval_step20/aggregate/summary.json

This establishes an optimized-scorer gain on validation prompts, not independent
confirmation of perceived quality or prompt adherence. Brightness fell 0.018345
and edge energy rose 0.003477; check for excessive darkening or sharpening at
the next quality review. The reserved 120-prompt final test remains unused.
No images were inspected or published at this milestone. The supervisor passed
its regression guard and launched stable_to_50 from the round-20 final checkpoint
with the same proven configuration on four GPUs.

## Round 50 training completion (2026-09-26)

`stable_to_50` completed all 50 rounds / 100 optimizer updates. All four rank
success receipts passed the campaign verifier. The final checkpoint metadata
records completed_epoch=50 and global_step=100; checkpoint.pt is 403,092,417 bytes.
CPU inspection of the saved adapter found all 33,554,432 elements finite, with
33,554,209 changed since round 45 (maximum absolute change 0.0012808731).
Acceptance evidence: `/mnt/nvme/outputs/qwen21_aesthetic_lora/checkpoint50_acceptance.json`.

Four native evaluation workers are now evaluating the same 64 validation prompts
and seeds used at round 20. No gallery publication or image review accompanies
this generation stage. The comparison utility at
`/mnt/nvme/outputs/qwen21_aesthetic_lora/compare_validation_checkpoints.py`
requires identical prompt/seed/metadata grids and base-image hashes before
reporting paired checkpoint changes. Run with arguments `20 50` after all four
round-50 score reports are complete. Training completion alone does not establish
visual quality improvement; the independent quality review and final test remain
outstanding.

Round-50 validation completed: aesthetic mean 5.920277, paired base delta
+0.050610 (95% CI [-0.020427, +0.117687]). Against round 20, the paired delta
was -0.064494 (95% CI [-0.134126, -0.002927]); base image hashes matched on all
64 cases. Brightness declined another 0.029644 relative to round 20, while edge
energy declined 0.009708. Do not promote round 50 over round 20.

A controlled continuation branches from round 20 with learning rate 2.5e-5
(previously 1e-4), retaining other settings and optimizer moments. Because resume
restores optimizer param-group learning rates, an explicit derived checkpoint
`resume20_lr25e6` changes only those LR fields; recursive tensor/array/state
comparison against the original confirmed all other state identical. Original
checkpoints remain unchanged. The first supervisor launch failed before training
because the verification script encountered a NumPy comparison error and the old
plan remained in place; this was corrected and verified before restarting the
failed unit. `qwen21-aesthetic-conservative.service` trains `conservative_to_40`
on four GPUs, then evaluates the same validation set. This LR choice is an
experimental response to observed drift, not a proven fix. Memory limits remain
260 GiB high / 300 GiB maximum, no swap and no automatic retries.

Relevant primary reference reviewed: https://github.com/yifan123/flow_grpo
(including its discussion of over-optimization and GRPO-Guard). No new algorithm
or clipping change was adopted from that reference in this controlled branch.

## Independent proxy cross-evaluation of rounds 20 and 50

CPU-only PickScore evaluation completed successfully on all 192 existing native
images (base, round 20, round 50; 64 paired prompts). Model revision
`a4e4367c6dfa7288a00c550414478f865b875800`; processor revision
`1c2b8495b28150b8a4922ee1c8edee224c284c0c`. Scoring uses normalized CLIP
features, model logit scale, and /26, in float32 with four CPU threads. Image
hashes were verified before scoring. No training GPU was allocated. The service
had a separate 16 GiB memory maximum and completed successfully.

Paired PickScore differences:
- Round 20 vs base: +0.000782, 95% CI [-0.003254, +0.004275], wins 36/64.
- Round 50 vs base: -0.009051, 95% CI [-0.013895, -0.004749], wins 20/64.
- Round 50 vs round 20: -0.009833, 95% CI [-0.013919, -0.006231].

Reports: `/mnt/nvme/outputs/qwen21_aesthetic_lora/cross_eval_pickscore_20_50/`.
This supports rejecting round 50 as the preferred candidate. Round 20 has an
optimized aesthetic-score gain but no demonstrated PickScore gain. PickScore is
independent of the optimized SigLIP aesthetic head, yet is another learned proxy,
not human judgment; it shares Pick-a-Pic lineage with the prompt source. No
claim of broad aesthetic improvement follows. The lower-LR branch remains an
experiment; its next validation should include both scorers. Final-test prompts
remain unused.

The reusable `cross_eval_pickscore.py` accepts `--steps`, `--output`, and `--reuse`.
It reuses scores only for matching image hashes and prompts with the same pinned
scorer provenance, and verifies each input image hash. Recomputing both round
20/50 paired summaries from cached scores reproduced the original JSON exactly;
Ruff and compilation passed. `qwen21-conservative-cross-eval.service` observes
only the current conservative supervisor PID 387625, without restarting or
stopping training. After successful training and native eval40, it verifies the
20/40 base-image grid and computes the new PickScore report on CPU. A changed
supervisor PID, failed campaign, or six-hour observation timeout fails this
observer and leaves training untouched. CPU scoring is capped at four cores and
16 GiB RAM. Expected output: `cross_eval_pickscore_20_40/` under the campaign root.

### Conservative round 40 and reserved-test candidate selection

The quarter-learning-rate continuation completed round 40 / global step 80.
`conservative_checkpoint40_acceptance.json` verifies finite 33,554,432 adapter
parameters, changes since round 35, checkpoint byte count, and restored LR 2.5e-5.
Native validation used all 64 paired prompts; base image hashes and prompt/seed
metadata exactly match round 20. Aesthetic delta versus base is +0.0696805
(95% bootstrap CI +0.0177281 to +0.122404), but versus round 20 is -0.0454231
(CI -0.0748447 to -0.0147226). Independent CPU PickScore versus base is
+0.00045935 (CI -0.00360273 to +0.00380902); versus round 20 it is -0.00032267
(CI -0.00161521 to +0.00097073). Neither PickScore difference is significant.
Thus the smaller LR avoids the large independent-score regression observed at
original round 50, but does not establish further quality improvement.

Round 20 was selected as the candidate before inspecting the reserved 120-prompt
test. The four-GPU native evaluation is running under
`qwen21-aesthetic-test120.service`, with output and frozen selection provenance in
`/mnt/nvme/outputs/qwen21_aesthetic_lora/test120_lora20/`. It uses 30 prompts per
GPU, 1024 square images, 40 native steps, one paired seed per prompt, and the
existing 300 GiB host-memory cap. No result from this test was used to select
round 20. Overall aesthetic improvement remains unproven pending this test and
independent quality assessment. If subsequent tuning uses these test findings,
this split must be described as exposed rather than a fresh final test.

### Reserved 120-prompt result and broader-batch experiment

Round 20 test aesthetic delta: +0.0348152, 95% CI [-0.00411166,
+0.0748061], 67/120 score wins. Independent PickScore delta: +0.00196574,
CI [-0.000897691, +0.00491559], 65/120 score wins. Neither establishes
improvement. Both evaluations completed successfully. The copied CPU script's
terminal message incorrectly says 64 prompts; the actual summary and paired
count were verified as 120. Brightness decreases by 0.0183841 on average;
edge energy increases by 0.00316048. These are diagnostics, not preference proof.

A bounded new experiment starts from base under
`qwen21-aesthetic-diverse-batch.service`. Root:
`/mnt/nvme/outputs/qwen21_aesthetic_lora/diverse_batch_trial`.
It changes prompts per rank from 1 to 4 (16 global), retaining G=8,
generation/train microbatch 2, PPO epochs 2, LR 1e-4, KL .04 and the
300 GiB memory cap. Five rounds precede automatic native validation on the
original 64 prompts. Hypothesis: broader prompt coverage reduces noisy updates;
this is not an established causal explanation or guaranteed improvement.
The old reserved test is now exposed and must not be presented as a fresh
final test for subsequent tuned candidates.

### Broader-batch round 2 acceptance and queued diagnostics (2026-09-26)

Round 2 completed at global step 4. All 33,554,432 exported LoRA parameters
are finite; 33,541,966 elements differ from round 1, maximum absolute change
0.0001944353. The checkpoint payload matches its declared 403,092,417 bytes.
Evidence: `diverse_batch_trial/checkpoint2_acceptance.json` under the output root.
This verifies parameter updates, not aesthetic improvement. Training continues
toward round 5 with the same supervisor and four ranks.

`qwen21-diverse-pickscore.service` waits for this exact supervisor to complete
successfully, then scores the new 64-prompt native evaluation on CPU. It reuses
base scores only when image SHA256 and prompt match, with scorer revisions
inherited from the original independent evaluation. CPU memory is capped at
16 GiB. The scorer now accepts `--root` so the new experiment does not overwrite
old results.

The user's requested portrait diagnostic is queued separately under
`qwen21-portrait-styles.service`: two adult subjects, eight styles each, all
16 prompts retained. Arms are base, original round 20, and broader-batch round 5;
paired prompt/seed, 1024 square, 40 native steps. This is exploratory, not a fresh
held-out test. Artifacts live in `portrait_styles/` under the output root.
`portrait_styles/build_site.py` validates all 48 original images and paired
metadata before building the local gallery; syntax checks passed, but the
builder has not run because generation is pending. Deployment is still required
once actual outputs exist. The existing 120-prompt results are published at
https://qwen21-aesthetic-review.rigue.chatgpt.site/test120.html .

### Broader-batch completed; portrait gallery; window trial (2026-09-26)

Broader-batch round 5 completed successfully on all four ranks. Native 64-prompt
validation aesthetic delta versus base is -0.0378418, 95% CI
[-0.0736653, +0.00265169]. Independent PickScore delta is -0.00265430,
CI [-0.00465440, -0.000915691], 25/64 wins. This experiment did not establish
improvement and is not being extended. Results reside in
`diverse_batch_trial/eval_step5/aggregate` and `diverse_batch_trial/pickscore`.

All 48 portrait images completed and were published at
https://qwen21-aesthetic-review.rigue.chatgpt.site/portrait_styles.html .
The 16 paired prompts span eight styles and two adult subjects. The broader-batch
round-5 aesthetic delta is -0.0279726, CI [-0.0675377, +0.0216793], 4/16 wins.
These are exploratory diagnostics. Image hashes and matching prompt/seed metadata
were verified before publishing; no examples were selected by outcome.

`qwen21-aesthetic-window2.service` (supervisor PID 478169) now runs a bounded
five-round experiment from base under `window2_trial`. It retains the broader
batch configuration but restricts stochastic transitions to a two-step window
within [4,36], with replay timestep selection `sde_window`. The hypothesis is
reduced mismatch between training and native generation; no quality gain is
claimed. This is not an exact reproduction of Flow-GRPO-Fast. Configuration and
provenance are in `window2_trial/experiment.json`. The original 64-prompt native
validation runs automatically afterward; independent CPU PickScore is queued
against the exact supervisor. First actual replay parity and checkpoint acceptance
remain pending. Four GPUs were verified at 100% utilization with about 34 GiB
each; host service memory was about 31 GiB, within the 300 GiB cap.

Window trial round 1 completed at global step 2. All 33,554,432 exported
LoRA parameters are finite; checkpoint payload matches 403,092,417 declared
bytes. First replay parity error is 2.384185791015625e-7, below 1e-5.
Evidence: `window2_trial/checkpoint1_acceptance.json`. This establishes
numerical acceptance only, not aesthetic improvement.

Window trial round 2 passed checkpoint acceptance at global step 4: all 33,554,432 parameters finite; 33,547,241 elements changed since round 1, maximum absolute change 0.0001984413248. Evidence: `window2_trial/checkpoint2_acceptance.json`. This is update verification, not a quality result.

Window trial round 3 passed checkpoint acceptance at global step 6: all 33,554,432 parameters finite; 33,549,952 elements changed since round 2, maximum absolute change 0.0001930497237. Evidence: `window2_trial/checkpoint3_acceptance.json`. Quality evaluation remains pending.

### Progress reporting clarification (2026-09-26 15:31 PDT)

Checkpoint metadata confirms these are separate experiment paths, not one run
that has only reached round 3 overnight:

| Path | Initialization / lineage | Latest completed round | Global step |
| --- | --- | ---: | ---: |
| stable_to_20 | Original campaign | 20 | 40 |
| stable_to_50 | Continues original round 20 | 50 | 100 |
| conservative_to_40 | Branches from original round 20 | 40 | 80 |
| diverse_batch_trial/train_to_5 | Restarts from base, broader prompt batch | 5 | 10 |
| window2_trial/train_to_5 | Restarts from base, two-step stochastic window | 3 | 6 |

Do not sum terminal round labels: resumed experiments share earlier training.
The current window experiment's supervisor started at 14:18:49 PDT; checkpoints
1, 2, 3 were saved at 14:44:02, 15:07:04, 15:30:06 PDT. Recent rounds take
about 23 minutes. At 15:31, approximately 45 minutes of training remained,
followed by native validation and CPU PickScore. This estimates experiment
completion, not time to achieve aesthetic improvement. No reliable quality
improvement has yet been established. Status updates must name the experiment
and distinguish training completion from quality evidence.

Window trial round 4 passed checkpoint acceptance at global step 8: all 33,554,432 parameters finite; 33,549,967 elements changed since round 3, maximum absolute change 0.0001935864566. Evidence: `window2_trial/checkpoint4_acceptance.json`. Quality evaluation remains pending.

Window trial completed all five rounds successfully on all four ranks at global
step 10. Final and round-5 adapters are tensor-identical; all 33,554,432 parameters
are finite. Since round 4, 33,551,312 elements changed (maximum absolute change
0.000193610555). Evidence: `window2_trial/checkpoint5_acceptance.json` and four
`training_run_result.rank-*.json` files. Round-5 training reward was 5.91644144;
this is not a quality conclusion. Automatic 64-prompt native evaluation started
at 16:17:04 PDT with four processes (509005–509008). Independent PickScore remains
queued. No extension or new training experiment is selected before these results.

### Window2 round-5 results and continuation decision (2026-09-26)

All training and native evaluation processes completed successfully. On the same
64 validation prompts, aesthetic delta versus base is +0.01788394, 95% CI
[-0.02286778, +0.05924987], 30/64 wins: no established aesthetic improvement.
Against original round 20, aesthetic delta is -0.09721963, CI
[-0.14873262, -0.04084053]. Independent CPU PickScore completed: versus base,
+0.003469263, CI [+0.000976987, +0.005783667], 41/64 wins. This is an exploratory
positive preference-proxy signal on repeatedly used validation data; the interval
is not adjusted for experiment selection. It does not prove visible aesthetic
improvement or fresh held-out generalization. Evidence is under
`window2_trial/eval_step5/{aggregate,comparison_to_20}` and
`window2_trial/pickscore/vs_base/summary.json`.

This mixed result warrants a bounded continuation rather than another reset.
`qwen21-aesthetic-window2-to15.service` resumes
`window2_trial/train_to_5/checkpoint-final` into `window2_trial/train_to_15`,
retaining optimizer/RNG through `trainer.resume_from`. The only configuration
changes are resume source, output directory, and total epochs 5 to 15. Training
continues for ten additional rounds with the same four GPUs, LR, reward, batch,
and stochastic window. Checkpoints save every round; native validation and CPU
PickScore run only after round 15, comparing against base and window2 round 5.
There is no automatic extension past round 15. The source checkpoint digest and
predeclared decision are recorded in `train_to_15/continuation_protocol.json`.
Service limits: MemoryHigh 260 GiB, MemoryMax 300 GiB, no swap, whole-cgroup kill
on OOM. Initial supervisor PID 512337; invocation ID
`bce13047ea3b4a43aeb590ad444b64f7`. Startup/resume acceptance must be checked from
live logs; service launch itself does not prove restored training or quality.

Per-round speed is not total campaign time saved: the prior from-base window
trial added experiment cost. Do not claim faster time to a useful model without
verified quality at a measured cumulative GPU-hour budget. The previously
exposed 120 prompts must not be relabeled as a new final holdout.

### Fresh holdout frozen during window2 continuation

Before round-15 results, `freeze_fresh_holdout.py` selected 256 prompts from the
repository's 1,024-prompt test source, excluding normalized matches against the
entire training manifest and string leaves in available aesthetic campaign,
qualification, full-parameter, and research JSONL artifacts. There were 835
eligible prompts. Selection uses fixed salted SHA256 ordering, independent of
scores. Normalization is NFKC/casefold/whitespace, not semantic deduplication.
The candidate is fixed to window2 round 15 before seeing its results. Manifest:
`fresh_test256_window15/prompts.jsonl`, SHA256
`45aaddbcc87feae1bb55080bcc62cda7e5744b8704ce11346a50431ee0d50d93`.
Protocol and exclusion evidence are in `fresh_test256_window15/protocol.json`.
No images were generated and no evaluation was launched. Recheck non-overlap
before use; do not silently replace prompts or select another candidate using
these outcomes. Native 1024px/40-step paired base/candidate evaluation and pinned
independent PickScore are planned; visible improvement still requires blinded
review. This preserves a genuinely separate test from the exposed 120 prompts
and repeatedly used 64-prompt validation set, within the audited artifacts.

Independent scorer now supports `--prompts-per-shard` (default 16 preserves
ongoing 64-prompt validation; use 64 for the frozen 256-prompt test). It verifies
that every arm contains each global prompt index exactly once before scoring.
Syntax validation passed, the actual completed 64-prompt grid passed, and
synthetic 120/256 grids accepted complete data and rejected duplicate indices.
No new score computation or image generation was launched by this change.

Fresh test automation is now queued as `qwen21-fresh256-window15.service`.
It waits for exact continuation PID 512337 / invocation
`bce13047ea3b4a43aeb590ad444b64f7`, terminal success, and the continuation's
`awaiting_quality_review` status. No GPU work overlaps the active training or
64-prompt validation. Before starting, it rechecks the frozen manifest hash,
training source hash, exact shard concatenation, and non-overlap against existing
JSONL evidence; initial preflight passed across 136 files. It pins the actual
round-15 checkpoint digest and requires all four training ranks to have succeeded.
Then it generates all 512 base/candidate native images, aggregates aesthetic
metrics, and runs CPU PickScore with 64 prompts per shard. Existing outputs or
contamination cause failure instead of silent restart/reselection. The queue
observer timeout leaves training untouched. Scripts and logs reside under
`fresh_test256_window15`; syntax/preflight validation passed, actual image
execution remains pending. This test has no reward-dependent candidate gate.

Window2 continuation round 6 completed at global step 12 (checkpoint timestamp
16:58 PDT). Service log tails did not reflect this completion; future polling
must check checkpoint directories and metrics, not just service.log. The first
resumed replay gate passed at global step 10 with max error
2.384185791015625e-7. Exported round-6 adapter tensors are finite and changed from
round 5; exact counts are in `window2_trial/checkpoint6_acceptance.json`.
Training continues on the same supervisor; no restart occurred. This verifies
resume/update integrity, not aesthetic improvement.

Window2 round 7 completed at global step 14. Checkpoint payload size matches
metadata; all 33,554,432 exported LoRA parameters are finite and changed from
round 6. Evidence: `window2_trial/checkpoint7_acceptance.json`. Same supervisor
continues toward round 15; no new quality evaluation has run.

Window2 round 8 completed at global step 16. Checkpoint payload size matches
metadata; exported LoRA tensors are finite and changed since round 7. Evidence:
`window2_trial/checkpoint8_acceptance.json`. The same supervisor continues to
round 15. No quality improvement is inferred from successful checkpoint updates.

Window2 round 9 completed at global step 18. Exported adapter tensors are finite
and changed since round 8; checkpoint payload size matches metadata. Evidence:
`window2_trial/checkpoint9_acceptance.json`. Same continuation service remains
active. Quality evaluation remains scheduled after round 15.

Fresh256 evaluation integrity guard (2026-09-26): the queued evaluator now checks
both arms' exact per-shard prompt/sample grid, rejects duplicate/missing cells,
compares each prompt and seed against the frozen shard, and verifies PNG SHA256
and 1024×1024 resolution before computing aesthetic aggregates. Verified the
actual guard against the completed window2 step5 GPU0 shard (32 images); missing,
duplicate, and wrong-seed perturbations were rejected. The queued process imports
this evaluator only after training finishes, so no training or queue restart was
needed. This verifies evaluation integrity, not aesthetic improvement.

Window2 continuation checkpoint10 verified: completed_epoch=10/global_step=20,
checkpoint payload 403092417 bytes matches metadata. All 33554432 exported
adapter elements finite; 33551283 changed versus checkpoint9, max absolute
change 0.0001947332639247179. Evidence: window2_trial/checkpoint10_acceptance.json.
This establishes a saved numerical update only; native quality evaluation remains
scheduled at round15. The existing continuation service remains active.

Window2 continuation checkpoint11 verified: completed_epoch=11/global_step=22,
checkpoint payload size matches metadata. All 33554432 exported adapter elements
finite; 33551914 changed versus checkpoint10, max absolute change
0.00020046322606503963. Evidence: window2_trial/checkpoint11_acceptance.json.
Numerical acceptance does not establish quality improvement; round15 evaluation
remains pending. Existing training service remains active.

Window2 continuation checkpoint12 verified: completed_epoch=12/global_step=24,
checkpoint payload size matches metadata. All 33554432 exported adapter elements
finite; 33552104 changed versus checkpoint11, max absolute change
0.0002018422819674015. Training reward_mean=5.855297088623047 (round11:
5.842645168304443); this single-batch increase does not establish a learning trend.
Evidence: window2_trial/checkpoint12_acceptance.json. Round15 native evaluation
remains pending; continuation service remains active.

Window2 continuation checkpoint13 verified: completed_epoch=13/global_step=26,
checkpoint payload size matches metadata. All 33554432 exported adapter elements
finite; 33552046 changed versus checkpoint12, max absolute change
0.00020433025201782584. Training reward_mean=6.0096235275268555 (round12:
5.855297088623047). This batch increase is not fixed-prompt quality evidence.
Evidence: window2_trial/checkpoint13_acceptance.json. Existing service remains
active, with native quality evaluation scheduled after round15.

User authorized extending the same lineage to paper-scale update counts; first
budget is 240 optimizer updates = 120 rounds (2 updates/round). Queued
qwen21-aesthetic-window2-to240.service, invocation
e53e9ba8b3fd4f8bbb4b104714a2dd09, waits for the exact existing fresh256 evaluation
invocation to succeed before resuming full round15 checkpoint. Milestones 60
and 120 use existing 64-prompt native validation plus independent CPU PickScore.
Only total_epochs, output_dir and resume_from change; optimizer/RNG state is
restored, with no base reset. Existing strong aesthetic regression guard remains.
MemoryHigh260G/Max300G/Swap0, no automatic restart. Fresh256 remains a round15
held-out result and cannot be represented as a fresh test of later checkpoints.
Script: docs/research/qwen21_aesthetic/continue_window2_to240.py. Dry-run override
comparison passed, and actual checkpoint13 passed the new checkpoint verifier.
The queued continuation has not yet run training; monitor runtime and final
quality separately. Estimate at 23.24 min/round: ~42h additional from round13,
excluding evaluation overhead. Matching optimizer counts is not matching sample
budget or reproducing the paper's reward/model configuration.

### Round 14 checkpoint acceptance (2026-09-26 PDT)

Verified `window2_trial/train_to_15/checkpoint-14`: completed epoch 14,
optimizer global step 28, payload size matches metadata (403092417 bytes).
All 33,554,432 LoRA elements are finite; 33,551,974 differ from round 13,
maximum absolute change 0.00020227197092026472. Global training reward mean
is 5.824727535247803 (round 13: 6.0096235275268555). Different prompts per
round prevent interpreting this change alone as a quality regression.
Evidence: `window2_trial/checkpoint14_acceptance.json`. Training remains live;
no new held-out result is available at this checkpoint.

### Round 15 checkpoint acceptance (2026-09-26 PDT)

Both checkpoint-15 and checkpoint-final metadata report completed epoch 15,
global optimizer step 30, with payload sizes matching metadata. Their adapter
files have identical SHA256. All 33,554,432 adapter elements are finite;
33,551,854 differ from round 14, maximum absolute change
0.00020768959075212479. Round 15 global training reward mean is
6.214529037475586. This training reward increase does not establish held-out
improvement. Evidence: `window2_trial/checkpoint15_acceptance.json`.

### Window2 round 15 native validation (64 prompts)

All 128 images completed. Independently verified exact base/lora15 grid,
prompt strings, seeds, PNG SHA256, and 1024-square resolution; evidence is
`window2_trial/eval_step15/pair_integrity.json`.
Aesthetic base 5.86966697126627 -> 5.907711610198021; paired delta
+0.03804463893175125, 95% bootstrap CI [-0.0367024689912796,
0.13016370683908463], wins 25/64, median delta -0.03558349609375.
This is not a statistically clear improvement. Brightness increases by
0.020942189963534474 (CI [0.01813245750963688, 0.023942910600453615]),
62/64 brighter; edge-energy change -0.00052210598369129 has CI spanning zero.
Independent CPU PickScore has started; fresh256 remains queued. Do not
infer aesthetic improvement from round 15's high training reward.

### Window2 round 15 independent PickScore (64 validation prompts)

CPU scorer completed 192 samples (base, round5, round15). Round15 score mean
0.8459609979763627, paired improvement over base +0.0040581114590168,
95% CI [0.000628315843641758, 0.007609205320477486], 41/64 wins.
Versus round5: +0.0005888482555747032, CI [-0.002868923358619213,
0.004077324643731117], 36/64 wins. Thus there is a small positive validation
signal against base, but no clear additional gain from round5 to round15.
Validation has been repeatedly inspected; this is not confirmation of visible
beauty improvement. Results: `window2_trial/pickscore_5_15/{vs_base,vs_lora5}`.
The to15 service exited successfully; the already queued fresh256 job is next.

### Fresh256 round15 native aesthetic result

All 512 images passed independent complete-grid, prompt, seed, SHA256 and
1024-square resolution checks (`fresh_test256_window15/eval_step15/pair_integrity.json`).
The preselected round15 checkpoint has paired aesthetic delta
+0.00672103650867939, 95% CI [-0.021488631144165993, 0.03480532206594944],
126/256 wins (49.21875%), median -0.005003690719604492. This does not
confirm an aesthetic improvement. Independent CPU PickScore is running.
The authorized extension to 240 optimizer updates remains queued after this
job; no new training hyperparameters or checkpoint selection were inferred
from the held-out results. This set is now exposed for future decisions.

### Fresh256 PickScore complete and extension handoff repair

Independent PickScore round15 minus base: +0.004108649445697665,
95% CI [0.002512310864403844, 0.005759068299084902], 151/256 wins.
Mean round15 score 0.8368720612488687. A small preference-proxy gain
replicates, while the aesthetic reward gain does not. No visible improvement
claim follows from these scores alone.

The fresh256 unit exited successfully. Extension startup was blocked before
training by an assertion requiring InvocationID even after systemd cleared it
on successful inactive transition. Fixed the terminal path to require inactive,
success, PID0 and an empty or matching ID; live tasks still require exact ID.
Added completed-checkpoint identity and both 256-pair report checks. Real
terminal preflight passed; restarted only the failed extension supervisor.
No training process or checkpoint was reset. Fresh256 site builder now has
complete inputs and generated 16 fixed-index original pairs locally.

### Fresh256 website published; extension resumed

Published owner-private `/fresh256.html` with full aesthetic/PickScore statistics,
16 fixed-index original pairs, and downloadable integrity evidence. Source commit
`5c92695929e702d9b1e135c6d8a44821676a1872`; deployment
`appgdep_6ab89eb9c03c8191b64109516b7e67e8` succeeded at
2026-09-27T04:42:52.734113+00:00. All 32 image links and JavaScript syntax
were checked; copied image hashes match the scored originals.

Extension supervisor PID589025, invocation9cca8fa1c11a4077b1607e5ef865c65e:
all four rank logs confirm resume from train_to_15/checkpoint-final with
start_epoch=15. Four GPUs observed at 100% during new rollout startup.
This is continuation, not a base reset. Target remains rounds60/120.

### User qualitative feedback on fresh256 gallery

User observed better fingers/legs in the Simpson and Moto-Moto examples.
Located gallery indices 0 (Homer Simpson in a psychedelic trip meditation,
seed20261001) and 213 (moto-moto from the movie madagascar, furry,
hippopotamus, seed20261214). Preserved in
`fresh_test256_window15/user_qualitative_observations.json` for later fixed
comparisons. This is user-reported, post-hoc qualitative evidence; it does
not establish a general anatomy improvement or causal reward mechanism.
No training membership, reward or checkpoint schedule was changed.

### Window2 continuation: rounds 16–20 verified

Continued from round15 with the same configuration and optimizer state.
Each checkpoint passed metadata/step, payload-size and finite-adapter checks;
all five adapters changed relative to the preceding checkpoint.

| Round | Optimizer updates | Mean training reward |
| --- | --- | --- |
| 16 | 32 | 5.970694 |
| 17 | 34 | 6.077457 |
| 18 | 36 | 5.889493 |
| 19 | 38 | 6.039351 |
| 20 | 40 | 5.802374 |

These batches contain different prompts; the rewards do not establish a
sustained quality improvement. No new native evaluation has been run for
these checkpoints. Planned validation milestones remain rounds60 and120.
Per-checkpoint evidence: `window2_trial/checkpoint{16..20}_acceptance.json`.

### Window2 continuation: rounds 21–22 verified

Round21 completed 42 optimizer updates with mean training reward 5.730157;
round22 completed 44 updates with reward 6.015204. Both checkpoints passed
metadata, payload-size and finite-adapter checks. Each adapter changed
relative to its immediate predecessor. Evidence is saved in
`window2_trial/checkpoint21_acceptance.json` and
`window2_trial/checkpoint22_acceptance.json`, including payload SHA256.
The round22 reward rebound is a different-prompt training measurement,
not evidence of improved native image quality. No additional native
evaluation was run; the next planned milestone remains round60.

### Window2 continuation: rounds 23–25 verified

| Round | Optimizer updates | Mean training reward |
| --- | --- | --- |
| 23 | 46 | 5.881426 |
| 24 | 48 | 5.982927 |
| 25 | 50 | 5.904591 |

All three checkpoints passed completed-round/update metadata, payload-size,
and finite-adapter checks. Each adapter changed relative to its preceding
checkpoint. Payload hashes and parameter-change evidence are recorded in
`window2_trial/checkpoint{23..25}_acceptance.json`. No native quality
evaluation was added for these rounds; the next milestone remains round60.
The same supervisor continued into round26 sampling on all four GPUs.

### User-requested round47 evaluation (2026-09-27)

Interrupted the uncheckpointed round48 rollout at the user's request to evaluate
immediately. Preserved checkpoint47 (94 optimizer updates), including optimizer
and RNG state. The previous supervisor is inactive. New supervisor:
`qwen21-aesthetic-eval47-resume.service`, script `evaluate47_resume.py`.
It evaluates round47, resumes to60, evaluates60, then resumes to120 and evaluates120.
Continuation directories are `window2_trial/train_to_{60,120}_after_eval47`.

Native evaluation: existing64 validation prompts, paired seeds, 1024 square,
40 steps. All128 images passed exact-grid/prompt/seed/hash/resolution checks;
`eval_step47/pair_integrity_verified.json` records explicit shard evidence.
All64 regenerated base images match round15 hashes exactly.

| Metric | Round47 minus base (95% paired CI) | Round47 minus round15 (95% paired CI) |
| --- | --- | --- |
| Aesthetic | +0.040190 [-0.026942, +0.123674] | +0.002146 [-0.047295, +0.051184] |
| PickScore | +0.002489 [-0.000363, +0.005339] | -0.001569 [-0.004486, +0.001218] |

Neither metric establishes additional improvement over round15. This is repeated
validation, not a new unseen test. Aesthetic win rate versus base is33/64;
PickScore win rate versus base is40/64 and versus round15 is26/64.
Reports: `eval_step47/aggregate`, `aesthetic_15_47`, `pickscore_15_47`.
Round47 absolute aesthetic mean is5.909857. Brightness delta versus base is
+0.001472 (95%CI [-0.006027,+0.007408]), without a clear brightness shift.
All four ranks logged successful resume from checkpoint47 with start_epoch47
at10:25:23 local. No base reset or training hyperparameter change.

### Round43 selection check and four-arm website (2026-09-27)

User requested round43 (highest observed training reward6.250262 through47)
and a webpage. Interrupted unsaved resumed round48; evaluated immutable43 and
resumed from newer47, not from43. Supervisor now
`qwen21-aesthetic-eval43-resume.service`; script `evaluate43_resume.py`;
continuations `train_to_{60,120}_after_eval43`. Same60/120 milestones and config.
Round43 native64 aesthetic mean5.933551, versus base+0.063884
95%CI[-0.001450,+0.149163],37/64 higher. Versus15+0.025840
CI[-0.017293,+0.069068]; versus47+0.023694 CI[-0.002907,+0.052240].
PickScore versus15-0.001282 CI[-0.004281,+0.001830]; versus47+0.000287
CI[-0.001382,+0.001964]. No significant pairwise advantage established.

`build_checkpoint_review.py` builds `/checkpoints.html` with all64 prompt groups,
all256 original base/15/43/47 images, same-prompt/seed/hash checks, global statistics,
direct43-vs15/47 confidence intervals, and curves. No score-based image selection.
Source site commit5906c424888ae1ae4c2b093978414a50c37be242.

### 2026-09-27 checkpoint 43 comparison site published
- Live private comparison: https://qwen-aesthetic-checkpoints.rigue.chatgpt.site
- Includes all 64 matched prompt/seed groups for base/15/43/47, statistical reports and curves. Display media are 1024px WebP quality 95 because hosting enforces a 256 MiB archive limit; scores use original generated files. Lossless originals remain local.
- Source: /mnt/nvme/work/qwen-checkpoint-comparison, commit df2b1eb7b69499834c1cd9d4e1da7b3169bab537.
- Project appgprj_6ab95a02a63881919276b703e5462cbc; version appgprj_6ab95a02a63881919276b703e5462cbc~appgver_d7dd81491d348191b730845ea0eab30f; deployment appgdep_6ab95c71f02c8191b806ead07db9da10 confirmed succeeded at 18:12:24 UTC.
- Prior oversized standalone deployment appgdep_6ab95b51757c8191a75152e25a2dbd4c failed; replaced by the successful smaller archive above. Historical review site remains unchanged in production.
- User service qwen21-aesthetic-eval43-resume.service active; all four GPUs observed 100% utilization during update after resuming checkpoint 47. No quality-based checkpoint replacement was made.

### Alice / multi-style portrait diagnostic frozen (2026-09-27 18:20 UTC)
- User requested visible improvement comparable to the Alice illustration in GRPO-Guard figure 1. Inspected actual figure and author project page: comparison is unguarded versus guarded RL under over-optimization, not base-versus-RL proof.
- Frozen /mnt/nvme/outputs/qwen21_aesthetic_lora/alice_portrait_probe/prompts.jsonl and protocol.json before generation: 12 prompts, 6 styles, 4 paired seeds, planned base/43/60 native 1024px 40-step evaluation. Normalized exact training overlap zero; no semantic novelty claim.
- This is a targeted diagnostic and cannot replace broad held-out assessment. Report all outputs, bootstrap by prompt, distinguish style change from improvement.
- Status: prepared only, no GPU job or automatic hook yet. Existing supervisor remains uninterrupted. Next continuation should integrate generation at an evaluation milestone without competing with four-card training or restarting it solely for this probe.
- Added evaluate_portrait_probe.py: default prepares 12 single-prompt jobs preserving identical four seeds per prompt; --execute runs four GPU lanes only when compute processes are absent and checkpoint60 exists. Three arms produce 144 images. It verifies grid completeness, exact prompts/seeds, 1024 resolution and image SHA256 before merging scores. No automatic scheduling hook yet; do not launch against live training.
- Configuration preflight passed via native evaluator --dry-run using existing checkpoint47 in place of not-yet-existing60; stdout/stderr stored in probe directory. This checks configuration resolution, not generation quality. Fixed protocol remains base/43/60.
