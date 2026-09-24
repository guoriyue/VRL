# Visual RL engine acceptance evidence

Status snapshot: 2026-09-22 18:35 UTC. The requested minimum 24-hour run is active;
its earliest completion time is 2026-09-23 07:36:41 UTC. This page distinguishes
implemented interfaces, real-model execution, and demonstrated capability gains.
The chronological [progress ledger](PROGRESS.md) contains commits and detailed logs.
Artifact paths below are relative to the repository root.

## Engine and reward system

| Area | Verified evidence | Remaining limit |
| --- | --- | --- |
| Qwen Image 2.1 generation and optimization | Local pinned model loads; reference/RGBA generation, real reward-weighted LoRA updates, native/replay checks and checkpoint continuation executed | One RTX 5090; no new multi-node acceptance claim |
| Independent rewards | Resumable content-bound scoring, nested axes, structured diagnostics, score comparison, frozen calibration/application, repeated-score variation and stress analysis work independently of the trainer | No human preference labels or human-calibrated deployed weights |
| Human review | Portable blinded A/B packets, original/reference media, explicit tie/unsure, checked answer import; real browser interaction smoke passed | The armchair demonstration contains one source and no human answers |
| Rollout admission | Per-prompt identities, detached source/task metadata, full reward axes, zero-advantage selections and unknown failure attribution are persisted | Zero advantage is not automatically task difficulty or verifier failure |
| Numerical consistency | BF16 compute can retain FP32 replay inputs; real Qwen native probes and controller updates passed their parity gates | Parity does not validate the reward's meaning |
| Checkpoint recovery | Real Qwen GPU training: SIGKILL after checkpoint-17, automatic restart, and final step-18 model/optimizer/progress/RNG matched uninterrupted training bitwise (1,027 tensors); CPU Ray descendant cleanup and controller restore also passed | One local GPU, one injected interruption at a published checkpoint; not a multi-node or arbitrary-crash guarantee |
| HTTP reward service | Typed transport, ownership/parking, cache identity/capacity and auxiliary integrity; v7 rejects stale service-instance mutations before model work | Long soaks are running; their CPU-oracle workload is not GPU-serving capacity evidence |
| CPU regression | Full suite 8 including stratified comparison, producer digests and foreground-alpha measurement: 4,844 passed, 17 skipped, 155 deselected | CPU lane excludes real GPU/distributed/optional/slow tests |

## Measured image capability

The frozen early-window generator checkpoint is
`outputs/qwen_image_21/rgba_early_window16/checkpoint-16`. It was trained on synthetic
opaque object extraction. Scores below use the original exact-reference objective;
they are not independent human preference scores.

| Evaluation | Base → trained mean | Paired result | Interpretation |
| --- | --- | --- | --- |
| Fresh 100 sources, native 256, two independent draws/source | 0.90649 → 0.91904 | +0.01256; original predeclared source-bootstrap interval [0.00755, 0.01849]; 79/100 improve | Narrow independently confirmed task gain |
| Fresh 50 sources, native 512, two draws/source | 0.79273 → 0.88776 | +0.09503; median delta −0.01666; 17/50 improve | Severe failures reduced, but most sources regress slightly |
| Fresh 50 sources at 40% foreground opacity, two draws/source | 0.04401 → 0.00767 | −0.03633; interval [−0.05973, −0.01843]; no positive source-average delta | Failed transparency-transfer gate |

Reports and explicitly rank-selected contact sheets are under
`outputs/qwen_image_21/rgba_early_confirmation_*`, `rgba_resolution512_*`, and
`rgba_translucent_*`. The earlier full-noise-window experiment improved its noisy
training sampler while regressing under native inference; that negative result
remains in the ledger and is not merged with the early-window result.

An optional dense RGBA axis now preserves ordering when the original clipped
match is zero. Its oracle/degeneracy tests pass.
A five-opacity ablation completed with fixed rollout-collection budgets,
identical initialization/data and only the selected reward scalar changed. Report
actual optimizer steps and skipped groups separately from collection epochs.
Both arms completed 32 collection epochs and 256 samples. The clipped arm kept
172 samples and made 28 actual Adam updates; the dense arm kept all 256 and made
32 updates. Checkpoint digests, per-parameter Adam counters and metric rows agree.
The 600-image paired native evaluation completed: original match means are
0.40774 base, 0.43609 clipped-trained and 0.45834 dense-trained. Dense training
improves combined scores on 40%/60% opacity tasks, but fully opaque sources regress 0.89175 to 0.76132
(delta interval [-0.21455, -0.07079]). Its dense-score mean itself is nearly
unchanged. This is a development tradeoff, not a universal reward improvement.
The comparison fixes collection budget, not accepted gradients or total compute.
Fresh confirmation is queued: 100 new sources, five opacity strata, two new draw
seeds, all three frozen arms, all five axes and no checkpoint selection. Exact
source/target overlap checks and 100 executable oracle checks passed.

Post-hoc pixel checks show the dense model still outputs target-core alpha about
0.928 for requested 0.4 and 0.886 for requested 0.6. Combined-score gains therefore
do not establish correct specified transparency. A new independent
`foreground_alpha_l1` axis avoids whole-canvas dilution by transparent background.
All 600 outputs were rescored under an explicit v3 recipe in new directories;
all nine historical axes remained exactly unchanged. Global foreground-alpha
error is nearly unchanged, while the opaque stratum clearly regresses. This new
endpoint is post-hoc and does not alter the frozen confirmation protocol.

## Agentic visual control

The bounded controller observes original/current images, chooses explicit edit or
stop actions, and pays a per-edit cost with terminal quality paid once. The Qwen
editor is frozen. Categorical likelihood replay, causal return-to-go, independent
policy identity, real gradients and optimizer/RNG restoration are implemented.
This is controller-only RL, not joint controller/generator training or unrestricted
tool use.

The current Qwen factory binds the exact packaged prompt template and Pillow
version into base identity v2. Changed templates reject incompatible restore;
loaded controllers keep immutable snapshots. Sixteen comparisons using the actual
local Qwen processor confirmed unchanged token and image tensors after extracting
the original template. Earlier experiments retain their frozen code; this contract
fix is not a policy-quality improvement.

The initial 64-episode experiment improved a two-source monitor from 0.60937 to
0.61990 net return, below fixed-one-edit at 0.80400 and random at 0.62271. It failed
the controller capability gate. The higher-exploration run completed 512 successful episodes and 16 optimizer
updates, all with zero replay error; checkpoint-16 integrity is verified. Its
monitor improved 0.54248 -> 0.61990, still below random 0.62271 and fixed
one-edit 0.80400; the baseline capability gate remains failed. Already-done
return regressed 0.74189 -> 0.71872 while needs-edit improved. Exact baseline
media and scores matched across checkpoints. Final paired evaluation completed
64 comparisons per checkpoint across 16 sources: controller 0.65948 -> 0.69305,
delta +0.03357 with descriptive source-bootstrap interval [-0.01789, 0.08348].
All 16 source averages remain below fixed one-edit (0.80530); the baseline gap is
-0.11224, interval [-0.14538, -0.08569]. Already-done return regressed
0.79055 -> 0.72714 and average calls increased 1.375 -> 1.78125. The controller
capability gate failed; this is not a demonstrated reliable stopping policy.
Last collection-policy
initial stop probability was 8.24% for already-done and 5.39% for needs-edit tasks;
this remains a training diagnostic, not evidence of a reliable stopping policy.

Current frozen runs use RGB composites. Optional `--controller-observation rgba`
adds original/current alpha masks, binds the mode to replay/checkpoints, and never
exposes verifier targets. Focused tests pass; a full real-model four-view gradient
probe passed exact replay and nonzero gradients/parameter updates; its recorded
action probability did not change at numerical resolution. A matching 512-episode
alpha-observation ablation is now running after successful GPU recovery, with
paired analysis predeclared.
This interface does not establish alpha reasoning.

Training-only CPU diagnostics now expose a stronger failure: opaque and transparent
states with pixel-identical white RGB composites get identical RGB action
probabilities, as expected. Supplying alpha masks does not by itself establish
correct stopping. On four training sources, generic completion wording prefers
stop in both states; an explicit alpha-mask completion explanation prefers edit
in both. No additional GPU training has been scheduled from these failed recipes.
The 24-source alias fixture has 4 training, 4 fresh monitor and 16 fresh final
sources; all 48 initial states pass the exact reward oracle check. Only its training
sources have been queried by the controller. Direct mask classification on those
four training sources passed 8/8 states in both single-image and four-view
presentations. This narrows the diagnosis to task/action interpretation or policy
learning; it does not establish a successful editing policy.

Optional exact-target masked RGB checks have also passed a 20-source / 100-image
procedural audit. All exact targets attain the optimum; all tested deviations
score lower, but noise can still outrank some wrong-color samples under pixel
distance. This is task-specific measurement, not general perceptual calibration.
The RGB native probe and two-seed untrained local-edit baseline are queued after
the alpha experiment. Their real-model qualification is not yet complete.
Rectangular portrait/landscape qualification and four existing local-photo tasks
are queued after that. The photo fixture has pinned originals, same-canvas sources,
two seeds per task and unchanged controls. Its manual coarse regions only measure
outside-region changes; semantic and human judgments remain separate. Two initial
region extents were corrected during visual input preflight before any GPU job;
the original waiting queue is retained as superseded, and v2 is the active fixture.
Independent learned EditReward scoring of the 12 photo candidates is queued after
mixed-opacity confirmation, using a temporary local v7 service. It will compare
semantic and locality rankings without fabricated preference labels; actual
learned scoring on these new candidates has not yet run.

## Active work and review entry points

- [Reward reliability baseline and verified artifact index](../reward_reliability/README.md)
- [Independent reward scoring and calibration](../../rewards_offline_evaluation.md)
- [Visual controller training, restore and comparison](../../visual_controller_rl.md)
- Frozen GPU queue: `outputs/qwen_image_21/followup_gpu_queue_20260922/`
- Controller run: `outputs/qwen_image_21/controller_rgba_exploration16/`
- Source-paired controller analysis: `outputs/qwen_image_21/controller_rgba_exploration_analysis.json`
- Mixed-opacity protocol: `outputs/datasets/rgba_mixed_opacity_20260928/`
- Mixed-opacity stratified results: `outputs/qwen_image_21/rgba_mixed_stratified_axes.json`
- Fresh confirmation queue: `outputs/qwen_image_21/mixed_confirmation_queue_20260922/`
- Actual GPU recovery comparison: `outputs/qwen_image_21/real_recovery_20260922/comparison.json`
- Original CPU HTTP soak: `outputs/reward_evaluation/service_soak_20260922_attempt2/`
- Fixed auxiliary-file soak: `outputs/reward_evaluation/service_auxiliary_soak_20260922/`
- Instance-bound v7 soak: `outputs/reward_evaluation/service_instance_soak_20260922/`
- Exact RGB qualification queue: `outputs/qwen_image_21/exact_edit_queue_20260922/`
- Rectangular photo queue: `outputs/qwen_image_21/natural_edit_queue_20260922_attempt2/`
- Independent photo reward queue: `outputs/qwen_image_21/photo_reward_queue_20260922/`
- Training-only CPU action-order diagnostics: `outputs/qwen_image_21/controller_action_order_cpu_20260922/`
- Alpha-alias input diagnostics: `outputs/qwen_image_21/controller_alias_cpu_20260922/`
- Explicit mask-semantics diagnostics: `outputs/qwen_image_21/controller_alpha_semantics_cpu_20260922/`
- Direct mask perception: `outputs/qwen_image_21/controller_alpha_perception_cpu_20260922/`
- Browser review: `outputs/reward_evaluation/armchair_blind_review/index.html`

All three soaks have owned processes and controlled restarts. The original runs
frozen commit 2aaf81fc6; the auxiliary-file soak runs 337702cb5; instance binding
runs cff89eea4 and has rejected an old client after real server replacement.
Each began after the 24-hour goal began, so none is a full 24-hour service test.
They must not be described
as testing later source changes. GPU jobs are serialized and stop on failure;
queued work is not counted as complete. User sprint edits and the untracked
PhyMotion tree remain untouched. No changes have been pushed or deployed.
