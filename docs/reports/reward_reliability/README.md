# Reward reliability baseline

Snapshot: 2026-09-22. The independent reward workflow is executable; general
preference calibration and reward-recipe promotion remain unproven. This report
collects existing evidence without labeling model or automated judgments as human
preferences. See [artifact_index.json](artifact_index.json) for content-bound run
identities and provenance/summary digests. All 2,808 indexed scoring records were
rechecked against their current media and declared auxiliary asset hashes.

## Natural-image diagnostic: semantic score and protected region disagree

Four outputs share one armchair photograph and the instruction to change only the
upholstery to blue. The common 256-pixel reference and explicit binary mask protect
a coarse magazine-holder rectangle. This mask does not measure the whole
background, and preserving that rectangle does not prove semantic correctness.

| Candidate | EditReward (higher) | Protected-region RGB L1 (lower) |
| --- | ---: | ---: |
| Unchanged source | −0.362050 | 0.000000 |
| First blue edit | 0.500816 | 0.197280 |
| Second blue edit | 0.489514 | 0.207311 |
| Explicit corrective edit | 0.175492 | 0.176056 |

Visual inspection found unwanted blue coloring on the magazine holder. The
corrective edit partially restored the outer panel but left its interior blue.
The two scores disagree on five of six candidate pairs. This is a concrete
reason to retain separate instruction-following and preservation diagnostics;
it is not evidence that locality alone is a better overall reward. The unchanged
source illustrates the opposing exploit: perfect preservation without following
the requested change.

Evidence: `outputs/reward_evaluation/masked_edit_audit/with_repair.jsonl`, the
`editreward_with_repair` and `locality_with_repair` scoring directories, and
`ranking_comparison_with_repair.json`. Each score directory retains its own frozen
recipe and sample records; joining them does not overwrite training rewards.
The real EditReward model was served independently over HTTP. There is only one
source, so six pairs cannot establish six independent tasks or a useful confidence
interval.

## Synthetic RGBA stress audit

Two hundred exact synthetic oracles each produce an unchanged sample and six
explicit RGB/alpha perturbations: 1,400 media files, independently scored by RGBA
reference matching and image sharpness (2,800 scoring records). The six changes
lower exact RGBA match on all 200 sources:

| Perturbation | Mean RGBA-match delta |
| --- | ---: |
| RGB blur | −0.01550 |
| RGB noise | −0.09412 |
| RGB checkerboard | −0.50000 |
| Black RGB | −0.39330 |
| Fully opaque alpha | −0.93830 |
| Empty alpha | −1.00000 |

Noise and checkerboards increase sharpness on every source. Even RGB-only blur
increases this scorer here: changing straight RGB while retaining alpha can
introduce dark compositing edges. Thus a probe name is not a universal quality
label, and sharpness should not independently decide image quality. The reference
verifier measures an explicitly known target; this does not validate it for
open-ended aesthetics or photographs without exact targets.

Evidence: `outputs/reward_evaluation/rgba_oracle_stress200/recipe.json`,
`media.jsonl`, `rgba_report.json`, and `sharpness_report.json`. Reports retain
individual deltas and failed/missing-pair semantics; repeated variants are grouped
by their source rather than treated as independent tasks.

## Calibration and promotion status

A portable blinded review packet is available at
`outputs/reward_evaluation/armchair_blind_review/index.html`. It contains six
candidate pairs, allows A/B/tie/unsure and leaves unanswered pairs absent. Its
browser interaction smoke test passed; there are **no collected human answers**.
The packet is a development demonstration on one source, not a source-disjoint
calibration/holdout dataset. No human-calibrated weights have been promoted.

The frozen early-window Qwen RGBA policy improved exact matching on fresh opaque
synthetic sources, but failed a translucent-source transfer challenge. An optional
dense RGBA scalar and matched short-training ablation are under evaluation. Their
outcome is separate from human preference calibration. Full positive and negative
training results are in the [engine acceptance report](../visual_rl_engine_20260922/ACCEPTANCE.md).

Remaining acceptance work includes actual preference annotations on multiple
source-separated calibration/holdout groups, independently justified failure
labels, declared tolerances before inspecting holdout, and cost/misranking
comparison before selecting a new general reward recipe. Video preference and
failure-mode coverage is not established by these image-only experiments. There
is no basis here to promote a general image/video reward configuration.

## Reuse

[Offline reward documentation](../../rewards_offline_evaluation.md) describes
standalone scoring/resume, compatible raw-axis reuse, candidate comparison,
perturbation audits and frozen calibration. These commands do not require a
trainer or generation worker. Scoring failures remain explicit errors, rather
than synthetic zero rewards. Task-specific exact checks and complementary learned
scores keep their existing reward interfaces; this report adds no new verifier
framework or training-time adaptation of weights.

## Repeat-scoring evidence

`outputs/reward_evaluation/rgba_repeatability_20260922/report.json` summarizes three
fresh CPU scoring executions of 600 existing media (200 sources, each with a
baseline/noise/empty-alpha variant). Each execution reports 600 scored and zero
reused records. All nine exact-reference axes have zero observed score range on
all samples. This verifies deterministic execution of this CPU oracle on these
inputs; it says nothing about learned reward uncertainty or open-ended quality.

The four armchair outputs were also re-executed through the real EditReward
service before/after its transport diagnostic update and restart. Both score axes
are exactly equal across those two executions. Report:
`outputs/reward_evaluation/masked_edit_audit/repeatability_two_executions.json`.
Treat this as narrow re-execution compatibility evidence, not a same-runtime
large-sample noise estimate. Neither study supplies human preference labels.

## Exact masked RGB task audit

An additional procedural audit covers 20 local-recoloring sources with five
candidates each. Explicit `exact_target=true` checks the editable region against
a known target and requires that the target preserve protected source pixels.
All 20 exact targets scored 1 and all 80 deviations scored lower. Mean target
matches were 0.6140 unchanged, 0.6090 wrong color, 0.6519 inside noise and 0.9393
outside spill. Noise outranking some wrong-color samples demonstrates why pixel
distance is not a general perceptual-quality ordering. These checks have not been
promoted as a general reward or demonstrated a learned local-edit improvement.

Evidence is under `outputs/reward_evaluation/exact_masked_edit_20260922/`:
`audit_report.json`, content-bound `scores/`, and standalone `scorer.json`.
The original artifact index above retains its original baseline scope. The new
100-record audit and its configuration can be reused through the ordinary CLI:

```bash
.venv/bin/python -m vrl.scripts.rewards.rescore_media \
  --manifest outputs/reward_evaluation/exact_masked_edit_20260922/media.jsonl \
  --config outputs/reward_evaluation/exact_masked_edit_20260922/scorer.json \
  --output-dir outputs/reward_evaluation/exact_masked_edit_20260922/scores --resume
```

The verified resume reused all 100 records with zero new inference and retained
run ID `e6345d52cc136d6276b305904d0692d8ffa995d7377c3887df8e59be6c1ca198`.
Real Qwen RGB qualification and two native baseline draws per source are queued
separately; procedural oracle checks do not establish that baseline's quality.
