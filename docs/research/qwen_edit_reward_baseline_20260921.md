# Qwen localized editing: learned reward baseline and region evidence

## Question and experiment

Can a reward distinguish commercially relevant localized-editing failures and help
select a better natural output? The initial scope is color variants of clothing
and furniture, upholstery changes that preserve geometry, and whole-item
replacement. Returning the original image is not a benchmark task here.

We run the official, untrained Qwen-Image-2.1 pipeline at a 1024 resolution budget
and 40 Euler steps with seeds 0–3 per request. Ten original requests use four real
photographs. Two source photographs define development; the other two define a
small heldout set. A separate development-only four-seed feedback retry is judged
against its original request. It has additional generation cost. This is an
inference and verifier experiment, not RL policy training.

Source/task/rule manifests: [`manifests/edit_locality`](../../manifests/edit_locality).
Local artifacts: `outputs/qwen_image_21_reward_study/`, including the standalone
`index.html`, raw RGBA PNGs, source images, per-candidate explanations, review labels,
model provenance, environment versions, input hashes, and `summary.json`.

## Faithful learned baselines

- [EditScore](https://github.com/VectorSpaceLab/EditScore), upstream commit
  `4609c5d2ebb62fdebf665d3c924686d896ef1f74`; `EditScore/EditScore-7B` revision
  `dcce052accfc75f2d48279c889a71755ecdf737b`. Use its original following/consistency
  and perceptual-quality prompts, 25-point internal scale, single pass, temperature
  0.7, judge seed 42, and `average_last` reduction. In this path the overall score
  is `sqrt(min(following, consistency) * perceptual_quality)`, normalized to 0–10.
- [EditReward](https://github.com/TIGER-AI-Lab/EditReward), upstream commit
  `77a93aaa461fe9187e0ff841b59ecc0d0620bb7f`; released
  `TIGER-Lab/EditReward-Qwen2.5-VL-7B` revision
  `51b92ab5246295637c4ab3bd71e54a26f0a5189d`. Use the original
  `EditRewardInferencer`, strict full-checkpoint loading, multi-head RankNet, and
  upstream mean pooling. Scores retain their raw unbounded scale; log-sigma is
  recorded but not treated as calibrated confidence. The original 200704-pixel
  image preprocessing is retained. SDPA replaces unavailable Flash Attention 2.

Both use the cached Qwen2.5-VL-7B base revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`. Reward execution uses a separate
Transformers 4.57.0 environment because the Qwen-Image-2.1 generation environment
uses Transformers 5.17.0. This does not change the repository lock or production
reward dependencies. RGBA is composited onto white for these RGB judges; raw
outputs remain available. Both receive the same reference, candidate, and original
editing instruction.

The first development failure is real color leakage: a request to recolor only an
armchair seat also turns the backrest or a separate magazine holder blue. All four
original seeds violate a protected region. Full-image EditScore nevertheless gives
approximately 9.4–9.6/10 and often explicitly claims that the backrest stayed grey.
A local-crop ablation still scores these outputs highly. Merely adding another
consistency weight would not fix an incorrect visual observation: EditScore already
uses the minimum of following and consistency in this configuration.

## Candidate correction and what it proves

For six color tasks, source-defined sparse patches specify the requested part and
parts that must retain their original appearance. Target-color membership uses
circular HSV hue distance with a soft saturation weight. A protected-region penalty
measures newly added target-color membership relative to the source. The corrected
score is the minimum of normalized EditScore, target-color membership, and protected
color preservation. V2 uses royal-blue hue 225 degrees and teal hue 180 degrees;
these scalar choices were frozen after development and before heldout inspection.
V3 fixes one source annotation error discovered during heldout inspection: a patch
labeled wooden frame overlapped editable upholstery. The frozen V2 scores are kept
separately; V3 is explicitly an exploratory annotation-corrected evaluation.

These patches provide grounded evidence for a narrow color constraint. They are
manually selected, do not cover whole object masks, and cannot establish material,
shape, exact print fidelity, or overall scene preservation. A model could optimize
patches while damaging other regions; these checks are not validated as a sole
long-running RL reward. Material/replacement tasks retain the original learned
reward. The crop ablation changes both image view and instruction to describe that
view; it does not isolate resolution alone.

Visual labels are assistant reviews written before viewing candidate reward scores,
not independent human ground truth. Exact print-fidelity cases remain unknown in
binary metrics instead of being declared correct from readable text alone. Report
within-request positive-versus-negative ordering, top-1 selection, all tied winners,
and lowest-seed tie breaking. Do not infer improvement from merely lowering a bad
image's score when every candidate is bad.

## Reusable implementation

- `vrl.scripts.eval.qwen_edit_locality`: official model candidate generation,
  CPU text encoder and untiled CPU VAE, GPU transformer, resumable outputs.
- `vrl.scripts.eval.edit_reward_baseline`: isolated upstream model inference,
  full-view baselines and a separately named local-crop ablation.
- `vrl.scripts.eval.edit_color_constraints`: fixed source-patch color evidence;
  it never reads review labels.
- `vrl.scripts.eval.edit_reward_report`: per-task ranking analysis and standalone
  visual report; it consumes the labels separately from reward computation.

The separation is intentional: generation and reward inference require different
dependency environments, numeric verification must remain independent of review
labels, and report rendering owns presentation. Prompt/region tables belong in the
manifest assets. Production reward registries, trainer APIs, and cross-family
adapters are unchanged; this study does not justify restructuring them.

## Measured results

Forty original candidates plus four feedback retries completed. Every candidate
was scored by both released 7B reward models. The review separates requested-edit
success from protected-region fidelity; fine print preservation remains unknown.

| Evaluation | EditScore | EditReward | EditScore + color evidence |
| --- | ---: | ---: | ---: |
| Original development top-1, 3 tasks with known labels | 2/3 | 2/3 | 2/3 |
| New-scene top-1, 6 tasks | 4/6 | 4/6 | 4/6 |
| Original/retry armchair pool: correct good-vs-bad pair credit | 11/16 | 16/16 | 16/16 |

These are small descriptive counts from assistant review, not benchmark claims or
independent human validation. The new-scene set has only one request with both
passing and failing candidates, giving four comparable pairs. It is too small to
establish overall reward superiority. The fixed-seed baseline and each learned
reward both pass 4/6 of those tasks; the candidate oracle passes 5/6. The color
addition has **not demonstrated improved heldout top-1 selection**. It also misses
non-target-color drift: a protected burgundy backrest can become darker/purpler
without acquiring the requested teal/blue color.

The concrete output improvement is feedback repair of the development armchair
request. Originally, 4/4 candidates incorrectly recolor the backrest or magazine
holder. With a single, explicit localization-feedback rewrite and the same four
seeds, 4/4 retain the grey backrest/arms and brown magazine holder while turning the
seat blue. No pixel restoration or compositing is used. The retry explicitly calls
the continuous lower front part of the seat region; a user should confirm this
interpretation of “seat surface.” The robust demonstrated gain is removal of the
original protected-region violations, not certification of every original pixel.

For these eight candidates, EditReward ranks every corrected candidate above every
original failure. Their raw score ranges are 0.553–0.737 versus 1.005–1.325. EditScore
has overlapping ranges: 9.398–9.592 versus 9.398–9.600. The region evidence separates
the specific color-leakage violation, but does not outperform EditReward on this
pair set. This supports retaining EditReward as a serious baseline rather than
assuming that a hand-built scalar must improve it.

`feedback_improvement.jpg` shows the original, initial seed-0 result, and seed-0
feedback retry. `index.html` contains all candidates, exact prompts, scores, review
evidence, and a click-to-enlarge view. This gain used an assistant-written feedback
prompt and extra inference; it is not an autonomous production repair policy or a
trained LoRA result.

## Next reward design justified by this run

1. Keep EditReward and EditScore as separate baselines. Compare within-request
   ranking and selection, not their raw magnitudes. Preserve per-criterion
   following/consistency/quality evidence where available.
2. Represent the user request as an editable part and explicitly protected parts.
   Use verified region annotations to supply evidence and concrete retry feedback.
   Source-color fidelity needs a broader comparison than target-color leakage;
   material and replacement require different checks for texture and geometry.
3. Use a verifier failure to request a bounded repair, then judge against the
   original instruction. This run demonstrates a useful repair, but automatic
   region localization and feedback generation still need independent evaluation.
4. Before RL scaling, collect new source scenes and independent human preferences
   with both successful and failed candidates per request. Test reward ordering,
   judge randomness, non-target color drift, and region-annotation mistakes. Only
   then compare equal-rollout-budget RL runs with the original learned reward and
   a corrected reward. The current color-only score is not ready to be the sole
   objective for broad image-editing RL.

## Judge randomness check

Repeating development EditScore with judge seed 43 changes the armchair-pool
pair credit from 11/16 to 13/16, but changes its top-1 outcome from pass to fail
under the declared tie policy. An original output with a blue magazine holder
receives 9.798/10, tied with two corrected outputs. This is a tied highest score,
not a unique preference for the failure. The fixed source-color checks still
identify its violation. A single sampled judge response and a high absolute score
are therefore insufficient acceptance criteria for this task. The original and
repeated scores are both retained; neither run is discarded.

## Validation and records

The run completed 44 real official-pipeline generations and 136 candidate reward
evaluations: 44 full-image EditScore, 44 EditReward, 28 local-crop EditScore, and
20 repeated development EditScore evaluations with judge seed 43. The local-crop
armchair pool reaches 10/16 pair credit and does not resolve the original false-high
scores. All four added evaluation commands pass Ruff checks and formatting; the
numeric color verifier and report builder execute against the complete artifact
set. The standalone HTML was rendered with headless Chrome for inspection.

The tracked [machine-readable record](qwen_edit_reward_baseline_20260921.json)
contains all candidate scores, original EditScore explanations, review evidence,
model revisions, dependency versions, and both frozen/corrected color scores.
Full-resolution image files remain in the local output directory. No production
reward registry, RL training loop, or policy weights were changed.
