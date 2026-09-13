# GOAL: find a reward that actually improves Anima, and prove it

You are working in `~/Desktop/VRL` (git repo, branch `main`). One RTX 5090, 32 GB,
shared with the operator's own jobs. The model is Anima (`cosmos-predict2-anima`,
a Cosmos-Predict2 anime image DiT) trained with Flow-GRPO-style LoRA RL.

**Objective.** Find a reward that (a) measures something an anime image model is
actually judged on and (b) provably moves this policy. Validate it on a small
fixed prompt set first, then train on the full set and prove the gain on
held-out prompts. Report negative results as negative.

**Scope clarification (2026-09-09).** The operator explicitly permits any
reward/target that makes Anima visibly better, not only pose/craft. Search color,
lighting, text, composition, line quality and reference/style adherence where
the scorer is qualified. The base must have meaningful headroom, the reward
must not pay for saturation/blur/content deletion, and trained ODE images must
look clearly better with gains on unseen prompts. A changed image or higher
training score does not meet this objective.

---

## Already established — do not re-derive

### Machinery: three defects, all fixed. Do not reintroduce them.

1. **Importance ratio / replay parity** was broken in every LoRA GRPO run before
   2026-08-28 (`torch_compile` x generation batch shape, 0.203 log-prob drift).
   Fixed. First-step parity gate must read 0.000000.
2. **The final denoise step got no gradient.** `actor.timestep_fraction < 1` with
   the default `timestep_selection: strided` yields indices `0..n-2` and drops
   the last step, which carries most of the gradient (fixing it moved grad_norm
   5.1x and approx_kl 9.9x). Use `timestep_fraction: 1.0`, or `< 1` **only** with
   `timestep_selection: random`.
3. **`rollout.noise_level` was 0.3**, the only preset in the repo below 0.7. It
   is the flow_grpo Eq.9 SDE exploration temperature and the only source of
   within-group reward spread a policy gradient can act on -- every sample also
   draws its own initial latent, which the policy does not control, so lowering
   the injected noise shrinks only the policy-attributable half of the spread and
   the advantage ends up mostly reflecting which latent got lucky. Measured on 8
   fixed prompts with a deterministic reward: **0.3 falls 3.0 SEM over 6 updates,
   0.7 rises 7.7 SEM over 10.** Now 0.7. Never lower it.

Every "this target/reward does not work" conclusion recorded before 2026-09-07
was measured on at least one of these defects. Treat those conclusions as void
unless re-measured.

### Reward properties that decide success

- **Granularity is the binding constraint.** A thresholded verdict takes 2-4
  distinct values inside a 16-sample group; the advantage collapses to "which
  half of the batch am I in" and the policy does not move (three separate
  configurations, all |t| < 0.7). Replacing each condition's yes/no with the
  continuous quantity it thresholds moved the same policy at **t = +9.52** over
  30 updates. `geneval_owl` and `wd_tagger` both expose a dense reading now,
  selected by `score_key`.
- **Select prompts for within-group contrast, not difficulty.** A condition the
  base satisfies 0% of the time gives zero contrast in a group of 16, so no
  gradient. Per-condition variance is about `q(1-q)`; pick prompts whose target
  conditions have base hit rate `q` near 0.5 and drop anything with `q <= 0.05`
  or `q >= 0.95`. Same reward, same everything else: hardest-first selection gave
  **t = -0.15**, contrast-first gave **t = +1.69**.
- **Domain match is the anti-hack property.** GenEval's OWLv2 is photo-domain.
  Training on it raised the score by making images *less* anime: characters
  dropped, backgrounds flattened to plain, sharpness median -12%, and several
  reward flips were detector errors (a character drawn with zebra ears counted
  as "zebra"; "three cups" passed with no visible cups). A scorer trained on the
  generation domain (WD14 tagger, danbooru) is a better-matched candidate, not
  proof against reward hacking. Its craft score still needs image-level checks
  for false positives, missing characters, and style or background shortcuts.
- **Rollout-to-clean rank transfer is a real filter.** GRPO consumes the
  within-group ordering of NOISY rollout samples while the evaluation measures
  clean samples. Spearman rho between the two orderings over the same initial
  latents: sharpness **+0.369** (learnable), GenEval verdict **+0.164** (not).
  Note the reward *level* transfers fine; it is the ordering that breaks.

### Rewards already ruled out — do not retry

| reward | why |
|---|---|
| AnimeReward | ranks saturation (rho +0.507 within a group), not quality |
| PickScore | tolerates loss of anime line/detail; contradicts AnimeReward on the same checkpoint |
| codex / Luna VLM judges | test-retest 0.1-0.5, ties on ~60% of prompts: noise is the same size as signal |
| CountGD person counting | photo detector; post-training audit found it rewarding distant people erased into fragments |
| GenEval OWLv2, verdict | granularity 2-4 values, unlearnable |
| GenEval OWLv2, dense | learnable (t=+9.52) but photo-domain; drifts the model out of anime |
| wd_tagger on the NSFW attribute set | base already 0.923 recall / 67% perfect, no headroom |

### Base weaknesses actually measured (where the headroom is)

- **Anime craft axes** (camera angle / framing / pose / arm placement / gaze),
  120 held-out prompts, clean sampling: WD recall **0.632**, only **25.8%** of
  prompts fully satisfied. Per-tag base hit rate: `dutch_angle` 0.00,
  `head_tilt` 0.00, `arms_behind_back` 0.14, `walking` 0.18, `arm_up` 0.31,
  `from_above` 0.40, `standing` 0.43, `kneeling` 0.44, `leaning_forward` 0.45.
  Dataset: `dataset/anime_craft`. A dense-reward probe on contrast-selected
  prompts reached t = +1.69 over 10 updates (not yet significant).
- **Hands / structural anatomy**: 0% perfect, 6-38% visibly broken. This is the
  largest real weakness and there is **no continuous anime-domain scorer for it**
  -- building one is the highest-value open problem.
- Saturated, do not target: tag adherence 0.98, natural-language adherence 0.94,
  negation 100%, NSFW compliance 97%.

---

## Protocol — run it in this order

### Step 0. Screen the candidate before spending GPU on training (cheap, ~1 hour)

1. **Granularity.** Score 16 samples of one prompt, count distinct reward
   values. Need roughly >= 10/16. If it is 2-4, densify the reward or discard it.
2. **Rank transfer.** Render each initial latent twice -- SDE at noise 0.7 (the
   training distribution) and deterministic ODE (the eval distribution) -- score
   both, and take the Spearman correlation of the two orderings over 16 samples,
   averaged over 8 prompts. Need rho >= ~0.35. Pattern:
   `docs/sprints/` references `reward_transfer_probe.py`; rebuild it if absent.
3. **Headroom.** Base score distribution must not be saturated. 0.6-0.8 mean
   with real per-condition failures is good; 0.92+ is not.
4. **Anti-hack.** Inside a group, correlate the reward against saturation,
   brightness, edge energy, background simplicity and character presence. A
   strong correlation with any of them predicts the hack you will get. Also
   confirm the scorer's training domain is anime, not photo.

### Step 1. Fixed-prompt probe (1 hour, then 3 more if promising)

8 prompts chosen by contrast (see above), 16 samples each, `noise_level 0.7`,
10 updates; continue the same run to 30 if the slope is positive and the screen
passes. Fixed prompts remove prompt-mixture variation, but **training reward is
not clean-sampling evaluation**. Fit an OLS slope and report its t as a training
diagnostic, not independent proof of improvement; correlated updates and
repeated checkpoint selection also limit its significance interpretation.
Reference values measured on this machine:

| probe | slope | t |
|---|---|---|
| sharpness (positive control) | +0.0199 | +12.83 |
| dense GenEval, 30 updates | +0.0025 | +9.52 |
| dense craft, contrast-selected, 10 updates | +0.0033 | +1.69 |
| any thresholded verdict | ~0 | \|t\| < 0.7 |

### Step 2. Held-out check on the probe checkpoint

Evaluate on a manifest that *contains* the trained prompts, then split the delta
into trained vs unseen. The trained subset proves optimizability; the unseen
subset is a degradation check, **not** a generalization claim (8 prompts cannot
generalize).

### Step 3. Look at the images. Always.

Build a side-by-side sheet, base vs checkpoint at identical seeds, labelled with
per-condition verdicts, covering wins **and** losses. Every reward that failed in
this program looked acceptable in the numbers first. Look specifically for:
objects or characters disappearing, backgrounds simplifying, style drifting out
of anime, and reward flips that are scorer errors rather than real changes.

### Step 4. Only then, the real run

Full rotating prompt set, checkpoint every 5-10 updates, seed-aligned paired
evaluation on a frozen held-out manifest at every checkpoint, and sharpness as a
hard guardrail (fail if the median drops more than 10% from base). With a
rotating prompt set **the training curve is not readable as learning** -- the
per-step prompt draw contributes about +/-0.063 of noise with 16 prompts, which
swamps the per-step gain. Only the held-out evaluation counts.

---

## Traps that already cost time here

- Judge a training curve only on a **fixed** prompt set. On a rotating set it
  measures prompt-draw luck.
- Rollout reward and eval reward are different scales; never compare them. A
  rollout at noise 0.7 scores far below the same prompt sampled cleanly, and
  that is correct, not a bug. Chasing that gap is what introduced defect 3.
- Any before/after must generate **both** arms from the same manifest with
  `seed + row_index`. One unaligned comparison produced a swing larger than any
  real effect, with the opposite sign.
- Validate any tag or condition vocabulary against the scorer's own label list.
  A self-invented tag reads as a 100% failure and once produced a fake
  0.731-vs-0.980 recall.
- A GPU memory parking failure on a shared card is usually another process.
  `gpu_used_bytes` now reads the driver's per-process table; check
  `nvidia-smi --query-compute-apps` before blaming the code.
- Never kill a GPU process without reading `/proc/<pid>/cmdline` first. The
  operator runs their own jobs on the same card.
- `ppo_epochs=1` with a strict on-policy schedule makes the ratio identically 1,
  so ratio clipping and trust regions are inert. That is expected; it does not
  mean the gradient is dead.

## Deliverable

A reward that passes Step 0, shows a significant positive slope in Step 1,
survives Step 3 by eye, and produces a held-out gain in Step 4 with the sharpness
guardrail intact. When a candidate fails, record **which of the four screening
properties it failed** and move on. Do not tune hyper-parameters against a reward
that fails the screen -- that is what the previous three months did.

## Where the evidence lives

- `docs/sprints/SPRINT_anima_geneval_spatial_rl.md` -- the whole investigation:
  sections 6.11 (noise root cause), 7 (the three defects), 7.5 (granularity and
  transfer measurements), 8 (dense reward result), 8.4 (eyeball review of the
  hack), 9 (anime-craft target and the contrast-selection lesson).
- `docs/sprints/SPRINT_anima_rl_target_search.md` -- earlier target search, with
  a dated correction at the end marking which of its null results are void.
- `docs/sprints/SPRINT_anima_rl_5090.md` -- the run log that first recorded the
  symptom ("something degrades the policy independently of what it optimizes")
  without identifying the cause.
- `outputs/eval_geneval_anime/examples/` -- side-by-side sheets showing what
  detector hacking looks like.

## Handoff audit — 2026-09-09

The craft candidate is promising but has not passed Step 0. The reported
90 distinct dense scores among 120 images are **across prompts**, not 16 samples
of each prompt. No craft-specific paired SDE/ODE rank-transfer or within-group
anti-hack results were located. Tag hit rates across prompts are only a
selection heuristic, not measured within-prompt hit rates.

The current manifests also require evaluation isolation before a full run:

- `outputs/probe_craft_contrast8_anime.jsonl` contains eight prompts from the
  original 200-row evaluation manifest, at zero-based indices
  `11, 21, 32, 49, 51, 69, 70, 105`. Report these as **seen**, separately from
  the remaining 192 prompts. The existing first-120 base set is 8 seen +112
  unseen. Because the base census informed target selection, this is a
  development set; final confirmation should use a separately frozen test set.
- The 2,000-row training manifest has 1,963 distinct prompt strings and shares
  eight exact prompts with evaluation (evaluation indices
  `5, 22, 76, 82, 86, 131, 148, 188`). Deduplicating by final prompt and excluding
  all original evaluation prompts would leave 1,955 training rows. Disjoint
  source post IDs alone do not prevent leakage after prompt projection.
- `outputs/build_anime_craft.py` selects appearance context using
  `sorted(tags & APPEARANCE)[:6]`. Probe row 1 consequently contains both
  black/brown hair and long/short hair without specifying separate characters.
  Exact deduplication does not repair this semantic ambiguity. Preserve the
  historical probe manifest for checkpoint comparisons; audit context
  construction before preparing the full training set.

Source manifest SHA-256 values for this audit:

| Manifest | SHA-256 |
|---|---|
| `datasets/danbooru/anime_craft/train_prompts.jsonl` | `10a1b2ad78104ebbf3618ee42f1ffc3e74e555309e34cf7448548144a311ca09` |
| `datasets/danbooru/anime_craft/eval_prompts.jsonl` | `292883d1c93fb9d2de133690520a97d229516e8a774400477c78788a00052d58` |
| `outputs/probe_craft_contrast8_anime.jsonl` | `4a68592a0745a90a7f8e36f637adcc85078fd7121b9906e5d0254e68355a1f32` |

At audit time, `outputs/anima_craft_contrast_probe/metrics.csv` contained epochs
0–11: reward slope +0.002412/update, naive OLS t = +1.699. Both
`checkpoint-10/checkpoint_meta.json` and `checkpoint-final/checkpoint_meta.json`
reported global_step 10; neither represents the two later logged updates.
`train_craft_contrast_probe_30.log` ended with
`supervisor stopped by operator signal; not restarting`, and the training
process was no longer present. Do not treat this as a live 30-update run or
automatically restart an operator-stopped job.

Next evidence required: craft-specific Step 0 paired images with per-image
scores, followed by seed-aligned base/checkpoint-10 comparisons separated into
seen and unseen subsets. No quality improvement has yet been established.

Preliminary visual inspection of `outputs/anime_craft_base/rollout_check.jpg`
found mixed changes rather than an obvious consistent win. For example, the
third row's first labelled seed changes an overhead arm to a forward-reaching
arm, while `arm_up` changes from a displayed miss to a hit. This is a candidate
scorer false positive to check at full resolution, not established improvement.
The generating script and completed log were subsequently recovered from
`/tmp/claude-1000/-home-mingfeiguo-Desktop-wm-infra/765fce32-5ff3-4b5c-96e4-cb32527847f3/scratchpad/`
(`craft_rollout_check.py` and `craft_rollout_check.log`). The script uses the
same eight probe prompts for both arms, initial seeds `7777 + row_index` and
`8888 + row_index`, 512px, 20 ODE steps, CFG 4.5, BF16/TF32, compilation off,
and the checkpoint-10 adapter path for the trained arm. The sheet labels omit
the row-index offset. The log reports tag hits **27/50 -> 31/50**, with per-row
counts `2->3, 4->4, 5->5, 5->4, 0->1, 4->5, 2->4, 5->5`.
These are correlated tag verdicts on 16 images per arm, not 50 independent
trials. This is useful exploratory paired evidence, but lacks unseen prompts,
saved full-resolution individual images, numerical sharpness measurements,
and a frozen resolved model identity. It does not satisfy the delivery gates.

### Authorized screening restart

The operator subsequently released the GPU for this task. A new one-shot
screen was launched with:

```bash
OMP_NUM_THREADS=4 .venv/bin/python -u outputs/craft_transfer_probe.py \
  --output outputs/craft_transfer_probe_20260909_v1
```

This screens the historical 8 probe prompts with 16 paired initial latents each,
SDE noise 0.7 versus native ODE, preserving all 256 PNGs, per-image scores and
target probabilities, resolved config, manifest hash, and per-group statistics.
The SDE RNG uses the production loop's separate generator with the sample seed,
not the older transfer script's XOR seed. Each pair checks identical initial
latents. The tagger runs on CPU. Character-presence and simple-background
probabilities are same-tagger diagnostic proxies, not independent visual judges.
Only `summary.json` indicates that all groups finished; partial scores are not
a completed screen. This is screening, not a resumed training run.

The screen completed successfully: all 256 PNG hashes and score-grid entries
were verified. Mean within-prompt SDE/ODE dense-score Spearman was **0.141856**;
individual values were `0.155, -0.006, 0.515, 0.026, -0.138, 0.312, 0.089, 0.183`.
A 10,000-resample prompt bootstrap (seed 99107) gave [0.0175, 0.2798]; this
describes only these eight selected prompts, not the full craft population.
SDE distinct-score counts were `13, 12, 16, 16, 16, 9, 15, 14`; overall dense
means were 0.71595 SDE and 0.74599 ODE. The historical prompt/reward combination
therefore does not pass the handoff's transfer screen and is not cleared for
automatic continuation to 30 updates.

Important calibration limitation: Laplacian-sharpness rank transfer on these
same pairs was also only **0.036765**, not the historical +0.369. The historical
control used different prompts and an XOR noise seed. Thus the +0.35 cutoff
is a screening heuristic, not a universal learnability theorem; these results
do **not** prove that craft or sharpness cannot be learned. Do not attribute
all of the failure to WD tagger without matching the control protocol.

Visual check also exposed a coverage blind spot: `p07_s06_sde.png` scores 1.0
on `from_side/profile/standing` but depicts two character views despite the
prompt's `1girl, solo`; `p07_s15_sde.png` depicts one figure and scores 0.5731.
This does not establish training-induced hacking (both are base samples), but
shows why craft-tag reward alone cannot certify full-prompt or overall quality.

### Prospective development screen

`outputs/craft_clean_data_probe_20260909/` contains candidate splits of
2,420 train /64 dev /128 test, with source/final-prompt/historical-projection
isolation and hashes in `report.json`. These are **not approved training or
test-generation manifests**: subsequent source-tag review found unsuitable
content and residual semantic conflicts. The split report proves isolation,
not content suitability. Preserve them for audit; filter and re-freeze safe
confirmation data before any full run.

The next screen uses `outputs/craft_dev8_screen_probe.jsonl`: six reviewed
dev rows (zero-based indices `12, 31, 14, 36, 57, 48`) plus two prospectively
authored, fully clothed adult pose prompts. All eight are development data,
not confirmation data. Both pinned tokenizers fit all prompts within 128 tokens.
No reward formula or training hyperparameter was changed. Command:

```bash
OMP_NUM_THREADS=4 .venv/bin/python -u outputs/craft_transfer_probe.py \
  --manifest outputs/craft_dev8_screen_probe.jsonl \
  --output outputs/craft_dev8_transfer_probe_20260909_v1
```

As above, inspect `summary.json` for completion, per-image scores and PNGs for
evidence, and the sharpness control before interpreting any transfer threshold.
Do not infer training has resumed from a screening process being live.

### Broader non-pose candidate screening

Read the operator's new task attachment and reopened the search beyond pose.
The first candidate is longer headlines on illustrated anime posters, using
the existing PP-OCRv6 reward with `all_text` and substring full credit disabled.
No reward implementation was added. The historic OCR null is not a decisive
negative for corrected-noise training, but its saturated baseline remains a
reason not to repeat that curriculum blindly.

`outputs/ocr_poster_headroom_probe_20260909/` completed 8 prospective prompts
x4 clean ODE samples: mean OCR **0.938469**, **24/32 exact**, and the complete
normalized target appeared somewhere in 26/32 transcripts. This last count is
a diagnostic, not a replacement reward. Examples:

- `p00_s01.png`: headline is correct; extra locomotive lettering lowered the
  whole-image OCR score to 0.667. It is not a spelling-headroom example.
- `p05_s03.png`: visibly misspells SECRET and LIBRARY and repeats the latter;
  this is a real rendering failure.
- `p07_s03.png`: the headline looks correct but OCR introduced an accent into
  DIFFERENT; verify recognition before claiming a generator defect.

These titles are mostly too easy; this is not clearance for an OCR long run.
The one-shot input is `outputs/ocr_poster_headroom_probe.jsonl`; original PNGs
and line-level scorer evidence are retained for visual audit.

The next baseline screen uses `outputs/lighting_palette_headroom_probe.jsonl`
with the existing WD scorer and sampler, at
`outputs/lighting_palette_headroom_probe_20260909/`. It tests dappled sunlight,
sidelighting, backlighting, light rays, candlelight, spotlight, limited palette
and monochrome, each x4 paired SDE/ODE samples. All eight target labels were
verified in the scorer's cached vocabulary. Four samples per condition are
only a preliminary headroom/judge check, not the 16-sample qualification gate.
Attribute adherence is not general aesthetic quality, and low detection
probability is not automatically a genuine visual failure.

The lighting screen completed all 64 images. Mean ODE scores by condition were
0.629756 / 0.043796 / 1 / 1 / 0.950990 / 1 / 0.981921 / 1 in the order above.
The low-scoring sidelighting example also has a small subject and large black
borders; a low-scoring dappled-light example visibly has patches of light.
Neither is sufficient evidence for a lighting training target. The other six
conditions provide little preliminary score headroom on these four seeds.

`outputs/mark_absence_reward_probe_20260909/` scored 256 existing development
images on CPU, using `1 - mean(raw signature/watermark/artist_name probability)`.
At a 0.35 detector threshold, 0/128 ODE and 1/128 SDE samples were detected;
mean paired rank correlation was -0.039706. Crucially, ODE files
`p03_s01_ode.png` and `p04_s02_ode.png` in the source archive visibly contain
bottom-edge pseudo-lettering despite remaining below threshold. This is a
judge-calibration warning, not proof that unwanted marks are absent. Do not
implement or train this candidate from probability spread alone.

The existing sharpness checkpoint-10 was evaluated with native ODE on the new
eight development prompts, two seeds each, against base (32 images total):
`outputs/sharpness_ode_before_after_probe_20260909/`. Reward scale was set to
1.0 only for evaluation to avoid the training scale's clipping. The paired
prompt-level reward delta was +0.001021, bootstrap 95% CI
[-0.000291, +0.002289] for `image_sharpness`; this does not establish a gain. Fixed-label before/after
sheets are under `fixed_model_mosaics/`, including original-resolution pairs.
Visual review found no consistent clarity improvement; examples lose
background detail or move toward white backgrounds (prompt 4 seed 15458,
prompt 8 seed 15466), and prompt 7 seed 15463 becomes monochrome. These are
qualitative counterexamples to treating a sharpness score as overall quality,
not a measured population-wide causal claim. Do not promote this checkpoint.
All 32 image hashes, dimensions and paired prompt/seed identities passed audit.
The current training manifest and evaluation prompts have zero text overlap;
however, the training run did not retain an independent manifest hash/snapshot,
so historical disjointness cannot be established from immutable evidence.
The current training prompts concern objects/spatial relations, making this a
cross-content transfer check rather than matched-distribution confirmation.

The next offline headroom screen uses
`outputs/ocr_spelling_headroom_probe.jsonl`: eight naturally themed illustrated
posters with longer, less common English words, four ODE seeds each. It reuses
the same existing OCR script and reward without a new model-specific config.
Its output is `outputs/ocr_spelling_headroom_probe_20260909/`; a launch is not
completion, and spelling errors must be checked against the actual image
before selecting training data.

The spelling screen completed 32/32 images (process exit 0): mean whole-image
OCR 0.835065, 12/32 exact. Both pinned tokenizers fit every prompt (maximum
Qwen 62 / T5 72 tokens). This aggregate mixes several error types:
`p00_s03.png` visibly misspells EXTRAORDINARY; `p02_s02.png` repeats ANNIVERSARY;
`p04_s02.png` misspells CORRESPONDENCE and includes manuscript writing. In
contrast, `p02_s01.png` renders the complete headline correctly but gets zero
because OCR appends small marginal lettering. Before authorizing training,
audit headline-only recognition separately from extra-text penalties using
the existing scorer's line-selection policy and saved line-level transcripts.
The result is evidence of some visible headroom, not a qualified reward yet.

#### Title-policy audit (no new generation)

Replayed all 32 saved recognition transcripts through the production scoring
functions in `outputs/ocr_title_policy_probe.py`; all-text decisions matched
the saved scores exactly. Results are persisted in
`outputs/ocr_spelling_title_policy_probe_20260909.json`:

- All text: mean 0.835065, 12 exact.
- Best contiguous lines: mean 0.958344, 14 exact.
- Best contiguous lines with existing duplicate threshold 0.5: mean 0.943501,
  still 14 exact. This is a diagnostic policy comparison, not a chosen recipe.

Independent visual inspection of all 32 originals found at least 15 with
clear text defects: 12 primary-title spelling/character errors and three
additional repeated/garbled headings. The conservative list is
`p00_s00/s01/s03`, `p01_s02/s03`, `p02_s02`, `p03_s02`,
`p04_s00/s01/s02`, `p06_s00/s01/s02/s03`, and `p07_s01`.
Ambiguous stylized characters in `p05_s01` and `p07_s02` were not counted.
Illustrated scenes are generally retained; the architecture prompts have
odd character/scene scale, which must not be mistaken for a title defect.

**Headroom interpretation correction:** character-normalized reward near 0.96
does not imply 96% of titles are correct. The Step 0 mean-score heuristic must
not reject this candidate on that basis: 14/32 exact recognition and at least
15/32 visually faulty images establish meaningful title-level headroom.
This does not yet establish learnability or unseen-prompt improvement.

The remaining reward issue is concrete: `p02_s02` gets 1.0 under both
headline policies despite a visibly repeated ANNIVERSARY. The existing
duplicate guard compares each leftover line to the *whole* two-line target;
a repeated fragment is too short to reach the 0.5 similarity threshold.
Do not merely lower that threshold to fit this one image, or call headline
selection alone a safe training reward. Whole-image scoring catches this
defect but also assigns a disproportionate zero to a correct headline with
small marginal text (`p02_s01`). No production reward changes were made.

#### Wrapped-line duplicate correction

The subsequent production change extends the existing opt-in duplicate guard
to compare unselected lines against selected, exact target-backed fragments
as well as the complete target. Fragment similarity uses the reference fragment's
length, matching target normalization; a longer candidate cannot dilute its
insertion errors (regressions cover THE/TIME and A/AT). No new
reward class, configuration field or model-specific preset was introduced.
The default policy and guard-disabled headline scoring remain unchanged.

Replaying the same 32 transcripts after the change produced exactly one score
change: `p2s2`, the repeated ANNIVERSARY case, 1.0 -> 0.5. The other 31 guarded
scores stayed unchanged; guarded mean is 0.927876 with 13 exact. Evidence:
`outputs/ocr_spelling_title_policy_fragment_fix_v2_probe_20260909.json`.
The OCR test module passed (26 passed, one opt-in test skipped), including
repeated first/second wrapped lines, unrelated text and low-confidence text.
This closes the observed loophole, not all possible reward hacks: segmentation,
short fragments, recognition errors and content preservation still require
validation on rollout images before claiming a usable training reward.
In particular, a misspelled selected fragment is not a reference: selecting
PHILHARMONIC + ANNIIVERSARY can still miss an extra ANNIVERSARY. Do not claim
this conservative guard handles arbitrary repeated or incorrectly recognized
fragments, or use it as the sole safety argument for a training run.

The next revision replaces exact-substring fragment recovery with the existing
Levenshtein dependency's alignment of the selected span to the target. Wrapped
references are recovered from target characters, not copied from recognized
misspellings; entirely unmatched pieces do not create fragment references.
Insertion/deletion/substitution regressions now cover the previously missed
ANNIIVERSARY + ANNIVERSARY case. OCR tests: 29 passed, one opt-in test skipped.
The existing 32-image scores are unchanged from the preceding correction;
replay evidence is `outputs/ocr_spelling_title_policy_alignment_probe_20260909.json`.
This closes the listed misspelled-fragment example, but does not establish
general recognition accuracy or eliminate short-fragment ambiguity.

The paired SDE/ODE screen is now launched (not training):

```bash
OMP_NUM_THREADS=4 .venv/bin/python -u outputs/craft_transfer_probe.py \
  --manifest outputs/ocr_spelling_headroom_probe.jsonl --reward ocr \
  --samples 16 --output outputs/ocr_spelling_transfer_probe_20260909
```

This reuses the one-shot sampler rather than adding a production experiment
entrypoint. Eight development prompts x16 paired seeds =256 images; OCR uses
best contiguous lines, substring credit disabled, duplicate threshold 0.5.
WD scores are retained only as diagnostic character/background proxies.
Completion requires the final summary and visual audit; neither source-test
success nor a live generation process qualifies the reward for long training.

Review then corrected trailing insertion recovery and split replacement spans
across OCR line boundaries rather than assigning the whole replacement twice.
Tests now pass 30 with one skipped; the explicit PHILHARMONIC / A / AT case
also detects the last line as a duplicate. The already-running paired screen
loaded the preceding alignment revision: its PNGs/transcripts remain useful,
but its saved guarded scores must be replayed with the final scorer after
completion before using rank statistics. Do not mix live-process semantics
with the subsequently edited source or restart generation just to rescore.

The transcript replay tool now accepts `--expected-count` (32 by default),
rejects incomplete/duplicate sample grids, and saves source-code/transcript
SHA256 plus original score/policy alongside all rescored variants. The final
32-image replay is `outputs/ocr_spelling_title_policy_final_probe_20260909.json`.
The paired run will require `--expected-count 256`; its original policy is
guarded headline scoring, so an all-text equality assertion must only apply
when the saved policy actually was all-text. Original records stay untouched.

The first completed paired group (EXTRAORDINARY EXPEDITIONS) has three SDE
score levels and four ODE levels; preliminary rho -0.241698. This is a warning,
not a final eight-prompt result or proof of unlearnability. Visual inspection
of `p00_s03_sde.png` vs `p00_s03_ode.png` confirms genuine title errors in the
former and correct text in the latter, with substantial illustrated scenes
retained in both. Do not generalize that one pair to the remaining images.

Before any spelling training, froze 16 additional illustrated-poster prompts
at `outputs/ocr_spelling_heldout_probe.jsonl`, SHA256
`0786168c608c53bb1c9467506ccb1d3f51294347e36bba80a00b2c4ef3163143`.
Both prompt text and complete target titles are disjoint from the eight
development/training candidates. The shared prompt loader accepts all rows;
maximum token lengths are Qwen 63 / T5 78. These are new titles/scenes within
the same poster template, not evidence of broad stylistic generalization.
Do not use these rows to choose rewards or tune training. If a short run is
authorized by the screen, compare base and the preselected final checkpoint
with native ODE, four paired seeds per prompt, seed 58031, and retain every
image. Report prompt-cluster uncertainty, exact-title recognition, visual
spelling/repetition errors and scene preservation separately. A favorable
mean character score alone does not meet the visible-improvement objective.

The 256-image paired screen completed successfully. Final-source transcript
replay and full image-hash/grid verification are saved in
`outputs/ocr_spelling_transfer_rescored_probe_20260909.json`; all guarded scores
equal the live run's scores despite the later boundary corrections. SDE mean
0.946060, 67/128 exact; ODE mean 0.953855, 66/128 exact. Mean within-prompt rank
correlation is 0.140094; SDE distinct counts are [3,3,4,2,4,3,4,5].

Decision: these are unfavorable screening diagnostics, not clearance for a
long run. Nevertheless, discrete edit scores do not make policy gradients
undefined, and latent rank transfer is not a theorem about policy improvement
(the prior sharpness control already demonstrated that limitation). Run one
bounded ten-update optimizability test, explicitly as a test of these weak
proxies rather than calling Step 0 passed. Do not extend it based on training
reward. Final checkpoint ODE on the frozen held-out prompts remains required.

Launch identity: `outputs/anima_ocr_spelling_probe_20260909`, existing
`experiment/anima_preview3/online_grpo` + `reward=ocr`, LoRA, CPU PP-OCRv6,
guarded contiguous-title policy, noise 0.7, random 25% timesteps, lr 0.0003,
KL 0.04, eight fixed prompts x16 samples, ten updates, save every five.
The sampler had exited before launch; the other user's GPU process was left
untouched. Launch is not evidence that a training update has completed.
Startup caveat: the inherited `global_std=true` is evaluated over each streaming
microbatch of two prompt groups, not over all eight groups. This run retains
the earlier probe setting for comparability; do not describe it as full-batch
global normalization. The launcher emitted this warning explicitly. Driver
1159626 started the online recipe; completed-update evidence is still pending.

The first completed update is now recorded as epoch 0 in the run's
`metrics.csv`: reward mean 0.9576, std 0.1045, grad norm 0.001062,
pre-update logprob max difference 0.000000, pre-update clip fraction 0.0000,
eight trained prompt groups, advantage zero rate 0.0000. These are rounded
CSV values, not an independent exact-zero tensor comparison. This establishes
a functioning first update, not a reward trend or visual improvement. The
same bounded run remains active; checkpoint-10 and held-out ODE are pending.

At five completed updates, reward means are
[0.9576, 0.9663, 0.9633, 0.9522, 0.9496]: no sustained improvement so far.
All five recorded pre-update difference/clip values remain zero at CSV
precision, with nonzero gradient norms. `checkpoint-5` is saved: metadata
global_step=5, checkpoint.pt size 330937455 bytes matches the metadata, and
the exported safetensors adapter opens with 448 tensor entries. This is an
intermediate recovery artifact, not a reward-selected evaluation checkpoint;
the fixed final comparison remains base vs checkpoint-10.

The bounded run completed all ten updates and exited successfully on September
9 at 20:48:36 (environment time). Reward means across the ten updates are
[0.9576, 0.9663, 0.9633, 0.9522, 0.9496, 0.9442, 0.9524, 0.9593, 0.9519, 0.9413]:
there is no sustained training reward improvement. All CSV numeric entries are
finite. Checkpoint-10 metadata reports global_step=10; checkpoint.pt is
330937455 bytes as declared, and its LoRA safetensors opens with 448 entries.
All four preregistered data/scorer hashes were rechecked and match.

The preregistered held-out ODE evaluation started after training exited and
the GPU process list was empty. Output:
`outputs/ocr_spelling_heldout_ode_probe_20260909`, base vs checkpoint-10,
16 held-out prompts x4 paired seeds (seed 58031), 128 images total.
Generation has started; evaluation results and visual conclusions are pending.

Held-out evaluation subsequently exited successfully with all 128 generated
images and a published report. OCR mean is 0.971994721 (base) vs 0.974076621
(checkpoint-10), paired prompt-level delta +0.002081900, bootstrap 95% CI
[-0.010097777, +0.015278224]. Of 16 prompt means, five improve, six decline,
and five tie. This is not reliable numerical improvement. The visual audit
and independent image-grid/hash verification are still pending; no claim of
visible benefit or goal completion follows from this small mean increase.

The image-grid/hash audit now passes: exactly 128 unique cells (16 prompts,
four seeds, base/ck10), every SHA256 matches, every source is 512x512, and
every pair shares its prompt and seed. Fixed-label, unresized pair sheets are
in `fixed_model_mosaics/` (32 sheets; left base, right ck10). The one-shot
renderer/verifier is `outputs/ode_pair_review_probe.py`; no production renderer
or experiment preset was added.

Two independent reviewers covered all 64 pairs without reading OCR scores
first. This was label-visible inspection, not blind evaluation or a human
gold-standard transcription. Conservative title-only judgments: six clear
or partial improvements, five regressions, four ambiguous/mixed, and 49 ties
(including persistent misspellings). Exact OCR scores decline from 44/64 to
40/64; sample-score changes are seven improvements, nine regressions, 48 ties.
These counts describe different measures and must not be conflated.

Concrete visual counterexamples, indexed as in the generation manifest:
- p01s2 removes an extra gibberish line but changes correct UNFORGETTABLE to
  UNFORGETABLE; the scalar score nevertheless increases.
- p02s1 loses a T in TERRITORIES; p02s2 repairs the first title word.
- p07s1 replaces a repeated CELEBRATION with a prominent gibberish box, not a
  clean title. Its OCR score increases from 0.5 to 0.833333.
- p11s0/s2 introduce duplicated or malformed P/R strokes in INCOMPREHENSIBLE.
- p13s3 deletes the sculptor and chair entirely, leaving the statue/courtyard;
  both titles were already correct. This was directly rechecked by the main
  agent on the native-pixel pair sheet.
- p09s2 deforms the vase; some other samples gain tiny extraneous lettering.
  Most illustrations retain substantial scenes, but that does not erase the
  specific content losses. Fonts often become heavier without fixing spelling.

Decision: this checkpoint fails the requested visible-quality/held-out benefit
criterion. Do not extend this run automatically or call a small mean increase
success. Three separate explanations remain distinguishable: a no-op weight
update is contradicted by visible changes; failure to optimize vs failure to
generalize still needs the trained-prompt ODE diagnostic; and reward/visual
disagreement is directly demonstrated above. The next diagnostic is the same
base/ck10 ODE comparison on the eight development prompts, not another training
run or retuning the held-out set. Another workspace's GPU job appeared after
the completed evaluation, so no competing generation process was launched.

After the GPU process list became empty, completed the trained-prompt ODE
diagnostic using the same generic evaluator and checkpoint, development
manifest, four paired samples, seed 68031:
`outputs/ocr_spelling_train_ode_probe_20260909`. All 64 images/32 pairs and
SHA256 hashes pass verification; the evaluator exited successfully.
OCR mean falls from 0.976288496 to 0.959917508; paired prompt delta
-0.016370989, bootstrap 95% CI [-0.030606561, -0.001443452]. Two of eight
prompt means improve, five decline, one ties. Exact OCR falls from 20/32 to
12/32. This is not merely a held-out generalization failure: the final policy
also fails to improve clean ODE on the training prompts in this diagnostic.
It does not establish that all OCR training recipes are impossible.

A CPU-only recognition recheck on held-out p01s0, p02s0, p07s1, p15s0
(both arms) ruled out a proposed explanation for these specific examples:
the OCR engine does recognize missing letters, returning UNFORGETABLE,
UNDISCOVRED/UNDISCOVEED, and CRYSTALOGRAPHIC instead of silently correcting
them. High character-average scores are not exact-title success. On p07s1,
base recognition contains CELEBRATION twice; ck10 recognition contains an
intervening `[6图]` line instead. A reward increase on that pair is not a
clean visual repair. No scorer or training parameters were changed during
these checks. All training and evaluation processes for this run have exited.

Post-failure shared-path audit (read-only; no further training):
- Ran the existing continuous GRPO and denoise flow-matching tests: 32 passed.
- An isolated CPU check used the real SDE transition and GRPO loss with 16
  samples, reward equal to the sampled scalar transition, global_std enabled,
  noise 0.7. One SGD step raised the transition mean by 0.002846846 and the
  advantage-weighted log probability by 0.092952795. Initial mean loss was
  approximately zero but the velocity gradient was 0.762207925. This validates
  the local update sign, not the entire Anima learning problem.
- Independently traced artifact-ID validation/reordering in rewards/inference,
  collector flatten/split ordering, and synchronized trajectory/advantage
  slicing. No identity mismatch was found in the inspected paths.
- Traced begin/backward/finish and strategy parking: gradients survive CPU
  parking, four microbatches accumulate before one optimizer step, and LoRA
  trainables are FP32. No evidence of zeroing accumulated gradients or
  low-precision update rounding was found. The already documented microbatch
  global_std limitation remains, not a newly discovered failure cause.

Upstream comparison was checked directly against the author repository:
[GRPO recipes](https://github.com/yifan123/flow_grpo/blob/main/config/grpo.py)
and [base defaults](https://github.com/yifan123/flow_grpo/blob/main/config/base.py).
The SD3 OCR recipe inherits lr=3e-4 and noise=0.7, uses 24 images per prompt,
timestep_fraction=0.99 and EMA enabled. Our 16-image, random-25%-timestep,
ten-update Anima probe is not an equivalent reproduction. These cross-model
differences are hypotheses, not proof that a particular knob caused failure;
do not label the shared implementation broken or silently extend the failed
probe on this evidence. The current local tests and source audit do not rule
out subtler estimator/optimization issues or establish reward learnability.

Alternative-method feasibility check, before adding training code:
- The offline DPO trainer is generic, but the shipped complete entrypoint is
  Wan-specific (`vrl/scripts/families/wan_2_1/train_dpo.py`) and rejects other
  families. The shared online factory rejects diffusion_dpo. Anima is not a
  drop-in supported DPO run; no new family script or preset was added.
- DiffusionNFT requires `diffusion_nft_prepare_transformer_input`; Anima does
  not implement that hook. Its availability elsewhere is not Anima support.
- `grounded_ocr` multiplies the existing character score by a binary visual
  guard; it does not expose a denser target-conditioned recognition score.

A new candidate, not an endorsed reward: ARC/SOLACE intrinsic self-confidence.
Primary method source: [arXiv v1](https://arxiv.org/html/2603.00918v1#S4),
with the later title/naming on the
[CVPR page](https://openaccess.thecvf.com/content/CVPR2026/html/Kim_Improving_Text-to-Image_Generation_with_Intrinsic_Self-Confidence_Rewards_CVPR_2026_paper.html).
It re-noises generated clean latents with shared antithetic probes and scores
negative-log noise-reconstruction error using conditional, unguided model
predictions. The paper explicitly reports blank/textureless collapse when
optimizing too much of the trajectory, motivating a later-step training
window. Published success on other models is not Anima evidence.
Next action is a read-only scoring feasibility/calibration check, including
the blank/blur/content-loss counterexamples, before any production integration
or training. Do not conflate OCR confidence with generator self-confidence,
and do not change the frozen OCR held-out protocol to make this candidate win.

Frozen-base self-confidence calibration completed:
`outputs/anima_self_confidence_probe_20260909` (one-shot script
`outputs/anima_self_confidence_probe.py`). Scored the existing vase-deformation
pair p09s2 and sculptor-deletion pair p13s3 with two shared probe seeds, eight
antithetic probes each, all 20 native schedule steps; also reported the last
12 steps separately. These previously inspected images are now calibration
material for this candidate, not fresh held-out evidence. A subsequent trained
candidate would require a new untouched evaluation set.

All eight records are finite, source PNG hashes match, and the process exited
successfully. Independent source review found the intended flow velocity,
sigma domain, shared probes, conditional-only prediction, and mean-MSE then
negative-log aggregation intact. Last 12 steps means a schedule-index suffix,
not sigma <= 0.6. This uses PNG posterior-mode re-encoding and a frozen base
scorer, not original rollout latents or online ARC training.

Degraded-minus-base scores for the two independent K=8 probe sets:
| Counterexample | All 20 steps | Last 12 steps |
| --- | --- | --- |
| Deformed vase p09s2 | -0.005445 / -0.007607 | -0.080516 / -0.089271 |
| Deleted sculptor p13s3 | +0.039124 / +0.045093 | +0.045741 / +0.056133 |

The scorer consistently penalizes this vase deformation but prefers the
image missing the requested sculptor. Therefore it is not cleared as a sole
general-quality/content-preservation reward. Do not launch training on this
calibration or treat reward self-confidence as verified visual quality. The
counterexample does not refute the paper's online method; it rejects the
unqualified frozen-base reward interpretation tested here. No production
module, reward preset, model weight, or image content was changed.

Follow-up calibration prepared, not yet launched: compare each image's
denoising error under its intended prompt against an explicit counterfactual
(deformed vase; empty statue courtyard without the sculptor). This follows
the conditional-error comparison idea of
[Diffusion Classifier](https://openaccess.thecvf.com/content/ICCV2023/html/Li_Your_Diffusion_Model_is_Secretly_a_Zero-Shot_Classifier_ICCV_2023_paper.html),
not a claim of exact likelihood evaluation for Anima's flow model.
The same one-shot script now accepts `--counterfactual-prompts` and `--output`;
original scoring defaults remain unchanged. Counterfactual text is isolated
in `outputs/anima_self_confidence_counterfactual_probe.json`. Ruff and CLI help
checks pass. The original calibration files remain untouched.

Pending command, only when the shared GPU is available:
```bash
OMP_NUM_THREADS=4 .venv/bin/python outputs/anima_self_confidence_probe.py \
  --counterfactual-prompts outputs/anima_self_confidence_counterfactual_probe.json \
  --output outputs/anima_counterfactual_confidence_probe_20260909
```
Pair the new per-step errors with the original records by image hash and probe
seed. Report counterfactual-MSE minus intended-MSE (positive favors intended
description), and preserve both repeats rather than selecting a window after
seeing results. It must favor the intact sculptor image over the missing-one
image before broader screening; passing two chosen pairs is not sufficient
qualification for training. At preparation time another workspace's live
Python/Ray job owned the GPU; no competing scorer was launched or queued.

The competing job subsequently exited; with an empty GPU process list,
ran that exact command. Counterfactual scoring exited successfully with eight
finite records. Matched all records to the original by image hash/probe seed
and verified identical sigma schedules. The difference below is reconstructed
MSE(counterfactual) minus MSE(intended); positive favors the intended caption.

| Image | All-step differences, two probe seeds | Last-12 differences, two probe seeds |
| --- | --- | --- |
| Intact vase | -0.00000134 / -0.00002855 | +0.00000968 / -0.00003900 |
| Deformed vase | +0.00027771 / +0.00027003 | +0.00007747 / -0.00000548 |
| Sculptor present | -0.00116799 / -0.00117790 | +0.00069669 / +0.00081180 |
| Sculptor missing | -0.00281058 / -0.00271498 | -0.00054897 / -0.00041137 |

Interpretation: the contrast consistently ranks the intact sculptor scene
above the subject-deleted scene; the predeclared last-12 window also yields
the expected positive/negative caption decisions. This is a useful local
content-alignment signal, not evidence of general reward reliability. It fails
the vase-quality distinction, and the all-step margin prefers the empty-scene
caption even when the sculptor is present. No post-hoc window selection or
threshold tuning is warranted. Broader object/subject-presence calibration
on complete groups is the next qualification question, not another OCR run.
Future evaluation must use fresh prompts because these inspected pairs are
now development material. No training was launched.

Complete-group presence headroom check: independently inspected all BASE
samples for prompts 00, 02, 03, 13 (16 images across eight native-pixel sheets).
Requested people are visible in 16/16, absent 0, uncertain 0. Small or occluded
people count as present; statues do not. This subset gives no observed base
headroom for a person-presence training objective. It does not establish zero
failure globally. The deleted-sculptor pair is a useful degradation guard
counterexample, not evidence that the base needs that specific repair.

Next calibration launched, no weight updates: conditional spelling contrasts
on complete BASE groups p01 and p15 (eight images). Previously reviewed title
labels: p01 samples 0/1/3 omit a T, sample2 is correct (but has extra lettering);
p15 samples 0/1/2 omit an L, sample3 is correct. This checks headline spelling
only, not overall image quality. Both comparison captions were programmatically
verified to differ from the originals solely by the specified one-letter
deletion. Foils: `outputs/anima_spelling_counterfactual_probe.json`.

The existing one-shot scorer now accepts `--base-prompt-groups`, verifies all
four samples per selected prompt, and records selected image paths. Ruff
checks pass; no production module or YAML was added. First command is live:
```bash
OMP_NUM_THREADS=4 .venv/bin/python outputs/anima_self_confidence_probe.py \
  --base-prompt-groups 1 15 \
  --output outputs/anima_spelling_confidence_intended_probe_20260909
```
Once it exits and the GPU is available, score the same groups with
`--counterfactual-prompts outputs/anima_spelling_counterfactual_probe.json`
and output `outputs/anima_spelling_confidence_foil_probe_20260909`.
Retain the original two probe seeds, K=8, all-step and last-12 summaries;
compare counterfactual-MSE minus intended-MSE by image hash/seed. Correct
examples should outrank incorrect examples within each group, reproducibly;
do not choose a new window after seeing the values. These eight images are
calibration examples, not future held-out evidence.

Both spelling-calibration processes now completed successfully: intended and
foil each have 16 records (eight images x two probe seeds), all finite and
paired by image SHA256 with identical schedules. The correct headline sample
is p01s2 and p15s3. Rankings by counterfactual-MSE minus intended-MSE:

| Group | Correct sample rank, all 20 steps | Correct sample rank, last 12 steps |
| --- | --- | --- |
| p01, both probe seeds | 1 / 4 | 1 / 4 |
| p15, both probe seeds | 2 / 4 | 1 / 4 |

Suffix correct-sample margins are +0.00032942/+0.00040865 (p01) and
+0.00038516/+0.00027539 (p15). This is reproducible within this small,
already inspected calibration set. Some incorrect samples also have positive
margins, so a zero-threshold spelling classifier is not established. Preserve
the all-step failure on p15; the predeclared suffix is a candidate, not a
post-hoc universal success claim. No weight update occurred.

This justifies checking more words/error types before integration, not training
from two repeated-letter deletion examples. The prompt-specific foil is a
material requirement: this is not a drop-in universal aesthetic score, and
constructing foils from held-out answers would invalidate a future evaluation.
Any eventual training must use development-only foils, independent quality
checks, and a newly frozen untouched held-out set.

### Expanded spelling calibration: visual preflight, not yet scored

Prepared `outputs/anima_spelling_expanded_calibration_probe.md` and
`outputs/anima_spelling_expanded_counterfactual_probe.json`. They select
complete base groups 2 and 5, with the existing scorer unchanged. Verified
programmatically that each foil deletes exactly one headline letter and
changes no scene wording. Keep both noise seeds, K=8, all 20 and last 12 steps.

Independent visual review corrected an important initial label: group 2
sample 3 says UNDISCCOVERED (extra C), not the correct UNDISCOVERED. Group 2
therefore has one correct headline (sample 1), two deletions (samples 0 and 2),
and one insertion (sample 3). The foil UNDISCOVRED matches only sample 0;
samples 2 and 3 explicitly test whether the contrast generalizes beyond its
named typo. Require reporting every correct/incorrect pair, not merely top-1.

Group 5 sample 0 has ambiguous overlapping initial strokes: the apparent
missing I is not a reliable binary label. Samples 1/2/3 are visually correct.
Retain sample 0 as uncertain, excluded from binary success claims. This group
cannot by itself establish discrimination. Inspected base groups 6 and 11
had no identified headline errors and were not counted as discrimination
evidence. All these readings are model-assisted, not independent human gold.

Recomputed the earlier two-group suffix results: correct-minus-best-incorrect
margins are 0.00029657/0.00037055 (group 1) and 0.00018715/0.00014335
(group 15). In each group and each probe seed, two incorrect images also
score positive. Ranking is the candidate signal; zero is not a validated
classification threshold.

Expanded GPU scoring has NOT launched. At the latest live check PID 1331015
was running `/tmp/vrl_sd3_execution_precision_probe.py` from the other
workspace `/home/mingfeiguo/Desktop/vrl2/VRL`, holding about 11 GB. Recheck
process state before using the GPU; do not terminate or compete with that job.
The next scoring commands use the existing script with `--base-prompt-groups
2 5`, first intended captions, then the expanded counterfactual JSON, in new
distinct output directories. No production reward, training, or YAML was
added in this continuation.

The external GPU process subsequently exited. After verifying an empty
compute-process list before each launch, expanded intended scoring (session
9071) and foil scoring (session 37380) both completed with exit 0. Each has
eight images / 16 records in
`outputs/anima_spelling_expanded_intended_probe_20260909` and
`outputs/anima_spelling_expanded_foil_probe_20260909`. Verified matched
image hashes/seeds, identical schedules, and finite results.

Group 2 passes: samples rank `[1, 3, 2, 0]` in both repetitions and both
windows, so the sole correct headline beats all three incorrect ones,
including the two errors not named by the foil. Its correct-minus-best-wrong
margins are 0.00011595/0.00017623 (all steps) and 0.00021410/0.00062509
(suffix). These repeated comparisons are not independent new word groups.
Group 5's uncertain sample ranks last in both repetitions/windows; do not
count that as a binary success. Full results and frozen labels are recorded
in `outputs/anima_spelling_expanded_calibration_probe.md`.

Three labeled development groups now pass suffix top-1 under both probe
seeds. This supports further qualification, not a trained-model improvement
claim. No weights changed. The main remaining risks are scene-quality and
content-deletion confounds, narrow word/error coverage, and the need for
prospective foils plus a fresh untouched evaluation set.

### Scene-regression qualification and integration boundary

Prepared `outputs/anima_spelling_scene_guard_probe.md` and
`outputs/anima_spelling_scene_counterfactual_probe.json`, before scoring.
Use the original four-image selection: base/ck10 group 9 sample 2 (distorted
vase) and group 13 sample 3 (sculptor deleted). Both arms' headlines appear
correct. Rechecked the original intended-score manifest and all image hashes;
its eight records can be reused after matching the new model protocol and
schedules. Each new foil changes only one headline letter, verified with
sequence alignment. This is a naturally occurring regression check, not a
controlled proof that content deletion itself causes a score change: fonts,
layout, and other image details also differ. A preference for the degraded
image would nevertheless disqualify a claim of standalone quality protection.

A read-only integration audit found that frozen-base scoring can reuse the
generator during LoRA training: `vrl/generation/steps/denoise/loop.py` already
uses `model.disable_adapter()` for reference forwards, and Anima exposes
`diffusion_pretraining_prediction`. This does not recover the initial model
after full-parameter updates. The full encoder/VAE/model lives in the
generation worker; `AnimaReplayModel` lacks these components. Collector
offloads generation before activating shared-GPU reward execution, so model
sharing is not a YAML-only reward plugin change. Future shared-model scoring
would need execution inside the generation worker's active lifecycle, with
the reward layer retaining score validation/composition ownership. Keep the
existing worker/runtime/reward protocol boundaries; no production module,
new YAML, or architectural rewrite is justified before qualification.

Also preserve the current PNG-posterior-mode definition: switching to online
terminal latents would be a changed scorer and require a new parity check.
The existing independent probe already owns only one frozen model and is
the appropriate qualification path. At this preparation point the external
SD3 process PID 1354449 was confirmed live, so scene-foil GPU scoring had not
started; recheck ownership before launch.

Scene-foil scoring subsequently completed (session 75946, exit 0), after an
empty compute-process list was observed. Output:
`outputs/anima_spelling_scene_foil_probe_20260909`, eight records / four
images. Verified the full model protocol and manifest match the reused
intended run, plus all image/seed pairs, sigma schedules and finite scores.

The spelling contrast fails as standalone quality protection: the distorted
vase wins both repetitions and windows (degraded-minus-base suffix
+0.00013898/+0.00010308; all steps +0.00016452/+0.00015492). The sculptor
deletion pair changes preference between noise repetitions (suffix
-0.00017706/+0.00010081; all steps -0.00009396/+0.00002761). Both headlines
appear correct. Fonts/layout also change, so these pairs do not identify a
causal deletion mechanism. Preserve the positive spelling-ranking results,
but do not promote them to a general-quality claim. Full frozen protocol
and results: `outputs/anima_spelling_scene_guard_probe.md`. No weights changed.

### Archived OCR objective audit: exact-match candidate, not a fix claim

Re-read `_best_candidate_score` in `vrl/rewards/models/ocr.py` and group
centering in `vrl/algorithms/advantages.py`. Current OCR reward is normalized
edit similarity (then optional duplicate/extra-line penalties). GRPO rewards
above the prompt-group mean receive positive advantages. This is intentional
relative optimization, not evidence of a sign bug.

Audited all 1,280 JSON scoring records from
`outputs/anima_ocr_spelling_probe_20260909/ocr_debug`, grouped by the actual
sample-ID prefix before `:sample:`. Verified 80 complete groups of 16.
Compared selected OCR text to normalized target using the existing
`normalize_ocr_text`, and verified agreement with raw-score equality to one:

- 549 selected headlines exactly match; 731 do not.
- 279 of the 731 OCR-mismatched headlines have reward above their own group
  mean and therefore positive pre-clipping advantage (38.2% of mismatches).
- No exact headline in these records was demoted by the duplicate penalty.
- Seven records have a duplicate penalty; the 279 count is not an artifact of
  classifying correctly spelled but duplicate-penalized headlines as misspelled.
- A counterfactual binary full-credit reward has nonzero within-group
  variance in 76/80 groups (95%); four groups contain no full-credit samples,
  and no group is entirely full-credit. This is archived-policy support,
  not a guarantee about future rollout distributions.

Example: MECHANICAL SYNCHROIZATION receives 0.96 in a group averaging
0.9475. Dense similarity can reinforce a near miss relative to worse misses.
That can be useful for a curriculum; it does not prove it caused the observed
training failure. Binary exact-match would align the update signal with
whole-headline success more directly and is not overwhelmingly sparse on
this archive. It remains a distinct candidate requiring an explicit bounded
experiment, not a diagnosed implementation fix or a trained improvement.

Do not require a specialist reward to solve every quality dimension. The
intrinsic spelling contrast's scene counterexamples reject its use as
standalone quality protection, not the possibility of specialist spelling
training with independent quality-preservation checks. No production scoring
semantics or model weights changed during this audit.

### Exact-match experiment implementation and launch

Implemented `ocr_match` as a second output of the existing `OCRRewardModel`,
selected through `OCRReward(score_key="ocr_match")`. The reusable OCR preset
keeps `score_key: ocr` as its default. No new reward class, Anima-specific
YAML, scoring-policy field, or shared-model intrinsic execution path was added.
Exact scoring requires selected normalized text equality and no configured
duplicate/extra-line rejection. It never grants substring credit. For video,
the denominator includes every sampled frame, unlike the historical positive-
frame-only similarity mean. The existing v5 debug sidecar gains the additive
`aggregate_match_score` field; historical `aggregate_score` and filename score
remain edit similarity. Training metrics `r_ocr` use the configured score key.

Kept the wrapper as a transport/public API boundary and the model as scoring
owner. The local two-value score-key validation is an output-schema boundary,
not a business vocabulary. Existing policy normalization, duplicate guard,
registry, and resource ownership remain unchanged.

Verification: touched-file Ruff check/format/check passed; OCR + grounded OCR
tests report 37 passed, one optional real-engine test skipped. Replayed all
1,280 archived recognized-line sets through the actual reward API using an
engine fake: every exact score equals the archived full-credit indicator,
with 549 positives. This validates scoring semantics, not OCR recognition
accuracy. Independent read-only review found no blocking issue. Config resolve
also passes using the previous resolved run plus the exact score-key override.

Frozen experiment: `outputs/anima_ocr_match_probe_20260909/evaluation_plan.json`.
Sixteen newly authored held-out titles in
`outputs/ocr_match_fresh_heldout_probe.jsonl` are distinct from training and
previous held-out targets. Its hash is
`de6b54265219aeeaa954d8a13387cc6be942156ab8d61592560fe8ea3a84389c`.
The plan also freezes training/scorer/wrapper/normalizer hashes, complete
launch/evaluation argv, fixed final checkpoint-10, four samples per prompt,
native ODE, seed 78031, all-pair visual review and prompt-cluster uncertainty.
The task remains poster spelling, not general aesthetic improvement.

After the GPU compute-process list was verified empty, launched the bounded
ten-update run at 21:54:26 on September 9 (environment time), session 88442,
driver PID 1370224, owned Ray generation worker PID 1371124. Same eight prompts
x16, lr 0.0003, KL 0.04, noise 0.7 and random 25% timestep recipe as the dense
OCR run; only optimization-score selection changes. Global std remains per
two-prompt streaming microbatch and the startup warning is preserved.
The first 16-image rollout group completed at 21:55:26. At this recorded point
no complete optimizer update or improvement has been established. Keep polling
the same live session; do not restart from an observation timeout.

The first exact-match optimizer update is now recorded in `metrics.csv`
(epoch 0): 8 groups x16, reward mean 0.5469, reward std 0.4595, grad norm
0.003175, advantage zero rate 0.0000, pre-update logprob difference 0.000000
and pre-update clip fraction 0.0000 at CSV precision. All numeric fields are
finite. Independently counted 70 exact positives among the first 128 OCR
sidecars (70/128 = 0.546875), agreeing with the rounded training mean. The
precision-drift guard reports `violated=false`, finite values and exact zero
for its recorded maximum logprob difference. Frozen data/code hashes were
rechecked during the run. Session 88442 remains live and has begun the next
round's generation. This proves a functioning update, not an improvement
trend or a successful held-out comparison. Other workspace processes changed
shared GPU occupancy during the update; they were not stopped or modified.

### Exact-match run interrupted by node-memory contention

Session 88442 is now terminal (exit 1), not live. It completed three updates:
exact positives 70/128, 69/128, 67/128; gradients 0.003175, 0.002415,
0.002326. All completed-update metrics are finite and recorded pre-update
parity/clip values are zero. Fourth-update generation began, but no fourth
optimizer update completed.

Authoritative `run_verdict.json` reports Ray `OutOfMemoryError`: node memory
87.80/91.87 GB (95.5675%) exceeded the 95% guard. Ray killed generation worker
1371124 (9.16 GB); the training driver used 9.38 GB. The same raylet event's
top-process table lists external probe PIDs 1418892 and 1418894 at 22.78 and
22.77 GB, the latter named `WeightDeliveryProbeWorker`, plus its driver
1418394 running `vrl.scripts.perf.weight_delivery_probe` (4.68 GB).
This is node RAM contention, not a CUDA allocation or reward/parity failure.
Ray's object store reported zero used objects/bytes, so its configured
capacity must not be mistaken for the observed allocation cause. Full event:
`/tmp/ray/session_2026-09-09_21-54-33_033096_1370224/logs/raylet.out`.
Ray documents this node-wide protection at
https://docs.ray.io/en/latest/ray-core/scheduling/ray-oom-prevention.html.

Verified all four training/probe PIDs exited. Host available memory rebounded
to about 82 GiB. A separate OpenWorld job, PID 1413976 with cwd
`/home/mingfeiguo/Desktop/OpenWorld`, still owned GPU memory, so no new GPU
run was launched. No external process was terminated and memory guards were
not disabled.

Recovery cannot resume weights: exhaustive output-file inspection found no
checkpoint/pt/safetensors artifact, consistent with save_freq=5 and the
`vrl/scripts/common/online.py` save-after-complete-update path. The exception
path shuts down and does not publish a partially completed checkpoint. Keep
all failed-run metrics/images; do not claim its weights were retained.

Prepared, NOT launched:
`outputs/anima_ocr_match_retry1_probe_20260909/evaluation_plan.json`.
It preserves reward, data hashes, optimizer recipe, maximum ten updates and
fresh held-out evaluation, but saves every completed update and uses new
training/evaluation output directories. It must restart from base, not from
the incompatible dense-OCR checkpoint. The three failed-attempt updates are
not included in the retry's ten. Frequent checkpoints mitigate lost work;
they do not solve cross-workspace RAM contention. Recheck host/GPU ownership
before launch and do not rerun while another workload owns the card.

The external GPU job then exited. Two subsequent compute-process checks were
empty and host available memory was about 85 GiB. All frozen hashes and the
retry config (ten updates, save_freq=1) passed verification. Launched retry
session 11566 at 22:25:07 (environment time), starting from base in the new
retry directory. Startup loaded the replay model and retained the known
microbatch-global-std warning. No retry update has completed at this point.
Keep monitoring session 11566; session 88442 is permanently terminal and must
not be polled as if still running. The original failed-attempt verdict remains
unchanged. This recovery mitigates an observed infrastructure interruption;
it is not evidence of better reward or image quality.

### Retry also terminal: exclusive resource ownership is unresolved

Session 11566 exited 1 during generation-model loading, before any completed
update or checkpoint. This is a different immediate failure from the previous
node-RAM event: CUDA OOM at `AnimaModel.from_build` / `transformer.to`, with
only 14.62 MiB free. The exception lists competing PIDs 1423735 (12.89 GiB)
and 1424588 (16.58 GiB), plus CUDA contexts. Do not call this fragmentation
or infer that the unchanged batch-1 model cannot fit by itself.

Read-only process inspection identified surviving competing driver 1423646:
`/home/mingfeiguo/Desktop/vrl2/VRL/.venv/bin/python -m vrl.scripts.train
--config experiment/sd3_5/online_grpo_ocr ... trainer.total_epochs=2 ...
trainer.output_dir=outputs/repro/sd3_5_ocr_fp16_b`. Its generation worker
1424588 also has cwd `/home/mingfeiguo/Desktop/vrl2/VRL`. PID 1423735 had
already exited when inspected, so its ownership is not independently known.
Our retry driver 1423790 and worker 1425415 are gone. The failed verdict is
preserved in the retry directory, along with an empty metrics table; no
checkpoint or trainable-weight artifact exists.

The empty-GPU preflight was real, but is not a cross-workspace reservation:
another job can start during loading or training. Repeating momentary-idle
restarts is not a recovery strategy. Do not launch retry2, shrink the frozen
experiment, disable memory protection, or stop another session's process
without resource-priority coordination. A stable exclusive training window
is needed; user direction on Anima versus the concurrent vrl2 SD3 task is
pending. This is the second consecutive resource-contention interruption,
not evidence the reward failed. Goal remains incomplete; neither old live
session handle should be reused.

The next goal continuation revalidated the same contention: vrl2 driver
1423646 / generation worker 1424588 remained live (about 18 GiB on the
worker), while OpenWorld PID 1427359 used about 13 GiB on the same GPU.
Resource priority was still unanswered. Across three consecutive goal turns
the same missing cross-workspace allocation prevented safe completion:
node-memory interruption, CUDA-memory interruption, then confirmed concurrent
occupancy. CPU qualification is complete; the remaining training and native
ODE comparison require GPU execution. Mark the persistent goal blocked on
resource coordination, not complete or scientifically disproven. No further
restart or external-job termination is authorized by this continuation alone.

### User-prioritized Anima restart

The user explicitly selected Anima priority. At the new preflight, the GPU
compute-process list was empty and host available memory was about 85 GiB;
no conflicting process needed termination. Verified all frozen data/scorer
hashes and config, then created
`outputs/anima_ocr_match_retry2_probe_20260909/evaluation_plan.json` with the
same ten-update experiment, per-update checkpoints and fresh evaluation,
plus the explicit user resource-priority decision. Existing failed outputs
remain unchanged. Confirmed conflicts may now be stopped gracefully under
that priority, preserving outputs; this does not authorize killing unrelated
services or assistant sessions.

New live session: 13543, driver PID 1447915, owned Ray generation worker
1448338. Training started at 22:56:34 on September 9 (environment time).
Model loading succeeded; first 16-image rollout group completed at 22:57:26.
No complete update or checkpoint yet at this observation. The earlier
sessions 88442 and 11566 remain terminal. Monitor the new session and verify
checkpoint-1 publication before claiming interruption recovery is proven.
