# GOAL: find a reward that actually improves Anima, and prove it

You are working in `~/Desktop/VRL` (git repo, branch `main`). One RTX 5090, 32 GB,
shared with the operator's own jobs. The model is Anima (`cosmos-predict2-anima`,
a Cosmos-Predict2 anime image DiT) trained with Flow-GRPO-style LoRA RL.

**Objective.** Find a reward that (a) measures something an anime image model is
actually judged on and (b) provably moves this policy. Validate it on a small
fixed prompt set first, then train on the full set and prove the gain on
held-out prompts. Report negative results as negative.

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
