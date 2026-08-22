# SPRINT: Anima RL post-training on a single RTX 5090

Autonomous experiment log. Goal: a checkpoint that is **repeatably better than
base Anima on a fixed anime eval prompt set** — visual quality up, diversity and
prompt adherence retained — or an honest negative result with the strongest
checkpoint kept and the blocking reason named.

## Environment

| Item | Value |
|---|---|
| GPU | 1x RTX 5090, 32.6 GB |
| Repo env | `.venv`, transformers 5.13.0 |
| Base model | `circlestone-labs/Anima` preview3 (Cosmos Predict2 DiT + Qwen3-0.6B text encoder + Qwen-Image VAE) |
| Trainer entrypoint | `vrl.scripts.train:train_online` |

## Pre-flight findings (before any training)

These came out of a rollout + reward dry run and change the experiment design.

### 1. `online_grpo_aesthetic` trains on DrawBench, not anime

`vrl/config/presets/experiment/anima_preview3/online_grpo_aesthetic.yaml`
composes `/dataset/drawbench_train_192`, so the eval manifest resolves to
`datasets/drawbench/eval_64.txt`. The images it produced (LEGO men, black
bananas, "a green cup and a blue cell phone") are **correct executions of
DrawBench prompts**, not model failures.

The anime dataset preset (`/dataset/anime_anatomy`) and its data
(`datasets/danbooru/anatomy/{train,eval}_prompts.jsonl`, 1000 eval prompts)
already exist and are unused by this experiment. Anime runs must compose it.

### 2. Base Anima is healthy; earlier "failures" were misdiagnosis

Given anime prompts it produces correct, clean anime figures at 50 steps /
CFG 7.0. Two intermediate diagnoses I made and then disproved:

- *"seed 20260816 is a bad noise region"* — wrong; the seed was fine, the
  prompts were DrawBench.
- *"20 steps is too few to converge"* — wrong; the probe that "fixed" it had
  also changed the prompt. Step count is a quality knob, not the bug.

Recorded because both are the kind of plausible-but-wrong root cause worth not
repeating.

### 3. Existing rewards do not separate good anime from garbage

Scoring the good batch against the collapsed batch:

| batch | aesthetic | pickscore |
|---|---|---|
| good (on-prompt) | 4.981 ± 0.459 | 0.8289 ± 0.0360 |
| collapsed (off-prompt) | 5.009 ± 0.499 | 0.8297 ± 0.0370 |

**Statistically identical.** The top-ranked image by aesthetic score was a
collapsed one. This is the single most important pre-flight result: optimizing
LAION-aesthetic alone cannot be expected to improve anime quality, and it is
why AnimeReward is worth the integration cost.

### 4. AnimeReward needs an isolated environment

The quality head is `Idefics2ForSequenceClassification` from the `mantis`
package (not transformers). Mantis targets transformers 4.x; the repo is on
5.13. Three separate breaks surfaced (`tie_weights(recompute_mapping=)`,
`DynamicCache.from_legacy_cache`, `get_usable_length`) with ~20 call sites of
4.x-era cache API in one file — shimming is too fragile.

Resolution: run it in a dedicated venv pinned to transformers 4.x, exposed over
HTTP, mirroring the existing `*_http.yaml` reward-service pattern and the
VBench precedent already documented in `pyproject.toml` ([tool.uv] conflicts).
The repo env is not downgraded.

Weights on disk (21 GB, verified against the safetensors index):
`~/.cache/huggingface/hub/models--IndexTeam--Index-anisora/snapshots/b134a8e6.../reward/weights/`
The IP/character-consistency stack was deliberately not kept — it scores identity
across frames and is undefined for single stills.

### 5. AnimeReward DOES discriminate on stills (the go/no-go result)

Probed the quality head on the same two batches:

| judge | good (on-prompt anime) | collapsed (off-prompt) | separation |
|---|---|---|---|
| LAION aesthetic | 4.981 ± 0.459 | 5.009 ± 0.499 | none (inverted) |
| PickScore | 0.8289 ± 0.0360 | 0.8297 ± 0.0370 | none |
| **AnimeReward quality** | **0.6963 ± 0.0267** | **0.4699 ± 0.0814** | **Cohen's d = 3.74** |

Ranges do not overlap (good 0.655–0.735, collapsed 0.299–0.598). This is what
justifies the integration cost.

## Integration

- `vrl/rewards/models/animereward.py` — `AnimeRewardQualityModel` (TorchRewardModel).
- `vrl/rewards/functions/animereward.py` — `AnimeRewardQualityReward`
  (DiskArtifactRewardFunction, so the HTTP transport is available).
- registered as `animereward_quality`; preset `/reward/animereward_quality_http`.
- `vrl/config/reward_service/animereward_quality.yaml` — service on port 8310,
  `generation_overlap_safe: false` (it shares the one physical GPU).
- Isolated env: transformers **4.49.0** + tokenizers 0.21.4. Not 4.44.2 — that
  pins tokenizers <0.20, which cannot read this checkpoint's `tokenizer.json`
  ("data did not match any variant of untagged enum ModelWrapper"). 4.49 is the
  window that reads the new tokenizer format and still predates the 5.x
  `GenerationMixin`/cache-API break.
- End-to-end verified: scores through registry -> disk artifact -> HTTP ->
  service match the direct-model probe exactly (0.6963/0.4699).

## Fixed evaluation protocol

`vrl/scripts/eval/anima_fixed_eval.py` — 24 held-out anime prompts, seed 7777
(per-index, so checkpoints differ only by weights), 30 steps, CFG 5.0. Reports
quality mean/std plus a model-free diversity pair (mean pairwise pixel L2 and
color-histogram L2), because reward optimization can present as pure mode
collapse that a quality mean cannot see.

**Baseline (base Anima, no LoRA):**

| metric | value |
|---|---|
| quality mean | **0.6914** |
| quality std | 0.0387 |
| quality min / max | 0.600 / 0.735 |
| diversity pixel_l2 | **32.639** |
| diversity color_hist_l2 | 0.1885 |

## Memory budget (single 32.6 GB card)

The reward service holds **17.5 GB** resident for its whole lifetime, leaving
~15 GB for trainer + rollout. This is the binding constraint on batch size and
is why the run below uses LoRA rather than full fine-tuning.

## Experiment log

| # | Config | Why | Result |
|---|---|---|---|
| — | pre-flight | see above | design changed: anime dataset + AnimeReward required |
| 1 | recipe defaults (clip_ratio 1e-4, lr 1e-5, 10-step rollout) | first honest attempt | **stopped at epoch 4 — policy frozen by the trust region** |
| 2 | clip_ratio 3e-3 | widen trust region past the measured update size | **stopped at epoch 10 — clip_fraction fixed (0.45 -> 0.11) but reward flat (slope +6e-4)** |
| 3 | + lr 1e-4 (LoRA rate) | run 2's lr was a full-param rate; repo LoRA configs use 1e-4..3e-4 | **stopped at epoch 12 — grad_norm rose ~3x, reward still flat/declining (slope -9e-4)** |
| 9 | + 20-step rollout, noise_level 0.3, 16x16 batch | fix the undersampled/over-noised rollout (see below) | **stopped at epoch 10 — reward fell monotonically 0.637 -> 0.452; checkpoint significantly WORSE than base** |

### Run 9 — the rollout fix works, and reveals a real negative result

Runs 1-8 were flat. Run 9 is the first run where the training signal moves with
statistical significance — and it moves **down**.

| epoch | 0 | 2 | 4 | 6 | 8 | 9 |
|---|---|---|---|---|---|---|
| `reward_mean` | 0.6369 | 0.5981 | 0.5737 | 0.5616 | 0.4888 | 0.4523 |

Fitted slope **-0.0153/epoch, t = -6.85 (df 7)**. Total decline 29%.

**Two null explanations were checked and rejected:**

1. *Loss sign error.* No. `vrl/algorithms/grpo/continuous.py:148-151` computes
   `unclipped_loss = -advantages * ratio` and minimizes the max of the clipped
   pair — positive advantage correctly pushes the ratio up.
2. *Prompt-draw luck.* The anatomy train set has 20,000 prompts and each epoch
   draws only 16, so every epoch sees a disjoint prompt set. But the expected
   epoch-to-epoch SEM from that draw is 0.0092 (from the measured between-group
   std 0.029 and within-group 0.09 at 16 prompts x 16 samples). The observed
   0.148 decline is **12x** that, and monotone in 8 of 9 transitions.

**Checkpoint-10 confirms it on clean sampling.** Identical 24 prompts, seed 7777:

| | base | run-9 ckpt-10 |
|---|---|---|
| quality mean | **0.6914** +/- 0.0396 | **0.6640** +/- 0.0352 |
| diversity pixel_l2 | 32.64 | 30.55 |
| diversity color_hist_l2 | 0.1885 | 0.2067 |

Paired t over the 24 matched prompts: **t = -3.25 (df 23), significant
degradation** (|t| > 2.07). Worsened on **20/24** prompts, improved on 4.

This matters because it is a *different* failure from run 3. Run 3 was "no
learning" (t = -1.53, ns). Run 9 is "learning the wrong thing" — the policy
demonstrably moves (`kl_penalty` roughly doubles, 2.2e-4 -> 4.4e-4) and every
step of that movement costs quality.

**Prime suspect: the KL anchor is not anchoring.** At `kl_coef: 0.004` the
logged `weighted_kl_loss` is ~1e-6 against a `policy_loss` of ~3.5e-4 — two
orders of magnitude smaller. The reference-KL term is numerically absent, so
nothing holds the policy near the base model while the surrogate chases a
signal it cannot actually climb. Diversity also narrows slightly (pixel_l2
32.64 -> 30.55), consistent with drift toward a narrower region rather than
toward quality.

**Not yet ruled out**, and the reason no config change should be made on the
strength of one run: with `n_samples_per_prompt: 16` on a *fresh* 16-prompt draw
each epoch, the advantage baseline is per-prompt and never revisits a prompt.
Whether the decline is the weak KL anchor, or GRPO overfitting a per-prompt
baseline it can never validate against, is not decided by the present data.

### Experiment A — KL anchor ruled out as the cause

`kl_coef` 0.004 -> 0.3, everything else identical to run 9. The intervention did
exactly what it was designed to do and **did not help**:

| | run 9 | experiment A |
|---|---|---|
| KL term as % of policy_loss | 0.3% | **17.3%** |
| `kl_penalty` drift over epochs | 2.2e-4 -> 5.5e-4 (rising) | 2.2e-4 -> 2.5e-4 (**flat**) |
| reward slope | -0.0177/epoch | **-0.0208/epoch** |

The anchor works mechanically — `kl_penalty` stops climbing, i.e. the policy is
genuinely held near the base model. Quality still declines, marginally faster.
**Stopped at epoch 4** (t = -6.77); a hypothesis this clearly falsified does not
deserve six more epochs of GPU.

### The actual root cause: the reward ranks style, not quality

The assumption behind all ten runs was that AnimeReward's **within-group**
ranking tracks quality. GRPO consumes nothing else — the group mean is
subtracted and the group std divides
(`vrl/algorithms/advantages.py:88-91`), so within-group ranking IS the entire
learning signal. That assumption had never been tested. Pre-flight validated the
reward on a *between-condition* contrast (good anime vs collapsed output, Cohen's
d = 3.74), which says nothing about resolving 16 samples of one prompt that
differ only by SDE noise.

Tested directly (`probe_rank_validity.py`, 16-sample real rollout group, one
prompt, 20 steps / noise 0.3):

- Scoring is **deterministic** (two passes identical, max diff 0.0000) — the
  spread is a real preference, not sampling noise.
- Range within the group is wide: 0.4725 to 0.6950.
- **But both extremes are competent illustrations.** No artifacts, no anatomy
  failure, no collapse in either. The top-ranked image has an elaborate outfit,
  saturated dark colors and dramatic hair; the bottom-ranked has a simpler
  outfit and a lighter, flatter palette. This is a *style* difference.

Correlating reward against simple image statistics over the group (n=16):

| statistic | r with reward |
|---|---|
| saturation | **+0.507** |
| ink coverage | +0.433 |
| edge energy | +0.406 |
| brightness | **-0.355** |

Within a group of equally-competent images the reward prefers **darker, more
saturated, busier** frames. And the trained policy moved exactly there: run 9's
checkpoint raised mean saturation **+6.3%** vs base (0.0447 -> 0.0475) while its
fixed-eval quality fell 0.6914 -> 0.6640.

**This is textbook reward hacking, and it explains every observation:** the
policy successfully climbs the reward's real preference; that preference is
only weakly related to quality; so training reward and eval quality move in
*opposite* directions. It also explains why the KL anchor did not help — the
policy was not drifting randomly, it was optimizing something real and wrong.

Consequence: **no amount of GRPO tuning fixes this.** kl_coef, lr, batch size,
clip_ratio and prompt-pool structure all change how fast the policy climbs a
signal that points slightly away from quality. The reward is the defect.

### Experiment C — swapping the reward helps, but does not stop the decline

`pickscore: 1.0 + animereward_quality: 0.25`, rollout settings identical to run 9.

| epoch | total | pickscore (w=1.0) | animereward (w=0.25) |
|---|---|---|---|
| 0 | 0.9358 | 0.7772 | 0.6345 |
| 1 | 0.9224 | 0.7693 | 0.6127 |
| 2 | 0.9143 | 0.7641 | 0.6008 |

The reward swap **does** change the dynamics — the driven objective decays about
3x slower than the undriven one:

| series | per-epoch relative change |
|---|---|
| run 9 animereward (driven) | -3.53%, -2.56% |
| C animereward (**not** driven, w=0.25) | -3.44%, -1.88% |
| C pickscore (**driven**, w=1.0) | **-1.02%, -0.67%** |

So the objective matters. But **everything still declines**, and that is the
finding that outranks the reward-hacking story:

**AnimeReward falls at nearly the same rate in all three runs regardless of what
is being optimized** — run 9 (AnimeReward alone) -0.0388 over two epochs,
experiment A (AnimeReward + 75x stronger KL) -0.0311, experiment C (PickScore
driving, AnimeReward at 1/4 weight) -0.0337. A purely reward-hacking explanation
predicts the decline should track the objective. It does not track it strongly
enough. Something degrades the policy **independently of what it optimizes**.

**Ruled out while chasing this** (each checked against the source, not assumed):

- *Advantage machinery.* Healthy in every run: full groups of 16,
  `adv_zero_rate` ~0, `adv_saturation` 0, `advantage_mean` exactly 0.
- *Reference model for KL.* Correct — `disable_adapter()` yields the true base
  model (`vrl/rollouts/evaluators/denoise/sde_logprob.py:131-138`).
- *`sde_window_size`.* Not set (0 = disabled), so all timesteps are eligible.

**Two real defects found in the shared defaults**, neither yet tested as a fix:

1. **The last denoise step never gets gradient.** `actor.timestep_fraction: 0.99`
   (`vrl/config/presets/base/actor.yaml:29`) is a compute-saving default
   inherited from flow_grpo's ~50-step SD3 setup. At this config's
   `num_steps: 20` it computes `int(20*0.99) = 19`, takes the strided branch, and
   yields indices 0..18 — **timestep 19, the final step that produces the actual
   image, is excluded from the loss on every update.** Dropping 1 of 50 steps is
   negligible; dropping the *last* of 20 is not.
2. **Evaluated checkpoints are EMA weights, not training weights.**
   `save_training_checkpoint` writes adapter artifacts from EMA
   (`vrl/trainers/checkpointing.py:421-424`) and the repo default is
   `ema.decay: 0.9, update_interval: 8` — far more aggressive than the class's
   own 0.9999 default. Every "checkpoint vs base" number in this document is
   therefore a measurement of a heavily-smoothed average, not of the policy the
   training curve describes. This does not explain the training-metric decline
   (that is measured on live weights at rollout) but it does mean training
   curve and eval result are not measuring the same weights.

### Experiment D — the timestep fix stops the decline

`timestep_fraction: 1.0` (all 20 denoise steps get loss) + `ema.enable: false`
(exported adapter IS the trained policy). Reward composition unchanged from C.

| epoch | total | pickscore | animereward |
|---|---|---|---|
| 0 | 0.9354 | 0.7777 | 0.6309 |
| 1 | 0.9232 | 0.7685 | 0.6186 |
| 2 | 0.9166 | 0.7642 | 0.6100 |
| 3 | 0.9160 | **0.7724** | 0.5744 |
| 4 | **0.9195** | 0.7715 | 0.5920 |

**The driven objective stops declining:**

| | slope | t |
|---|---|---|
| C pickscore (`timestep_fraction` 0.99) | -0.00655 | **-8.40** |
| D pickscore (`timestep_fraction` 1.00) | **-0.00085** | **-0.48 (flat)** |

A 7.7x reduction in slope, and no longer statistically distinguishable from
flat. Epoch 3 is the first *rise* in any run of this sprint (+0.0082), and
epochs 2-4 trend mildly positive (slope +0.0037, t = +1.39 — not yet
significant at n=3).

**The excluded timestep was carrying most of the gradient**, which is why this
was never a tuning problem:

| | C (tf 0.99) | D (tf 1.00) |
|---|---|---|
| `grad_norm` | 0.00148 | **0.00757** (5.1x) |
| `approx_kl` | 1.13e-05 | **1.12e-04** (9.9x) |
| `ratio_abs_dev_mean` | 0.00163 | 0.00431 |

`approx_kl` rising 10x means the policy is finally *moving* per update. And
`ratio_abs_dev_mean` now sits well above the systematic rollout->replay mismatch
floor (~0.0017), so the importance ratio reflects real policy change instead of
being dominated by numerical bias — previously the bias alone consumed ~56% of
the `clip_ratio: 3e-3` trust region.

**Correction — the mid-run rise did not hold.** At epochs 2-7 pickscore was
rising (slope +0.0038, t = +6.14) and this document previously called that a
significant improvement. That was computed on a mid-run window. Epochs 8-9
reversed it (0.7837 -> 0.7695), and over the **full 10 epochs**:

| series | slope | t | verdict |
|---|---|---|---|
| pickscore (driven) | +0.00059 | +0.84 | **flat, not significant** |
| animereward | -0.00463 | **-2.96** | still declining |
| total | -0.00056 | -0.60 | flat |

A trend fitted to a hand-picked sub-window is not a result. The honest reading
is that the timestep fix took the driven objective from *significantly falling*
(C: t = -8.40) to *flat* (t = +0.84) — real progress on the mechanism, but not
learning.

**Checkpoint-10 eval settles it** — 24 fixed prompts, seed 7777, scored by both
rewards plus an independent edge-energy statistic (no EMA this time, so these
weights ARE the trained policy):

| metric | base | D ckpt-10 | delta | t | verdict |
|---|---|---|---|---|---|
| AnimeReward | 0.6914 | 0.6664 | -0.0250 | **-4.11** | **significantly worse** |
| PickScore (driven) | 0.7862 | 0.7915 | +0.0053 | +1.41 | not significant |
| Edge energy | 0.0235 | 0.0218 | -0.00168 | -1.65 | worse in 16/24 |
| diversity pixel_l2 | 32.64 | 30.70 | — | — | narrowed |

**The driven objective did not significantly improve on held-out prompts, while
anime quality significantly degraded and detail density fell.** Visual check of
`eval_0021` confirms the numbers: cleaner than run 9's sketch-like output, but
flatter and less detailed than base — muted palette, minimal shading.

So experiment D fixed a real mechanism defect (the final denoise step now gets
gradient; `grad_norm` 5x, `approx_kl` 10x) **without** producing a better model.
The remaining blocker is the reward, exactly as the section below concludes.

### The two rewards significantly DISAGREE — and AnimeReward is the right one

Scoring the *same* 24 base images and the *same* 24 run-9-checkpoint images with
both rewards (`vrl/scripts/eval/anima_fixed_eval.py --with-pickscore`, and
`score_pickscore_dir.py` to backfill the existing reports):

| reward | base | run-9 ckpt | delta | t | better |
|---|---|---|---|---|---|
| AnimeReward | 0.6914 | 0.6640 | **-0.0274** | **-3.25** | 4/24 |
| PickScore | 0.7862 | 0.7941 | **+0.0078** | **+2.18** | 17/24 |

The same checkpoint is significantly *worse* by one reward and significantly
*better* by the other. So "which reward is right" is now load-bearing, and it
cannot be settled by either reward's own number.

Settled by looking at the images where they disagree most:

- `eval_0021`: run-9's version is flat, muted and sketch-like, with far less
  line detail than base and a barely-rendered face. **AnimeReward penalized it
  (-0.100); PickScore rewarded it (+0.031).**
- `eval_0000`: both versions are clean; the checkpoint merely renders a
  *different* outfit and a younger-looking face. PickScore prefers it.

Measured across all 24 pairs, the run-9 checkpoint lost edge energy (line/detail
density) in **14/24** images (mean -0.00100).

**This corrects the earlier entry above.** PickScore's advantage was inferred
from one 16-image group where it caught two anatomy failures AnimeReward missed.
That observation stands, but it does not generalize: PickScore is a *general*
text-image preference model (CLIP ViT-H/14 on Pick-a-Pic, mostly photographic
and general-illustration pairs). It rewards clean, simple, well-composed images
and is largely indifferent to anime line quality and rendering density — so on
this model's failure mode (flattening toward sketch-like output) it points the
wrong way.

Neither reward is trustworthy alone here: AnimeReward tracks saturation and
complexity within a group; PickScore tracks generic composition and tolerates
loss of anime craft. **This is the central unresolved problem of the sprint** —
not the GRPO configuration, which experiment D shows is now working.

**HPSv3 as a third opinion.** Downloaded (16 GB + Qwen2-VL-7B base) and probed
on the same evidence. Scoring it required fixing a real bug first — under the
repo's pinned `transformers>=5.13` the Qwen2-VL vision tower returns
`BaseModelOutputWithPooling`, but `vrl/rewards/models/hpsv3.py:115` assumed the
old tensor return and called `.to()` on it, so **HPSv3 raised
`AttributeError` on every image**. Fixed by reading `.pooler_output` (the merged
patch embeddings) with a `getattr` fallback for the older tensor return.

With it working, the three-reward picture on the same 24 images:

| | run-9 ckpt | D ckpt |
|---|---|---|
| AnimeReward | **t = -3.25 worse** | **t = -4.11 worse** |
| PickScore | t = +2.18 better | t = +1.41 ns |
| HPSv3 | t = -0.67 ns | t = -1.29 ns |
| Edge energy | 14/24 worse | 16/24 worse |

HPSv3 sides with AnimeReward's *direction* on both checkpoints (both negative,
neither significant). Two of three rewards plus the independent detail statistic
agree there is no improvement; PickScore is the lone dissenter, and the images
show it rewarding exactly the flattening the others penalize.

HPSv3 is **not** the clean replacement, though: its within-group saturation
correlation is **+0.471**, nearly as style-driven as AnimeReward's +0.556 (and
worse than PickScore's +0.344). All three available rewards rank partly on
style within a group of equally-competent images.

**The obvious next idea — ensemble the three — was tested and REJECTED before
spending any GPU on it.** The hope was that averaging z-scored rewards would
cancel each model's idiosyncratic style bias while reinforcing the shared
quality signal. Measured on the same 16-sample group:

| reward | within-group rho with saturation |
|---|---|
| AnimeReward | +0.556 |
| HPSv3 | +0.471 |
| PickScore | +0.344 |
| **z-average of all three** | **+0.600 (worst)** |

Averaging made it *worse*. The saturation preference is **shared** across all
three models, not idiosyncratic, so averaging reinforces it instead of cancelling
it. Their pairwise rank agreement (anime-pick +0.45, anime-hps +0.41, pick-hps
+0.64) is consistent with a common bias plus independent noise.

**Consequence for anyone continuing this work:** the blocker is not "pick a
better reward from the shelf" — all three on the shelf, and their ensemble, rank
partly on saturation/complexity within a group of equally-competent anime
images. That is precisely the axis GRPO optimizes, and precisely the axis that
does not correspond to quality on this model.

### Experiment E — nsfw_safety as the sole reward: objective reached, then two incidents

Run at the user's explicit direction: `nsfw_safety` alone (weight 1.0) on the
safety-stress prompt set, where the signal is live (19/24 base images trigger,
penalty mean -0.786). Launched 2026-08-17 23:39, config
`online_grpo_nsfw_only` (since deleted — one-shot experiment, this section is
its record).

**The objective was reached.** Observed during the run: metrics hit the goal
(penalties driven to ~zero) around epoch 14-17. Convergence then exposed two
defects, one benign and one serious:

1. **Crash on convergence (benign, still open).** Once every group scores a
   uniform 0.0 penalty, every streamed microbatch is filtered; at step 18
   (05:46) the trainer skipped the optimizer as designed but then died writing
   metrics: `metrics.initial_replay` is `None` when all microbatches are
   filtered, and `OnlineMetricRow.from_step_metrics`
   (`vrl/trainers/metrics_io.py:117`) reads `.clip_fraction` off it
   unconditionally. Because `save_freq` was 10, the converged epochs 14-17 were
   never checkpointed — only pre-convergence `checkpoint-10` survives, and the
   quality-collapse check (AnimeReward on the anatomy eval set) was never run
   against a converged checkpoint. The composite config's `save_freq: 5` is the
   direct lesson.

2. **Sign-flip incident (serious, resolved 2026-08-18).** During launch
   debugging, `vrl/rewards/models/nsfw_safety.py` was left in the working tree
   with its return flipped to `return float(penalty)` — positive. Every
   consumer weights this component positively (the preset's whole contract is
   "only returns zero or negative scores", and `vrl/config/schema.py` /
   `vrl/config/builders.py` guard against negative weights), so a positive
   return turns the guardrail into an NSFW *bonus*. Two runs trained on the
   inverted signal before it was caught via their own metrics
   (`r_nsfw_safety` logged +0.6..0.8 — impossible under the correct sign):
   the 12:26 resume of this experiment (8 epochs, SIGTERM'd) and the first ~5
   epochs of the composite animereward+nsfw run (relaunched fresh after the
   revert; the poisoned `outputs/anima_animereward_nsfw_grpo` was quarantined
   with a `_signflip_tainted` suffix). The flip also overwrote this
   experiment's original epoch 0-17 `metrics.csv` on resume, which is why the
   convergence claim above cites the observed numbers rather than a surviving
   CSV.

**Verdict:** the penalty is trainable where it has variance, converges in ~15
epochs at these settings, and the standing risk (nothing anchors quality when
the penalty is the only signal) was never measured because of defect 1. The
composite run — AnimeReward carrying the gradient, `nsfw_safety` at 0.5
bending it — supersedes this configuration.

### Experiment F — "quality consistent + NSFW trigger down": a three-attempt ladder

Goal: keep anatomy quality at base level while reducing the safety trigger rate
(base: 19/24 images on the fixed safety set). Three configs were run and are
**since deleted — one-shot experiments, this section is their record**. Only the
lever changed between them; everything else (lr 1e-4, clip_ratio 3e-3,
timestep_fraction 1.0, EMA off, 20-step rollout at noise 0.3, 16x16 batch) was
held at the Experiment-D settings.

| # | config (deleted) | lever | outcome |
|---|---|---|---|
| 1 | `online_grpo_quality_nsfw_guardrail` | 85/15 anatomy/safety mix, nsfw weight **0.15** | anatomy quality drifted DOWN (0.691 base -> 0.683 ck4 -> 0.673 ck6), trigger rate **unmoved** (19/24) |
| 2 | `online_grpo_quality_nsfw_klanchor` | 50/50 mix, nsfw **0.3**, **kl_coef 0.05** (~12x) | trigger rate **unmoved** (19/24) |
| 3 | `online_grpo_quality_nsfw_strong` | 50/50 mix, nsfw **0.5**, kl back to 0.004 | trigger **0/24** — by collapsing into a yellow grid that destroys the image |

**What each attempt taught, in order:**

1. A guardrail weight small enough not to disturb quality is also too small to
   move the trigger rate. It did not buy safety, and it still cost quality — the
   weight is not a free dial between the two objectives.
2. A global KL anchor is the wrong instrument here. It pins the whole policy to
   base, and **base is exactly what produces the triggering images**, so KL
   blocks the behavior change on explicit prompts that safety needs. It protects
   quality by forbidding the very move being asked for.
3. Given real strength, the policy does find a way to zero the penalty — the
   cheapest one, which is to stop producing images. The yellow-grid collapse
   eliminated NSFW completely (0/24) while destroying every image.

**The load-bearing finding of attempt 3: AnimeReward was blind to the collapse.**
It scored the grid 0.61, the same range as real anime. A reward-robustness probe
over base real-anime frames vs the ck30 grid frames:

| reward | real anime | yellow grid | separates? |
|---|---|---|---|
| AnimeReward | 0.691 | 0.533 | no (blind, small d) |
| PickScore | 0.814 | 0.634 | **yes** (Cohen d +6.0) |
| LAION aesthetic | 5.81 | 3.99 | **yes** (Cohen d +5.3) |

That is why the ladder continues with PickScore as the quality anchor
(`online_grpo_pickscore_nsfw`, kept) rather than AnimeReward: a quality model
that cannot see the degenerate solution cannot prevent it. The caveat recorded
in that config still stands — PickScore/aesthetic separate the *grid* collapse
but not subtle on-prompt-vs-off-prompt collapse (0.829 vs 0.830), so runs are
judged by eye plus the model-free diversity metric, never a single score.

**Dataset note.** These three runs trained off a mixed prompt manifest built on
the side by a `build_anime_quality_safety_mix` script. Both the script and its
output are gone: the mix ratio is now declared in the dataset preset itself
(`data.manifest` as a `{manifest path: prompt count}` mapping plus
`data.mix_seed`), so no derived manifest is materialized. The surviving 50/50
preset reproduces those runs' prompt set exactly, prompt for prompt.

### Run 3 — the decisive negative result

Checkpoint-10 evaluated against base on the identical 24 prompts and seeds:

| | base | ckpt-10 |
|---|---|---|
| quality mean | **0.6914** +/- 0.0396 | **0.6857** +/- 0.0467 |
| diversity pixel_l2 | 32.639 | 33.026 |

Paired t-test over the 24 matched prompts: **t = -1.53 (df 23), not significant**
(|t| > 2.07 needed). Improved on 5/24, worsened on 10/24, tied on 9/24.
Diversity held, so this is not collapse — it is simply **no learning**.

**What was ruled out, with evidence:**

- *Gradients not flowing.* Ruled out. All 224 `lora_B` tensors moved off their
  zero init (mean norm 3.8e-2). The LoRA trains and checkpoints correctly.
- *Reward signal too weak for GRPO.* Ruled out — there is ample within-group
  variance. (These run-3-era numbers, "within 0.0327 vs between 0.0292, ratio
  1.12", were measured on the *broken* 10-step/noise-0.7 rollout and no longer
  describe the fixed config. Remeasured on run 9's 3712 scored samples across
  232 full groups: within-group 0.0577, between-group 0.1843, ratio 0.31. The
  conclusion survives the correction — with `global_std: false` each group is
  normalized by its own std, so a low ratio is not itself a defect: mean |advantage|
  is 0.81 and only 2.2% of groups are near-degenerate.)
- *Reward is noise.* Ruled out in pre-flight (Cohen's d = 3.74) — but note that
  test was *between conditions*, not within a group. See "the reward ranks style,
  not quality" above: the within-group ranking, which is the only thing GRPO
  consumes, turned out to be the real defect.
- *Learning rate.* Addressed in run 3; grad_norm rose ~3x with no reward effect.
- *Trust region.* Addressed in run 2; clip_fraction fell to 0.11.

**The remaining structural cause.** With `ppo_epochs=1` under
`strict_on_policy`, behavior and target policy are identical on the single
replay pass, so the importance ratio is identically 1. The trainer documents
exactly this as "the documented flat-curve root cause" in
`_validate_trust_region_engages` — but that guard only fires for algorithms
that *declare* `requires_active_trust_region` (Flow-DPPO / GRPO-Guard). Plain
GRPO is allowed through, and it degenerates to a plain policy-gradient step
whose magnitude, at this batch size (4 prompts x 8 samples = 32 samples per
update), is too small to move a 2B model in tens of epochs.

**Highest-value next experiment** (not yet run): raise the per-update sample
count substantially (more prompts per batch, and/or `ppo_epochs>1` via the
legacy full-batch path the guard's message describes) so each update sees
enough signal to overcome gradient noise. Scaling `lr` further without
addressing batch size would trade one instability for another.

### Run 1 — stopped: the trust region strangles learning

Ran cleanly (parking loop healthy, ~26 s per rollout batch, reward service
answering) but learned nothing:

| epoch | reward_mean | approx_kl | grad_norm | clip_fraction |
|---|---|---|---|---|
| 0 | 0.2228 | 1e-6 | 6.5e-3 | 0.448 |
| 1 | 0.2009 | 1e-6 | 2.4e-3 | 0.462 |
| 2 | 0.2231 | 1e-6 | 8.0e-3 | 0.465 |
| 3 | 0.2042 | 1e-6 | 3.2e-3 | 0.451 |

Root cause, measured rather than guessed:

```
ratio_abs_dev_mean = 6.1e-4      <- typical policy update
clip_ratio         = 1.0e-4      <- trust region ceiling
```

The mean ratio deviation is **6x the clip threshold**, so ~45% of the batch is
clipped every step and `approx_kl` stays at 1e-6 — the policy is numerically
frozen. `clip_ratio: 1e-4` is inherited from
`/recipe/online/flow_matching_grpo`, where the comment attributes it to
flow_grpo's SD3 setup. Whatever holds for SD3, on Anima it is far too tight.

Second finding from the same run: rollouts score **0.20-0.22**, while the same
prompts at eval settings score **0.69**. Chased this down because a broken
reward signal would invalidate everything downstream:

- Step count is **not** the cause. Re-running the fixed eval at the rollout's
  own 10 steps / CFG 4.5 still scored **0.654** — nearly the 30-step number.
- The cause is `denoise_mode: sde` with `noise_level: 0.7`. GRPO rollouts
  inject exploration noise by design, and that is what costs ~0.45 of score.
- The signal survives it. Across 184 logged rollout samples the reward spans
  **0.114-0.403** (mean 0.221, std 0.050, CV 0.227), so within-group variance —
  the thing GRPO actually consumes — is healthy.

Consequence for reading later results: **rollout `reward_mean` (~0.22) and
fixed-eval quality (~0.65) are different scales and must never be compared to
each other.** Only eval-to-eval comparisons decide whether a checkpoint is better.

### Environment note: an untracked in-progress refactor

`vrl/nn/optimization/` is untracked in this working tree (pre-existing, not
mine). Run 1's first launch died with `NameError: apply_rollout_optimizations`
inside the Ray worker even though the import in
`vrl/models/steps/denoise/build.py:26` is correct — the worker imported the
package while its bytecode was still being written (`__pycache__` timestamps
land inside the failing run's window). Relaunching resolved it. Worth knowing
if a fresh clone reproduces the same error.

### Environment note: parking residual tolerance

The colocated handoff validator compares **device-wide** used bytes before and
after parking, so it also sees the reward service's 17.3 GB. Run 1 tripped it by
**20 MiB** over the 256 MiB default (`residual 18.387` vs `baseline 18.117 GiB`)
— the cumem allocator itself parked correctly, freeing 5.66 GiB. This is the
case `VRL_CUDA_RESIDUAL_BYTES_LIMIT_MIB` exists for (documented in
`vrl/utils/cuda_memory.py`), so runs use `VRL_CUDA_RESIDUAL_BYTES_LIMIT_MIB=1024`.
The safety check was raised, not disabled.
