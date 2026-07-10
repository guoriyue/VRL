# SPRINT: Config duplication audit — factor more, don't flatten

> **Diagnosis (flips the initial feeling).** The `configs/` tree is NOT over-organized —
> it is a correct 3-tier Hydra composition (`base/` atoms → `recipe/` family+algo combos
> → `experiment/` runs, plus orthogonal `model/` `sampling/` `dataset/` `reward/`).
> Flattening it would RE-introduce massive duplication. The pain the eye reads as "too
> much organization" is actually **under-factoring**: the same override blocks are
> copy-pasted across 30+ experiments because (a) some are redundant re-statements of a
> default, (b) family+algo-constant values (like `entrypoint`) have no home above the
> experiment, and (c) there is no per-family recipe layer to own shared tuning. The fix
> is MORE factoring, not less. All counts below measured 2026-06-30 over 133 yaml /
> 42 experiments.

## Tiers today (keep these — they earn their keep)

```
base/ (22)     actor · algorithm/* · rollout/* · distributed/* · trainer   — atoms, one concern each
recipe/ (10)   online/diffusion_grpo etc. — compose ~6 base files + set family-boundary values
experiment/(42) compose a recipe + model + sampling + reward + dataset + per-run overrides
model/(15) sampling/(16) dataset/(13) reward/(13)  — orthogonal, composed into experiments
```
`recipe/` saves each experiment from re-listing ~6 base defaults → without it, 42×6 repetition.
`base→recipe→experiment` is the standard anti-duplication pattern. **Non-goal: flattening any tier.**

## The measured duplication (the real problem)

Most-overridden keys across the 42 experiments (each line = how many experiments repeat it):

| key | count | nature | fix |
|---|---|---|---|
| `kl_coef: 0.0` | **29** | **redundant** — `base/algorithm/grpo.yaml:10` already = `0.0` | delete the no-op line |
| `same_latent: false` | **12** | **redundant** — `base/rollout/diffusion.yaml:17` already = `false` | delete the no-op line |
| `trainer.entrypoint` | 38 | **family+algo-constant** (all 10 cosmos identical; all 5 sd3_5; etc.), but `model/` doesn't own it → no home but per-experiment | move to a family+algo recipe |
| `lr: 1.0e-4` | 34 (18 same value) | family-common tuning (no base default: `actor.lr: ???`) | family recipe default |
| `n_samples_per_prompt` / `prompts_per_batch` | 36 / 36 | rollout sizing, family-common | family recipe default |
| `samples_per_chunk: 1` | 32 | memory-driven, family-common for video | family recipe default |
| `sde` (`type: cps`) | 27 | family-common | family recipe default |
| `ppo_epochs: 1` | 22 | family-common tuning | family recipe default |
| `clip_ratio: 1.0e-3` | 10 | **wrong-level default**: base=0.2 → diffusion recipe=1e-4 → cosmos/wan want 1e-3 | video sub-recipe default |
| `debug.first_step: true` | 15 | debug flag repeated | recipe default (or drop) |
| `torch_compile` / `gradient_checkpointing` | 18 / 15 | perf flags repeated | recipe/model default |
| `output_dir` | 38 | **genuinely per-run** — NOT duplication | leave inline |

A single experiment is ~46–55 non-comment lines today; the **actual per-run difference**
(shape, checkpoint path, output_dir, reference_mode) is ~8–15 lines. The other ~30–40 are
the repeated blocks above.

## Three root causes

1. **Redundant default re-statement.** `kl_coef: 0.0` (×29) and `same_latent: false` (×12)
   re-state values that are already the composed default — pure no-op noise.
2. **Homeless family+algo constants.** `entrypoint` is fixed by (family, algorithm/task):
   `cosmos_predict2` → `train_cosmos_predict2_grpo` (all 10); `sd3_5` → `train_sd3_5_grpo`
   (all 5); `wan_2_1` → grpo / i2v_grpo / dpo by task; `flux` → grpo / diffusion_nft by algo.
   `model/` configs don't carry it (they are algo-agnostic), and no recipe carries it either,
   so every experiment repeats it.
3. **Missing per-family recipe layer.** There is one recipe per (algo) — `diffusion_grpo` — but
   not per (family, algo). So the cosmos/wan video-GRPO tuning (lr=1e-4, clip_ratio=1e-3,
   ppo_epochs=1, sde=cps, samples_per_chunk=1, n_samples=8) has nowhere to live but each
   experiment. The `kling` + `v2w_reference` + `v2w_reference_480p` cosmos experiments repeat
   **4/4** of the tuning block; the droid/fullparam ones repeat 2–3/4.

## The fix — add one layer, delete no-ops, re-home constants

**Step 1 — delete redundant default re-statements (safe, mechanical).**
Remove bare `kl_coef: 0.0` (×29) and `same_latent: false` (×12) where they match the composed
default and carry no explanatory comment. KEEP the few that document a real "why off here"
(e.g. cosmos kling's `kl_coef: 0.0` has a beta-too-strong rationale comment — keep that one).

**Step 2 — introduce per-(family, algo) recipes that own the constants + shared tuning.**
```
recipe/online/cosmos_predict2_grpo.yaml   # composes /recipe/online/diffusion_grpo, sets:
  trainer.entrypoint: ...cosmos...train_cosmos_predict2_grpo
  actor.optim.lr: 1.0e-4
  algorithm: { clip_ratio: 1.0e-3 }
  rollout: { n_samples_per_prompt: 8, sde.type: cps }        # the family-common block
```
Then each cosmos experiment: `defaults: [/recipe/online/cosmos_predict2_grpo, /model/..., /sampling/..., /reward/..., /dataset/...]`
and keeps ONLY its real diffs (shape override, model.path, reference_mode, output_dir, save_freq).
Do the same for `sd3_5_grpo`, `wan_2_1_grpo` / `wan_2_1_i2v_grpo` / `wan_2_1_dpo`, `flux_grpo` /
`flux_diffusion_nft`. (Recipes composing recipes is supported by the OmegaConf overlay loader,
`vrl/config/loading.py`.)

**Step 3 — fix the wrong-level `clip_ratio` default.** Either set the video-family recipes to
`1e-3` (Step 2 covers this) or add a `recipe/online/diffusion_grpo_video` between them; leave
the AR-safe base `0.2` and the flow-matching `1e-4` where they are.

**Step 4 (minor) — collapse the 2 single-file groups.** `profile/` (1 file) and
`reward_rubrics/` (1 file) are the only genuine over-organization; inline or merge them. Low value.

### Before / after (illustrative — `online_grpo_v2w_reference_480p`)
```
BEFORE (~25 non-comment lines): defaults(6) + actor.optim.lr + algorithm(clip_ratio,kl_coef)
       + rollout(n_samples, prompts, samples_per_chunk, sde, same_latent) + cosmos + trainer(...)
AFTER  (~10 lines): defaults(/recipe/online/cosmos_predict2_grpo + model + sampling/480p + reward + dataset)
       + sampling.num_frames/fps override + model.path + cosmos.reference_mode + trainer.output_dir/eval
```
The tuning/rollout block disappears into the recipe; only the genuine per-run diff remains.

## Axis 2 — file-level near-duplication (the "too many grpo configs" feeling)

Separate from repeated *lines*, there is repeated *files*: variant clusters where the
same experiment appears N times for different shape / precision / distributed topology.
The pattern is **inconsistent** — and that inconsistency is the real problem:

- **Thin extension (correct)** — `cosmos25 kling_*` topology variants: `defaults:
  [/experiment/.../online_nft_kling_video_reward]` + ~12 lines overriding only the
  distributed block. Named, reproducible, no duplication.
- **Fat copy (duplication)** — `cosmos v2w_reference_480p` / `_fullparam_240p`, the
  `droid_*` shape variants, `sd3_5 ocr_*`: re-declare the FULL defaults list + all tuning
  instead of extending their base experiment. ~26-29 lines each where a thin extension
  would be ~10-15.

**But not every name-cluster is collapsible.** `sd3_5 ocr_crossnode_debug` differs from
`ocr` by ~34 real lines (different shape 128², different tuning, different topology+reward)
— it is a genuinely distinct experiment that merely shares the "ocr" stem. The 42
experiments are mostly real variety (reward × model × shape × algo × topology), NOT
duplicates; the true near-duplicates are the fat-copy resolution/precision variants.

**Fix:** adopt the thin-extension pattern uniformly for the fat-copy near-duplicates —
`defaults: [/experiment/.../<base>]` + only the real diff (sampling group override,
`model.path`, precision, `output_dir`). Verify with the resolved-config baseline
(byte-identical before/after). Candidates: `v2w_reference_480p`, `v2w_reference_fullparam_240p`,
`droid_full_target_480p`, `droid_target_240p` (9-line diff — the clearest), `droid_full_target_480p_lora`.
Leave the genuinely-distinct ones (`ocr_crossnode_debug`, etc.) as full configs.

**Secondary:** YAML style is inconsistent (inline `optim: {lr: 1e-4}` vs block) — this
inflates apparent diffs and hurts readability. A style-normalize pass (pick block form)
is a cheap readability win, orthogonal to the factoring.

## Non-goals

- **Do not flatten** `base/`→`recipe/`→`experiment/`. That reintroduces the duplication.
- **Do not over-parametrize.** Genuinely per-run values (`output_dir`, checkpoint `model.path`,
  shape overrides, per-experiment eval cadence) stay inline — pulling them into defaults would
  create a config-explosion of near-identical recipes.
- **Do not delete commented/intentional default re-statements** — a re-stated default WITH a
  "why" comment is documentation, not noise (respects the explicit-config preference).

## Open judgment calls (decide before Step 1/2)

1. **Bare `kl_coef: 0.0` / `same_latent: false`** — delete as no-ops, or keep for explicit
   auditability? (Recommend: delete the bare ones, keep the commented ones.)
2. **Per-family recipes** — is the repeated tuning the intended shared default for a whole
   family, or is each experiment independently tuned and only coincidentally equal? If shared,
   Step 2 is a clean win; if they drift per-run, the constant stays inline and only `entrypoint`
   + the redundant defaults get factored.

## Effort / risk

Steps 1 + 4 are mechanical and safe (no-op deletions + trivial group merge), verified by
`tests/config/test_load_all_experiments.py` (all experiments must still load+validate). Step 2
is the real win but touches many experiments — do it family-by-family, re-running the load test
after each family. No behavior change if the recipe defaults exactly equal what the experiments
currently set (assert via a resolved-config diff before/after per experiment).
