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
