# Reward card: HPSv3 + SigLIP V2.5 (Qwen-Image-2.1 text-to-image, shared-noise early-window sampler)

- **Kind**: learned composite.
- **Axis**: overall visual appeal.
- **Preset**: `vrl/config/presets/reward/hpsv3_siglip_aesthetic.yaml`.
- **Weights**: HPSv3 1.0 + SigLIP 2.75, so each term contributes about one within-group standard deviation.
- **Ready for a training key**: yes, with one logged blind spot (mild grain, gate 3).

## Sampler this card applies to

It applies only to the training sampler below; a card for another sampler says nothing here.

- Qwen-Image-2.1 revision `b3179ad3`, 1024 × 1024, 40 steps, guidance 1.0.
- Flow-GRPO SDE with noise 0.7 and 2 stochastic steps in window `[0, 6]`.
- `rollout.group_shared_noise: true`.

Within-group spread (gate 4), measured under this sampler with a shared initial latent: median group standard deviation is
HPSv3 0.553, SigLIP 0.200, PickScore 0.012. The previous `[4, 36]` / noise 0.2 window gave SigLIP 0.033 (see
`docs/research/qwen21_aesthetic/DEBUG_20260927.md`).

## Data

- **Prompts**: the 64 frozen qualification prompts, `docs/research/qwen21_aesthetic/qualification_prompts.jsonl`
  (8 categories × 8).
- **Images**: 8 samples each (512 images) at the sampler above, with one shared initial latent per prompt.
- **Location**: `/mnt/nvme/aes2/qual/`, with scores in `scores.jsonl`.

## Gate 1: repeatability

PASS. All three scorers are deterministic in float32/bf16 inference (the earlier SigLIP card measured a maximum range of 0.0
over 2 passes). HPSv3 uses a single deterministic forward pass.

## Gate 2: blind preference

Per prompt, the reward's top pick is compared with a uniform-random member of the same group (fixed seed). An identical pick
counts as a tie (0.5). The judges were three independent model judges (Claude subagents) who saw only the side-by-side pair
and the prompt, never scores or arm names. Pair A/B order was randomized.

- **Judge agreement**: exact appeal agreement on 30 overlapping pairs was 26/30. A later judge matched the earlier labels on
  4/4 decoys.
- **Bootstrap**: 10,000 resamples over prompts.
- **Gate rule**: the lower bound of the utility CI must be above 0.5, and the adherence (0–3) lower bound must be at least −0.15.

| reward | utility | 95% CI | W/L/T | adherence Δ [95% CI] | gate |
|---|---|---|---|---|---|
| SigLIP V2.5 | 0.578 | 0.461–0.688 | 33/23/8 | +0.000 [−0.148, +0.148] | FAIL |
| PickScore | 0.641 | 0.531–0.750 | 37/19/8 | +0.000 [−0.133, +0.141] | PASS |
| HPSv3 | 0.727 | 0.625–0.820 | 41/12/11 | +0.086 [−0.062, +0.234] | PASS |
| **HPSv3 + 1.0·SigLIP (z)** | **0.664** | **0.555–0.766** | | +0.008 [−0.133, +0.156] | **PASS** |
| HPSv3 + 2.0·SigLIP (z) | 0.633 | 0.523–0.742 | | +0.016 [−0.133, +0.172] | PASS |

The labels are model judgments, not human preference.

## Gate 3: shortcuts

For one candidate per prompt, the share of transformed images that scored at or above the genuine image:

| transform | HPSv3 | SigLIP (earlier card) | composite w = 1 |
|---|---|---|---|
| Gaussian blur r = 3 | 0/64 | 38% | 0% |
| pixel noise σ = 25 | 33% | 12% | 9.4% |
| pixel noise σ = 10 | 56% | — | 28% |
| black | 0/64 | 11% | — |
| checkerboard | 0/64 | 6% | — |

Mean change in within-group z-score for mild noise (σ = 10): HPSv3 **+1.01**, SigLIP −2.04, composite about −1.0.

- HPSv3 alone *rewards* grain, which is an exploit direction. The SigLIP term reverses the sign, and HPSv3 covers SigLIP's blur
  blind spot.
- **Blind spot (logged)**: the composite still scores mildly grainy images at or above the original 28% of the time.
- **Post-training check required**: measure output noise level and blind-judge base vs checkpoint for grain. Stop if grain
  increases.

## Gate 5: after training

Pending. Base vs checkpoints on the 120 frozen test prompts
(`docs/research/qwen21_aesthetic/test_prompts.jsonl`), with deterministic native sampling, blind pairwise appeal and
adherence, and a grain check.
