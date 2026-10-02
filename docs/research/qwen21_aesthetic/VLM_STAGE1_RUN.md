# Stage 1 with the Qwen3-VL-32B pairwise judge (`vlm_stage1`)

Started 2026-10-01 01:42. The Seedream 3.0 / RewardDance recipe applied to the shared-noise sampler: the
reward is a large generative VLM judging each rollout against a deterministic base-model image of the same
prompt, not a CLIP-style scalar. Reading note: `READING_seedream_rlhf.md`; judge card:
`docs/rewards/cards/vlm_preference_qwen3vl32b_qwen21_t2i.md` (pairwise mode passes gates 1-4).

## Setup

- Preset `experiment/qwen_image_21/online_grpo_aesthetic_shared_noise_4gpu` + `/reward=vlm_preference_http`;
  3 FSDP ranks on GPUs 0-2 (colocated trainer/rollout), judge service on GPU 3
  (`vrl/config/reward_service/qwen3vl32b_pairwise.yaml`, port 8318, 41-43 GB).
- Sampler: 1024 px / 40 steps, SDE noise 0.7 on 2 steps in [0, 6], shared initial latent per group of 8.
- Optimizer: lr 1e-4, 4 optimizer steps per batch, ppo_epochs 1, 2 prompts per round (16 rollouts),
  LoRA rank 32, `shard_trainable_only`. 60 epochs, checkpoint every 5.
- Data: 400 prompts sampled (seed 20261001) from `manifests/pickscore_sfw/train_safe.txt`, each with a base
  reference image `aes_refs/stage1/{i:03d}.png` (native 1024/40, seed 7000+i) named by
  `metadata.judge_reference` in `/mnt/nvme/aes2/stage1_prompts.jsonl`.
- Reward: `vlm_preference` = weighted mean of P(candidate preferred | both orders) over criteria
  overall (w=2), structure, clean. 0.5 means "as good as the base image for this prompt".
- Cost: ~15 min per round (6 min generation per prompt per rank, ~2 min judge, ~2 min park/update);
  judge adds no wall time because it overlaps generation.

## Progress

| epochs | mean reward (5-round avg) | reward_std range |
|---|---|---|
| 0-4 | 0.49 | 0.18-0.24 |
| 5-9 | 0.56 | 0.15-0.25 |
| 10-14 | 0.64 | 0.15-0.21 |

grad_norm 1e-4 and approx_kl ~0 at the first optimizer step are the same as `main_run1` (which learned);
the reward is moving while its within-batch std holds, which is the RewardDance collapse signal we watch.

Finished 2026-10-01 15:13 (60 epochs, one attempt). Rounds 30-59 stayed at a 5-round mean of 0.73-0.77 with
reward_std 0.06-0.25; checkpoints every 5 epochs.

## Held-out evaluation (115 safe test prompts, deterministic 1024/40 seed 0, one image per prompt and arm)

Eval dirs `/mnt/nvme/eval/vlm_stage1_s{0,1,2}`; base images are bit-identical to the earlier evals (seed 0), so
they double as the judge references.

Judge diagnostic (training reward, P(better than base), 0.5 = base) and image statistics:

| arm | judge | overall | structure | clean | brightness | saturation |
|---|---|---|---|---|---|---|
| base | 0.500 | 0.500 | 0.500 | 0.500 | 0.383 | 0.124 |
| ck10 | 0.669 | 0.726 | 0.617 | 0.607 | 0.397 | 0.129 |
| ck20 | 0.787 | 0.884 | 0.707 | 0.674 | 0.466 | 0.146 |
| ck30 | 0.825 | 0.942 | 0.732 | 0.681 | 0.479 | 0.142 |
| ck40 | 0.823 | 0.925 | 0.748 | 0.693 | 0.472 | 0.131 |
| ck50 | 0.841 | 0.942 | 0.758 | 0.721 | 0.481 | 0.138 |
| ck60 | 0.833 | 0.927 | 0.746 | 0.732 | 0.519 | 0.145 |

Independent rewards never used in training (`/mnt/nvme/aes2/score_eval.py`, win = share of prompts where the
checkpoint beats base):

| reward | base | ck10 | ck30 | ck50 | ck60 | ck30 win |
|---|---|---|---|---|---|---|
| HPSv3 | 8.91 | 9.49 | **10.41** | 10.36 | 10.34 | 85% |
| PickScore | 0.828 | 0.834 | 0.845 | **0.847** | 0.842 | 65% |
| SigLIP aesthetic | 5.95 | 6.04 | 6.09 | 6.13 | **6.23** | 57% |

Blind A/B (4 independent model judges, 230 pairs, sides randomized, 32 overlap pairs: 24/32 agree):

| arm | preferred over base | W/L/T | adherence delta (1-5 scale) | drift flags ck / base |
|---|---|---|---|---|
| ck30 | **80.4%** [73.5, 87.0] | 90/20/5 | **+0.37** [+0.08, +0.64] | 17 / 10 (of 131) |
| ck60 | 69.1% [60.9, 77.4] | 77/33/5 | +0.02 [-0.27, +0.31] | 17 / 11 |

Reading: ck30 is the best checkpoint of any run so far (previous best: HPSv3+SigLIP `main_run1` ck10 at 72.5%),
and it is the first one whose prompt adherence improved rather than merely held. From ck30 to ck60 the judge
score barely moves but blind preference and adherence fall back while brightness keeps rising: the judge's
"overall" question saturates (0.94) and the remaining optimization pressure goes into a brighter, glossier
look that the judge does not penalize. The judges' notes name the drift: brighter/pastel "polished digital
art" defaults, dropped style qualifiers (named painters, "muted", "charcoal"), occasional melted or duplicated
structure.

Decision: ck30 is the stage-1 result. Stage 2 (`vlm_stage2`, started 18:53) warm-starts its LoRA, moves the two
trained SDE steps to [10, 30] and judges with `qwen3vl32b_pairwise_stage2.yaml` (structure 1.0, clean 1.0,
overall 0.5 with "washed-out brightness" added to the color-cast clause).

## Open items

- Brightness drift is the shortcut of this judge: add a brightness/contrast transform to the shortcut gate and,
  if it passes, a "no washed-out brightness" clause did not suffice, add an explicit exposure criterion.
- Blind labels are model judgments, not human preference.

## Anchored run `vlm_natural_tonal1` (natural criteria + tonal anchor, from base, 40 epochs, 2026-10-02)

Reward `/reward=vlm_preference_http_tonal`: natural judge (`qwen3vl32b_pairwise_natural.yaml`) × 1 + `tonal_anchor` × 2
(bands 0.04 luma / 0.06 saturation around the base reference). Training reward rose only from 0.27 to ~0.40 (5-round
means); the judge term 0.48 → 0.58, the anchor −0.12 → −0.09, luma delta kept inside ±0.05.

Held-out (115 safe prompts, same protocol; `s1ck30` = stage-1 checkpoint-30 as a reference arm):

| arm | natural judge | fidelity | structure | brightness | HPSv3 | PickScore | blind pref vs base | adherence Δ | flagged fake (≥1 of 2 judges) |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.500 | 0.500 | 0.500 | 0.383 | 8.91 | 0.828 | -- | -- | 8 / 115 |
| ck40 | 0.558 | 0.671 | 0.698 | 0.335 | 9.97 | 0.840 | **68.3%** [60.0, 76.1] | +0.28 | 24 / 115 |
| s1ck30 | 0.465 | 0.693 | 0.797 | 0.479 | 10.41 | 0.845 | 80.4% (earlier eval) | +0.37 | 20 / 115 |

Head-to-head ck40 vs s1ck30 (115 pairs): s1ck30 preferred 58.5% [49.3, 67.6]; ck40 better adherence (+0.10) and fewer
fake votes (12 vs 21 of 128), but the per-image flag counts above are a wash. Judge overlap agreement 20/32.

What the fake flags correlate with (44 flagged of 230 trained-arm images): the natural judge's own score, AUC 0.59
(fake images score lower); brightness 0.52 and saturation 0.51 (nothing); edge energy 0.63 (fake images are smoother).
The flagged look is texture and style -- glossy toy-like eyes, airbrushed skin and fur, glow halos, sparkle FX -- not
global tonality.

Reading: the anchor did what it was built for (brightness 0.335 vs 0.479, adherence up) and the natural judge rates the
anchored policy above base where it rates stage-1 below base, but RL under either judge still doubles-to-triples the
share of images the blind judges call fake (8 → 20-24 of 115), and nothing we score today detects that look well
(best AUC 0.63). Run continued to 80 epochs for the appeal curve; human fake labels (labelling page) are the next
input before another reward change.
