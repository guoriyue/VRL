# Reward Configuration Guide

Reward configs are single-component building blocks. Experiment configs may
compose them with `reward.components` and override per-component
`reward.kwargs.<component>.score_key` when a compound recipe needs a narrower
signal than the base reward default.

## Video Score Keys

Do not treat every video reward's default score as an orthogonal training
signal. In particular, Kling VideoReward's `overall_reward` is
`VQ + MQ + TA`, so it already includes prompt/text alignment.

The aggregate score key differs per reward: it is `overall_reward` in Kling
VideoReward, `overall` in VideoCon-Physics, and `overall` in VideoScore2 (the
mean of its three axes). Each reward keeps its own deliberate vocabulary — see
the per-reward score-key table below.

Use these conventions for compound video recipes:

| Scenario | Component | Score key | Reason |
|---|---|---|---|
| Single Kling baseline | `kling_video_reward` | `overall_reward` | A standalone baseline can use the model's aggregate score. |
| Motion or physics compound | `kling_video_reward` | `motion_quality` | Keeps Kling focused on motion and avoids duplicating prompt-alignment rewards. |
| Visual-quality compound | `kling_video_reward` | `visual_quality` | Keeps Kling focused on visual quality. |
| Physical commonsense | `videocon_physics` | `physical_commonsense` | Avoids mixing VideoCon semantic adherence into prompt-alignment rewards. |
| Physical/common-sense judge | `videoscore2` | `physical_common_sense` | VideoScore2's naturalness/physics axis; pair with Kling `motion_quality` instead of duplicating text alignment. |
| Visual-quality judge | `videoscore2` | `visual_quality` | VideoScore2's clarity/artifact axis as a learned second opinion to Kling `visual_quality`. |
| Rubric / cloth-physics judge | `unified_reward_video` | `physics` | UnifiedReward-2.0's physics axis; steer it at the dress/skirt question via `worker_config.rubric_path`. |
| Human dynamics (external) | `phymotion` | `overall` | SMPL+MuJoCo kinematic/contact/dynamic feasibility via an external PhyMotion env (opt-in). |
| Robot target matching | `target_video_similarity` | `target_similarity` | Compares generated V2W output with manifest `target_video` / `target_image`; use as a probeable task signal, with Kling as a guard. |

Keep `configs/reward/kling_video_reward.yaml` on `overall_reward` for single-reward
baselines. Compound experiments should override the score key explicitly, e.g.
`configs/experiment/diffusion/wan_2_1/online_grpo_physics.yaml` uses
`kling_video_reward.score_key=motion_quality` and
`videocon_physics.score_key=physical_commonsense`.

The current `reward.components` schema is a mapping, so one recipe cannot use
two differently configured instances of the same component. If a future recipe
needs both Kling `visual_quality` and `motion_quality`, add explicit registry
aliases such as `kling_video_reward_vq` / `kling_video_reward_mq`, or change the schema to
a list of component instances in a separate sprint.
