# Reward Configuration Guide

Reward configs are single-component building blocks. Experiment configs may
compose them with `reward.components` and override per-component
`reward.kwargs.<component>.score_key` when a compound recipe needs a narrower
signal than the base reward default.

## Video Score Keys

Do not treat every video reward's default score as an orthogonal training
signal. In particular, Kling VideoReward's `overall_reward` is
`VQ + MQ + TA`, so it already includes prompt/text alignment.

Use these conventions for compound video recipes:

| Scenario | Component | Score key | Reason |
|---|---|---|---|
| Single Kling baseline | `video_reward` | `overall_reward` | A standalone baseline can use the model's aggregate score. |
| Motion or physics compound | `video_reward` | `motion_quality` | Keeps Kling focused on motion and avoids duplicating prompt-alignment rewards. |
| Visual-quality compound | `video_reward` | `visual_quality` | Keeps Kling focused on visual quality. |
| Physical commonsense | `videocon_physics` | `physical_commonsense` | Avoids mixing VideoCon semantic adherence into prompt-alignment rewards. |
| Claude video judge | `claude_video` | rubric overall | Keep rubric axis weights inside the rubric rather than duplicating them in `MultiReward`. |

Keep `configs/reward/video_reward.yaml` on `overall_reward` for single-reward
baselines. Compound experiments should override the score key explicitly, e.g.
`configs/experiment/diffusion/wan_2_1/online_grpo_physics.yaml` uses
`video_reward.score_key=motion_quality` and
`videocon_physics.score_key=physical_commonsense`.

The current `reward.components` schema is a mapping, so one recipe cannot use
two differently configured instances of the same component. If a future recipe
needs both Kling `visual_quality` and `motion_quality`, add explicit registry
aliases such as `video_reward_vq` / `video_reward_mq`, or change the schema to
a list of component instances in a separate sprint.
