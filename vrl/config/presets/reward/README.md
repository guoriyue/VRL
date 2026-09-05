# Reward Configuration Guide

Reward configs are reusable building blocks, not generator-specific reward
implementations. Select them at launch with `+reward=...` over a reward-neutral
execution recipe; do not add one experiment YAML for every model/reward/dataset
combination. Multiple selections merge in command order. See
[runtime composition](../../../../docs/CONFIGURATION.md).

Experiment runs may compose them with `reward.components` and override per-component
`reward.kwargs.<component>.score_key` when a compound recipe needs a narrower
signal than the base reward default.

## Inference Deployment

Reward execution is selected per component. `in_process` is the default and is
the only supported heavy-reward mode when trainer, rollout, and reward share one
GPU. An operator-owned service uses typed transport config in the
`reward.inference` section, keyed by component name:

```yaml
reward:
  components:
    videoscore2: 1.0
  kwargs:
    videoscore2:
      artifact_dir: /shared/vrl/reward_artifacts
  inference:
    videoscore2:
      kind: http
      endpoint: http://reward.internal:8300
      timeout_s: 1800
      expected_model: videoscore2-v1
      expected_model_version: VideoScore2@pinned-revision
```

`reward.kwargs.<component>` holds constructor arguments only; transport config
never nests inside it.

Do not put `worker_config`, `device`, or parking fields on an HTTP component.
Those belong to the standalone service config. HTTP scoring currently uses
integrity-checked shared-filesystem paths, so the trainer and service must see
the same absolute `artifact_dir`. External-only rewards receive no local Ray
resource bundle. `expected_model` is required; set `expected_model_version` for
fixed reward protocols so preflight also rejects a same-name service running a
different checkpoint, revision, or threshold. If a transport failure leaves the remote request state
unknown, VRL retains that request's artifacts for operator cleanup rather than
deleting files that the service may still be reading.

Generation/reward streaming is capability-derived. It is enabled only when no
GPU phase handoff is required and every reward component is both non-blocking
and physically isolated from generation. HTTP is only a transport: an external
service stays fail-closed unless its `/info` response advertises the
`generation_overlap_safe` capability. The standalone service emits that
capability for an explicit `generation_overlap_safe: true` operator attestation,
or for an explicitly configured CPU device. Never attest a GPU service that can
resolve to any trainer or generation GPU. Strict scheduling keeps an unverified
service on the batched-serial path; continuous scheduling rejects it because even
one reward call would overlap trainer backward. In-process rewards retain one
batched scoring call even on a dedicated GPU; this avoids trading batch
throughput for fake event-loop concurrency.

## CountGD object counting

Select `+reward=countgd_http` with any reward-neutral generator recipe. Start
the shared CPU service from the existing isolated CountGD installation:

```bash
data/external/countgd/env/bin/python -m vrl.rewards.service.server \
  --config vrl/config/reward_service/countgd.yaml
```

Each JSONL prompt row supplies its own counting target, for example:

```json
{"prompt": "Three cats sitting on a sofa.", "metadata": {"object_class": "cat", "expected_count": 3}}
{"prompt": "Two cups on a wooden table.", "metadata": {"object_class": "cup", "expected_count": 2}}
```

One batch can contain different classes; no per-class reward or experiment YAML
is needed. `object_class` is a non-empty text description, not a closed label
list. Use one class or noun phrase per image (for example, `red apple`), not
multiple classes separated by periods or question marks (these are rejected).
The adapter normalizes case, whitespace at
the ends, and trailing periods before passing the description to CountGD.
`expected_count` must be an integer >= 0. The reward is 1 only when the detected
count exactly matches; it is otherwise 0. Neither value is inferred from the
generation prompt. This is image counting, not video tracking.

The generic names replace `countgd_person_count` / `countgd_person_count_http`;
old run configs must use the new component/preset and convert `expected_people`
to `object_class: person` plus `expected_count`. Restart the service with the
new config: its protocol version intentionally differs from the person-only
version, so an old running service is rejected. The historical Anima paired
evaluator translates its old person-count archives at the model boundary.

[CountGD](https://github.com/niki-amini-naieni/CountGD) supports text-conditioned
open-world counting, but accepting a class description does not establish its
accuracy on that class or style. The previous Anima person-count experiment
found detector-only gains under optimization. Audit held-out images before
using this signal for training; zero-count rewards can also reward missed
detections. Checkpoint/runtime pins and the 0.23 detection threshold are unchanged.

## WD tagger

Select `+reward=wd_tagger` with any reward-neutral generator recipe. The default
model is `SmilingWolf/wd-swinv2-tagger-v3`, loaded in-process on CPU. Each prompt
row must supply the general tags to check, using the model's `selected_tags.csv`
vocabulary:

```json
{"prompt": "A smiling girl with long hair.", "metadata": {"adherence_tags": ["long_hair", "smile"]}}
```

The `wd_tagger` score is the fraction of requested tags detected at the configured
threshold (default 0.35). It does not penalize extra tags or measure full prompt
understanding, spatial relationships, or aesthetics. A repeatable tagger can
still misclassify images; validate its predictions on the intended data before
training. Character and rating tags are not scored.

`wd_tagger` replaces the old `tag_adherence` component, preset, and score key;
update existing launch overrides accordingly. The `adherence_tags` metadata
field and scoring behavior are unchanged.

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
| Robot V2W perceptual anchor | `target_dino_similarity` | `target_dino_similarity` | Frozen-DINOv2 cosine + temporal term vs manifest `target_video` / `target_image`; keeps frames on the real-image manifold. Zero-training (pretrained). |
| Robot V2W motion guard | `motion_dynamics` | `motion_dynamics` | RAFT optical-flow Dynamic Degree; a hard floor under static/blur collapse. |
| Frame-aesthetic preference (Flash-GRPO) | `hpsv3` | `top_frame_mean` | HPSv3 image preference over the best 30% of frames (unbounded scale, ~-1..+12). Zero temporal signal — judge motion by watching samples, and watch `frame_min` in the debug records for best-frames-only hacking. |

Keep `reward/kling_video_reward` on `overall_reward` for single-reward
baselines. Compound experiments should override the score key explicitly, e.g.
`experiment/wan_2_1/online_grpo_physics` uses
`kling_video_reward.score_key=motion_quality` and
`videocon_physics.score_key=physical_commonsense`.

The current `reward.components` schema is a mapping, so one recipe cannot use
two differently configured instances of the same component. If a future recipe
needs both Kling `visual_quality` and `motion_quality`, add explicit registry
aliases such as `kling_video_reward_vq` / `kling_video_reward_mq`, or change the schema to
a list of component instances in a separate sprint.
