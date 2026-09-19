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

Reward execution is selected per component and always runs outside the
trainer process: a reward scoring in the driver competes with the launch-bound
replay for the interpreter (measured +61 s per epoch on SD3.5 under continuous
scheduling). The default, `ray`, runs each component in a driver-owned Ray
actor with placement and memory parking derived from the run topology.
Media travels through Ray without driver-side image or video files. Models
that need a file decoder materialize temporary inputs inside their scoring
process. `kind: http` connects to
an operator-owned service instead, with typed transport config in the
`reward.inference` section, keyed by component name:

```yaml
reward:
  components:
    unified_reward_video: 1.0
  inference:
    unified_reward_video:
      kind: http
      endpoint: http://reward.internal:8300
      timeout_s: 1800
      expected_model: unified-reward-2.0
      expected_model_version: UnifiedReward-2.0@pinned-revision
```

`reward.kwargs.<component>` holds constructor arguments only; transport config
never nests inside it.

Do not put `worker_config`, `device`, or parking fields on an HTTP component.
Those belong to the standalone service config. HTTP scoring uploads media bytes;
the trainer and service do not need a shared filesystem. External-only rewards receive no local Ray
resource bundle. `expected_model` is required; set `expected_model_version` for
fixed reward protocols so preflight also rejects a same-name service running a
different checkpoint, revision, or threshold.

Scoring does not require `media_type`, `artifact_format`, `artifact_dir`, or
`retain_artifacts` constructor settings. The model owns its decoder format.
For an explicit experiment archive, set `reward.kwargs.<component>.archive_dir`;
these saved copies are separate from the inputs sent for scoring. `debug_dir`
remains available for reward-specific diagnostic records. GenEval's evaluator
`artifact_dir` is a separate model argument, not a transport directory.

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

## GenEval (OWLv2 + CLIP)

Select `+reward=geneval_owl` with a manifest whose rows carry `metadata.geneval`
(the Flow-GRPO GenEval manifests under `manifests/geneval`, including the anime
restyle selected by `+dataset=geneval_anime`). The official GenEval decision
rules run in-process over `google/owlv2-base-patch16-ensemble` detections and
`openai/clip-vit-large-patch14` colour classification, so no mmdet stack or
reward server is needed; absolute scores are therefore not comparable to
published GenEval tables, only before/after on this detector.

The `geneval_owl` score is the fraction of satisfied conditions (object
presence, count, colour, relative position); the strict all-or-nothing verdict
is a per-prompt reduction of the same conditions. The model stays resident
on the reward device (~2.3 GB). `geneval` remains the adapter for an external
evaluator supplied by `import_path`.

## Video Score Keys

Do not treat every video reward's default score as an orthogonal training
signal. In particular, Kling VideoReward's `overall_reward` is
`VQ + MQ + TA`, so it already includes prompt/text alignment.

The aggregate score key differs per reward: it is `overall_reward` in Kling
VideoReward, and `overall` in VideoCon-Physics. Each reward keeps its own deliberate vocabulary — see
the per-reward score-key table below.

Use these conventions for compound video recipes:

| Scenario | Component | Score key | Reason |
|---|---|---|---|
| Single Kling baseline | `kling_video_reward` | `overall_reward` | A standalone baseline can use the model's aggregate score. |
| Motion or physics compound | `kling_video_reward` | `motion_quality` | Keeps Kling focused on motion and avoids duplicating prompt-alignment rewards. |
| Visual-quality compound | `kling_video_reward` | `visual_quality` | Keeps Kling focused on visual quality. |
| Physical commonsense | `videocon_physics` | `physical_commonsense` | Avoids mixing VideoCon semantic adherence into prompt-alignment rewards. |
| Rubric / cloth-physics judge | `unified_reward_video` | `physics` | UnifiedReward-2.0's physics axis; steer it at the dress/skirt question via `worker_config.rubric_path`. |
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

## OCR over the standalone PaddleOCR service

Two ways to keep PaddleOCR out of the trainer process (measured: in-process
OCR under continuous scheduling cost 61 s per epoch of launch-bound replay):

- `/reward/ocr` uses a driver-owned Ray actor by default. The actor receives
  `reward.kwargs.ocr` and scores media through Ray; no HTTP server or manual
  startup is needed. The removed `kind=service` override must be dropped or
  replaced with `kind=ray`.
- `/reward=ocr_http` (self-contained preset): an operator-run service
  (`vrl/config/reward_service/ocr_paddle.yaml`):

```bash
.venv/bin/python -m vrl.rewards.service.server \
  --config vrl/config/reward_service/ocr_paddle.yaml
```
