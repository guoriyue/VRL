# Sprint: VideoCon-Physics reward + VideoPhy prompts for Wan 2.1 GRPO

## Goal

Add a physics-grounded reward signal to the running Wan 2.1 1.3B GRPO loop and
combine it with the existing Kling MQ reward so the policy gradient sees both
motion quality AND physical commonsense. Replace the generic PickScore prompt set
with VideoPhy's 688 physics-grounded prompts so every rollout exercises a real
physical interaction.

## Background

Current run (`outputs/wan_1_3b_motion_physics_run01`) uses Kling VideoReward MQ as
the only reward. Kling MQ's "Naturalness" sub-rubric covers physical-law alignment
qualitatively, but it is a single learned signal trained on generic motion
quality, not a physics-specialized model. Off-the-shelf alternative
`videophysics/videocon_physics` (Qwen2-VL-7B VLM judge, ICLR 2025) is a drop-in
clone of the Kling Ray-actor pattern that outputs an explicit
`physical_commonsense` Likert scalar.

**Answer to the originating user question** ("just reward or also dataset?"): the
reward + dataset are technically decoupled in the RewardFunction contract
(`vrl/rewards/base.py:80-90` only sees `(prompt, video)`), so a new reward works
on any manifest. BUT generic prompts (portraits, abstract art) give a near-zero
physics gradient because most rollouts never attempt a physical interaction. The
dataset switch is what makes the new reward useful in practice.

## Scope (in)

1. **New reward**: VideoCon-Physics as a second Ray-actor reward, cloning the
   Kling pattern.
2. **Reward composition**: `MultiReward` weighted sum
   `{video_reward: 0.7, videocon_physics: 0.3}` — both rewards declare
   `share_with_rollout: true` + `release_after_score: true` and run sequentially
   per epoch on the single 1×RTX 5090.
3. **New dataset**: VideoPhy released 688 physics prompts (solid-solid,
   solid-fluid, fluid-fluid) → `datasets/videophy_688/{train,eval}.txt`.
4. **New experiment yaml**: `online_grpo_physics.yaml` that wires reward +
   dataset + recipe together.

## Scope (out)

- Training new physics models from scratch.
- Replacing Kling MQ entirely (decision: combine, not replace).
- Adding optical-flow / RAFT-based hand-rolled physics signals (Kling MQ already
  covers amplitude/stability; redundant).
- 14B Wan variant — 1.3B first, scale later if reward curves trend up.

## Deliverables

### Code (new files)

| Path | Purpose | Template to clone |
|---|---|---|
| `vrl/rewards/models/videocon_physics.py` | `VideoConPhysicsModel(RewardModel)` — loads `videophysics/videocon_physics`, runs Qwen2-VL inference, returns `{semantic_adherence, physical_commonsense, overall}` | `vrl/rewards/models/kling_video_reward.py:181-251` (helpers `_resolve_model_root`, `_torch_dtype`, frame sampling are reusable as-is) |
| `vrl/rewards/functions/videocon_physics.py` | `VideoConPhysicsReward(RewardFunction)` thin wrapper calling `_init_reward_model(inference_runtime="ray", media_type="video", model_factory=...)` | `vrl/rewards/functions/video_reward.py:23-93` |
| `configs/reward/videocon_physics.yaml` | Reward block: components, kwargs, distributed resources (share_with_rollout, release_after_score) | `configs/reward/video_reward.yaml` |
| `datasets/videophy_688/train.txt` | One prompt per line, ~619 prompts (90% of 688) | `datasets/pickscore_sfw/train.txt` |
| `datasets/videophy_688/eval.txt` | One prompt per line, ~69 prompts | `datasets/pickscore_sfw/test.txt` |
| `configs/dataset/videophy_688.yaml` | `loader: prompt_manifest`, paths to the two txt files | `configs/dataset/video_t2v_pickscore_sfw.yaml` |
| `configs/experiment/diffusion/wan_2_1/online_grpo_physics.yaml` | End-to-end experiment: defaults + `reward.components: {video_reward: 0.7, videocon_physics: 0.3}` + score_keys + output_dir | `configs/experiment/diffusion/wan_2_1/online_grpo_video_reward.yaml` |

### Code (modified files)

| Path | Change |
|---|---|
| `vrl/rewards/functions/registry.py:31-55` | One-line import + `_REWARD_REGISTRY["videocon_physics"] = VideoConPhysicsReward` |

### Data ingestion (one-off, not in repo)

A small script (does not need to live in repo) to fetch
`videophysics/videophy_test_public` from HuggingFace, extract the 688 captions,
write them line-by-line to `datasets/videophy_688/{train,eval}.txt`. Run once,
commit the resulting `.txt` files only.

## Critical files to reference (don't rewrite)

- `vrl/rewards/models/kling_video_reward.py:181-251` — Ray reward-model template
- `vrl/rewards/functions/video_reward.py:23-93` — thin RewardFunction wrapper
  showing the Ray-actor + artifact-store pattern
- `vrl/rewards/functions/registry.py:96-115` — `MultiReward.from_dict` already
  supports weighted combination, **no changes needed**
- `vrl/rewards/base.py:103-130` — `_init_reward_model` boilerplate that all
  model-backed rewards reuse
- `vrl/trainers/data/prompts.py:27-50` — auto-detecting `.txt`/`.jsonl` manifest
  loader, no registry to update for a new dataset

## Sequencing

1. **Data first** (no GPU, cheap): write the HF→txt fetch script, produce
   `datasets/videophy_688/{train,eval}.txt`, then `configs/dataset/videophy_688.yaml`.
   Verify with `python -c "from vrl.trainers.data.prompts import
   load_prompt_manifest; print(len(load_prompt_manifest('datasets/videophy_688/train.txt')))"`.

2. **Reward model wrapper**: write `vrl/rewards/models/videocon_physics.py`,
   register in registry, write `configs/reward/videocon_physics.yaml`. Smoke-test
   in isolation with a single offline video before wiring into a training run.

3. **Wire experiment yaml**: write `online_grpo_physics.yaml` with weighted
   components. Sanity-check with `load_config(...)` (same dry-validate pattern
   used earlier today: `cfg.reward.components` should print
   `{'video_reward': 0.7, 'videocon_physics': 0.3}`).

4. **Launch one training run**, watch `metrics.csv` for the new per-component
   columns (`r_video_reward`, `r_videocon_physics`) and a weighted `reward_mean`
   that sits in the convex hull of the two component means. Per-epoch wall-clock
   will grow from ~84s to ~110s (two reward models load sequentially).

## Risks + mitigations

| Risk | Mitigation |
|---|---|
| Two Ray-actor rewards both declare `share_with_rollout: true` — resource resolver may not serialize cleanly. | Verify `vrl/generation/ray/launcher.py` actually tears down one reward actor before loading the next. If it deadlocks, fall back to a v2 design that loads both reward weight sets into a single Ray actor with two `__call__` paths. |
| VideoCon-Physics scores on a different scale than Kling MQ z-scores; weighted sum may be dominated by the larger-magnitude reward. | After 10 epochs, inspect per-component `last_components` distribution. If one dominates, normalize VideoCon-Physics by its own inference_config (same trick `kling_video_reward.py:337-347` uses with `_mean`/`_std`). |
| VideoPhy 688 is small; GRPO may overfit prompt-specific tricks. | If `reward_mean` keeps climbing but eval prompts don't improve, expand the manifest with motion-keyword-filtered pickscore_sfw entries. v1 is `videophy_688` only by user choice. |
| Per-epoch time grows ~30%. | Measured 84s→110s on 1×5090. Acceptable for 1×GPU sprint. If unblocking 14B later, batch both rewards into one actor. |
| Checkpoint save bug (already fixed at `vrl/trainers/checkpointing.py:153`) regression on the new run. | First save at epoch 20 is the canary. Existing run already verified the fix works. |

## Verification

End-to-end smoke on 1×RTX 5090 32GB:

1. **Manifest count check**: `wc -l datasets/videophy_688/train.txt` → ~619.
2. **Reward unit smoke**: instantiate `VideoConPhysicsReward(...)` outside any
   Ray cluster, confirm class import + registry lookup work.
3. **Config compile**: `python -c "from vrl.config.loading import load_config;
   c = load_config('experiment/diffusion/wan_2_1/online_grpo_physics');
   print(dict(c.reward.components), c.data.manifest)"` — expect
   `{'video_reward': 0.7, 'videocon_physics': 0.3}` and the videophy manifest path.
4. **First epoch ground truth**: launch the experiment, watch
   `outputs/wan_1_3b_physics/metrics.csv` for non-NaN `r_video_reward` AND
   `r_videocon_physics` columns within ~110s.
5. **Checkpoint at epoch 20**: must produce `checkpoint-20/checkpoint.pt` +
   `lora_weights/adapter_model.safetensors` + `checkpoint_meta.json` (the JSON
   file is the regression canary for the recently-patched bug).
6. **Visual sanity**: pick 4 held-out prompts from `eval.txt` (one solid-solid
   collision, one solid-fluid pour, one fluid-fluid mix, one free choice),
   generate from the trained LoRA, compare to checkpoint-20 of the
   MQ-only run. Physics-related failure modes (intersection, gravity violation,
   liquid passing through container) should be visibly less frequent.
