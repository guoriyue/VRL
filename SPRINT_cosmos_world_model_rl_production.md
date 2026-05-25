# SPRINT: Cosmos World Model RL Production Run

## 0. Core Decision

本 sprint 只负责 Cosmos world model production RL。

目标是把已经验证过的 Cosmos world datasets 接到真实 `video_reward`，完成两条可审计 production route：

```text
Cosmos Predict2 Video2World + per-sample reference_image + real video_reward
Cosmos Predict2.5 DiffusionNFT Text2World + real video_reward
```

不在本 sprint 里重新设计 dataset。Prompt-only manifests 的 source of truth 是 committed `datasets/**` manifests, their `report.json` provenance files, and task/dataset configs. Reference-image / external artifact contract 的 source of truth 是：

```text
vrl/scripts/data/video_world.py        # video-world-bridge first-frame importer
vrl/scripts/diffusion/cosmos/train.py  # reference_image resolution rules
```

核心判断：

- Cosmos Predict2 Video2World 是 reference-conditioned world generation，必须证明 `reference_image` 真实进入 rollout。
- Cosmos Predict2.5 DiffusionNFT 是 Text2World world-generation route，必须保持 `model.use_lora=true`。
- Cosmos 验收重点是 reference consistency、motion quality、visual quality、physical plausibility。
- 完成标准是 production run artifact、非平 reward、权重变化、base-vs-trained eval，不是单纯 config load。
- "Production" means source-backed data, real reward scoring, auditable artifacts, and
  acceptance evidence. Do not add `_production` to config filenames just to signal intent.
  Stable route names are preferred; production readiness is proven by the run report.

## 1. Upstream Dataset Dependency

本 sprint 依赖 upstream dataset sprint 产出并验证：

```text
datasets/video_world/v2w_train.jsonl
datasets/video_world/v2w_eval.jsonl
datasets/video_world/references/
source-backed text-conditioned world/video train/eval manifests
```

`v2w_train.jsonl` / `v2w_eval.jsonl` are not present in the repo yet. Add the dataset
config only after a source-backed importer creates and validates those manifests.

V2W dataset acceptance is intentionally strict. The manifest is not "real" merely because
it has prompts and image paths. It counts only if every committed row is traceable to an
approved upstream source, the referenced first-frame image is decoded from that source
rather than generated or hand-authored, and the dataset report records the upstream repo,
split, episode/video id, frame index, decoding method, row count, and validation summary.

本 sprint 不重新定义：

```text
coverage_buckets
approved_sources
caption rules
filtering rules
manifest row examples
minimum dataset sizes
```

### 1.1 Resolved: video_world importer decodes real LeRobot v2.0 + v2.1 data

抓手：`vrl/scripts/data/video_world.py` 的 `video-world-bridge`
(`_iter_lerobot_first_frames`)。

之前 importer 假设 `datasets.load_dataset(streaming=True)` 每行带 inline image +
caption，真实 LeRobot 不是这样，所以跑不通。现已按真实格式重写，按
`codebase_version` 自动分流两种 layout：

```text
v2.1 (DROID):  meta/tasks.parquet + data/chunk-*/file-*.parquet（多 episode 聚合）
               + videos/<cam>/chunk-*/file-*.mp4，按全局 index 解码第一帧
v2.0 (Bridge): meta/tasks.jsonl + meta/episodes.jsonl（caption 直接在 episodes）
               + data/.../episode_N.parquet + videos/.../episode_N.mp4（每 episode 一个
               mp4，第一帧=frame 0）
```

依赖：`pyarrow` + `av` + `huggingface_hub` + `pillow`（均已装，无新增重依赖；不需要
`lerobot` 库）。默认 `--repo-id lerobot/droid_100`；Bridge 用
`IPEC-COMMUNITY/bridge_orig_lerobot`。frame 经 PyAV HTTP 流式解码。

验证（真实数据，2026-05-25）：

```text
droid_100  (v2.1): --limit 4 => 真实 320x180 RGB PNG + caption "Put the marker in the pot"
bridge_orig(v2.0): --limit 3 => 真实 256x256 RGB PNG + caption "put small spoon from basket to tray"
两者 manifest 均通过 cosmos _normalize_per_sample_reference_images。
```

已知边界（非阻塞）：v2.1 v1 只读第一个 data/video file（`--limit` 上限即该 file 内的
episodes，跨 file 全量是后续）；v2.0 每 episode 独立 mp4，`--limit` 直接生效。fetch
路径依赖网络，无 unit test（transform 有离线 test，fetch 靠真实 run 验证）。

注意：本项只解决 **dataset** 侧。真实 Cosmos training run 仍另外依赖 reward
backend；Kling VideoReward inference is repo-owned under `vrl/rewards/models`,
so this route must not depend on an external VideoAlign checkout on `PYTHONPATH`
or a git submodule.

## 2. Current Repo Reality

已有 Cosmos 训练入口：

```text
Cosmos Predict2 Video2World:
  configs/experiment/diffusion/cosmos_predict2/online_grpo_video_reward.yaml
  vrl.scripts.diffusion.cosmos.train:train_cosmos_predict2_grpo

Cosmos Predict2.5 Text2World / DiffusionNFT:
  configs/experiment/diffusion/cosmos_predict2_5/online_nft_video_reward.yaml
  vrl.scripts.diffusion.cosmos.train:train_cosmos_predict25_diffusion_nft
```

已有 prompt loader：

```text
vrl/trainers/data/prompts.py
```

`PromptExample` 已支持本 sprint 需要的字段：

```text
prompt
reference_image
reference_video
references
task_type
request_overrides
metadata
```

已有 reward runtime：

```text
configs/reward/video_reward.yaml
vrl/rewards/functions/video_reward.py
vrl/rewards/models/kling_video_reward.py
vrl/rewards/ray/runtime.py
vrl/rewards/ray/worker.py
```

当前 production reward contract：

```text
driver: VideoReward 负责把 rollout video materialize 成 artifact，并发给 Ray reward actors。
worker: RewardModelWorker 加载 KlingVideoRewardModel；内部 model_factory 由 reward_name 派生。
public YAML: 只暴露 reward_name、score_key、media_type、artifact/debug dirs 和简短 worker_config。
legacy fields: backend / endpoint URLs / scorer / import_path / model_factory 都不是 public route config。
```

`reward_name=KlingTeam/VideoReward@main` 是 public model selector。Kling loader
把 VideoReward 原始输出映射成这些 public score keys：

```text
overall_reward
visual_quality
motion_quality
text_alignment
```

## 3. Production Routes

### 3.1 Cosmos Predict2 Video2World Route

Goal:

- Consume `datasets/video_world/v2w_train.jsonl`.
- Every row must carry `reference_image`.
- `cosmos.reference_mode` must be `per_sample`.
- Train with real `video_reward`.
- Evaluate on reference-conditioned fixed eval rows.

Required experiment:

```text
configs/experiment/diffusion/cosmos_predict2/online_grpo_v2w_reference.yaml
```

Future production defaults after manifest import:

```text
/recipe/online/diffusion_grpo
/model/diffusion/cosmos/predict2_2b
/sampling/video/704p_93f
/sampling/denoise/35_step_cfg_7
/reward/video_reward
future /dataset/video_world_v2w after manifest import
```

Required config decisions:

```yaml
cosmos:
  reference_mode: per_sample

model:
  reference_image: ""

reward:
  kwargs:
    video_reward:
      inference_runtime: ray
      reward_name: KlingTeam/VideoReward@main
      score_key: overall_reward
      media_type: video
      artifact_format: mp4
      worker_config:
        model_path: ""      # optional local Kling VideoReward checkpoint root
        dtype: bfloat16
```

Production guard:

- The V2W manifest must be source-backed; placeholder, synthetic, generated, or
  hand-authored reference images do not satisfy this sprint.
- Training must fail if all rows lack `reference_image`.
- Training must fail if referenced image files are missing.
- Debug report must record active reference paths.

### 3.2 Cosmos Predict2.5 DiffusionNFT Route

Goal:

- Consume a source-backed text-conditioned world/video train manifest.
- Train text-conditioned world generation with real `video_reward`.
- Keep `model.use_lora=true`.
- Prove optimizer step, non-flat reward, and changed LoRA weights.

Required experiment:

```text
configs/experiment/diffusion/cosmos_predict2_5/online_nft_video_reward.yaml
```

Production defaults after source-backed dataset import:

```text
/recipe/online/diffusion_nft
/model/diffusion/cosmos/predict2_5_2b
/sampling/video/512p_93f
/sampling/denoise/10_step_no_cfg
/reward/video_reward
future /dataset/<source_backed_world_t2v>
```

Production acceptance must record the dataset source/report in `outputs/cosmos_world_model_production_report.md`.

Required config decisions:

```yaml
model:
  use_lora: true

reward:
  kwargs:
    video_reward:
      inference_runtime: ray
      reward_name: KlingTeam/VideoReward@main
      score_key: overall_reward
      media_type: video
      artifact_format: mp4
      worker_config:
        model_path: ""      # optional local Kling VideoReward checkpoint root
        dtype: bfloat16
```

Do not set:

```text
model.use_lora=false
```

Reason:

```text
DiffusionNFT requires default + previous adapters.
```

## 4. Implementation Tasks

### Phase 1: Cosmos Route Configs

Add only when the route does not already have a clear config. Do not create duplicate
`*_production.yaml` files when an existing route config can carry the production-ready
settings.

Add or verify:

```text
configs/experiment/diffusion/cosmos_predict2/online_grpo_v2w_reference.yaml
configs/experiment/diffusion/cosmos_predict2_5/online_nft_video_reward.yaml
```

Edit:

```text
tests/config/test_load_all_experiments.py
vrl/config/validation.py
```

Required checks:

- Cosmos Predict2 V2W route config loads.
- Cosmos Predict2.5 DiffusionNFT route config loads.
- Route configs use `/reward/video_reward`.
- Route configs use `reward_name=KlingTeam/VideoReward@main`.
- Route configs use `score_key=overall_reward` unless explicitly testing a component score.
- Route configs do not expose legacy loader fields in `worker_config`
  (`import_path`, `model_factory`, `score_key_map`, `backend`, `backend_code_dir`).
- Runtime derives internal `worker_config.model_factory` from `reward_name`.
- Cosmos V2W config uses `cosmos.reference_mode=per_sample`.
- Cosmos Predict2.5 route config keeps `model.use_lora=true`.

### Phase 2: Reference Metadata Wiring

Edit or verify:

```text
vrl/trainers/data/prompts.py
vrl/rollouts/collector/requests.py
vrl/scripts/diffusion/cosmos/train.py
tests/rollouts/test_video_world_reference_metadata.py
```

Required checks:

- `PromptExample.reference_image` loads from JSONL.
- per-sample `reference_image` reaches collector request metadata.
- Cosmos per-sample reference mode rejects missing `reference_image`.
- relative reference paths resolve from manifest parent when appropriate.

### Phase 3: Cosmos Reward Runtime Contract

Edit or verify:

```text
configs/reward/video_reward.yaml
vrl/rewards/functions/video_reward.py
vrl/rewards/models/kling_video_reward.py
vrl/rewards/ray/runtime.py
tests/rewards/test_video_reward.py
tests/rewards/test_video_reward_versioning.py
tests/rewards/test_reward_resource_lifecycle.py
```

Required checks:

- reward debug rows record `reward_model_version`.
- reward debug rows record selected `score_key`.
- reward artifact manifest is written.
- actor runtime can be released after score when configured.
- Cosmos route configs reject legacy endpoint/scorer/loader fields and use the
  Kling VideoReward production contract.

### Phase 4: Cosmos Fixed Eval and Report

Add:

```text
outputs/cosmos_world_model_production_report.md
```

Report must include:

```text
config path
dataset manifest path
checkpoint path
reward name
reward score key
reward model version
reward backend code availability
number of train/eval rows
reward mean/std before training
reward mean/std after training
sample video artifact paths
active reference image paths for V2W
known failures
reward-hacking inspection notes
```

The report should link to:

```text
metrics.csv
reward_artifacts/manifest.jsonl
reward_debug/video_reward_requests.jsonl
reward_debug/video_reward_results.jsonl
checkpoint-final/
resolved_config.yaml
```

## 5. Production Run Requirements

Environment:

```text
No external VideoAlign PYTHONPATH or git submodule is required. The production
reward loader uses the repo-owned Kling VideoReward inference backend under
vrl/rewards/models/kling_video_reward.py.
# optional: set reward.kwargs.video_reward.worker_config.model_path to a local
# KlingTeam/VideoReward checkpoint root; otherwise the loader resolves
# KlingTeam/VideoReward@main through Hugging Face.
```

Required run artifacts:

```text
metrics.csv
reward_artifacts/manifest.jsonl
reward_debug/video_reward_requests.jsonl
reward_debug/video_reward_results.jsonl
checkpoint-final/lora_weights
resolved_config.yaml
```

Required metrics:

```text
reward_mean is finite
reward_std is finite
at least one reward component is non-flat
global_step > 0
trainable weights changed
```

Run acceptance:

- A Cosmos run without real video reward does not count as production completion.
- A Cosmos V2W run whose manifest cannot prove source-backed first-frame references does
  not count as production completion.
- A Cosmos run with flat real reward does not count as success.
- A Cosmos run with generated videos but no reward debug artifacts does not count as success.
- Cosmos V2W without active `reference_image` does not count as success.
- Cosmos Predict2.5 with `model.use_lora=false` does not count as success.

## 6. Acceptance Criteria

This sprint is complete only when:

- Cosmos Predict2 V2W reference route config loads.
- Cosmos Predict2 V2W train/eval manifests are source-backed and provenance-checked.
- Cosmos Predict2 V2W production run completes with real reward and per-sample reference image.
- Cosmos Predict2.5 DiffusionNFT route config loads.
- Cosmos Predict2.5 DiffusionNFT production run reaches optimizer step with real reward.
- Unit tests pass for config loading, reference image wiring, and production reward config.
- At least one Cosmos run writes `reward_artifacts/manifest.jsonl`.
- At least one Cosmos run writes `reward_debug/video_reward_results.jsonl`.
- At least one Cosmos run records non-flat reward.
- At least one Cosmos run proves trainable weights changed.
- `outputs/cosmos_world_model_production_report.md` records base-vs-trained results.

## 7. Out of Scope

Do not include in this sprint:

- Dataset source selection and coverage design. That belongs in committed dataset manifests, dataset reports, dataset configs, and the `vrl/scripts/data/` population scripts.
- Wan T2V production training.
- OCR reward acceptance for Cosmos world model.
- Action-conditioned robot world model.
- Multiview driving generation.
- Cosmos Transfer controlnet training.
- Full SFT over video datasets.
- Full-model finetune for Cosmos Predict2.5 DiffusionNFT.
- Replacing the reward service architecture.

## 8. Risks and Guards

Risk: real video reward is flat.

Guard:

- Require reward component debug rows.
- Require per-component score inspection.
- Fail production acceptance if reward std is zero across the training run.

Risk: Cosmos V2W silently falls back to zero conditioning.

Guard:

- Dataset validation requires `reference_image` upstream.
- Config uses `cosmos.reference_mode=per_sample`.
- Runtime/debug report records active reference paths.
- Tests assert per-sample request metadata carries `reference_image`.

Risk: prompt-only T2W is mistaken for Video2World conditioning.

Guard:

- Treat Predict2.5 T2W as Text2World production route only.
- Cosmos Predict2 V2W remains the route that proves reference-conditioned world generation.

Risk: dataset quality causes reward hacking.

Guard:

- Consume only validated upstream manifests.
- Reject manifests whose rows cannot be traced back to upstream dataset artifacts.
- Keep train/eval split.
- Keep KL enabled for GRPO.
- Track reference consistency, text alignment, visual quality, physical plausibility, and motion quality separately.
- Add manual inspection notes to the production report.

Risk: production route is too expensive.

Guard:

- Reduce batch size or resolution only through explicit production overrides.
- Do not replace the approved production reward implementation.
- Do not accept config-only completion as production readiness.

## 9. Verification Commands

Config and unit checks:

```bash
python -m pytest tests/config/test_load_all_experiments.py
python -m pytest tests/rewards/test_video_reward.py tests/rewards/test_video_reward_versioning.py
python -m pytest tests/rollouts/test_video_world_reference_metadata.py
python -m compileall vrl tests
```

Run-level checks:

```bash
test -f outputs/<cosmos-run>/metrics.csv
test -f outputs/<cosmos-run>/reward_artifacts/manifest.jsonl
test -f outputs/<cosmos-run>/reward_debug/video_reward_results.jsonl
test -d outputs/<cosmos-run>/checkpoint-final
```

## 10. References

Local files:

```text
/home/mingfeiguo/Desktop/VRL/vrl/scripts/data/video_world.py
/home/mingfeiguo/Desktop/VRL/docs/papers/cosmos_predict2_5_world_simulation_with_video_foundation_models_for_physical_ai_2511.00062v2.pdf
/home/mingfeiguo/Desktop/VRL/vrl/trainers/data/prompts.py
/home/mingfeiguo/Desktop/VRL/vrl/rewards/functions/video_reward.py
/home/mingfeiguo/Desktop/VRL/vrl/rewards/models/kling_video_reward.py
/home/mingfeiguo/Desktop/VRL/vrl/rewards/models/KLING_VIDEO_REWARD_NOTICE.md
/home/mingfeiguo/Desktop/VRL/vrl/rewards/ray/runtime.py
/home/mingfeiguo/Desktop/VRL/configs/reward/video_reward.yaml
/home/mingfeiguo/Desktop/VRL/configs/experiment/diffusion/cosmos_predict2/online_grpo_video_reward.yaml
/home/mingfeiguo/Desktop/VRL/configs/experiment/diffusion/cosmos_predict2_5/online_nft_video_reward.yaml
```

External references:

```text
https://arxiv.org/abs/2511.00062
https://github.com/nvidia-cosmos/cosmos-predict2.5
https://huggingface.co/KlingTeam/VideoReward
https://github.com/KwaiVGI/VideoAlign
```
