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

## 1. Upstream Dataset Dependency

本 sprint 依赖 upstream dataset sprint 产出并验证：

```text
datasets/video_world/v2w_train.jsonl
datasets/video_world/v2w_eval.jsonl
datasets/video_world/references/
source-backed text-conditioned world/video train/eval manifests
```

`v2w_train.jsonl` / `v2w_eval.jsonl` are not present in the repo yet. Add the production dataset config only after a source-backed importer creates and validates those manifests.

本 sprint 不重新定义：

```text
coverage_buckets
approved_sources
caption rules
filtering rules
manifest row examples
minimum dataset sizes
```

### 1.1 Open: video_world importer does not yet decode real LeRobot data

抓手：`vrl/scripts/data/video_world.py` 的 `video-world-bridge`
(`_iter_lerobot_first_frames`)。

现状（不要包装成"已完成"）：importer 目前**跑不通真实数据**。它假设
`datasets.load_dataset(streaming=True)` 的每行带 inline PIL image + caption
string，但真实 LeRobot v2.1 不是这样：

```text
帧：     videos/observation.images.<cam>/chunk-*/file-*.mp4   (mp4 视频，非 inline)
caption：meta/tasks.parquet                                   (task_index -> 文本)
逐帧：   data/chunk-*/file-*.parquet                          (episode_index/frame_index/task_index)
```

所以现在 column 检测拿不到 image/language，importer 直接报
`No usable episodes`（loud fail，不产假数据），等于真实 video_world dataset 仍未就绪。

已核对的真实 repo（shipped 默认 `lerobot/bridge_orig` 是 404，要改）：

```text
IPEC-COMMUNITY/bridge_orig_lerobot   # Bridge
lerobot/droid_100                    # DROID，小（10 files），适合验证
```

修复方案（不引新依赖，`pyarrow` / `av` / `imageio_ffmpeg` 已装）：

```text
1. 读 meta/tasks.parquet            -> task_index -> caption(prompt)
2. 读 data/*.parquet               -> 每个 episode 的 frame_index==0 行 + task_index
3. 用 meta/info.json 把 episode 映射到 mp4 chunk/file/timestamp，解码第一帧
4. 落 references/*.png + 带 caption 的 manifest（沿用现有 build_video_world_rows）
5. 修正默认 --repo-id
```

完成标准：`populate video-world-bridge --repo-id lerobot/droid_100` 写出真实
first-frame PNG + 带 caption 的 manifest，并对 `droid_100` 验证通过；transform
已有离线测试，fetch 路径补一条真实 droid_100 的解码冒烟验证。

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
vrl/rewards/ray/runtime.py
vrl/rewards/ray/worker.py
```

## 3. Production Routes

### 3.1 Cosmos Predict2 Video2World Production RL

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
      worker_config:
        scorer: import_path
        import_path: ???
        reward_model_version: ???
```

Production guard:

- Training must fail if all rows lack `reference_image`.
- Training must fail if referenced image files are missing.
- Debug report must record active reference paths.

### 3.2 Cosmos Predict2.5 DiffusionNFT Production RL

Goal:

- Consume a source-backed text-conditioned world/video train manifest.
- Train text-conditioned world generation with real `video_reward`.
- Keep `model.use_lora=true`.
- Prove optimizer step, non-flat reward, and changed LoRA weights.

Required experiment:

```text
configs/experiment/diffusion/cosmos_predict2_5/online_nft_video_reward_production.yaml
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
      worker_config:
        scorer: import_path
        import_path: ???
        reward_model_version: ???
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

### Phase 1: Cosmos Production Configs

Add:

```text
configs/experiment/diffusion/cosmos_predict2/online_grpo_v2w_reference.yaml
configs/experiment/diffusion/cosmos_predict2_5/online_nft_video_reward_production.yaml
```

Edit:

```text
tests/config/test_load_all_experiments.py
vrl/config/validation.py
```

Required checks:

- Cosmos Predict2 V2W production config loads.
- Cosmos Predict2.5 DiffusionNFT production config loads.
- Production configs use `/reward/video_reward`.
- Production configs use `worker_config.scorer=import_path`.
- Production configs require `VIDEO_REWARD_SCORER`.
- Production configs require `VIDEO_REWARD_MODEL_VERSION`.
- Cosmos V2W config uses `cosmos.reference_mode=per_sample`.
- Cosmos Predict2.5 production config keeps `model.use_lora=true`.

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
- Cosmos production configs only accept the production import-path scorer contract.

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
reward scorer import path
reward model version
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
VIDEO_REWARD_SCORER=<real import path>
VIDEO_REWARD_MODEL_VERSION=<real reward model version>
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
- A Cosmos run with flat real reward does not count as success.
- A Cosmos run with generated videos but no reward debug artifacts does not count as success.
- Cosmos V2W without active `reference_image` does not count as success.
- Cosmos Predict2.5 with `model.use_lora=false` does not count as success.

## 6. Acceptance Criteria

This sprint is complete only when:

- Cosmos Predict2 V2W reference production experiment loads.
- Cosmos Predict2 V2W production run completes with real reward and per-sample reference image.
- Cosmos Predict2.5 DiffusionNFT production experiment loads.
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
- Keep train/eval split.
- Keep KL enabled for GRPO.
- Track reference consistency, text alignment, visual quality, physical plausibility, and motion quality separately.
- Add manual inspection notes to the production report.

Risk: production config is too expensive.

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
/home/mingfeiguo/Desktop/wm-infra/vrl/scripts/data/video_world.py
/home/mingfeiguo/Desktop/wm-infra/docs/papers/cosmos_predict2_5_world_simulation_with_video_foundation_models_for_physical_ai_2511.00062v2.pdf
/home/mingfeiguo/Desktop/wm-infra/vrl/trainers/data/prompts.py
/home/mingfeiguo/Desktop/wm-infra/vrl/rewards/functions/video_reward.py
/home/mingfeiguo/Desktop/wm-infra/vrl/rewards/ray/runtime.py
/home/mingfeiguo/Desktop/wm-infra/configs/reward/video_reward.yaml
/home/mingfeiguo/Desktop/wm-infra/configs/experiment/diffusion/cosmos_predict2/online_grpo_video_reward.yaml
/home/mingfeiguo/Desktop/wm-infra/configs/experiment/diffusion/cosmos_predict2_5/online_nft_video_reward.yaml
```

External references:

```text
https://arxiv.org/abs/2511.00062
https://github.com/nvidia-cosmos/cosmos-predict2.5
https://github.com/Vchitect/VBench
https://github.com/KwaiVGI/VideoAlign
```
