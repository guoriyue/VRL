# SPRINT: Wan Video RL Production Run

## 0. Core Decision

本 sprint 只负责 Wan T2V production RL。

目标是把已经验证过的 Wan video dataset 接到真实 `video_reward`，完成可审计的 production run：

```text
Wan T2V + real video_reward + base-vs-trained video eval
```

不在本 sprint 里重新设计 dataset。Prompt-only manifests 的 source of truth 是 committed `datasets/**` manifests, their `report.json` provenance files, and task/dataset configs.

核心判断：

- Wan 路线是 text-to-video，不需要 `reference_image`。
- Wan 验收重点是 motion quality、visual quality、text alignment。
- SD3.5 OCR 已经证明基础训练链路；Wan 必须用真实 video reward 和真实 video artifacts 验收。
- 完成标准是 production run artifact、非平 reward、权重变化、base-vs-trained eval，不是单纯 config load。

## 1. Upstream Dataset Dependency

本 sprint 依赖 upstream dataset sprint 产出并验证：

```text
source-backed Wan T2V train/eval manifests
```

No hand-written seed prompt config counts as an upstream dataset dependency.

本 sprint 不重新定义：

```text
coverage_buckets
approved_sources
caption rules
filtering rules
manifest row examples
minimum dataset sizes
```

## 2. Current Repo Reality

已有 Wan 训练入口：

```text
configs/experiment/diffusion/wan_2_1/online_grpo_ocr.yaml
vrl.scripts.diffusion.wan_2_1.train:train_wan_2_1_grpo
```

已有 prompt loader：

```text
vrl/trainers/data/prompts.py
```

已有 reward runtime：

```text
configs/reward/video_reward.yaml
vrl/rewards/functions/video_reward.py
vrl/rewards/ray/runtime.py
vrl/rewards/ray/worker.py
```

## 3. Production Route

### 3.1 Wan T2V Production RL

Goal:

- Train Wan T2V with real video-quality / motion / text-alignment reward.
- Consume a source-backed T2V train manifest.
- Evaluate on a source-backed T2V eval manifest.
- Produce generated videos, reward artifacts, reward debug rows, and a base-vs-trained report.

Required experiment:

```text
configs/experiment/diffusion/wan_2_1/online_grpo_video_reward.yaml
```

Production defaults after source-backed dataset import:

```text
/recipe/online/diffusion_grpo
/model/diffusion/wan_2_1/1_3b
/sampling/video/240p_33f
/sampling/denoise/20_step_cfg_4_5
/reward/video_reward
future /dataset/<source_backed_video_t2v>
```

Required config decisions:

```yaml
reward:
  kwargs:
    video_reward:
      inference_runtime: ray
      worker_config:
        scorer: import_path
        import_path: ???
        reward_model_version: ???

rollout:
  n: 4
  rollout_batch_size: 1
  sample_batch_size: 1
  noise_level: 0.7
  sde:
    type: cps

trainer:
  output_dir: outputs/wan_1_3b_video_reward
```

## 4. Implementation Tasks

### Phase 1: Wan Production Config

Add:

```text
configs/experiment/diffusion/wan_2_1/online_grpo_video_reward.yaml
```

Edit:

```text
tests/config/test_load_all_experiments.py
vrl/config/validation.py
```

Required smoke checks:

- Wan smoke config loads.
- Wan production config uses `/reward/video_reward`.
- Wan config uses a source-backed video dataset config.
- Wan production config uses `worker_config.scorer=import_path`.
- Wan production config requires `VIDEO_REWARD_SCORER`.
- Wan production config requires `VIDEO_REWARD_MODEL_VERSION`.

Production acceptance must record the dataset source/report in `outputs/wan_video_production_report.md`.

### Phase 2: Wan Reward Runtime Contract

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
- Wan production config only accepts the production import-path scorer contract.

### Phase 3: Wan Fixed Eval and Report

Add:

```text
outputs/wan_video_production_report.md
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

- A Wan run without real video reward does not count as production completion.
- A Wan run with flat real reward does not count as success.
- A Wan run with generated videos but no reward debug artifacts does not count as success.
- A Wan run that does not compare base vs trained outputs does not count as success.

## 6. Acceptance Criteria

This sprint is complete only when:

- Wan video reward production experiment loads.
- Wan production run completes with real video reward.
- Wan production run writes `reward_artifacts/manifest.jsonl`.
- Wan production run writes `reward_debug/video_reward_results.jsonl`.
- Wan production run records non-flat reward.
- Wan production run proves trainable weights changed.
- Unit tests pass for config loading and production reward config.
- `outputs/wan_video_production_report.md` records base-vs-trained results.

## 7. Out of Scope

Do not include in this sprint:

- Dataset source selection and coverage design. That belongs in committed dataset manifests, dataset reports, and dataset configs.
- Cosmos Predict2 Video2World.
- Cosmos Predict2.5 DiffusionNFT.
- Reference-image conditioning.
- OCR reward acceptance for Wan video.
- Replacing the reward service architecture.

## 8. Risks and Guards

Risk: real video reward is flat.

Guard:

- Require reward component debug rows.
- Require per-component score inspection.
- Fail production acceptance if reward std is zero across the training run.

Risk: dataset quality causes reward hacking.

Guard:

- Consume only validated upstream manifests.
- Keep train/eval split.
- Keep KL enabled for GRPO.
- Track text alignment, visual quality, and motion quality separately.
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
python -m compileall vrl tests
```

Run-level checks:

```bash
test -f outputs/<wan-run>/metrics.csv
test -f outputs/<wan-run>/reward_artifacts/manifest.jsonl
test -f outputs/<wan-run>/reward_debug/video_reward_results.jsonl
test -d outputs/<wan-run>/checkpoint-final
```

## 10. References

Local files:

```text
/home/mingfeiguo/Desktop/wm-infra/vrl/trainers/data/prompts.py
/home/mingfeiguo/Desktop/wm-infra/vrl/rewards/functions/video_reward.py
/home/mingfeiguo/Desktop/wm-infra/vrl/rewards/ray/runtime.py
/home/mingfeiguo/Desktop/wm-infra/configs/reward/video_reward.yaml
/home/mingfeiguo/Desktop/wm-infra/configs/experiment/diffusion/wan_2_1/online_grpo_ocr.yaml
```

External references:

```text
https://arxiv.org/abs/2505.05470
https://github.com/Vchitect/VBench
https://github.com/KwaiVGI/VideoAlign
```
