# SPRINT: Cosmos Robotic Data Factory — 只接真实公开数据源

状态：**planned / public-source substrate landed（2026-06-27）**。

核心边界：训练数据必须来自可下载、可解析、有 provenance 的真实公开数据集。仓库不再保留 `sidewalk_world` / `home_world` 这类本地手工 manifest bridge，也不把目标域名字伪装成数据集。

## 0. 核心结论

先不要问“调哪个 reward”。先问：

1. 生成数据要喂给哪个下游系统：perception、prediction、planning，还是 policy？
2. 目标域真实 heldout 里 Cosmos 现在失败在哪个维度？
3. 失败维度是数据覆盖/SFT 问题，还是可排序、可验证、适合 RL 的问题？
4. 训练样本是否能追溯到真实公开数据源，而不是手工拼出来的本地占位 manifest？

第一条可跑路线现在收窄为：

| 路线 | 真实来源 | 作用 |
|---|---|---|
| **DROID / LeRobot target V2W** | `lerobot/droid_100` on Hugging Face | 从真实机器人 manipulation demo 解析首帧和 target clip，给 Cosmos V2W 做 target-aware RL |

Sidewalk delivery 和 household chores 仍然是重要目标域，但必须等具体公开源 importer 接入后再进入训练配置。

## 1. 已落地 substrate

保留：

- `PromptExample` artifact schema 新增 `target_image` / `target_video`。
- target artifact 从 prompt example 传到 collector metadata，再进入 reward artifact metadata。
- `target_video_similarity` reward：读取 manifest 里的 `target_video` / `target_image`，比较生成视频和真实 target clip。
- `target_video_similarity_probe`：离线检查 target reward 是否能正常读真实 target media。
- `video-world-targets` importer：从公开 LeRobot/Hugging Face 数据集下载/解析真实 robot videos，生成 `reference_image` + `target_video` manifest。
- `droid_target_v2w` dataset config。
- `online_grpo_droid_target_240p` Cosmos config。

删除：

- `vrl/scripts/data/robot_world.py`
- `sidewalk-world-local` / `home-world-local`
- `sidewalk_delivery_v2w` / `home_manipulation_v2w`
- `online_grpo_sidewalk_delivery_240p` / `online_grpo_home_manipulation_240p`

原因：这些是“本地已有素材转 manifest”的桥，不是公开数据集下载/解析器。它们容易让人误以为仓库已经支持某个真实 dataset。

## 2. 当前真实数据路线：DROID / LeRobot target clips

准备数据：

```bash
python -m vrl.scripts.data.setup video-world-targets \
  --repo-id lerobot/droid_100 \
  --name droid_targets \
  --limit 50 \
  --eval-limit 8 \
  --max-target-frames 33
```

输出：

```text
data/external/video_world/manifests/droid_targets_train.jsonl
data/external/video_world/manifests/droid_targets_eval.jsonl
data/external/video_world/droid_targets_report.json
data/external/video_world/references/*.png
data/external/video_world/targets/*.mp4
```

manifest row 形状：

```json
{
  "prompt": "Put the marker in the pot",
  "reference_image": "video_world/references/droid_000001_first.png",
  "target_video": "video_world/targets/droid_000001_target.mp4",
  "task_type": "video2world",
  "metadata": {
    "source": "droid",
    "source_repo": "lerobot/droid_100",
    "source_split": "main",
    "source_episode": "000001",
    "source_video": "videos/observation.images.exterior_image_1_left/chunk-000/file-000.mp4",
    "source_frame_index": 0,
    "decode_method": "pyav_http_target_clip",
    "conditioning": "first_frame"
  }
}
```

训练配置：

```bash
python -m vrl.scripts.data.setup for-experiment diffusion/cosmos_predict2/online_grpo_droid_target_240p
```

实际训练仍然需要先跑 discrimination probe：

```bash
python -m vrl.scripts.eval.target_video_similarity_probe \
  --manifest data/external/video_world/manifests/droid_targets_eval.jsonl \
  --out outputs/droid_target_similarity_probe.jsonl
```

## 3. Reward 策略

`target_video_similarity` 是 baseline 学习信号，不是最终 robot quality verifier。

它能学到：

- 真实 demo continuation 的粗视觉/时序相似性。
- 首帧 conditioned V2W 是否朝真实 target clip 的方向发展。
- 比纯 Kling visual/motion reward 更贴近任务数据。

它不能单独保证：

- 接触物理一定正确。
- 物体状态谓词一定满足。
- 机器人动作一定可执行。
- 生成数据一定能提升真实 policy。

当前 recipe：

```yaml
reward:
  components:
    target_video_similarity: 0.80
    kling_video_reward: 0.20
```

原因：

- `target_video_similarity` 是从真实 demo 解析出的任务信号。
- Kling 只做视觉/运动质量 guard，避免生成结果退化成低质量视频。

## 4. 后续公开数据源，不许用本地占位桥

Sidewalk delivery 目标域下一步应该接：

- JRDB / JackRabbot：真实 mobile robot 视角，人群、室内外校园、social navigation。
- 接入要求：下载器、序列解析器、首帧/clip 导出、source report、train/eval split、provenance metadata。
- 在这些代码落地前，不保留 sidewalk delivery 训练 config。

Home chores / household manipulation 目标域下一步应该接：

- RoboCasa：公开模拟家庭/厨房任务 demo。
- DROID / BridgeData / Open X-Embodiment：真实 robot manipulation demos。
- 接入要求：不要要求用户先手工写本地 manifest；importer 必须直接解析公开 dataset layout。

## 5. 工程边界

保持不变：

- `RewardInferenceArtifact` / `RewardInferenceRequest` 协议层保留。
- `RewardFunction._init_disk_artifact_reward` 保留。
- trainer / GRPO / Cosmos replay/logprob runtime 不按 target domain 分叉。
- thin reward function 和 model 文件保留：`functions/` 是训练 runtime adapter，`models/` 是可离线调用的 scoring implementation。

新增真实数据约束：

- source-backed V2W validation 对普通 V2W 仍要求 `reference_image`。
- 当 experiment 使用 `target_video_similarity` 时，production validation 额外要求 `target_video` artifact 存在。
- source report 必须记录 `repo_id`、`source_split`、`decode_method`、train/eval rows、manifest paths 和 validation summary。

非目标：

- 不用 `Kling overall_reward` 作为 Robotic Data Factory 主 reward。
- 不在没有 target domain audit 前直接跑长 RL。
- 不把 rare-event frequency 当主 reward；rare event 用采样和过滤控制。
- 不加没有 detector/label 支撑的空壳 pedestrian/contact reward。
- 不再接受“本地路径 JSONL bridge”作为公开数据集接入。

## 6. 验证

- `python -m py_compile vrl/scripts/data/video_world.py vrl/scripts/eval/target_video_similarity_probe.py`
- `pytest tests/data/test_setup.py tests/data/test_video_world_manifests.py tests/data/test_artifact_manifest_validation.py tests/config/test_load_all_experiments.py::test_cosmos_target_v2w_production_validation_requires_target_clip -q`
- `pytest tests/data tests/rewards/functions tests/rewards/inference tests/config/test_load_all_experiments.py tests/config/test_schema.py -q`

## 7. 外部参考

- DROID: https://droid-dataset.github.io/
- LeRobot DROID sample: https://huggingface.co/datasets/lerobot/droid_100
- JRDB: https://jrdb.erc.monash.edu/
- RoboCasa: https://robocasa.ai/
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- Cosmos Predict2: https://github.com/nvidia-cosmos/cosmos-predict2
- Cosmos Predict2 Video2World model card: https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World
