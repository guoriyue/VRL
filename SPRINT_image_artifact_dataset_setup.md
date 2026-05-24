# Sprint: Image / Video Data Population Setup

## 核心结论

不要做通用 artifact infrastructure。这个 sprint 的目标不是给 dataset layer 加抽象，而是让用户可以用少量命令把真实数据放到本地，然后让现有 loader 直接读取普通 manifest。

保留的原则：

- 文本 prompt 数据可以直接 commit 到 `datasets/**`。
- 图片、视频、reference frame、preference image pair 不直接 commit 到 repo。
- 大文件放在 `data/external/**` 或 HuggingFace cache。
- 训练配置只指向普通 manifest，不依赖 `ArtifactManifest` / `VRL_DATA_ROOT` 这类额外框架。

已经移除的基础设施：

```text
vrl/trainers/data/artifacts.py
vrl/scripts/data/setup.py
tests/data/test_artifact_manifest_validation.py
tests/data/test_video_world_manifests.py
```

保留的代码边界：

```text
vrl/trainers/data/prompts.py
vrl/trainers/data/preferences.py
vrl/scripts/data/danbooru.py
vrl/scripts/data/populate.py
vrl/scripts/diffusion/cosmos/train.py
```

`vrl/rewards/artifacts.py` 和 `vrl/rollouts/collector/artifacts.py` 不属于 dataset population；它们处理 reward/video 输出生命周期，不在本 sprint 删除范围内。

---

## 推荐用户入口

统一入口是：

```bash
python -m vrl.scripts.data.populate <dataset>
```

这个脚本只做 dataset-specific population，不提供通用 artifact validator。

### 1. Pick-a-Pic

默认只下载 metadata 版本，避免用户误下完整图片对：

```bash
python -m vrl.scripts.data.populate pickapic
```

用于真实 DPO image training 时显式下载带图片版本：

```bash
python -m vrl.scripts.data.populate pickapic --with-images
```

相关代码：

```text
vrl/scripts/data/populate.py
vrl/trainers/data/preferences.py
configs/dataset/pickapic_v2.yaml
configs/experiment/diffusion/wan_2_1/offline_dpo_pickapic.yaml
```

规则：

- Pick-a-Pic 图片留在 HuggingFace cache。
- 不 commit `jpg_0` / `jpg_1`。
- `configs/dataset/pickapic_v2.yaml` 继续直接用 `pickapic_preference` loader。

### 2. Danbooru / Anime prompts

生成已 commit 的 prompt manifest：

```bash
python -m vrl.scripts.data.populate anime-prompts
```

如果已有本地 metadata：

```bash
python -m vrl.scripts.data.populate anime-prompts \
  --metadata /path/to/danbooru/posts.jsonl
```

输出：

```text
datasets/danbooru/anatomy/train_prompts.jsonl
datasets/danbooru/anatomy/eval_prompts.jsonl
datasets/danbooru/anatomy/prompt_report.json
```

相关代码：

```text
vrl/scripts/data/populate.py
vrl/scripts/data/danbooru.py
configs/dataset/anime_anatomy.yaml
```

### 3. Danbooru positive images / hand crops

真实图片不进 repo，只生成指向本地图片的 manifest：

```bash
python -m vrl.scripts.data.populate anime-positives \
  --metadata /path/to/danbooru/posts.jsonl \
  --image-root data/external/danbooru/images \
  --output datasets/danbooru/anatomy/positive_images.jsonl \
  --hand-crops-output datasets/danbooru/anatomy/hand_crops.jsonl
```

规则：

- `data/external/danbooru/images/**` 不 commit。
- `positive_images.jsonl` 可以 commit，但里面的 image path 必须是本机可读路径或 repo-relative path。
- 如果要跨机器复现，重新跑 populate 命令，不依赖 git 存储图片。

### 4. Cosmos Video2World tiny smoke data

先提供一个 tiny local dataset，保证 per-sample reference plumbing 能跑通：

```bash
python -m vrl.scripts.data.populate video-world-tiny
```

默认输出：

```text
data/external/video_world/manifests/tiny_train.jsonl
data/external/video_world/manifests/tiny_eval.jsonl
data/external/video_world/references/tiny_train_ref.ppm
data/external/video_world/references/tiny_eval_ref.ppm
```

训练时用 manifest override：

```bash
python -m vrl.scripts.train \
  --config experiment/diffusion/cosmos_predict2/online_grpo_video_reward \
  data.manifest=data/external/video_world/manifests/tiny_train.jsonl \
  cosmos.reference_mode=per_sample
```

`reference_image` 解析规则很简单：

- absolute path: 直接用。
- relative path: 优先按 manifest 所在目录解析。
- 如果 manifest-relative 不存在，再按当前工作目录解析。

相关代码：

```text
vrl/scripts/data/populate.py
vrl/scripts/diffusion/cosmos/train.py
tests/data/test_populate.py
tests/rollouts/test_video_world_reference_metadata.py
```

---

## 不做的事情

不要重新引入这些东西：

```text
ArtifactManifestError
ArtifactManifestReport
ResolvedArtifact
resolve_artifact_path()
validate_artifact_manifest()
validate_artifact_manifest_pair()
VRL_DATA_ROOT-only path contract
```

原因：

- 这些是 framework，不是 data population。
- 用户真正需要的是“下载/生成数据到哪里、训练命令怎么指向它”。
- 具体 dataset 的 importer 更容易 debug，也更容易删除。

---

## 最小完成标准

这个 sprint 完成时应该满足：

- `python -m vrl.scripts.data.populate video-world-tiny` 可以生成本地 tiny video-world manifest 和 reference frame。
- `tests/data/test_populate.py` 覆盖 tiny population。
- `tests/rollouts/test_video_world_reference_metadata.py` 覆盖 Cosmos per-sample reference 进入 rollout metadata。
- Pick-a-Pic 可以通过 `populate pickapic` 预下载 cache。
- Danbooru prompt 和 positive image manifest 通过 `populate anime-*` 包装已有 `danbooru.py` 命令。
- repo 仍然 ignore `data/external/`、`data/cache/`、`outputs/`。

验证命令：

```bash
python -m pytest -q tests/data/test_populate.py tests/rollouts/test_video_world_reference_metadata.py
python -m compileall -q vrl tests
```

---

## References

Source files:

```text
/home/mingfeiguo/Desktop/wm-infra/vrl/scripts/data/populate.py
/home/mingfeiguo/Desktop/wm-infra/vrl/scripts/data/danbooru.py
/home/mingfeiguo/Desktop/wm-infra/vrl/trainers/data/preferences.py
/home/mingfeiguo/Desktop/wm-infra/vrl/trainers/data/prompts.py
/home/mingfeiguo/Desktop/wm-infra/vrl/scripts/diffusion/cosmos/train.py
/home/mingfeiguo/Desktop/wm-infra/tests/data/test_populate.py
/home/mingfeiguo/Desktop/wm-infra/tests/rollouts/test_video_world_reference_metadata.py
/home/mingfeiguo/Desktop/wm-infra/configs/dataset/pickapic_v2.yaml
/home/mingfeiguo/Desktop/wm-infra/configs/dataset/anime_anatomy.yaml
```

External dataset references:

```text
https://huggingface.co/datasets/yuvalkirstain/pickapic_v2
https://huggingface.co/datasets/yuvalkirstain/pickapic_v2_no_images
https://rail-berkeley.github.io/bridgedata/
https://droid-dataset.github.io/
https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github
```
