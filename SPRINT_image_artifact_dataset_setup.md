# Sprint: Image / Video Data Population Setup

## 核心结论

不要做通用 artifact infrastructure。这个 sprint 的目标不是给 dataset layer 加抽象，而是让用户可以用少量命令把真实数据放到本地，然后让现有 loader 直接读取普通 manifest。

保留的原则：

- 文本 prompt 数据可以直接 commit 到 `datasets/**`。
- 图片、视频、reference frame、preference image pair 不直接 commit 到 repo。
- 大文件放在 `data/external/**` 或 HuggingFace cache。
- 训练配置优先指向普通 manifest。本 sprint 不要求新增或扩展 `ArtifactManifest` / `VRL_DATA_ROOT` 这类通用框架。

## 当前 repo 状态

`populate.py` 已经存在，不是待新增项。现在应该把它当成主入口继续收敛，而不是再设计一套新的 setup/infrastructure。

当前已有入口：

```text
python -m vrl.scripts.data.populate pickapic
python -m vrl.scripts.data.populate anime-prompts
python -m vrl.scripts.data.populate anime-positives
python -m vrl.scripts.data.populate video-world-tiny
```

当前已有但不应继续扩展为主入口的代码：

```text
vrl/scripts/data/setup.py
```

`setup.py` 现在主要是旧的目录创建和 anime metadata wrapper。它可以保留给已有测试和兼容路径，但新的用户数据填充不要继续往这里加；新能力放进 `populate.py`。

当前仍缺的是真实 source-backed population：

- `video-world-bridge` 或 `video-world-droid`：从真实视频/episode 抽 reference frame 并写 manifest。
- Danbooru positive image / hand crop 的真实本地图片准备流程：`anime-positives` wrapper 有了，但 repo 里还没有真实 `positive_images.jsonl` / `hand_crops.jsonl`。
- Pick-a-Pic 只需要 cache/prefetch，不应该把 image pair 放进 repo。

已有代码可以保留，但不要把它变成本 sprint 的主线：

```text
vrl/trainers/data/artifacts.py
vrl/scripts/data/setup.py
tests/data/test_artifact_manifest_validation.py
tests/data/test_video_world_manifests.py
```

判断标准：

- 如果现有训练路径或测试已经用它，先保留。
- 如果只是为了这个 sprint 的计划文本服务，不继续扩展。
- 新的数据填充能力放在 dataset-specific `populate.py`，不要再加一层通用 manifest framework。

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

这个脚本只做 dataset-specific population，不要求调用通用 artifact validator。

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

如果用自定义数据根目录，训练时同时设置：

```bash
export VRL_DATA_ROOT=/path/to/external/data
```

`reference_image` 解析规则按现有代码走：

- absolute path 默认不允许用于 production manifest。
- relative path 解析到 `VRL_DATA_ROOT`。
- 如果没有设置 `VRL_DATA_ROOT`，默认解析到 repo 下的 `data/external`。
- 所以 `populate video-world-tiny` 写出的路径是 `video_world/references/...`，不是 `../references/...`。

相关代码：

```text
vrl/scripts/data/populate.py
vrl/scripts/diffusion/cosmos/train.py
tests/data/test_populate.py
tests/rollouts/test_video_world_reference_metadata.py
```

---

## 不做的事情

本 sprint 不新增这些东西，也不把它们设为完成条件：

```text
ArtifactManifestError
ArtifactManifestReport
ResolvedArtifact
resolve_artifact_path()
validate_artifact_manifest()
validate_artifact_manifest_pair()
```

原因：

- 这些是 framework，不是 data population 的核心。
- 用户真正需要的是“下载/生成数据到哪里、训练命令怎么指向它”。
- 具体 dataset 的 importer 更容易 debug，也更容易删除。
- 现有 helper 如果已经被训练路径使用，可以保留；不要为了清理 sprint 文档去删代码。

---

## 最小完成标准

当前完成状态：

- Done: `python -m vrl.scripts.data.populate video-world-tiny` 可以生成本地 tiny video-world manifest 和 reference frame。
- Done: `tests/data/test_populate.py` 覆盖 tiny population。
- Done: `tests/rollouts/test_video_world_reference_metadata.py` 覆盖 Cosmos per-sample reference 进入 rollout metadata。
- Done: Pick-a-Pic 可以通过 `populate pickapic` 预下载 cache。
- Done: Danbooru prompt 和 positive image manifest 通过 `populate anime-*` 包装已有 `danbooru.py` 命令。
- Done: repo ignore `data/external/`、`data/cache/`、`outputs/`。
- Not done: 真实 Bridge/DROID/Cosmos Video2World importer。
- Not done: 真实 Danbooru positive image / hand crop manifest 生成并审计。

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
