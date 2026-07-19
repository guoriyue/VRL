# SPRINT: 清掉 smoke dataset，禁止把 smoke 输入当 RL 目标

状态：**done（2026-06-27）**。目标：把仓库里所有可被训练误用的 smoke/scratch 输入数据列清、移出长期数据路径，并给配置和文档加上边界：**smoke 只能是一次性验证产物，不能作为 RL 曲线、production recipe、paper parity 或可信 eval 的 dataset**。

## 0. 结论

清理前仓库里实际存在的 smoke/scratch 输入数据只有两组；本 sprint 已删除：

| 类别 | 路径 | 内容 | 结论 |
|---|---|---|---|
| V2W smoke dataset | `data/external/perf_smoke/v2w_smoke.jsonl` | 2 行 prompt，全部指向同一张 `perf_smoke/ref.png` | 已删除；不能喂给 Cosmos Predict2 V2W RL |
| V2W smoke reference | `data/external/perf_smoke/ref.png` | 1280x704 RGB 随机噪声图 | 已删除；只适合旧 perf/smoke，不是 reference image |
| 单 prompt scratch dataset | `outputs/_debug_cosmos25_fixed_prompt.txt` | 1 行：`Stone pendant hanging by a string loop.` | 已删除；它是调试固定 prompt，不是 dataset |

历史输出里还有几类 resolved config 指向 smoke 路径，但这些是 run 记录，不是当前长期输入源：

| 历史引用 | 当前状态 | 处理 |
|---|---|---|
| `outputs/*/resolved_config.yaml` → `data/external/perf_smoke/v2w_smoke.jsonl` | 目标文件已删除 | 保留历史 run 记录；历史记录只作 provenance |
| `outputs/*/resolved_config.yaml` → `outputs/_debug_cosmos25_fixed_prompt.txt` | 目标文件已删除 | 保留历史 run 记录；历史记录只作 provenance |
| `outputs/predict2_parity_g1/resolved_config.yaml` → `outputs/model_smoke_20260609/v2w_smoke.jsonl` | 目标文件不存在 | 记录为 dangling historical smoke reference |
| `outputs/cosmos_pred2_2b_kling_1gpu/run/resolved_config.yaml` → `datasets/drawbench/train_fixed4.txt` | 目标文件不存在 | 记录为 dangling historical mini-dataset reference |
| `docs/sprints/info/SPRINT_cross_model_smoke.md` → `outputs/model_smoke_20260609/*` / `data/external/model_smoke_20260609/*` | 文档声明已删除且本地不存在 | 保留文档；它是已归档的一次性 smoke 记录 |

## 1. 证据

### 1.1 实际存在的 smoke V2W manifest

`data/external/perf_smoke/v2w_smoke.jsonl`：

```jsonl
{"prompt": "A red ball rolls off a wooden table and bounces on the floor.", "reference_image": "perf_smoke/ref.png"}
{"prompt": "Water pours from a glass pitcher into a clear cup on a counter.", "reference_image": "perf_smoke/ref.png"}
```

问题不是行数少本身，而是两条样本都绑定同一张随机噪声 reference。它能证明代码路径能跑，但不能定义可信 Video2World RL 目标。

### 1.2 实际存在的 single-prompt scratch manifest

`outputs/_debug_cosmos25_fixed_prompt.txt`：

```text
Stone pendant hanging by a string loop.
```

这是固定 prompt 调试输入。它不在 `datasets/` 或 `configs/dataset/` 管理下，也没有 source report，不能作为长期 dataset。

### 1.3 当前真正的长期 dataset 配置

这些配置本身应该保留，问题是本地外部数据是否已 materialize：

```yaml
# configs/dataset/video_world_v2w.yaml
manifest: data/external/video_world/manifests/robot_train.jsonl
eval_manifest: data/external/video_world/manifests/robot_eval.jsonl
task_type: video2world
conditioning: reference_image
```

```yaml
# configs/dataset/videophy_i2v.yaml
manifest: data/external/videophy_i2v/manifests/train.jsonl
eval_manifest: data/external/videophy_i2v/manifests/eval.jsonl
task_type: image_to_video
conditioning: reference_image
```

这两个是 canonical external-data contracts，不是 smoke dataset。当前本地没有对应 manifest 文件，所以不能假装它们已可用于 RL。

## 2. 应该改什么

1. 已删除 `data/external/perf_smoke/`：
   - `data/external/perf_smoke/v2w_smoke.jsonl`
   - `data/external/perf_smoke/ref.png`

2. 已删除 `outputs/_debug_cosmos25_fixed_prompt.txt`，依赖它的历史结论只保留在 run/sprint 文档里。

3. 已给 training config validation 与 dataset readiness 加 guard：训练配置解析到以下路径时会 fail fast，没有配置逃生阀：
   - 路径包含 `/perf_smoke/`
   - 路径位于 `outputs/` 且被当作 `data.manifest`
   - basename 包含 `smoke`, `scratch`, `debug`, `fixed_prompt`, `fixed4`, `minigroup`

4. 已更新 `SPRINT_cosmos_predict2_2b_trustworthy_curve.md`：它目前仍围绕 Predict2 V2W full-param 曲线，但没有好 reference-image dataset，状态改为 blocked-by-dataset，不能再把 `perf_smoke/ref.png` 或 global random reference 当候选。

5. 已更新 proof-run 文档里的“7 样本够 smoke”表述：如果使用本地最小 I2V manifest，只能叫 `*_smoke` 或 `_scratch_*`，不得放在 canonical `data/external/videophy_i2v/manifests/{train,eval}.jsonl` 名字下冒充正式数据。

## 3. 应该保持不变

1. 保留长期文本/图像 prompt datasets：
   - `datasets/videophy/{train,eval}.txt`
   - `datasets/drawbench/*.txt`
   - `datasets/ocr/*.txt`
   - `datasets/pickscore_sfw/*.txt`
   - `datasets/geneval/*.jsonl`
   - `datasets/danbooru/**`

2. 保留 canonical external dataset configs：
   - `configs/dataset/video_world_v2w.yaml`
   - `configs/dataset/videophy_i2v.yaml`

3. 保留 smoke recipe 配置本身，只要它们使用正式 dataset config，而不是 smoke input files：
   - `configs/experiment/diffusion/flux/online_grpo_smoke_single_gpu.yaml` 用 `/dataset/ocr`
   - `configs/experiment/diffusion/qwen_image/online_grpo_smoke_single_gpu.yaml` 用 `/dataset/ocr`

4. 保留测试里的 tmp_path smoke 构造。例子：`tests/data/test_videophy_i2v.py::test_for_experiment_rejects_partial_videophy_i2v_smoke_data` 是防止 partial smoke data 冒充完整数据的 guard，不是长期 dataset。

5. 保留历史 `outputs/*/resolved_config.yaml`，因为它们是 run provenance。不要为了“清 smoke dataset”去改历史 resolved config。

## 4. ALL_CAPS / thin boundary review

本 sprint 不新增 ALL_CAPS 常量，也不需要把 dataset blacklist 写成手维护的业务词表常量。若实现 guard，优先从路径属性派生判断：

- `Path(parts)` 是否包含 `outputs` 或 `perf_smoke`
- manifest basename 是否包含 smoke/scratch/debug 标记

落地位置：

- `vrl/config/validation.py::training_data_path_rejection_reason`
- `vrl/config/validation.py::_validate_training_data_paths`
- `vrl/scripts/data/bootstrap.py::resolve_experiment_dataset_plan`

应保留的 thin boundary：

- `tests/data/test_videophy_i2v.py::test_for_experiment_rejects_partial_videophy_i2v_smoke_data`：这是测试 guard，保留。
- `configs/dataset/*.yaml`：这是 dataset public contract，不因 LOC 少而合并。

非目标：不为了减少文件数扁平化 dataset configs；这些配置的 uniform shape 对 grep、dry-load 和生产 run 对齐更重要。

## 5. 验收标准

清理完成后跑这些检查：

```bash
rg -n "data/external/perf_smoke|outputs/_debug_cosmos25_fixed_prompt|outputs/model_smoke_20260609|datasets/drawbench/train_fixed4" configs vrl tests docs/sprints
```

期望：只剩下本 sprint 文档和明确的 negative test / historical note。

```bash
find data/external -path '*perf_smoke*' -print
find outputs -maxdepth 1 -name '_debug_cosmos25_fixed_prompt.txt' -print
```

期望：零输出。

```bash
pytest -q tests/data/test_videophy_i2v.py tests/config/test_load_all_experiments.py
```

期望：全绿，证明 canonical dataset configs 仍能解析，partial smoke data 仍被拒绝。

实际结果：

```text
pytest -q tests/data/test_setup.py tests/data/test_videophy_i2v.py tests/config/test_schema.py tests/config/test_load_all_experiments.py
94 passed, 1 warning in 3.11s
```

## 6. 非目标

- 不生成新数据。
- 不删除历史 run 目录。
- 不删除 `datasets/videophy`：它是正式 T2V prompt dataset，不是 smoke dataset。
- 不把 smoke recipe 全部删掉；smoke recipe 可以存在，但不能依赖 smoke input dataset。
- 不把 `data/external/video_world` 或 `data/external/videophy_i2v` 的 canonical 路径伪造为空壳数据。

## 7. 参考路径

- `data/external/perf_smoke/v2w_smoke.jsonl`
- `data/external/perf_smoke/ref.png`
- `outputs/_debug_cosmos25_fixed_prompt.txt`
- `configs/dataset/video_world_v2w.yaml`
- `configs/dataset/videophy_i2v.yaml`
- `configs/dataset/videophy.yaml`
- `configs/experiment/diffusion/flux/online_grpo_smoke_single_gpu.yaml`
- `configs/experiment/diffusion/qwen_image/online_grpo_smoke_single_gpu.yaml`
- `tests/data/test_videophy_i2v.py`
- `docs/sprints/parked/SPRINT_cosmos_predict2_2b_trustworthy_curve.md`
- `docs/sprints/parked/SPRINT_wan_2_1_i2v_proof_run.md`
- `docs/sprints/parked/SPRINT_wan_2_2_proof_run.md`
- `docs/sprints/info/SPRINT_cross_model_smoke.md`
