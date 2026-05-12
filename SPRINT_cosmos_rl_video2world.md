# SPRINT：Cosmos-Predict2 2B 有效 Video2World RL

## 0. 目标

把当前 `cosmos_predict2_2b_grpo` 从可运行 smoke recipe 改成有意义的 Video2World RL mechanism run。

本 sprint 只做 Cosmos-Predict2 2B：

```text
model = nvidia/Cosmos-Predict2-2B-Video2World
algorithm = 本 repo 当前 GRPO + FlowMatchingEvaluator
condition = 真实 reference image
reward = 先从 aesthetic 升级到 prompt/reference 可审计 reward debug
```

不在本 sprint 做 DiffusionNFT，也不升级 Cosmos-Predict2.5。那两件事移到：

```text
SPRINT_cosmos_rl_diffusionnft_predict25.md
```

## 1. 当前状态

已经完成：

- 下载 Cosmos-Predict2.5 paper：

```text
docs/papers/cosmos_predict2_5_world_simulation_2511.00062.pdf
```

- `model.reference_image` 为空或路径不存在时 fail-fast，不再 warning 后继续 zero-conditioning。

当前问题：

- 只支持全局 `model.reference_image`，还不支持每个训练样本自己的 reference image。
- `aesthetic` reward 不能判断生成结果是否符合 reference / prompt。
- 输出目录没有保存 reference 条件和 reward debug artifact，后续无法审计训练到底学了什么。

## 2. 本 Sprint 要改的文件

### `vrl/scripts/cosmos/train.py`

需要改：

- 保留当前 fail-fast：`model.reference_image` 为空或不存在时直接报错。
- 在训练输出目录保存 reference image 副本或 reference manifest。
- 支持 Cosmos 专用 JSONL manifest：

```json
{"prompt": "...", "reference_image": "path/to/frame.png"}
```

- 当 manifest 里有 `reference_image` 时，按样本加载 reference image，而不是使用全局 `model.reference_image`。
- 第一版只支持 `rollout.rollout_batch_size=1` 的 per-sample reference；如果大于 1，先 fail-fast。
- 调用 collector 时传入当前样本的 reference image：

```python
await trainer.step(example_batch, reference_image=...)
```

如果 `OnlineTrainer.step()` 不能直接透传，则需要在 trainer/collector 边界加最小扩展，不能绕过 collector。

- `metrics.csv` 增加 reference 条件审计字段：

```text
reference_image_present
reference_image_path
```

### `vrl/trainers/online.py`

只有在 `train_cosmos_predict2_grpo()` 无法把 per-sample `reference_image` 传给 collector 时才改。

需要改：

- 允许 `step()` 接收 collector kwargs，或允许 `PromptExample.metadata` 透传到 collector。
- 保持现有 SD3 / Wan / Janus / NextStep 行为不变。
- 增加测试覆盖：没有 metadata 时旧路径不变，有 `reference_image` 时 collector 能收到。

### `vrl/trainers/data.py`

需要改：

- 扩展 manifest loader，或新增 Cosmos 专用 loader。
- 支持纯文本旧格式：

```text
A robot walking through a warehouse.
```

- 支持 JSONL 新格式：

```json
{"prompt": "A robot walking through a warehouse.", "reference_image": "data/ref/warehouse.png"}
```

- 对 Cosmos per-sample mode：缺 `reference_image` 必须 fail-fast。

### `configs/experiment/cosmos_predict2_2b_grpo.yaml`

需要改：

- 明确这个 recipe 需要真实 reference image。
- 保留 `model.reference_image` 作为 single-reference smoke / mechanism run 的 CLI override。
- 增加 manifest schema 说明注释。
- 如果加入新开关，只允许显式命名，例如：

```yaml
cosmos:
  reference_mode: global   # global | per_sample
```

不允许隐式从文件格式猜训练语义。

### `configs/model/cosmos/predict2_2b.yaml`

需要改：

- 保留 `reference_image: ""` 作为必须由用户 override 的空默认。
- 注释写清楚：空值会 fail-fast，不再 zero-conditioning。

### `vrl/rewards/*`

第一阶段不要直接做复杂 video reward service。先做可审计 frame-based reward。

可能改：

```text
vrl/rewards/codex_image_qa.py
vrl/rewards/multi.py
```

需要完成：

- 复用 `codex_image_qa` 对 generated middle/final frame 做 prompt alignment。
- 如果要评分 reference consistency，新增小型 reward component，例如：

```text
vrl/rewards/reference_codex_image_qa.py
```

它的输入必须包含 reference image、generated frame、prompt，输出 `[0, 1]` finite score。

### `vrl/rollouts/packers/diffusion.py`

可能需要改。

需要确认：

- `RolloutBatch` / `extras` 是否保留 generated video frames、reference path、prompt。
- reward debug 是否能拿到 generated frame。

如果当前 packer 已经足够，不改；如果 reward 层拿不到 reference/generation 对，就补 `extras`，不要让 reward 从全局变量读状态。

### `tests/scripts/test_cosmos_reference_image.py`

已经新增基础测试。继续扩展：

- 空 `model.reference_image` fail-fast。
- 路径不存在 fail-fast。
- RGBA reference image 会转换为 RGB。

### `tests/scripts/test_cosmos_manifest.py`

需要新增：

- 纯文本 manifest 仍能加载为 prompt。
- JSONL manifest 能加载 `prompt + reference_image`。
- Cosmos per-sample mode 缺 `reference_image` 会 fail-fast。
- `rollout_batch_size>1` 且 per-sample reference mode 会 fail-fast。

### `tests/trainers/test_online.py` 或 `tests/rollouts/test_collector_runtime.py`

如果改 trainer/collector 透传，就补测试：

- collector 收到 `reference_image` metadata。
- 不带 reference metadata 的旧训练路径不变。

### `README.md`

需要改：

- Cosmos recipe 文档必须写明：

```text
cosmos_predict2_2b_grpo requires model.reference_image or per-sample reference_image.
```

- 给出最小命令：

```bash
python -m vrl.scripts.train \
  --config experiment/cosmos_predict2_2b_grpo \
  model.reference_image=/path/to/reference.png
```

### `docs/cosmos_rl_gap.md`

需要新增：

- 当前 repo Cosmos-Predict2 2B GRPO 和 `/home/mingfeiguo/Desktop/cosmos-rl` 的差距。
- 明确当前不是 Cosmos-Predict2.5，不是 DiffusionNFT，不是 remote reward service。
- 记录后续迁移到 `SPRINT_cosmos_rl_diffusionnft_predict25.md`。

## 3. 实现顺序

1. 保持并测试 `model.reference_image` fail-fast。
2. 加 reference artifact 保存。
3. 加 Cosmos JSONL manifest loader。
4. 支持 per-sample `reference_image`，先限制 `rollout_batch_size=1`。
5. 补 reward debug artifact：reference、generated frame、prompt、raw score。
6. README 和 `docs/cosmos_rl_gap.md` 写清边界。
7. 跑相关测试。

## 4. 验证命令

```bash
python -m pytest -q \
  tests/scripts/test_cosmos_reference_image.py \
  tests/scripts/test_cosmos_manifest.py \
  tests/rollouts/test_runtime_inputs.py::test_cosmos_runtime_inputs_include_reference_image_from_cfg \
  tests/config/test_load_all_experiments.py::test_experiment_yaml_loads[cosmos_predict2_2b_grpo] \
  tests/config/test_load_all_experiments.py::test_validate_training_config_passes_for_all_active_experiments[cosmos_predict2_2b_grpo]
```

如果改了 trainer/collector 透传，再跑：

```bash
python -m pytest -q tests/trainers/test_online.py tests/rollouts/test_collector_runtime.py
```

最后跑：

```bash
python -m pytest -q tests/config tests/scripts tests/rollouts/test_runtime_inputs.py
```

## 5. 完成标准

这个 sprint 完成时必须满足：

- 没有 `model.reference_image` 时 Cosmos 训练在模型加载前 fail-fast。
- 全局 reference image override 可以进入 Cosmos training setup。
- JSONL manifest 支持 `prompt + reference_image`。
- per-sample reference image 在 `rollout_batch_size=1` 下走通 collector。
- 输出目录保存 reference 条件和 reward debug artifact。
- `metrics.csv` 能审计 reference image 是否存在。
- README 写清 Cosmos Video2World 不能 zero-conditioning 训练。
- `docs/cosmos_rl_gap.md` 写清当前 repo 与 cosmos-rl / Cosmos-Predict2.5 的差距。

最终 artifact：

```text
outputs/cosmos_pred2_2b_grpo_real_v2w/
  resolved_config.yaml
  metrics.csv
  reference_manifest.jsonl
  reference/
  reward_debug/
  samples/
  COSMOS_RL_GAP.md
```
