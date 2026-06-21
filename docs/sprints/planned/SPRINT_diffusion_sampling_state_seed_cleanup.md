# SPRINT: diffusion `*SamplingState.seed` 死字段清理（planned）

状态：未开始（2026-06-20）。
范围：删除 6 个 diffusion `*SamplingState` dataclass 上「`prepare_sampling()` 写入、但从不被 `state.seed` 读回」的死 `seed` 字段。纯机械删，跨 5 文件同一模式。
来源：repo-wide dead-dataclass-hunt（194 structs 审计 → 对抗验证）confirmed-dead，置信 high。承接 [[SPRINT_segment_signal_dead_field_cleanup]] 的派生结构体审计方法论。

## 0. Core Decision

种子值的**唯一活跃读取点是 `request.seed`**（`GenerationRequest`），用来即时初始化 `torch.Generator`：

```python
# vrl/models/diffusion/wan_2_1/model.py:361
seed = request.seed if request.seed is not None else random.randint(0, sys.maxsize)
```

而把这个值再拷一份存进 `SamplingState.seed` 的那个字段，全仓库**零读取**。`grep -rEn "\.seed\b" vrl/ tests/` 过滤掉 `request.seed` / AR 的 `params.seed`（`ARSamplingParams`，活字段，见 `janus_pro/runtime.py:304,452`）/ `trainer_config.seed` 后，没有任何 `state.seed` 读者：state 上的 `seed` 是 write-only 存档副本。

> ⚠️ **勿连坐**：AR 侧 `ARSamplingParams.seed` 是**活字段**（`vrl/models/ar/janus_pro/runtime.py:304,308,451,452,718,719,813,814` 用 `torch.manual_seed(params.seed)` 控流），`GenerationRequest.seed` / `TrainerConfig.seed` 也是活的。本 sprint **只删 diffusion `*SamplingState` 上的 `seed`**，不碰这三个。

## 1. 现状实锤

6 个死 `seed` 字段（均 `seed: int`，构造写入、从不读回）：

| struct | 定义 | 构造写入点 |
|---|---|---|
| `WanT2VSamplingState.seed` | `vrl/models/diffusion/wan_2_1/model.py:66` | `model.py:387` |
| `WanI2VSamplingState.seed` | `vrl/models/diffusion/wan_2_1/model.py:85` | `model.py:709` |
| `CosmosPredict2SamplingState.seed` | `vrl/models/diffusion/cosmos/predict2/model.py:72` | `model.py:301` |
| `CosmosPredict25SamplingState.seed` | `vrl/models/diffusion/cosmos/predict2_5/model.py:95` | `model.py:363` |
| `AnimaSamplingState.seed` | `vrl/models/diffusion/cosmos/anima/model.py:44` | `model.py:289` |
| `SD3SamplingState.seed` | `vrl/models/diffusion/sd3_5/model.py:70` | `model.py:278` |

合证：各 `model.py` 的 `export_batch_context()` / `export_replay_tensors()` 均不导出 `seed`；replay 侧 `restore_eval_state()` 路径不读它（种子在 replay 由 batch_context 的别的字段或硬编码决定）。`seed` 存进 state 后既不进控制流、不进 compute、不进序列化导出。

## 2. 落地方案

逐文件删字段 + 删构造传参：
- `wan_2_1/model.py`：删 `:66`、`:85` 两个 `seed: int` 字段，删 `:387`、`:709` 构造处 `seed=seed`。
- `cosmos/predict2/model.py:72` + 构造 `:301`。
- `cosmos/predict2_5/model.py:95` + 构造 `:363`。
- `cosmos/anima/model.py:44` + 构造 `:289`。
- `sd3_5/model.py:70` + 构造 `:278`。
- `prepare_sampling()` 里计算 `seed` 的局部变量**保留**（它仍用于初始化 `Generator`），只删「把 `seed` 塞进 state」这一步。

## 3. 验证（finishing criteria）

- `grep -rEn "\.seed\b" vrl/models/diffusion/` 仅剩 `request.seed`（活）；无 `state.seed` / `self.seed` 命中。
- `grep -rn "seed:" vrl/models/diffusion/` 零命中（6 个字段定义全删）。
- `pytest tests/models/diffusion/ -q` 全绿；diffusion rollout/replay 冒烟（任一 `*_sample.py`）输出与删前逐像素一致（种子逻辑未变，仅删未读副本）。

## 4. 非目标 / Non-Goals

- 不动 `GenerationRequest.seed`、`ARSamplingParams.seed`、`TrainerConfig.seed`（均活）。
- 不改 `prepare_sampling()` 里 `Generator` 的初始化逻辑。

## References
- `vrl/models/diffusion/wan_2_1/model.py:66,85,361,387,668,709`
- `vrl/models/diffusion/cosmos/predict2/model.py:72,254,301`
- `vrl/models/diffusion/cosmos/predict2_5/model.py:95,325,363`
- `vrl/models/diffusion/cosmos/anima/model.py:44,254,289`
- `vrl/models/diffusion/sd3_5/model.py:70,246,278`
- 关联：[[SPRINT_segment_signal_dead_field_cleanup]]
