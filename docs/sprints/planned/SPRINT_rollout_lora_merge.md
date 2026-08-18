# SPRINT：rollout 侧 merge LoRA → dense —— 拿全参的 GEMM 形状，不付全参的显存

状态：**planned（2026-08-17）**。基线 main @ `abb8e4da`。
前置：无（不依赖多卡、不依赖 FP8）。

## 0. 结论先行

`SPRINT_gemm_utilization.md` 的杠杆表里，非-FP8 路径**最大的一条**是
「full-param 替 LoRA」（P1.5），干掉 ~47% elementwise + lora_A/lora_B 瘦 GEMM。
但它挂在「显存 / 多卡」上，LoRA 家族（sd3.5 / wan / predict2.5）今天吃不到。

同一张表的**下一行**给了一条不用付显存的替代路径：

> | **rollout 侧 merge LoRA → dense** | 35 步×CFG 的推理前向变单个大 GEMM |
> 真要写（merge/unmerge 要跟 colocated 训练 + 权重同步配合） |
> 只接了 `disable_adapter`（给 KL ref 用），**没有 `merge_adapter`** |

`docs/sprints/done/SPRINT_gemm_utilization.md:195`

这条至今没人写。本 sprint 写它，但**把范围收在 Ray disaggregated rollout
worker 上** —— 那里 merge 是干净的，colocated 不是（§2.3 说明为什么）。

## 1. 为什么值得做

`SPRINT_cross_model_performance.md` §0 结论 2：

> **elementwise 风暴 ~70% 来自 LoRA plumbing**（280 个 PEFT 层的双 mm +
> fp32/bf16 cast + scaling mul）。cosmos 转全参后自然消失；sd3_5/wan 仍在
> LoRA 路径上。

rollout 是 35 步 × CFG 的重复前向 —— LoRA 的每层双 mm + cast + scale 要重复
35×2 次。merge 之后每个 linear 回到单个 dense GEMM，**和全参 rollout 的形状
完全一样**，但训练侧仍然只存 LoRA 的优化器状态。

代价只有一份 merge 后的权重副本，而在 disaggregated 模式下 rollout worker
**本来就持有自己的一份权重**（weight sync 推过去的），所以增量显存 ≈ 0。

## 2. 设计

### 2.1 落在优化层，不落在家族

`vrl/nn/optimization/passes.py:52-88` 的 `OptimizationPass` protocol 已经是
本 sprint 需要的形状。新增 `LoraMergePass`：

```python
name = "lora_merge"
introduces_replay_drift = True   # 见 2.2 —— 这条必须是 True
replaces_modules = False         # 原地改 base weight，不换 root 对象
```

`enabled(build)`：`build.use_lora` 且运行在 rollout worker（非 colocated）。
`conflicts(build)`：与 colocated 冲突（§2.3）。

家族侧零改动 —— 这正是 `SPRINT_plug_and_play_optimization_layer.md` 建这个
seam 的目的。

### 2.2 必须声明 replay drift（不要漏这条）

merge 之后 rollout 算的是 `(W + BA)·x` 一次 GEMM；trainer replay 算的是
`W·x + B(A·x)` 两次。bf16 下**累加顺序不同 → 结果不逐位相同**。

`PassResult.introduces_replay_drift` 的注释写得很清楚
（`vrl/nn/optimization/passes.py:45-48`）：

> a pass that makes the rollout forward differ from the trainer's exact
> replay forward must be paired with a drift correction.

所以 `LoraMergePass` 必须 `introduces_replay_drift=True`，由现成的 drift guard
接管（量化已经走这条路）。**红线**：`trainer.py` 的 logprob parity 均差 ≤ 0.01。
如果 merge 后过不了这条线，本 sprint 就地停止并把数字记进 §5 —— 不放宽阈值。

### 2.3 为什么排除 colocated

colocated 模式下 rollout 和训练**共用同一个 model 对象**。merge 会把 base
weight 原地改成 `W + BA`：

- trainer 的 LoRA 梯度路径依赖 base 保持 `W`；
- `disable_adapter()`（`vrl/models/families/wan_2_1/model.py:361-366`，KL ref
  用）在 merge 后语义就错了 —— 关掉 adapter 也回不到 base；
- 每步 weight sync 后要 unmerge→re-merge，bf16 下反复
  `W += BA` / `W -= BA` 会累积误差，直接顶到 §2.2 的 parity 红线。

这三条都是真问题，不是保守。**colocated 明确列为非目标**，由
`conflicts()` 硬拒绝（一个不能生效的 guard 比没有 guard 更糟 ——
`passes.py:186-188` 的原话）。

### 2.4 weight sync 之后要重新 merge

rollout worker 每次收到新 LoRA state 后必须重新 merge。落点在
`update_weights` 的接收侧（`vrl/generation/ray/weight_sync.py` 推过去之后）。
merge 必须在 **install 之后、第一次采样之前**，且失败要响 —— 半 merge 的
权重跑出去的轨迹是脏数据。

沿用 `require_every_core_quantized`（`passes.py:91-119`）的自证反模式：
**检查模块本身，不信 pass 的自我汇报**。Wan 双专家半量化那个 bug
（`SPRINT_plug_and_play_optimization_layer.md` §1.3）在这里会以「只 merge 了
一半专家」的形式重演，所以要写对应的 `require_every_core_merged`。

## 3. 验收标准

- **数值**：merge 前后 rollout 前向 `rel_diff < 1e-3`（bf16 量级），并且
  **rollout-vs-replay logprob parity 均差 ≤ 0.01**（硬红线，`trainer.py`）。
- **覆盖完整**：`require_every_core_merged` 能抓到「多专家家族只 merge 一半」
  的构造用例（照 Wan 双专家 bug 写测试）。
- **性能**：sd3_5 或 wan 上，rollout 段 wall time + launch 数 before/after；
  用 `vrl/scripts/perf/compile_benchmark.py` 同口径。elementwise kernel 占比
  应显著下降（这是本 sprint 的直接目标）。
- **weight sync 循环**：连续 N 步训练 + sync + re-merge，parity 不随步数漂移
  （防 §2.3 说的累积误差；即便走 disaggregated 也要测）。
- **colocated 被拒**：colocated 配置下开 `lora_merge` 必须**启动即报错**，
  不是静默忽略。
- 既有 LoRA 路径默认行为零变化（默认关）；`make verify` 绿。

## 4. 非目标

- **colocated 模式**（§2.3，三条硬理由）。
- 不改 LoRA 训练本身（rank、target_modules、优化器状态都不动）。
- 不做 QKV 融合（`SPRINT_gemm_utilization.md` P1 已实测低 ROI 且不落地）。
- 不做 FP8（P3，用户 2026-06-14 暂缓）。
- 不替代 P1.5「full-param 替 LoRA」—— 那条在显存够时仍然更彻底；本 sprint
  是显存不够时的路径。

## 5. 执行记录

（待填：parity 数字、rollout wall time before/after、elementwise 占比变化、
逐家族 merge 覆盖核对）

## 6. 相关

- 杠杆出处：`docs/sprints/done/SPRINT_gemm_utilization.md:195`（杠杆表）
- elementwise 风暴归因：`docs/sprints/info/SPRINT_cross_model_performance.md` §0
- 挂载 seam：`docs/sprints/SPRINT_plug_and_play_optimization_layer.md`
- 半覆盖反模式：`vrl/nn/optimization/passes.py:91-119`
