# SPRINT：训练步同步审计 —— 实测否决，不实施

状态：**done / 不实施（2026-08-17）**。计划在实施前先跑 KILL-RISK 门，
**门没过**：收益实测 0.2%（中位数），不值得动 trainer 最关键的那段代码。
本文保留为**否决记录**，防止同一提案被重新提出。

基线 main @ `abb8e4da`。测量硬件 RTX 5090（32GB，测时有其他负载，故用交替
采样取中位数）。

## 0. 结论先行

原始命题是对的一半：**同步确实存在，数量也确实是 16 次**（§1 的计数逐行核对
无误）。错的是**代价**。

实测：那 16 次同步合计 **75 µs**，而它们所在的那次 replay 迭代是 **138 ms**。
占比 **0.05%**。端到端交替 A/B 的中位数差是 **0.2%**，min-to-min 甚至是
**−3.2%**（即噪声大于信号）。

**根因**：同步的代价不是 memcpy 本身（4.7 µs 一次），而是它**禁止 CPU 跑在
GPU 前面**。当 GPU 每次迭代要干 138 ms 而 CPU 侧 plumbing 只有 ~1 ms 时，
CPU 本来就无事可做 —— 禁止它跑前面没有任何损失。这个损失只在
**CPU 与 GPU 时间可比**时才出现，而真实 DiT replay 比那个 regime 重两个数量级。

## 1. 计数是对的（保留，供以后引用）

每次内层迭代 ~16 次阻塞 D2H：

**A. parity 统计 8 次** —— `vrl/algorithms/logprob_mismatch.py:116-123`：
`bool(isfinite(delta).all() and isfinite(ratio).all())` 是 2 次（Python 的
`and` 先取左操作数真值），6 个 `float(0-dim tensor)` 各 1 次。
调用点 `vrl/trainers/online/trainer.py:964` 无条件执行。

**B. GRPO 指标 ~8 次** —— `vrl/algorithms/grpo/continuous.py:173-221`。

**这些值在循环内没有控制流消费者**（逐条核对过：`if not sample_batch.is_dummy`
读 Python bool，`if keep is not None` 是 tensor 身份判断，`if pc.tis_mode == "off"`
读配置）。所以**技术上可以推迟** —— 只是不值得。

## 2. 实测

### 2.1 单次同步的裸代价（空队列）

| 操作 | 耗时 |
|---|---|
| `1 × .item()` | 4.74 µs |
| `16 × .item()` | 74.90 µs |
| `torch.stack(16).tolist()` | 11.98 µs |

即：合并成一次传输能把 75 µs 降到 12 µs，**省 63 µs**。

### 2.2 放进真实量级的 replay 迭代

合成 DiT（24 层 / d=1536 / seq 1024 / batch 2，bf16，forward+backward，用真实的
`compute_logprob_mismatch_stats` 与 GRPO 指标块），交替 A/B、各 9 轮取中位数：

| 臂 | median | min |
|---|---|---|
| per_item（今天：16 次同步） | 138.59 ms | 132.64 ms |
| deferred（0 次同步，device 累加） | 138.31 ms | 136.94 ms |
| **节省** | **0.2%** | **−3.2%** |

63 µs / 138 ms = **0.05%**。测量噪声（GPU 竞争下 ±20%）**远大于**信号。

### 2.3 收益只在哪个 regime 出现

同一套代码扫 CPU/GPU 配比（合成负载）：

| regime | per_item | deferred | 节省 |
|---|---|---|---|
| GPU-bound（GPU 1.9 ms / CPU ~0） | 1.899 ms | 1.844 ms | 2.9% |
| **balanced（GPU ~1 ms / CPU 0.8 ms）** | 1.873 ms | 1.154 ms | **38.4%** |
| CPU-bound（CPU 1.5 ms） | 2.164 ms | 1.770 ms | 18.2% |
| CPU-bound（CPU 3.0 ms） | 3.501 ms | 3.233 ms | 7.7% |

balanced 那行的 38% 是真的，但**它要求每次迭代的 GPU 工作量在 1 ms 量级**。
真实 replay 是 138 ms。生产配置不可能落进这个窗口 —— 除非极小模型 + 极低分辨率，
而那不是要优化的对象。

## 3. 因此不实施

不动 `_ReplayMetrics` / `LogprobMismatchStats` / `PolicyUpdateStats` 的形态，
不改 Algorithm 的返回类型。理由：

1. 收益 0.2%，在测量噪声以下。
2. 代价是重构 `trainer.py` 最关键的那段（parity correctness gate 的数据通路）
   + 改动每个 algorithm 的返回类型。**风险与收益完全不成比例。**
3. `torch.stack(...).tolist()` 那个「便宜版」（省 63 µs）同样在噪声以下，
   也不做 —— 加一层间接换 0.05% 不划算。

## 4. 这个否决**没有**回答的问题

`SPRINT_cross_model_performance.md` §0 的「训练段 GPU 只有 ~64% 在做事」**仍然
成立且仍未解释**。本 sprint 排除了一个嫌疑人，没有找到真凶。

实测说明内层迭代本身是 **GPU-bound** 的（138 ms GPU vs ~1 ms CPU plumbing），
所以那 36% 的空转**不在内层循环里**。剩下的嫌疑人（未测）：

- `move_training_batch_to_device`（每个 sample_batch 一次 H2D，replay tensor 可能很大）
- optimizer step / grad clip / EMA
- `_all_ranks_have_work` 的 per-microbatch collective（多卡时）
- microbatch 之间的数据准备与 Python plumbing
- 权重同步与 rollout 交接

**下一个接手的人：先用 nsys 在真实 run 上定位那 36% 落在哪个区间，再立 sprint。**
不要再从「哪里有 `.item()`」这个角度找 —— 本文已经证明那条路是死的。

## 5. 方法学留档

- **交替 A/B + 中位数是必需的**，不是讲究：单次顺序测量给出过
  per_item 128 ms / deferred 137 ms（看起来 deferred 更慢），再测一轮变成
  162 / 96（看起来快 40%）。两个都是噪声。这与 2026-08-16 CUDA graph 复测
  记的教训一致：**GPU 竞争下要比就比两个都实际会跑的臂，且要重复取中位数。**
- 收益为正但在噪声内 → 当作零。

## 6. 相关

- 未解释的起点：`docs/sprints/info/SPRINT_cross_model_performance.md` §0
- 同样「实测后否决」的先例：`docs/sprints/done/SPRINT_gemm_utilization.md`
  的 QKV 融合（P1）与 CUDA graph（2026-08-16 复测）
- 父 program：`docs/sprints/planned/SPRINT_train_phase_efficiency_program.md`
