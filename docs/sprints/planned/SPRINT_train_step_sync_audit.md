# SPRINT：训练步同步审计 —— 把 per-timestep 的 host sync 收成一次

状态：**planned（2026-08-17）**。基线 main @ `abb8e4da`。
父 program：[Train-phase efficiency](SPRINT_train_phase_efficiency_program.md)

## 0. 结论先行

`SPRINT_cross_model_performance.md` 的核心结论是「**挤性能的对象是 launch 开销和
传输/序列化，不是 kernel 本身**」，同一份 trace 还记了「cosmos **训练段** GPU 实际
只有 ~64% 在做事」。但后续所有杠杆（compile、QKV 融合、CUDA graph 复测、量化）
**都打在 rollout 段**，训练段那 36% 空转从没有人去查根因。

本 sprint 查一个具体的、已定位的根因：**训练内层循环每次迭代做约 16 次阻塞式
device→host 同步**，而这些值在循环内**没有任何控制流消费者**——它们只是被 append
进 list，等到 optimizer 边界才聚合。

这不是猜测，是可以逐行数出来的。同时修法在本仓已有先例
（`continuous.py:367` 的 stack-then-one-collective），不需要发明新模式。

## 1. 证据

### 1.1 内层循环的形状

`vrl/trainers/online/trainer.py:1390-1408`，双层循环，每次迭代调一次
`_compute_replay_loss`：

```python
for j in self._sample_batch_train_indices(
    sample_batch, train_indices, cfg.timestep_selection,
):
    loss, metrics = self._compute_replay_loss(
        group_batch, batch_adv, j, algorithm_adapter=algorithm_adapter,
    )
    loss = loss * sample_batch.loss_weight / loss_scale
    self._backward(loss)
    if not sample_batch.is_dummy:
        agg.add(metrics, weight=sample_batch.loss_weight, capture_initial_replay=True)
```

单次 optimizer update 的迭代次数 = `len(_balanced_training_sample_batches(...))`
× `len(train_indices)`。

### 1.2 每次迭代的同步计数

**A. parity 统计（无条件，8 次）** —— `vrl/algorithms/logprob_mismatch.py:116-123`：

```python
finite = bool(torch.isfinite(delta).all() and torch.isfinite(ratio).all())
return LogprobMismatchStats(
    logprob_abs_diff_mean=float(abs_diff.mean()),
    logprob_abs_diff_max=float(abs_diff.max()),
    ratio_abs_dev_mean=float(ratio_dev.mean()),
    ratio_abs_dev_max=float(ratio_dev.max()),
    mismatch_kl=float((-delta).mean()),
    mismatch_k3_kl=float((ratio - delta - 1.0).mean()),
    finite=finite,
)
```

`bool(...)` 那行是 2 次（Python 的 `and` 先对左操作数取真值，再对右操作数取），
6 个 `float(0-dim tensor)` 各 1 次 = **8 次**。

调用点 `vrl/trainers/online/trainer.py:964` 是**无条件**的，注释写明理由是
correctness gate：

```python
# Parity is a trainer/evaluator fact, not an objective-specific metric.
# Measure every replayed segment here so TokenGRPO, trust-region variants,
# and multi-segment objectives cannot accidentally leave a false zero that
# lets the pre-optimizer correctness gate pass.
metrics.logprob_mismatch = compute_logprob_mismatch_stats(
    torch.cat(fresh_parts), torch.cat(old_parts),
)
```

**这个 gate 必须保留** —— 本 sprint 不动它的语义，只动它什么时候落到 host。

**B. GRPO 指标（~8 次）** —— `vrl/algorithms/grpo/continuous.py:173-221`，
`active_clip_fraction` / `tis_clip_fraction` / `rs_seq_masked_fraction` /
`clip_fraction` / `approx_kl` / `loss` / `policy_loss` / `kl_penalty` /
`weighted_kl_loss` 各一次 `.item()`。

合计 **≈16 次阻塞 D2H / 迭代**。

### 1.3 为什么可以推迟（关键安全论证）

循环内**没有任何分支读这些值**。逐条核对：

- `if not sample_batch.is_dummy` —— 读的是 Python bool，不是 metric。
- `continuous.py:168` `if keep is not None` —— tensor 身份判断，不取值。
- `continuous.py:178-181` `if pc.tis_mode == "off"` —— 读配置。

消费者只有 `agg.add()`（`trainer.py:328-338`，纯 append）和
`finish_optimizer_update` 里的加权聚合（`trainer.py:1465`）。也就是说
**这 16 个标量在 optimizer 边界之前无人需要**。

### 1.4 碎片化 collective（同一病灶的第二面）

`vrl/trainers/online/trainer.py:507-543` 三个 helper，每个都是
「新建 1 元素 tensor → `.to(device)` H2D → all_reduce → `.item()` D2H」：

```python
tensor = torch.tensor([float(value)], dtype=torch.float64)
if dist.get_backend() == "nccl":
    tensor = tensor.to(device)
dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
return float(tensor.item())
```

`_distributed_parity_verdict`（:546-560）连着打 **2 次** collective（一次 MIN 判
finite，一次 MAX 求 max_abs_diff），这两个可以合成一次 2 元素 reduce。

本仓已有正确写法可抄 —— `vrl/algorithms/grpo/continuous.py:367`：

```python
stats = torch.stack([values.sum(), values.new_tensor(float(n))])
if dist.get_backend() == "nccl":
    stats = stats.cuda()
dist.all_reduce(stats, op=dist.ReduceOp.SUM)
```

### 1.5 一个**不能**动的点，先记下来

`vrl/trainers/online/trainer.py:1355` 的 `_all_ranks_have_work()` 是**每个
microbatch 一次 collective + 一次 D2H**，看起来是同类问题，但它**不可消除**：
返回值直接决定 `return`（跳过 backward），而 rank 间不一致会 NCCL 死锁
（:114-124 有完整论证）。host 必须拿到这个值。

本 sprint **不碰它**。诚实记下：per-microbatch 那次同步保留，本 sprint 只收
per-timestep 的那 16 次。

## 2. 范围

**P1 — 指标改为 device-resident 累加。**
`LogprobMismatchStats` / `PolicyUpdateStats` 增加一个 device-tensor 形态：内层循环
把 6+8 个标量 `torch.stack` 成一个 1-D tensor 累加进 `_ReplayMetrics` 的 device
buffer（加权和 + 权重和），**不落 host**。`finish_optimizer_update` 做一次
`.tolist()`，把聚合结果还原成现有的 dataclass 字段。

对外接口零变化：`TrainStepMetrics` 的字段名、类型、语义、CSV 列全部不动。

**P2 — collective 合并。**
`_distributed_parity_verdict` 的两次 all_reduce 合成一次。做法：finite 编码进同一
个 float64 向量（`inf` 已经是 non-finite 的天然哨兵，:557 已经在用这个技巧），
一次 MAX reduce 出结果。三个单标量 helper 保留（其他调用点还在用），但 parity
路径改走合并版。

**P3 — 量化。** 用 `vrl/scripts/perf/backward_mfu_probe.py` 或 nsys，在
cosmos predict2（全参、compile 默认开）上测同步次数和训练段 wall time，
数字进本文 §5。

## 3. 验收标准

- **数值等价**：同一 seed、同一 batch，改前改后 `TrainStepMetrics` 每个字段
  逐位一致（float64 聚合顺序可能变，允许 `abs_diff <= 1e-12`；若不满足则说明
  聚合顺序改了语义，必须查清而不是放宽阈值）。
- **parity gate 不失效**：构造一个人为的 logprob drift（注入噪声让
  `logprob_abs_diff_max` 超 `trainer.py` 的 0.01 红线），确认 gate 仍然拦截。
  这是本 sprint 最重要的一条 —— 优化不能把 correctness gate 优化掉。
- **同步计数下降**：nsys 计数 `cudaMemcpyAsync` D2H + `cudaStreamSynchronize`，
  内层循环部分下降 ≥90%（保留的是 §1.5 的 per-microbatch 那次）。
- **多卡**：2 卡 gloo/nccl 下 parity verdict 与合并前一致（含一个 rank
  non-finite 的用例）。
- 既有测试全绿；`make verify` 绿。

## 4. 非目标

- 不动 `_all_ranks_have_work` 的 per-microbatch collective（§1.5，load-bearing）。
- 不动 parity gate 的判据、阈值、触发时机 —— 只动数值何时跨 host 边界。
- 不改 `TrainStepMetrics` 的对外 schema（CSV/wandb 列不变）。
- 不做 rollout 段的同步审计（那边的结论已经在 `SPRINT_gemm_utilization.md`，
  且 teacache 的 `.item()` 有自己的注释说明为何可接受）。
- 不引入新的 metrics 后端或异步日志线程。

## 5. 执行记录

（待填：同步计数 before/after、训练段 wall time before/after、GPU busy% 变化）

## 6. 相关

- 起点证据：`docs/sprints/info/SPRINT_cross_model_performance.md` §0
  （训练段 ~64% busy、「瓶颈是 launch/传输不是 kernel」）
- rollout 段的同类工作（已收口）：`docs/sprints/done/SPRINT_gemm_utilization.md`
- 现成的正确写法：`vrl/algorithms/grpo/continuous.py:367`
