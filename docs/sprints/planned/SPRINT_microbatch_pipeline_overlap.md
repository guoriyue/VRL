# SPRINT: microbatch pipeline overlap —— 用同一个 microbatch 贯通 rollout/reward/train（planned）

状态：proposed / bridge design。本 sprint 是两份现有工作的桥接层：

- `SPRINT_streaming_rollout_accumulation.md` 已经修了 Cosmos paper-shaped batch 的 host OOM：把一个 optimizer target batch 切成多个 microbatch，逐个 `collect -> backward -> release`。
- `SPRINT_memory_budgeted_microbatch.md` 继续把“切几次”改成“每刀多大”：`microbatch_size` 是唯一手填切片旋钮，`gradient_accumulation_steps` 派生。
- `SPRINT_continuous_scheduler_redesign.md` 讨论真正异步调度、staleness、weight sync barrier，但还没有把 microbatch 定义成 rollout/reward/train 共用的 pipeline item。

本 sprint 补这个缺口：**把 `microbatch_size` 对应的完整 prompt group 集合定义为流水线 item**，让
`rollout(m+1)` 可以和 `train(m)` 重叠，同时 optimizer step 仍然只在完整 target batch 累积完之后发生。

---

## 0. 核心判断

**应该共享同一个 `microbatch_size`，但要把它升级成调度 item，而不只是内存切片。**

当前 streaming path 的真实形状是串行的：

```python
for microbatch in microbatches:
    batch = await trainer.collect_training_batch(microbatch)
    try:
        await trainer.backward_on_training_batch(batch, total_groups=rollout_batch_size)
    finally:
        del batch
```

这个已经正确解决“不要一次持有 32 个 prompt × 8 samples”的 host OOM，但它没有形成真正流水线。下一步应该变成：

```text
begin optimizer update

submit collect microbatch 0
submit collect microbatch 1  # if capacity allows

while target_groups_trained < rollout_batch_size:
  ready = await next ready microbatch
  submit next collect microbatch when a slot opens
  backward ready microbatch
  release ready rollout/replay tensors

optimizer.step / EMA / after_optimizer_step / rollout weight sync
```

也就是：

```text
rollout/reward microbatch 1  ||  train microbatch 0
rollout/reward microbatch 2  ||  train microbatch 1
...
optimizer.step only after target batch is complete
```

这和“再加一个 training microbatch knob”不是一回事。当前 diffusion replay 训练已经在 trainer 里按 timestep 切：

```python
for b, adv_b in zip(device_batches, device_advs, strict=True):
    for j in train_indices:
        ...
        self._backward(loss)
        await asyncio.sleep(0)
```

GPU 侧已经有 `timestep × rollout_microbatch` 的天然切片。先不要再加第二个 train-only microbatch；只有当
`microbatch_size=1` 且单 timestep 仍然 GPU OOM 时，才需要单独开新的 GPU 子切片 sprint。

---

## 1. 当前系统边界

### 1.1 已落地：host OOM 修复，但顺序执行

`vrl/scripts/common/online.py:_run_streaming_optimizer_update` 做了三件正确的事：

1. 按 `rollout_batch_size // gradient_accumulation_steps` 切 prompt microbatch。
2. 每个 microbatch 单独 `collect -> backward -> release`。
3. `finish_optimizer_update()` 只调用一次，所以 optimizer step / EMA / weight sync 仍然按 target batch 计数。

它的问题不是数学，而是调度：**下一刀 collect 要等当前刀 backward 完才开始**。对视频 RL 来说，generation/reward 很慢，训练期间不让下一刀先跑，会浪费 rollout actor 的时间。

### 1.2 continuous scheduler 有异步框架，但粒度还是“整 iteration”

`ContinuousRolloutProducer` 当前的 item 是单 prompt group：

```python
class ContinuousRolloutItem:
    group_key: int
    rollout_policy_version: int | None
    batch: RolloutBatch
```

`ContinuousRolloutConsumer.drain_for_iteration()` 等到 `min_groups` 个 same-policy group 后，一次构造成 `RolloutIteration`：

```python
selected = self.queue.select_iteration(
    min_groups=min_groups,
    current_version=current_version,
    staleness=self.staleness,
)
```

而 `ContinuousRolloutSchedule.next_iteration()` 传进去的是 `min_groups=len(prompts)`。也就是说 continuous 现在仍然以“本次 trainer 请求的整批 prompts”为消费边界。它能预取 group，但没有“一个 optimizer update 内的 microbatch pipeline”这个概念。

### 1.3 weight sync 仍是 full drain barrier

`ContinuousRolloutSchedule.after_train_step()` 当前是：

```python
self.producer.pause_admission()
await self.producer.drain_inflight()
await self.lifecycle.sync_weights_after_train(phase_times)
self.producer.resume_admission()
```

这个 barrier 对 strict correctness 保守正确，但它会把跨 optimizer update 的深流水围住。microbatch overlap 的第一阶段不需要立刻拆掉这个 barrier：**先在一个 optimizer update 内重叠 collect 和 train，optimizer step 后仍然 sync 一次**。后续再接 `SPRINT_continuous_scheduler_redesign.md` 的 async weight sync / staleness 放宽。

---

## 2. 目标语义

### 2.1 配置语义

保持单根设计：

```text
rollout.rollout_batch_size        # target prompt conditions per optimizer update
rollout.microbatch_size   # pipeline item size: prompt conditions per collect/train item
actor.gradient_accumulation_steps # derived = rollout_batch_size / microbatch_size
rollout.sample_batch_size         # generation backend chunk size only
```

> 现状对齐（Evidence-First）：`microbatch_size` **目前还没落地**（由
> `SPRINT_memory_budgeted_microbatch.md` 引入）。当前代码里手填的是 `actor.gradient_accumulation_steps`，
> microbatch 大小 = `rollout_batch_size // gradient_accumulation_steps`（`vrl/trainers/core/types.py`
> 强制整除，且 `gradient_accumulation_steps>0` 时强制 `ppo_epochs==1`）。本 sprint 与两种参数化都兼容；
> 若 `microbatch_size` 尚未合入，应在依赖里标注它先行。

示例：Cosmos Predict2.5 + Kling paper-shaped batch：

```text
rollout_batch_size = 32
n_samples_per_prompt = 8
microbatch_size = 1

=> 32 microbatch items
=> each item has 1 complete prompt group = 8 videos/replay trajectories
=> one optimizer.step after 32 items
```

### 2.2 Pipeline item 合同

一个 pipeline item 必须是：

```text
RolloutMicrobatch
  update_id
  microbatch_index
  target_groups
  group_count
  n_samples_per_prompt
  rollout_policy_version
  prompts / PromptExample payloads
  TrainingBatch or RolloutIteration payload
  phase stats
```

关键点：

- `group_count == microbatch_size`，最后一刀不允许短缺；`rollout_batch_size` 必须整除。
- 每个 prompt group 必须完整包含 `n_samples_per_prompt` 个样本，不能把一个 GRPO/NFT group 拆开。
- `rollout_policy_version` 必须可追踪；strict path 下一个 optimizer update 内不能混版本。
- item 训练完必须释放 replay tensors；队列里不能长期保留已经 backward 的 `TrainingBatch`。

不一定要马上新增一个公开类。如果第一阶段只在 online recipe 内做 prefetch，可以先用内部 `_PendingMicrobatch` dataclass；但只要它跨 `scripts/common/online.py`、trainer、continuous scheduler 边界，就应该放到 orchestration/types 这类协议边界文件里。

---

## 3. 正确性不变量

> **承重决策（必须先定，否则下面的不变量 1 不成立）：`global_std` 的归一化范围。**
>
> 在 `vrl/algorithms/advantages.py` 里，advantage 的 **mean 永远是 per-group**（可流式），但
> **std 在 `global_std=true` 时是跨“传入的全部 rewards”的一个标量**：
>
> ```python
> global_std_value = rewards.std(unbiased=False)          # 跨整个传入 batch 的标量
> std = global_std_value if global_std else group_rewards.std(unbiased=False)
> ```
>
> 而 streaming path 每刀只把一个 microbatch 的 prompts 传进 `collect_training_batch`
> （`online.py:_run_streaming_optimizer_update` 先把 `rollout_batch_size` 切成 `micro` 再逐刀采集，
> `trainer.collect_training_batch` 只 concat 这一刀的 batches 算 advantage）。所以**现在的 streaming
> path 已经在按 per-microbatch 算 `global_std`**——它和 legacy full-batch 路径（`step()` 先收满整批再
> 算 advantage）在 `global_std=true` 下**本来就不数值等价**（`online.py` docstring 的 “gradient-
> equivalent” 只对 loss-scale 和 `global_std=false` 成立）。
>
> 把 microbatch 升级成 pipeline item **不会引入**这个分歧，但会放大它（item 越小，scale 越局部）。
> 必须显式三选一并写进配置/文档：
>
> - **(A) per-group std（`global_std=false`）**：item 完全自洽、可流式，**推荐作为 pipeline 默认**。
> - **(B) 保留 full-batch global std**：backward 前要对整个 target batch 做一次 std reduction barrier
>   → “item 进队列即训练” **不成立**，必须先收齐再定 scale。
> - **(C) running / EMA global std**：可流式但改语义，需单独实验验证，并应改名（不再是 “global”）。

1. **Advantage 的 mean 按 prompt group 归一化、可在 item 内计算；scale 取决于上面的 `global_std` 决策。**
   走 (A) 时 item 完全自洽，不需要等 32 个 prompt 到齐；走 (B) 时 item 内只能算到 centered advantage，
   最终 scale 要等 target batch 收齐。不要无条件断言“advantage 可以在 item 内算完”。
2. **loss scale 仍按 target batch。** `backward_on_training_batch(..., total_groups=rollout_batch_size)` 的含义不变；每刀 loss 仍除以 `target_groups × train_timestep_count`。
3. **optimizer boundary 不动。** `optimizer.step()`、GradScaler update、EMA、`algorithm.after_optimizer_step()`、checkpoint/global step、rollout weight sync 都只在完整 target batch 后执行一次。
4. **reward state 每个 optimizer update reset 一次。** `reward_fn.reset_components()` 不能改成每个 microbatch reset，否则 component-level aggregation 和 metric row 会变形。
5. **metrics 按样本数聚合。** reward mean/std、adv diagnostics、phase times 要继续跨 microbatch sample-weighted 或累加，不能用最后一个 item 覆盖整步结果。
6. **strict-on-policy 默认不变。** 第一阶段不默认打开 stale rollout 训练；如果 later 允许 `max_stale_policy_versions > 0`，必须显式进入 continuous/off-policy 配置。

---

## 4. 分阶段计划

### T1 — 先把 streaming collect 变成可 prefetch 的 stateless item

当前 `OnlineTrainer.collect_training_batch(prompts)` 会写 `self.prompts = prompts`，这让并发 collect 容易互相覆盖。第一步要把采集路径改成显式输入：

```text
collect_training_batch(prompts) should not mutate trainer prompt state in the streaming path
```

目标：

- 保留 `trainer.step()` 的 legacy API 行为。
- streaming path 使用显式 prompt microbatch，不依赖 `self.prompts` 这个可变共享状态。
- 加测试证明两个 pending collect 不会串 prompts。

### T2 — 在 online recipe 内实现 bounded microbatch prefetch

在 `_run_streaming_optimizer_update` 内增加一个很小的 pipeline executor：

```text
max_prefetch_microbatches = 1 or 2

submit collect task until prefetch slots full
for each microbatch in FIFO order:
  batch = await collect task
  submit next collect task
  await trainer.backward_on_training_batch(batch, total_groups=target)
  del batch
finish_optimizer_update()
```

第一版建议 FIFO consumption，不做 out-of-order train。理由：

- 保持 metric/debug 顺序稳定。
- 避免 reward component ordering 变化。
- 容易证明和当前串行 streaming 梯度等价。

`max_prefetch_microbatches` 不应该成为新的长期业务旋钮；先可以是 continuous/pipeline 内部 capacity，默认 1。等指标证明收益稳定，再决定是否暴露。

### T3 — 可观测性：证明真的有 overlap

必须加 metrics，不然“异步了”很容易自欺：

```text
microbatch.collect_wait_s
microbatch.train_s
microbatch.prefetch_ready_depth
microbatch.collect_task_inflight
microbatch.trainer_idle_wait_s
microbatch.rollout_actor_idle_gap_s  # if available from Ray runtime stats
```

验收不是只看 wall-clock，而是看：

- train 当前 item 时，下一个 collect task 已经在 Ray actor 上运行；
- trainer 等 collect 的空窗下降；
- host RSS 仍接近一个 microbatch，而不是回到 target batch 峰值；
- reward curve 和串行 streaming 在同 seed 下无系统性漂移。

### T4 — 接入 continuous scheduler 的 group queue

等 T1/T2 在 strict/streaming path 证明正确后，再把 continuous scheduler 的消费边界从“整 iteration”降到“microbatch item”：

```text
drain_for_microbatch(min_groups=microbatch_size)
```

保留现有 `ContinuousRolloutItem` 作为单 prompt group ready item；consumer 负责把 N 个 same-policy group 合成一个 microbatch item。不要把 queue 里的基础 item 直接改成整 microbatch，否则会损失 group 级 staleness/drop/backpressure 的灵活性。

### T5 — 再接 async weight sync / staleness

本 sprint 的第一收益是 optimizer update 内的 overlap。跨 update 深流水需要接 `SPRINT_continuous_scheduler_redesign.md`：

- admit 时预测 policy version；
- `max_stale_policy_versions >= 1` 的 bounded stale 训练；
- chunk-boundary / shadow-buffer weight sync；
- 去掉每步 full drain 的分钟级 barrier。

这一步风险更高，必须在 T1-T4 的 microbatch item 边界稳定后做。

---

## 5. 应该改什么 / 保持什么

应该改：

- streaming collect path 去掉对 `self.prompts` 的并发依赖。
- `_run_streaming_optimizer_update` 支持 bounded prefetch，而不是严格串行等待每刀 collect。
- 增加 microbatch pipeline metrics，证明 collect/train overlap 和 host RSS。
- continuous consumer 后续增加 `drain_for_microbatch`，从 same-policy group queue 中组装 microbatch item。

应该保持：

- `rollout_batch_size` 仍表示 optimizer target prompt conditions。
- `microbatch_size` 是唯一用户关心的切片大小；不新增 train-only microbatch knob。
- `sample_batch_size` 仍只管 generation backend chunk，不表达 RL batch。
- `ContinuousRolloutItem` 继续代表单 prompt group。它是 group-level queue/staleness/backpressure 的协议边界，不要为了“少一层”改掉。
- strict-on-policy 默认保持保守；staleness 放宽必须显式配置。

ALL_CAPS / thin file 处理：

- 不新增 ALL_CAPS hardcoded policy table。
- 不新建只有一两个转发函数的 helper 文件。
- 如果需要跨模块共享 pipeline item dataclass，把它放在现有 orchestration protocol/types 文件中；这是协议边界，保留 thin types 文件是合理的。

---

## 6. 非目标

- 不重新讨论 denoise step sweep；paper 的 denoise step 按 paper 配置走。
- 不替代 `SPRINT_memory_budgeted_microbatch.md` 的 RSS fail-fast / auto-tune。
- 不做 physical stage pipeline；这里的 pipeline 是 RL microbatch lifecycle，不是 denoise/vae/text encoder 物理拆 stage。
- 不默认开启 off-policy stale training。
- 不加第二个用户级训练 microbatch knob。
- 不改变 GRPO/NFT advantage 数学、loss scale、optimizer step 语义。

---

## 7. 验收标准

最小验收：

- 单元测试证明 prefetch 版 streaming 与现有串行 streaming 梯度等价。
- 单元测试钉住 `global_std` 行为(§3 承重决策)：`global_std=false` 时 streaming 与 legacy 的 advantage
  数值一致；`global_std=true` 时把 per-microbatch 与 full-batch 的差异作为**已知/受控**结果断言出来，
  而不是让它静默漂移。
- 单元测试证明两个 pending collect 不会因为 `self.prompts` 共享状态串 prompt。
- 单元测试证明 `reward_fn.reset_components()` 每个 optimizer update 只调用一次。
- 单元测试证明异常路径释放 pending task / batch，不遗留 inflight task。

真机验收：

- Cosmos Predict2.5 + Kling，`rollout_batch_size=32`、`n_samples_per_prompt=8`、`microbatch_size=1`，不发生 host OOM。
- 与串行 streaming 相比，`trainer_idle_wait_s` 下降或 rollout actor idle gap 下降。
- `metrics.csv` 仍然一行代表一个 optimizer update，不是一行一个 microbatch。
- e20/e40 reward 曲线不比串行 streaming 差；重点确认吞吐提升没有破坏 RL 信号。

---

## 关键文件引用

- streaming 基线：`vrl/scripts/common/online.py:_run_streaming_optimizer_update`
- advantage 归一化（global_std 承重点）：`vrl/algorithms/advantages.py:group_relative_advantages`
- trainer 边界：`vrl/trainers/online/trainer.py:collect_training_batch`、`backward_on_training_batch`、`finish_optimizer_update`
- 配置校验：`vrl/trainers/core/types.py:TrainerConfig.__post_init__`
- continuous 调度：`vrl/rollouts/orchestration/continuous/schedule.py`、`producer.py`、`consumer.py`、`queue.py`、`types.py`
- 已有 sprint：
  - `docs/sprints/planned/SPRINT_streaming_rollout_accumulation.md`
  - `docs/sprints/planned/SPRINT_memory_budgeted_microbatch.md`
  - `docs/sprints/planned/SPRINT_continuous_scheduler_redesign.md`
