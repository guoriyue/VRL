# SPRINT: Slime overlap strategy 对齐

状态：proposed。目标是把 slime 的 async rollout 经验吸收到 VRL continuous rollout，
但不把 slime 的实现细节机械搬过来。

## 0. 一句话

slime 的 async overlap 不是“裸 generation 和 training overlap”，而是：

```text
rollout_manager.generate(...)
  = generate samples
  + reward / group reward
  + reward post-process
  + convert to train_data
  + split by DP

actor_model.async_train(...)
  = consume train_data
  + compute log_probs / ref log_probs / values
  + compute advantages / returns
  + train
```

VRL continuous rollout 的正确方向相同：**ready queue 只放已经打分、可以训练的 batch**。
trainer 不应该等待 reward，也不应该消费半成品 rollout。

## 1. Slime 代码事实

### 1.1 async 训练 overlap 的对象是完整 rollout future

`train_async.py` 先提交下一轮 rollout future，再训练当前 rollout：

```python
rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
...
rollout_data_curr_ref = ray.get(rollout_data_next_future)
...
rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)
...
ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))
```

这说明 slime overlap 的生产单元是 `rollout_manager.generate(...)`，不是只包住
SGLang decode。

### 1.2 reward 在 rollout function 内完成

默认 SGLang rollout 路径是 `generate_and_rm`：

```python
sample = await generate(args, sample, sampling_params)
...
sample.reward = await async_rm(args, sample)
```

group reward 也是同一生产侧边界内完成：

```python
group = await asyncio.gather(*tasks)
...
rewards = await batched_async_rm(args, group)
```

所以 slime 的 actor 侧拿到的 sample 已经有 `sample.reward`。

### 1.3 rollout manager 生成训练数据时做 reward post-process

`RolloutManager.generate()` 调用顺序是：

```python
data, metrics = self._get_rollout_data(rollout_id=rollout_id)
...
data = self._convert_samples_to_train_data(data)
return self._split_train_data_by_dp(data, self.train_parallel_config["dp_size"])
```

`_convert_samples_to_train_data()` 写入：

```python
train_data = {
    "tokens": ...,
    "response_lengths": ...,
    "rewards": rewards,
    "raw_reward": raw_rewards,
}
```

### 1.4 GRPO/GSPO reward normalization 默认在 rollout manager 内做

slime 的参数默认值：

```python
parser.add_argument("--disable-grpo-std-normalization", action="store_false",
                    dest="grpo_std_normalization")
parser.add_argument("--disable-rewards-normalization", action="store_false",
                    dest="rewards_normalization")
```

没有传 disable flag 时，`rewards_normalization=True` 且
`grpo_std_normalization=True`。当 `n_samples_per_prompt == 1` 时，代码会强制：

```python
args.grpo_std_normalization = False
```

GRPO/GSPO 路径里 `_post_process_rewards()` 做：

```python
rewards = rewards - mean
...
rewards = rewards / (std + 1e-6)
```

actor 侧的 `compute_advantages_and_returns()` 对 GRPO/GSPO 不再做 group reward
normalization；它把传入的 `rewards` 扩成 token-level returns：

```python
returns = get_grpo_returns(rewards, kl)
advantages = [r for r in returns]
```

## 2. VRL 当前事实

### 2.1 continuous producer 已经生产完整打分 batch

`RolloutCollector.collect()` 当前顺序是：

```python
output = await self.runtime.generate(collector_request.request)
...
rewards = await self.reward_scorer.score(...)
batch = batch_builder.build(rewards)
```

`ContinuousRolloutProducer` 调用 `collect_prompt_batches(...)`，然后把完成的
`RolloutBatch` 放进 ready queue。也就是说，VRL continuous queue 里已经是带
`batch.rewards` 的 trainer-ready batch。

### 2.2 trainer 侧统一计算 advantages

VRL trainer 当前在拿到完整 iteration 后做：

```python
all_rewards = torch.cat([b.rewards for b in all_batches])
all_group_ids = torch.cat([b.group_ids for b in all_batches])
advantages_all = self.algorithm.compute_advantages_from_tensors(
    all_rewards,
    all_group_ids,
)
```

`group_relative_advantages()` 是 GRPO / TokenGRPO / DiffusionNFT 共享的数学入口。
这比把 normalization 放进 collector 更适合 VRL：collector 只负责 reward I/O 和
batch assembly，algorithm 负责 advantage 语义。

### 2.3 VRL 与 slime 的数学差异

slime 的 std 是 `torch.std(..., dim=-1)`，PyTorch 默认 `unbiased=True`。
VRL 当前用：

```python
group_rewards.std(unbiased=False)
```

这不是明显 bug。Flow-GRPO 上游 `stat_tracking.py` 用 `np.std`，默认就是
population std。VRL 在这一点上更接近 Flow-GRPO。

SD3.5 配置当前显式设置：

```yaml
algorithm:
  global_std: true
```

这又和 slime 默认的 per-group std 不同。是否使用 global std 是算法实验选择，
不属于 overlap 架构本身。

## 3. 核心决策

1. **保持 reward 在 rollout production 侧。**
   trainer 只消费已经打分的 batch。continuous ready queue 不接受未打分 sample。

2. **不把 reward normalization 下沉到 collector。**
   VRL 保留 raw reward；advantage normalization 继续由 algorithm/trainer 层统一完成。
   这是更清晰的职责边界，也避免不同 collector 家族复制 GRPO 语义。

3. **把 slime 的 async boundary 学过来，不照搬 slime 的 reward math 位置。**
   slime 的优势是完整 rollout future 与 train overlap；VRL 已经有 producer/queue/consumer
   的更强边界，应该强化这个边界，而不是把算法逻辑推回 rollout manager。

4. **weight sync 前必须 drain generation + reward。**
   slime 在 update weights 前等 pending generation 完成；VRL continuous 的
   `pause_admission -> drain_inflight -> sync_weights -> resume_admission` 是同一原则。
   因为 VRL 的 in-flight collect 包含 reward，所以 drain 必须覆盖 reward。

## 4. 实施计划

### T1. 固化 ready queue contract

新增/扩展 continuous rollout 测试，证明：

```text
ContinuousRolloutProducer 入队的 batch 一定已经有 rewards
reward failure 作为 producer failure 传播给 consumer
consumer 永远不会拿到未打分 batch
```

验收：

```text
pytest -q tests/rollouts/orchestration/continuous/test_schedule.py
pytest -q tests/rollouts/collector/test_runtime.py
```

### T2. 覆盖 reward-in-flight weight sync barrier

新增一个慢 reward fake：generation 已完成、reward 仍在等待时触发
`after_train_step()`，断言 sync 不会在 reward 完成前发生。

验收：

```text
after_train_step waits for in-flight collect, including reward_score
policy version does not change inside one generated/scored batch
```

### T3. 修正 continuous collect phase metrics 的归属

当前 `RolloutCollector.last_collect_phases` 是 collector 实例上的共享 mutable 状态。
在 continuous mode 多个 in-flight collect 并发时，这个字段不适合作为 per-item
指标来源。

改法：

```text
collect() 返回 batch 时，把 collect.engine_generate / collect.reward_score 写入
batch.context 或 ContinuousRolloutItem.phase_times
consumer build iteration 时聚合这些 phase_times
```

验收：

```text
continuous phase_times 能同时报告 engine_generate 和 reward_score
并发 collect 不互相覆盖 last_collect_phases
strict mode 继续保留现有 profile 输出
```

### T4. 保留 raw reward，显式记录 normalized advantage

不在 collector 改写 `batch.rewards`。增加或确认日志字段：

```text
reward_mean / reward_std       来自 raw rewards
advantage_mean / adv_zero_rate 来自 algorithm-produced advantages
```

如果需要对齐 slime 的 `raw_reward` 概念，可以只在 debug context 里记录原始 reward
副本，不改变训练输入语义。

### T5. 只有 profiling 证明 reward 是瓶颈时，才拆独立 reward stage

默认不要引入：

```text
generation queue -> reward workers -> ready queue -> trainer
```

这个三段 pipeline 只有在满足以下条件时才值得做：

```text
collect.reward_score 占 rollout production wall time 的主要部分
reward 有独立 CPU/GPU/Ray/remote service 资源
ready queue starvation 来自 reward，而不是 generation
reward artifacts 的内存生命周期已经可控
```

否则独立 reward stage 只会增加状态机、backpressure 和 failure handling 复杂度。

## 5. Architecture Hygiene

### 应该改

- 增加 continuous contract tests：确保 ready queue 只收完整打分 batch。
- 增加 reward-in-flight sync barrier test：确保权重更新不穿过 reward 阶段。
- 把 continuous per-collect phase metrics 从 shared collector field 移到 batch/item 归属。

### 不应该改

- 不把 `RolloutCollector.collect()` 拆成裸 generation API 和 reward API。
  这个函数是 rollout production facade，当前 thin boundary 有实际含义。
- 不把 `ContinuousRolloutProducer` 改成懂 reward internals 的类。
  producer 只调 collector；collector 负责 request -> generation -> reward -> batch。
- 不把 GRPO reward normalization 复制到 collector 或 family-specific builder。
  algorithm 层是 single source of truth。
- 不因为 slime 有 `_post_process_rewards()` 就把 VRL 的 raw reward contract 打破。

### ALL_CAPS / thin files 判断

本 sprint 不引入新的 ALL_CAPS hardcoded tables。

需要保留的薄边界：

```text
vrl.rollouts.batch.__init__      public API facade
RolloutCollector.collect         rollout production facade
ContinuousRolloutProducer        cadence / backpressure owner
GRPO.compute_advantages_from_tensors algorithm-owned math boundary
```

这些边界提供 public API、职责隔离或跨家族一致性，不按 LOC 压平。

## 6. Non-goals

- 不改变 GRPO 数学默认值（`global_std`、`unbiased=False`、`adv_clip_max`）。
  这些属于算法实验，不属于 overlap 架构。
- 不把 slime 的 rollout-manager normalization 位置照搬到 VRL。
- 不做 fully async / off-policy replay buffer；当前目标是 bounded ready queue。
- 不在没有 profiling 证据前拆 reward worker pipeline。
- 不处理 unrelated reward model 配置或 Kling reward 改动。

## 7. 验收标准

```text
代码契约：
  continuous queue 只包含已打分 RolloutBatch
  weight sync drain 覆盖 generation + reward
  reward failure 能 fail fast 到 consumer/trainer

可观测性：
  continuous mode 能看到 collect.engine_generate / collect.reward_score
  phase_times 不受并发 collect 覆盖

算法边界：
  batch.rewards 仍是 raw reward
  advantages 仍由 algorithm/trainer 统一计算
  raw reward metrics 和 advantage metrics 同时保留
```

## 8. 参考

- `/home/mingfeiguo/Desktop/slime/train_async.py`
- `/home/mingfeiguo/Desktop/slime/slime/rollout/sglang_rollout.py`
- `/home/mingfeiguo/Desktop/slime/slime/ray/rollout.py`
- `/home/mingfeiguo/Desktop/slime/slime/backends/megatron_utils/loss.py`
- `/home/mingfeiguo/Desktop/slime/slime/utils/arguments.py`
- `/home/mingfeiguo/Desktop/flow_grpo/flow_grpo/stat_tracking.py`
- `vrl/rollouts/collector/core.py`
- `vrl/rollouts/orchestration/continuous/producer.py`
- `vrl/rollouts/orchestration/continuous/consumer.py`
- `vrl/trainers/online/trainer.py`
- `vrl/algorithms/advantages.py`
