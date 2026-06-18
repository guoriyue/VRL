# SPRINT: Slime overlap strategy 对齐

状态：T1/T2/T3 已落地（contract tests + inflight-reward barrier in 8e3b3c9；phase-timing 收敛到 per-item stats in 1cf1bff，last_collect_phases 已删除）；T4（raw reward + advantage metrics）已基本被现有 pre_filter_reward_mean/std + advantage_mean/adv_zero_rate 覆盖但未作专项；T5（独立 reward stage）按 profiling gate 暂不做；T6（straggler control: over-sample/first-completed-wins/abort tail + wasted_groups telemetry）未动。
（2026-06-12 读完 `docs/sprints/reading/slime.md` 后复核：主方向仍成立；
补充 straggler control / 非 draining barrier 的后续边界，见 §3a / §4）。
目标是把 slime 的 async rollout 经验吸收到 VRL continuous rollout，但不把 slime 的实现细节
机械搬过来。

**Scope guard（2026-06-17）**：这里只研究完整 rollout future / ready batch 与 trainer 的 overlap。
不把 slime 的 async 经验解释成 microbatch/minibatch prefetch；sync microbatch 仍由 streaming
accumulation 文档负责。

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
这里的 batch 边界是 rollout production 边界，不是 training microbatch async 边界。

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

4. **当前阶段 weight sync 前仍 drain generation + reward；但这不是终局。**
   slime async 在 update weights 前等 pending generation 完成；VRL continuous 的
   `pause_admission -> drain_inflight -> sync_weights -> resume_admission` 是同一原则。
   因为 VRL 的 in-flight collect 包含 reward，所以当前 strict barrier 必须覆盖 reward。
   但 `reading/SPRINT_framework_lessons_vrl.md` 的 P1-1 也说明：长视频上全局 drain 会让
   trainer 被 straggler 卡住。后续要单独做非 draining barrier：按 request/chunk 边界换权重，
   由 `StalenessPolicy` 接住 mixed-version tail，保证单条 denoise trajectory 不跨 policy。

### 3a. Reading refresh：哪些 slime 机制该吸收，哪些不该照搬

读完 `docs/sprints/reading/slime.md` 后，本 sprint 的判断需要更精确：

```text
保留：
  ready queue 只放完整、已打分、trainer-ready 的 batch。
  reward failure fail-fast；不要让 trainer 消费半成品。
  reward normalization 不下沉到 collector；VRL algorithm 层继续做 advantage 语义。

新增：
  collect phase timing 必须绑定到 item/batch/request，不能读共享 last_collect_phases。
  over-sample / first-completed-wins / abort-tail 是 straggler control 的正确方向。
  每次 surplus / abort / wasted group 必须有 telemetry，不能像 slime 一样静默丢 finished work。

暂不做：
  partial-rollout resume。AR token 可以从 partial tokens 恢复；diffusion latent trajectory
  不能安全跨 policy/step 续跑。若未来做 diffusion partial buffer，buffer 内容必须进 checkpoint，
  不能复制 slime 的"buffer 不随 cursor 保存"缺口。
  非 draining weight sync。本 sprint 只先把当前 strict barrier 测准；非 draining barrier
  是后续独立 sprint。
```

## 4. 实施计划

### T1. 固化 ready queue contract — DONE (2026-06-12)

落地于 `tests/rollouts/orchestration/continuous/test_contracts.py`
（`test_ready_queue_gets_items_only_after_reward_scoring` /
`test_failed_reward_scoring_never_enqueues`）和
`tests/rollouts/orchestration/continuous/test_schedule.py`
（`test_reward_failure_fails_fast_and_never_reaches_queue`）。

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

### T2. 覆盖 reward-in-flight weight sync barrier — DONE (2026-06-12)

落地于 `test_schedule.py::test_weight_sync_waits_for_inflight_reward`（schedule 级，
gated reward fake）和 `test_contracts.py::test_drain_inflight_waits_for_generation_and_reward`
（producer 级，generation 与 reward 双 gate），外加
`test_contracts.py::test_items_carry_policy_version_captured_at_submission`
钉住 submission-time version stamping。

新增一个慢 reward fake：generation 已完成、reward 仍在等待时触发
`after_train_step()`，断言 sync 不会在 reward 完成前发生。

验收：

```text
after_train_step waits for in-flight collect, including reward_score
policy version does not change inside one generated/scored batch
```

### T3. 修正 continuous collect phase metrics 的归属 — DONE (2026-06-12)

`RolloutCollector.last_collect_phases` 已删除。phase 时间随 collect 调用流动：
`UnscoredRollout.phases`（engine_generate per call；call 级 reward_score/batch_build
只记在第一个 group 上避免重复计数）→ `collect_prompt_batches(phase_times=...)` 聚合
→ strict 写进 `iteration.phase_times`，continuous 经 `ContinuousRolloutItem.phase_times`
（每个 collect 调用只挂在首个 item 上）由 consumer 求和进 `iteration.phase_times`；
trainer 不再读 collector 共享字段（`vrl/trainers/online/trainer.py`）。
测试：`test_runtime.py::test_collect_phase_timings_are_per_call_not_shared`、
`test_prompt_collection.py::test_phase_times_accumulate_per_call`、
`test_contracts.py::test_consumer_aggregates_item_phase_times`。

原始问题：`RolloutCollector.last_collect_phases` 是 collector 实例上的共享 mutable 状态。
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

这个任务现在优先级上调。`reading/slime.md` 的 per-request tracing / nested metric
accumulator 证明：phase timing 应随 sample/request 生命周期流动，而不是挂在单例 collector
字段上。VRL 不需要复制 slime 的 `Sample` dataclass，但应把 `collect.engine_generate` /
`collect.reward_score` 放进 `ContinuousRolloutItem.phase_times` 或 batch context，由 consumer
聚合到 `RolloutIteration.phase_times`。

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

### T6. Straggler control：over-sample / first-completed-wins / abort tail

slime 最值得借的 runtime 策略不是把 reward 拆成单独 stage，而是：

```text
submit > needed groups
collect first N completed groups
abort or cancel tail
record wasted / dropped / aborted groups
```

VRL 当前 continuous producer 固定维持 `max_inflight_groups` collect jobs，consumer 等
`min_groups` same-policy groups。长视频或慢 reward 下，一个慢 group 会拉长整轮等待。

首版只做完整 group 级别，不做 partial resume：

```text
admit N + spare complete groups
consumer 取 first-completed 的 min_groups
tail group 在 request/chunk 边界 cancel 或让其完成后丢弃
所有 surplus/drop 都写 telemetry
```

验收：

```text
同 seed 下被接受 group 的 group_id / sample order deterministic
surplus 完成但未消费的 group 有 wasted_groups / wasted_samples telemetry
abort tail 不产生 trainer-visible batch
不引入 partial latent resume 或跨 policy continuation
```

## 4a. Multi-GPU placement strategy（记录：当前 auto 规则不是最优解）

### 现状（2026-06-10 落地的保守默认）

`share_with_rollout` 已三态化（`vrl/ray/resources.py`）：unset = auto——有空余卡给
reward 专卡，否则共享 rollout 池。它解决的是**单卡 footgun**（显式 true 在多卡机上仍
强制 reward/rollout 挤一张卡、每轮 7s+ 装卸churn），是正确性修复，**不是吞吐最优的
多卡分配策略**。

### 为什么"reward 专卡"很可能是错的多卡策略

GPU-seconds 账（SD3.5/cosmos 实测）：一轮里 rollout denoise ≈ 78s，reward 推理
≈ 1.1s（93ms/video × 12）。auto 在 3 卡机上给 reward 整卡 = 把一张卡分给**忙 1% 的
角色**，而真正的瓶颈（generation）只有 1 卡。吞吐最优很可能是：

```text
trainer=0  rollout=1+2(DP)  reward 与 trainer 时分共享 GPU0
```

reward 与 trainer 天然时分互补：trainer 在 generation/scoring 阶段闲置，reward 恰好
在那时运行——同卡不冲突，且都可常驻（显存够时）或 warm offload（之前评估过的
P1.5 `model.to('cpu')` 方案在这个组合里复活）。

### slime 的三种 placement（都不是"reward 专卡"）

```text
--colocate (+--offload)   推理引擎与 actor 同卡时分复用,阶段间 offload
                          (slime/utils/arguments.py:70-93)
disaggregate              训练与推理分池(默认形态)
--rm-type remote_rm       reward 外包成远程服务,不占训练集群 GPU
                          (slime/utils/arguments.py:1180-1207)
```

### 未来决策规则（多卡机器到位后用这个，不要用 auto 的直觉）

1. **按实测 phase wall time 分配边际 GPU**：先量 rollout / reward / train 三段墙钟，
   边际卡永远给最大消耗者（当前数据下 = rollout DP worker）。
2. **reward 专卡排最后**：只有 reward wall time 成为主要部分（大 VLM reward、高
   fps/分辨率、大 batch）才值得独立卡——与本 sprint §T5 的 gate 同一判据。
3. **优先考虑 reward↔trainer 同卡时分**，其次 remote rm 服务化（slime 路线），
   最后才是专卡。
4. auto 规则保留为安全默认（它保证能跑、消除 churn），但多卡 production 配置应当
   显式写 placement，并附 phase-time 依据。

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
- 不做 microbatch/minibatch async；不在 `_run_streaming_optimizer_update` 内做 prefetch。
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
