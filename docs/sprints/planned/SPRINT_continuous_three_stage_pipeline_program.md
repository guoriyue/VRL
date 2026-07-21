# SPRINT PROGRAM：Continuous rollout / reward / training 三段流水线

状态：**planned（2026-07-21）**。当前尚未开始实施；首个执行项是
[Stage contracts and baseline](SPRINT_continuous_stage_contracts_and_baseline.md)。

## 0. 结论先行

四张 L4 的正确目标不是让 trainer、rollout、reward 三个模型在四张卡上反复换入换出，
而是保持当前固定隔离：

```text
GPU 0   trainer
GPU 1-2 rollout generation
GPU 3   UnifiedReward service
```

在这个拓扑上，把当前的一个复合 collect task 拆成真正有背压的三段流水线：

```text
versioned prompt batches
  -> generation slots
  -> bounded unscored queue
  -> reward batcher
  -> scored trainer-ready queue
  -> trainer
```

trainer 只能读取已经完整打分的 batch。eval 不在这条热路径中，不拥有队列 admission，
不阻塞 reward，不参与 optimizer step，也不是本 program 的完成条件。需要质量对比时，使用
固定输入或 checkpoint 的离线检查；生产流水线只由训练正确性、吞吐、资源和故障不变量验收。

## 1. 当前代码现实

### 1.1 已经具备的能力

- `RolloutCollector.collect_unscored()` 已把 generation 的结果表示为 `UnscoredRollout`；
  `score_rollouts()` 已能接收多个 unscored group，一次调用 reward 并构造 trainer batch。
- `collect_prompt_batches()` 已能在单次 collect 内做 generation/reward overlap，并且最多只拥有
  一个 reward task，取消时会等待 reward cleanup。
- continuous owner 已运行在独立线程和 event loop，trainer 的同步 forward/backward 不会阻塞
  producer cadence。
- `ContinuousRolloutQueue` 只保存完整的 `RolloutBatch`，consumer 会验证同一 policy version、
  group 完整性和 staleness window。
- continuous topology 已 fail closed：trainer、rollout、reward 必须占用互不冲突的 accelerator；
  这个 program 保留该边界。

### 1.2 真正的气泡根因

当前 producer 的一个 slot 调用：

```python
batches = await collect_prompt_batches(...)
```

这个 await 覆盖 generation、reward 和 batch build。只有三者全部完成，slot 才从
`_inflight` 释放并进入 ready queue。`_ActivePromptBatch` 又只允许安装一个 finite prompt
batch；下一批要等当前 batch 被 trainer 完整消费后才安装。

因此 `max_inflight_groups=4` 只能把当前四个 group 暴露出来。GPU 1/2 生成完这四组后，
如果 GPU 3 仍在串行打分，producer 没有下一批可生成，rollout GPU 仍会空闲。继续增大
`max_inflight_groups` 不会创造新的 slot。

此外，UnifiedReward 当前只实现逐 artifact `__call__()`；reward service 可以接受完整请求，
但模型前向仍是 batch=1。这个问题与 producer 解耦相关，却不是同一个实现 sprint。

## 2. 目标 ownership

| 边界 | 唯一 owner | 说明 |
|---|---|---|
| prompt batch identity / policy version | continuous scheduler | batch admission 时冻结，不能由 reward 或 trainer 改写 |
| generation task | generation pump | 完成后立即释放 rollout slot，不等待 reward |
| unscored artifact lifetime | unscored queue | 从 generation receipt 到 reward 成功、取消或 terminal cleanup |
| reward batching / retry | reward pump | 保序映射结果；不改变 sample/group identity |
| trainer-ready state | existing ready queue | 只接收已打分、batch build 完整的数据 |
| same-version selection | consumer/scheduler | 一个 optimizer iteration 不混版本、不缺组、不重复组 |
| optimizer / weight commit | trainer + lifecycle | 不进入 producer 或 reward pump |
| eval | 无 hot-path owner | 只允许离线或非阻塞旁路，不反向控制生产队列 |

`ContinuousRolloutSettings` 继续是 resolved runtime knobs 的单一来源。新增字段必须有实际的
admission、queue、runtime 调用或 validation consumer；不能增加只打印但不生效的配置。

## 3. Sprint 顺序与 gate

### Sprint 0 — Stage contracts and baseline

[文档](SPRINT_continuous_stage_contracts_and_baseline.md)

先建立 stage identity、ownership、时间区间、队列和 GPU duty 的统一测量。退出条件是能解释
一轮更新中 generation、reward、ready wait、training 和 weight sync 的 wall-clock，并用测试
钉住当前行为。它不改变调度。

### Sprint 1 — Generation / reward pump split

[文档](../parked/SPRINT_continuous_generation_reward_pump.md)

触发：Sprint 0 完成。新增 bounded unscored queue 与独立 reward pump，让 generation slot 在
artifact 生成完成后立即释放。退出条件是 generation 可以在前一组 reward 期间继续提交，且
trainer 永远看不到 unscored 数据。

### Sprint 2 — Versioned two-batch lookahead

[文档](../parked/SPRINT_continuous_versioned_lookahead.md)

触发：Sprint 1 完成。当前 recipe 已把 `prompts` 和 `next_prompts` 同时交给 schedule，本 sprint
允许 owner 在当前 batch 尚未打分完成时就安装一个 next batch。窗口固定受 policy staleness 和
host-memory cap 约束；不是无限预生成。

### Sprint 3 — UnifiedReward adaptive batching

[文档](../parked/SPRINT_unified_reward_adaptive_batching.md)

触发：Sprint 2 的实测证明 GPU 3 inference 或 request queue 是可见瓶颈。producer 先合并多个
unscored group 为一个 reward request；UnifiedReward 再实现真正的 model micro-batch。必须先过
batch=1 parity 和显存 gate。

### Sprint 4 — Capacity calibration

[文档](../parked/SPRINT_continuous_stage_capacity_calibration.md)

触发：三段流水线和 reward batching 均可单独运行。建立安全上限：generation inflight、unscored
bytes、reward batch 和 ready bytes。结果是可复现的 capacity profile，不是凭 GPU utilization
猜 batch size。

### Sprint 5 — Adaptive backpressure

[文档](../parked/SPRINT_continuous_adaptive_backpressure.md)

触发：Sprint 4 给出硬上限。controller 只在这些上限内调整 admission 和 reward batch，不改变
GPU role，不改变 trainer batch semantics。目标是最大化 useful optimizer updates/hour，而不是
追求每张卡每秒都显示 100%。

### Sprint 6 — Recovery and cleanup

[文档](../parked/SPRINT_continuous_pipeline_recovery.md)

触发：adaptive path 稳定通过 deterministic tests。补齐取消、reward retry、artifact ownership、
terminal drain、checkpoint restart 和 request idempotency；不把大型视频队列写进 checkpoint。

### Sprint 7 — 12-hour GA

[文档](../parked/SPRINT_continuous_three_stage_12h_ga.md)

触发：Sprint 0-6 全部完成。用四张 L4 做 current-vs-new A/B、故障注入和 12 小时长跑，达到
吞吐、内存、staleness、reward/gradient health 与零泄漏 gate 后才转默认。

## 4. 跨 sprint 正确性不变量

1. 一个 prompt group 的每个 sample 只有一个稳定 identity；重试不能复制或重排样本。
2. policy version 在 batch admission 时冻结；generation、reward、ready queue 和 trainer 使用
   同一个 version，不从全局 mutable state 重新读取。
3. 一个 trainer iteration 只包含一个 batch identity、一个 policy version 和完整的 distinct
   group slots。
4. unscored item 绝不能进入 trainer-ready queue；reward 失败不得产生默认分数或部分 batch。
5. staleness 超窗的 work fail closed；不得为了保持 GPU 忙而训练过期轨迹。
6. 所有队列同时受 item 和 byte cap 约束；cap 到达后 upstream 停止 admission，不靠事后
   eviction 维持稳态。
7. weight sync failure 关闭 admission；任何 stage 不得跨两个 active weight version 执行。
8. cancellation 只有一个 artifact owner；成功、失败、取消和 shutdown 最终都释放同一组资源。
9. strict-on-policy 和不支持 off-policy 的算法保持现有串行路径。
10. eval 不参与以上任一状态转换。

## 5. 统一 benchmark contract

每个 child sprint 都使用同一组 primary metrics：

```text
optimizer_updates_per_hour
useful_trajectories_per_hour
trainer_wait_for_ready_s
generation_gpu_duty_ratio
reward_gpu_duty_ratio
unscored_queue_items / bytes / oldest_age_s
ready_queue_items / bytes / oldest_age_s
generation_service_s_ewma
reward_service_s_ewma
reward_queue_wait_s
rollout_policy_staleness
discarded_or_retried_groups
artifact_live_count / bytes
```

GPU duty 是诊断指标，不是单独的通过条件。允许短暂 kernel launch、weight sync、checkpoint 和
队列切换空隙；不能用无效工作、超窗 rollout 或重复 reward 把 utilization 数字填满。

## 6. 应改变、应保持与非目标

### 应改变

- producer 从一个复合 collect task 演进为 generation pump、unscored queue 和 reward pump。
- finite prompt batch 演进为有 batch identity 的 current + one-lookahead 窗口。
- admission 从固定计数扩展为 queue bytes、service time、staleness 和 capacity envelope 的统一决定。
- UnifiedReward 在 parity/显存 gate 后获得真正的 micro-batch inference。

### 保持不变

- `RolloutCollector.collect_unscored()` / `score_rollouts()` 是现有 stage seam，先复用，不再发明
  第二套 collector API。
- `ContinuousRolloutQueue` 仍是 scored-only trainer boundary。
- `ContinuousRolloutSchedule` 的 accelerator isolation guard 保留。
- `RolloutSchedule` facade 保持薄；strict path 不吸收 continuous 的 queue 实现。
- trainer 的 advantage、PPO replay、optimizer 和 checkpoint ownership 不移动到 owner loop。

### 非目标

- 不让 trainer 与 rollout 在同一张 L4 上并发常驻。
- 不在运行中迁移三类模型或动态重新分配物理 GPU。
- 不把 eval worker、eval queue、固定 eval cadence 或模型选择逻辑接入 reward pipeline。
- 不把 denoise 拆成 physical stage pipeline；已有 chunk/data-parallel 路径保持独立。
- 不以增加 prompts、PPO replay 或无效 sampling 的方式制造利用率。

## 7. Program Definition of Done

- [ ] Sprint 0-7 各自通过并按状态移动到 `done/`。
- [ ] 四卡拓扑持续满足 trainer/rollout/reward accelerator isolation。
- [ ] rollout 可以在 reward backlog 和 trainer backward 期间继续做有用 generation。
- [ ] 所有队列在 12 小时运行中有界，host/GPU memory 无单调爬升。
- [ ] 没有 mixed version、duplicate/missing group、partial reward 或超窗训练。
- [ ] 相同 batch=1 control 上 reward 和 trainer 数值 contract 保持一致。
- [ ] 自动 controller 的 fallback 可一键退回经过验证的静态 capacity profile。
- [ ] eval 仍与 reward/training hot path 解耦。

## 8. Negative result / rollback

如果三段流水线在真实四卡 run 上不能提高 `optimizer_updates_per_hour`，或收益来自 staleness、
重复 work、host-memory 增长、reward 数值漂移，则不转默认。保留 Sprint 0 telemetry 和已证明
独立有益的 reward batching；运行时退回当前 finite-batch continuous path。不要用扩大队列掩盖
吞吐不匹配。

## 参考

- `vrl/rollouts/collector/core.py`
- `vrl/rollouts/orchestration/prompt_collection.py`
- `vrl/rollouts/orchestration/continuous/producer.py`
- `vrl/rollouts/orchestration/continuous/owner.py`
- `vrl/rollouts/orchestration/continuous/scheduler.py`
- `vrl/rollouts/orchestration/continuous/queue.py`
- `vrl/rollouts/orchestration/continuous/consumer.py`
- `vrl/rewards/service/server.py`
- `vrl/rewards/models/unified_reward_video.py`
- `vrl/config/presets/experiment/wan_2_1/online_grpo_robotics_physics_4x_l4_continuous.yaml`
- [Continuous scheduler redesign](../done/SPRINT_continuous_scheduler_redesign.md)
- [Slime overlap strategy](../done/SPRINT_slime_overlap_strategy.md)
- [Reward service](../done/SPRINT_reward_service.md)
- [Historical async rollout/train overlap](../parked/SPRINT_async_rollout_train_overlap.md)
- [Historical batched reward inference](../parked/SPRINT_reward_batched_inference.md)
- [Rollout finalize overlap GA](SPRINT_rollout_finalize_overlap_ga.md)
