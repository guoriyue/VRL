# SPRINT：Continuous adaptive backpressure

状态：**parked（2026-07-21）**。等待
[Stage capacity calibration](SPRINT_continuous_stage_capacity_calibration.md) 产生已验证的 hard limits。

父 program：[Continuous three-stage pipeline](../planned/SPRINT_continuous_three_stage_pipeline_program.md)

## 0. 结论先行

实现一个确定性、可解释、有滞回的 controller，在 capacity profile 的硬边界内调整：

```text
generation admission target
reward request batch target / bounded wait
current-vs-lookahead priority
unscored and ready queue target watermarks
```

它不调整 trainer batch、samples、PPO epochs、staleness window 或物理 GPU role。目标函数是
`useful optimizer updates/hour`，不是瞬时 `GPU-Util=100%`。当 telemetry 缺失、异常或 controller
不确定时，退回已验证的 static profile，不能继续盲调。

## 1. Root cause / current behavior

Current producer admission checks only:

```text
inflight_count < max_inflight_groups
```

The ready-byte limit is a fail-fast guard requiring the finite batch to fit in
full; it does not participate in admission. The producer does not know the
unscored backlog, reward service time, trainer demand, batch priority, or
capacity profile.
固定 `max_inflight_groups=4` 在 reward 变慢时可能生成过多 unscored artifact；在 reward 变快时又
可能不足以填满 rollout workers。

把所有 caps 手工调大不是自适应：它会把瓶颈从 GPU idle 转成 host RAM、disk、queue age 或
stale work。

## 2. Goal and ownership boundary

新增一个 scheduler-owned `PipelineController`。输入只来自 typed snapshot：

```text
current/lookahead batch state
generation inflight and service-time EWMA
unscored queue items/bytes/oldest age
reward inflight, service-time EWMA and queue wait
ready queue items/bytes/oldest age
trainer demand / recent train-step service time
current policy version and allowed staleness
capacity profile hard limits
```

输出 typed decision：

```text
generation_target
reward_batch_target
reward_batch_wait_ms
admit_current
admit_lookahead
decision_reason
fallback_active
```

每个输出字段必须进入 scheduler/reward pump control flow 或 validation；`decision_reason` 若只用于
显示，在定义处标注 display/provenance-only。

## 3. Correctness and resource invariants

1. hard capacity、byte cap、staleness 和 batch ordering 是不可覆盖的 guards；controller 只能在其
   内部选择 target。
2. current batch missing work 永远优先于 lookahead。
3. controller 不创造 prompts，不改变 group size/sample count，不重复成功 work。
4. 一个 decision 在固定 snapshot 下确定；不依赖 wall-clock task race。
5. service-time 使用 bounded EWMA/robust window，异常值有上限；不能被一次 cold compile 永久拖偏。
6. target 变化有 hysteresis 和最小 dwell ticks，避免 generation/reward batch 每次 poll 抖动。
7. metrics 缺失、non-finite、counter 回退、profile mismatch 时立即 static fallback 并记录一次原因。
8. controller 不能增加 `max_stale_policy_versions`。

## 4. Controller policy

第一版使用明确规则，不使用强化学习、黑盒 Bayesian optimizer 或线上随机探索：

### Priority 0 — Correctness guards

```text
terminal/fatal error -> close admission
weight sync pause -> no new generation
capacity hard cap -> block upstream
current batch incomplete -> do not consume lookahead
staleness bound -> block invalid lookahead
```

### Priority 1 — Avoid trainer starvation

- current batch 缺 generation：把可用 generation slots 优先给 current。
- current batch 有 unscored item 且 trainer 在等：reward batch wait 降到最小，立即 dispatch。
- current batch reward 已齐：不再为它保留 reward capacity。

### Priority 2 — Fill useful overlap

- trainer 正在 replay 且 lookahead 尚未达到一个完整 batch：允许 lookahead generation。
- reward backlog 高于目标：降低 generation target，避免 unscored bytes/age 继续增长。
- reward 空闲且 unscored backlog 可组成更大 batch：在 bounded wait 内聚合。

### Priority 3 — Stabilize watermarks

目标不是让队列永远满，而是保持：

```text
enough current work to avoid trainer wait
at most one useful lookahead batch
small reward batching reservoir
bounded ready residency
```

## 5. Implementation stages

### T0 — Pure decision model

- controller 是无 I/O 的 pure state transition；fake snapshots 覆盖所有 decision reasons。
- static profile 是同一 decision schema 的固定实现，不写平行 scheduler path。

### T1 — Runtime integration

- The producer remains the inflight-admission owner; the consumer and
  `StalenessPolicy` remain the version-validation and full-batch-selection
  owners. The controller may compose these boundaries but must not reintroduce
  one scheduler that owns both decisions.
- producer/reward pump 只执行 decision，不重新解释 queue/staleness。
- 现有 max knobs 迁移为 capacity/default target 的单一 typed source；不保留两套同义设置。

### T2 — Reward batch adaptation

- reward pump 在 controller 输出的 size/wait 内聚合。
- current trainer demand 可打断 wait 并提交 partial batch。
- model OOM fallback 降低 observed safe target，但不能超过 capacity profile；是否持久化新值由 profile
  sprint 的 schema 决定，不在 metrics 中偷偷成为配置。

### T3 — Fallback and operator visibility

- config 支持 `static` / `adaptive` 两个明确 mode；默认先 static。
- adaptive telemetry 异常自动 fallback，不终止健康训练。
- correctness guard 触发仍 fail closed，不能被 fallback 掩盖。
- metric row 记录 active mode、target、reason、fallback transition。

### T4 — Simulation and real smoke

- 用 deterministic simulation 重放 fast/slow generation、reward burst、trainer slowdown、OOM-reduced
  capacity、weight sync 和 transient error。
- real four-L4 smoke 对比 static/adaptive 的 updates/hour、trainer wait、queue high-watermark 和 useful
  duty，不用 eval 分数做 controller 输入。

## 6. Failure, cancellation and recovery semantics

- controller exception 立即切 static profile，并保留 exception 摘要；不能让 owner cadence task 死亡。
- invalid correctness snapshot（future version、duplicate identity、negative bytes）不是可 fallback 的
  telemetry 问题，必须 terminal fail。
- restart 时 controller 从 static target 和空 EWMA 开始 warmup；不从 metrics.csv 猜内部状态。
- reward/generation task cancellation 由各 stage owner 处理，controller 只停止新 admission。

## 7. Telemetry

```text
continuous.controller_mode
continuous.controller_fallback
continuous.generation_target
continuous.reward_batch_target
continuous.reward_batch_wait_target_ms
continuous.controller_decision_changes
continuous.controller_reason
continuous.current_starvation_events
continuous.unscored_high_watermark
continuous.ready_high_watermark
continuous.useful_generation_ratio
continuous.useful_reward_ratio
```

reason 可映射稳定 enum 到 metric；完整文本只进日志，避免高基数字段。

## 8. Verification and acceptance gates

- fixed snapshot -> fixed decision，所有 hard guard 优先级有单测。
- current starvation 时 reward wait 被打断，lookahead 不抢 current slots。
- reward slowdown 使 generation target 下调，queue 恢复后有 hysteresis 地回升。
- missing/NaN/counter reset 触发 static fallback；future/mixed version 仍 terminal fail。
- controller 永远不输出超 capacity profile 的 target。
- simulation 中 adaptive 不产生更多 stale/discarded logical groups，并降低 trainer wait 或 queue peak。
- real smoke 的 updates/hour 不低于 static，且 queue memory/age 不恶化；否则 negative exit 保持 static。

## 9. What changes / what stays

### 改变

- scheduler admission 使用 stage snapshot 和 controller decision。
- generation/reward target 可在安全 envelope 内调整。
- telemetry 能解释每次 target 变化。

### 保持

- capacity profile 是 hard limit。
- batch/staleness/version/queue correctness guards。
- fixed GPU role 和 model residency。
- trainer mathematical batch 与 optimizer semantics。

## 10. Non-goals

- 不做动态 GPU placement。
- 不让 trainer 和 rollout 共享同一 L4。
- 不在线增加 prompts、samples、timesteps 或 PPO replay。
- 不以 reward 值高低控制吞吐调度。
- 不把 eval 反馈给 controller。

## 11. Definition of Done

- [ ] pure controller、static fallback 和所有 guards 有测试。
- [ ] 所有输出 knobs 都有行为 consumer。
- [ ] adaptive 不越过 capacity/staleness/batch bounds。
- [ ] simulation 覆盖主要快慢/故障组合。
- [ ] four-L4 smoke 不低于 static throughput 且不增加 queue risk。
- [ ] negative result 时 adaptive 保持非默认。

## 12. References

- `vrl/rollouts/orchestration/continuous/scheduler.py`
- `vrl/rollouts/orchestration/continuous/producer.py`
- `vrl/rollouts/orchestration/continuous/owner.py`
- `vrl/rollouts/orchestration/continuous/types.py`
- `vrl/rollouts/orchestration/continuous/staleness.py`
- `tests/rollouts/orchestration/continuous/test_scheduler.py`
- `tests/rollouts/orchestration/continuous/test_contracts.py`
