# SPRINT：Continuous stage contracts and baseline

状态：**planned（2026-07-21）**。

父 program：[Continuous three-stage pipeline](SPRINT_continuous_three_stage_pipeline_program.md)

前置：[Online metrics IO contract](../done/SPRINT_online_metrics_io_contract.md)。这是 program 的首个
continuous 实施 sprint，但新增 telemetry columns 前先建立单一 CSV row schema，避免继续同步
header、row dict 与 format order 三份定义。

## 0. 结论先行

先把 generation、reward、trainer-ready 和 training 四个边界的 identity、ownership 与时间区间
钉住，再改并发。当前 metrics 能看到 collect 总阶段和 ready queue，但不足以回答“哪一段在等
谁、队列里是哪一个 batch、GPU 空闲是背压还是饥饿”。本 sprint 只补 contract、telemetry 和
baseline，不改变生产调度。

eval 不属于该 contract。真实 reward、group 完整性、policy parity 和 optimizer health 足以
验收；离线 checkpoint comparison 可以作为补充证据，但不进入事件循环。

## 1. Root cause / current behavior

`ContinuousRolloutItem` 只有 `group_key` 和 `rollout_policy_version`。单 finite batch 时足够，
但未来同时存在 current 和 lookahead batch 后，两个 batch 都会出现 `group_key=0..N-1`；没有
稳定 `batch_id` 就无法正确选择或诊断。

当前 producer state 记录全局 `submitted_count` / `completed_count` / `error_count`，ready queue
记录 items/bytes/age；缺少：

```text
batch_id
stage item id
generation queue wait / service interval
unscored ownership interval
reward queue wait / service interval
trainer-ready wait
per-stage retry / cancellation / discard reason
backpressure reason duration
```

`collect.generation_reward_overlap` 只统计一次 `collect_prompt_batches()` 调用内部的 interval。
多个并发 producer task 之间发生的 overlap 不会被它完整表达，因此不能单独用它证明四卡流水线。

## 2. Goal and ownership boundary

定义一个最小、可贯穿 stage 的 identity contract：

```text
batch_id
group_slot
policy_version
attempt
generation_request_id
reward_request_id
```

- `batch_id + group_slot` 是逻辑 work identity。
- `attempt` 只表达同一逻辑 work 的重试，不改变 sample seeds 或 group mapping。
- generation/reward request id 是外部调用 provenance，不替代逻辑 identity。
- policy version 在 batch admission 时冻结。

优先把 identity 放进现有 typed item/state；不建立松散 metadata 字典作为控制面。若某字段只有
日志消费者，必须在定义处标注 display/provenance-only；否则必须被 selection、validation、
runtime request 或错误分支消费。

## 3. Correctness and resource invariants

1. 新 telemetry 为 observation-only；static schedule 的提交顺序、batch size 和结果逐位不变。
2. 指标不能从 mutable global reward cache 读取；每个 item 自己携带对应 timing。
3. interval 使用 monotonic clock；跨进程只传 duration 或各进程自己的时间线，不能直接比较
   未同步 wall-clock。
4. stage counters 必须区分 logical item 和 attempt，避免 retry 被误报为吞吐。
5. GPU duty probe 是 one-shot validation artifact；结论写入 `docs/sprints/info/` 后删除 scratch
   CSV，不让 scratch output 进入 import graph。
6. 不读取或记录 prompt 文本、secret、完整 model output。

## 4. Implementation stages

### T0 — Identity contract

- 为 finite batch 建立 owner-assigned monotonic `batch_id`。
- 将 batch identity 贯穿 producer item、ready item、consumer selection 和 metric row。
- 为 request retry 建立 `attempt`，并锁定 seed/sample identity 不变。
- 添加 architecture test，禁止 trainer/reward 重新派生 batch identity。

### T1 — Stage interval telemetry

- 通过 `OnlineMetricRow` 的单一 field/mapping扩展稳定 CSV，不手改 header/format list。
- 记录 generation admission wait、service、receipt。
- 记录 reward admission wait、service、receipt。
- 记录 ready queue residence 和 trainer demand wait。
- 记录 training replay、optimizer step、weight sync 的现有 phase，并在 update 级合并。
- 每种 backpressure reason 记录 count 和 duration，而不是只打印最近一次字符串。

### T2 — Queue/resource snapshot

- 输出 item count、estimated bytes、oldest age、distinct batches、distinct versions。
- 通过固定采样 probe 记录 GPU duty/power/memory，不把 `nvidia-smi` polling 塞进生产 hot loop。
- 记录 useful logical completions、attempts、retries、discarded/stale groups。

### T3 — Four-L4 baseline

用当前 robotics continuous recipe 跑可重复 baseline，至少覆盖：

```text
rollout generation active
reward backlog
trainer backward active
weight sync
current batch consumed and next batch installed
```

将 timeline、测量口径和结论写入一个 `docs/sprints/info/` 长期档案；scratch probe 文件按 one-shot
生命周期清理。

## 5. Failure, cancellation and recovery semantics

- telemetry 写入失败不能悄悄改变调度；必要字段无法构造时 fail fast 于 admission 前。
- producer retry 必须产生新 attempt，但沿用原 identity。
- cancellation 的最后状态只写一次；不能同时计为 success 和 cancelled。
- metrics CSV partial append、process restart 和 counter reset 的语义要有测试。

## 6. Telemetry contract

建议稳定命名：

```text
continuous.generation_queue_wait_s
continuous.generation_service_s
continuous.reward_queue_wait_s
continuous.reward_service_s
continuous.ready_residence_s
continuous.trainer_wait_for_ready_s
continuous.backpressure_inflight_s
continuous.backpressure_unscored_bytes_s
continuous.backpressure_ready_bytes_s
continuous.logical_groups_completed
continuous.group_attempts
continuous.group_retries
continuous.groups_discarded
continuous.active_batches
continuous.active_policy_versions
```

最终名字需与 `RolloutStats` 现有 phase/gauge/counter 约定对齐；不同时保留一组别名。

## 7. Verification and acceptance gates

- fake-clock 单测证明 interval 无负数、retry 不重复 logical completion。
- producer/queue/consumer contract tests 证明 batch identity 和 policy version 不丢失。
- current static path 的 batches、group ids、reward tensors 和 selection 顺序与修改前相同。
- 两个并发 collect 的 timing 不互相覆盖。
- baseline 能解释 GPU 1/2 idle 时是 `inflight_full`、`no_prompt_batch`、reward backlog、ready cap、
  weight sync 或 failure 中的哪一种。
- `git diff --check`、相关 continuous/reward stats tests 全绿。

## 8. What changes / what stays

### 改变

- typed identity 和 stage timing。
- owner/producer/queue metrics 的 batch-aware 聚合。
- 长期保留一份四卡 baseline 信息文档。

### 保持

- 一个 active finite batch。
- `collect_prompt_batches()` 仍是 producer 的复合 task。
- max inflight、ready queue 和 weight-sync 行为。
- trainer batch、reward 和 optimizer 数值。

## 9. Non-goals

- 不新增 unscored queue。
- 不改变 prompt lookahead 数量。
- 不实现 reward batching 或 adaptive controller。
- 不做 eval worker、eval queue 或 inline checkpoint evaluation。
- 不以 GPU utilization 为唯一成功标准。

## 10. Definition of Done

- [ ] Identity contract 有 production consumer 和测试。
- [ ] 所有 stage wait/service 和 backpressure reason 可在一次 update 中重建。
- [ ] 四卡 baseline 已记录到 `docs/sprints/info/`。
- [ ] static path 数值与顺序无变化。
- [ ] scratch artifacts 已按生命周期清理。
- [ ] Sprint 1 的 unscored queue 可以直接复用这些 identity 和 metrics。

## 11. References

- `vrl/rollouts/orchestration/continuous/types.py`
- `vrl/rollouts/orchestration/continuous/producer.py`
- `vrl/rollouts/orchestration/continuous/queue.py`
- `vrl/rollouts/orchestration/continuous/consumer.py`
- `vrl/rollouts/orchestration/continuous/owner.py`
- `vrl/rollouts/orchestration/prompt_collection.py`
- `vrl/trainers/online/trainer.py`
- `tests/rollouts/orchestration/continuous/test_contracts.py`
- `tests/rollouts/orchestration/continuous/test_scheduler.py`
- `tests/rollouts/orchestration/continuous/test_owner.py`
