# SPRINT：Continuous generation / reward pump split

状态：**parked（2026-07-21）**。等待
[Stage contracts and baseline](../planned/SPRINT_continuous_stage_contracts_and_baseline.md)
完成并给出可复用 identity/timing contract。

父 program：[Continuous three-stage pipeline](../planned/SPRINT_continuous_three_stage_pipeline_program.md)

## 0. 结论先行

把 continuous producer 的一个复合 collect task 拆为两个 owner-loop task：generation pump 和
reward pump。generation 完成后，typed unscored item 进入 bounded queue，rollout slot 立即释放；
reward pump 独立取出 item、打分、build batch，成功后才进入现有 trainer-ready queue。

```text
generation pump -> GeneratedRolloutQueue -> reward pump -> ContinuousRolloutQueue
                                                       scored-only -> trainer
```

使用已有 `RolloutCollector.collect_unscored()` 和 `score_rollouts()` seam，不创建第二套 reward
API。eval 不加入任一 pump。

## 1. Root cause / current behavior

`ContinuousRolloutProducer._collect_group()` 当前 await `collect_prompt_batches()`。这个 helper
内部已经把 collector 拆成 `collect_unscored()` 和 `score_rollouts()`，但 producer 看见的仍是
一个直到 reward 完成才结束的 task。

当四个 current-batch group 都完成 generation 并排队等待 GPU 3 时：

```text
producer inflight slots remain occupied by reward
pending_slots is empty
next prompt batch is not installable
GPU 1/2 have no useful generation work
```

`max_inflight_groups` 只限制复合 task 数；调大它不能解决有限 prompt batch 没有更多 slot 的问题。

## 2. Goal and ownership boundary

新增一个 typed item，至少行为消费：

```text
batch_id
group_slot
policy_version
attempt
unscored_rollout
estimated_bytes
generated_at
artifact ownership token
```

新增 `GeneratedRolloutQueue`：

- 只持有 generation 已完成、reward 未完成的 item；
- 同时受 `max_unscored_groups` 与 `max_unscored_bytes` 约束；
- 不做 policy selection，不构造 advantage，不暴露给 trainer；
- queue put/take/cancel 明确转移 artifact ownership。

现有 `ContinuousRolloutQueue` 不泛化成“任意 stage queue”。它继续只保存 trainer-ready
`RolloutBatch`，这一薄边界是正确性防火墙。

## 3. Correctness and resource invariants

1. generation receipt 先做 version/staleness validation，再进入 unscored queue；reward receipt 在
   发布 ready item 前再次验证，覆盖 reward 运行期间 trainer version 前进的情况。
2. reward 结果必须与输入 item 一一对应、保序、数量完全相等；否则整次 reward batch 失败。
3. batch build 完成前不得向 ready queue 发布 item。
4. generation retry 复用 batch/group/sample identity；成功 item 不再生成第二次。
5. reward retry 不重新 generation；同一 artifact/request fingerprint 不得重复发布 ready item。
6. unscored queue 满时停止 generation admission；不能靠丢弃队头保持利用率。
7. ready queue 满时 reward pump 停止取新 item，并保留当前结果 ownership；不能反向拖垮
   service 的 active request cap。
8. reward/GPU 仍必须和 trainer、rollout accelerator 隔离。

## 4. Implementation stages

### T0 — Typed generated item and queue

- 建立 `GeneratedRolloutItem` 和 bounded queue。
- byte estimator 覆盖 trajectory、video/artifact provenance 与明显 tensor fields；不能只按 item
  个数假设所有视频大小相同。
- queue API 只暴露 snapshot/put/take/close 和必要 stats，不嵌入 reward policy。

### T1 — Generation pump

- producer group task 直接调用 `collector.collect_unscored()`。
- 保留 PromptExample 的 `generation_input()`、`reward_metadata()` 和 request overrides mapping；
  不在 producer 重写业务字段映射。
- generation 成功即释放 generation inflight count；把 item 移交 unscored queue。
- generation failure 沿用现有 per-slot retry/fail-fast 语义。

### T2 — Reward pump

- 独立 coroutine 从 unscored queue 取 item。
- 第一版允许一个 reward batch in flight；batch aggregation policy 留给后续 sprint，只需接口能
  接收 `list[UnscoredRollout]`。
- 调用 `collector.score_rollouts()`，将返回 batch 按 identity 映射到 ready item。
- reward failure 记录独立 attempt/reason，并按 bounded retry policy 重试；不能把 reward error
  计成新的 generation failure。

### T3 — Admission and shutdown

- scheduler 同时考虑 generation inflight、unscored items/bytes 和 ready bytes。
- shutdown 顺序固定：close generation admission -> settle/cancel generation -> settle/cancel reward
  -> close queues -> runtime shutdown。
- terminal error 由 owner 统一 quarantine，consumer 看到原始 stage/root cause。

### T4 — Compatibility

- strict schedule 继续使用 `collect_prompt_batches()`。
- continuous 可用 feature flag 做 static-vs-split A/B；flag 只在迁移期存在，GA 后删除或明确保留
  为 rollback，不留无人测试的双实现。

## 5. Failure, cancellation and recovery semantics

| 失败点 | 处理 |
|---|---|
| generation error | 同 slot、同 identity 重试；超预算 terminal fail |
| unscored queue admission fail | 不发布 partial item；停止 upstream admission |
| reward 429/transport retryable | 同 reward request identity 有界重试 |
| reward parse/model error | 不生成默认分数；保留 root cause，按配置 fail-fast |
| ready queue publication error | 不重新 score；owner 持有 scored result 直到 publish 或 terminal cleanup |
| cancellation | 取消拥有该 item 的唯一 task，等待 artifact cleanup 后转 terminal state |

大型 artifact 不在本 sprint 持久化为 checkpoint state；process restart 语义由 recovery sprint 完成。

## 6. Telemetry

除 program 统一指标外增加：

```text
continuous.generation_inflight
continuous.reward_inflight_batches
continuous.unscored_items
continuous.unscored_bytes
continuous.unscored_oldest_age_s
continuous.reward_batch_groups
continuous.reward_retries
continuous.generation_retries
continuous.stage_handoff_s
```

每个 metric 必须能区分逻辑 group 与 attempt。

## 7. Verification and acceptance gates

- gate-controlled fake collector 证明 reward N 在等待时 generation N+1 可以完成。
- trainer demand 在 unscored queue 非空、ready queue 为空时必须继续等待，不能读取 partial data。
- reward 返回 wrong count/order/identity 时 fail closed。
- queue item/byte caps 分别触发 generation backpressure，cap 解除后恢复。
- generation error 不取消已独立运行且 ownership 清楚的 reward；terminal error 会完整清理两者。
- same inputs 的 batch/group ids、reward tensors、trajectory 和 stats 与复合 path 一致。
- `tests/rollouts/orchestration/continuous/`、collector/reward service cancellation tests 全绿。

## 8. What changes / what stays

### 改变

- continuous producer 不再把 reward latency 算作 generation inflight lifetime。
- owner 新增 unscored queue 和 reward pump。
- scheduler 有三个 stage budget 的行为输入。

### 保持

- collector 的 generation/reward public seam。
- trainer-ready queue 与 consumer selection。
- 一个 finite active prompt batch；跨 batch lookahead 留给下一 sprint。
- trainer、rollout、reward GPU role 固定隔离。

## 9. Non-goals

- 不同时支持多个 prompt batch。
- 不实现 UnifiedReward model micro-batch。
- 不做 adaptive controller。
- 不引入 generic physical-stage framework。
- 不接入 eval。

## 10. Definition of Done

- [ ] generation 和 reward 有独立 inflight 生命周期。
- [ ] unscored queue 有 item/byte cap 和完整 ownership tests。
- [ ] trainer-ready queue 仍严格 scored-only。
- [ ] generation/reward overlap 在跨 task timeline 上可见。
- [ ] static numerical control parity 通过。
- [ ] 下一 sprint 可以在不改 stage contract 的前提下加入第二个 prompt batch。

## 11. References

- `vrl/rollouts/collector/core.py`
- `vrl/rollouts/orchestration/prompt_collection.py`
- `vrl/rollouts/orchestration/continuous/producer.py`
- `vrl/rollouts/orchestration/continuous/owner.py`
- `vrl/rollouts/orchestration/continuous/queue.py`
- `vrl/rollouts/orchestration/continuous/scheduler.py`
- `tests/rollouts/orchestration/test_prompt_collection.py`
- `tests/rollouts/orchestration/continuous/test_contracts.py`
- [Reward service](../done/SPRINT_reward_service.md)
