# SPRINT：Continuous versioned two-batch lookahead

状态：**parked（2026-07-21）**。等待
[Generation / reward pump split](SPRINT_continuous_generation_reward_pump.md) 完成。

父 program：[Continuous three-stage pipeline](../planned/SPRINT_continuous_three_stage_pipeline_program.md)

## 0. 结论先行

允许 continuous owner 同时持有 current batch 与一个 next batch，使 GPU 1/2 在 current batch 的
reward tail 尚未结束时就能生成 next batch。窗口不是任意长度：当前 GRPO recipe 只允许
`max_stale_policy_versions=1`，因此安全默认就是两批，并且两批在 admission 时各自冻结 policy
version。

recipe loop 已经在同一次 `next_iteration(prompts, next_prompts=...)` 调用中提供确定性 preview；
无需让 owner 偷读 sampler 或重新派生下一个 epoch。

## 1. Root cause / current behavior

当前 `_ActivePromptBatch` 只有一个实例，`set_prompt_batch()` 明确拒绝：

```text
pending slots remain
generation/reward work remains in flight
ready items remain unconsumed
```

`owner.next_iteration()` 也只在 consumer 取走当前完整 batch 后才安装 `next_prompts`。这保证了
单批语义，却让 current reward tail 期间没有下一批 generation work。

`ContinuousRolloutConsumer._select_iteration()` currently assumes that the ready
queue contains only one policy version globally, each `group_key` appears once,
and the item count exactly equals `min_groups`. Supporting two batches requires
validation by batch identity; it cannot simply delete these checks.

## 2. Goal and ownership boundary

将 `_ActivePromptBatch` 演进为有界的 batch deque：

```text
current batch: trainer 正在等待或即将消费
lookahead batch: 已知 prompts，可 generation/reward，但尚不能越过 current 被消费
```

每个 batch state 自己拥有：

```text
batch_id
policy_version
prompts
group_size
pending / generating / unscored / rewarding / ready / failed slots
```

schedule 只接收 recipe loop 提供的 preview。sampler、epoch RNG 和 checkpointed draw 仍由
`vrl/scripts/common/online.py` 拥有；owner 不导入 data sampler。

## 3. Correctness and resource invariants

1. current 和 lookahead 的 `group_slot` 可以相同，但 `(batch_id, group_slot)` 必须唯一。
2. consumer 只能选择队头 batch，不能因为 lookahead 先 ready 就跳过 current，避免训练数据顺序
   和 checkpoint sampler 语义变化。
3. 一个 trainer iteration 必须是同 batch、同 policy version、完整 distinct slots。
4. lookahead 在接收时冻结当时 active policy version；weight sync 后不得改写已有 item version。
5. lookahead 数量不能超过 staleness window 可证明的范围。默认两批；配置要求更大时 fail closed，
   不能把 `max_stale_policy_versions` 当作“尽量遵守”。
6. current 未消费时，lookahead 只能占用明确的 unscored/ready byte budget。
7. preview prompts 在下一次 trainer call 呈现时必须 identity/equality 匹配；不匹配 terminal fail。
8. prompt order、sample seeds 和 group normalization 与 serial recipe 相同。

## 4. Implementation stages

### T0 — Batch-aware state

- 用 `PromptBatchState` deque 替代单 `_ActivePromptBatch`。
- 所有 slot state transition 通过一处 typed method，禁止多个 loose set 互相漂移。
- validation 从 batch state 派生 allowed slots；不手写重复 `_VALID_*` 表。

### T1 — Early lookahead installation

- `next_iteration()` 在验证 `prompts` 和 `next_prompts` 后，一开始就安装 current + lookahead，
  而不是 consumer 返回后才安装 lookahead。
- streaming gradient accumulation 的 microbatch preview 仍保持现有顺序。
- 没有 `next_prompts` 时窗口自然退化为单批，不制造 synthetic work。

### T2 — Batch-aware queues and selection

- unscored/ready item key 改为 `(batch_id, group_slot)`。
- scheduler 按队头 batch 验证 complete/homogeneous version；允许 queue 同时包含不同 batch/version。
- current batch 未完整时返回 wait，不把 lookahead item 算作 current readiness。
- current 消费后原子推进 batch deque；下一次呈现的 prompts 必须和已安装 batch 匹配。

### T3 — Weight sync and staleness

- pause admission 时冻结所有 batch state transitions 的提交面；允许已提交 task 按 runtime capability
  安全完成。
- sync 成功后验证所有 unscored/ready item version 仍在 window 内。
- 超窗 batch 不做局部 slot drop；整批 fail closed，因为局部再生成会混合 policy version。

### T4 — Bounded fairness

- current batch 的 missing slots 优先于 lookahead。
- lookahead 不能独占 generation inflight 或 reward batch，导致 current trainer starvation。
- fairness policy 是 scheduler control flow，有确定性测试，不依赖 task completion timing 偶然性。

## 5. Failure, cancellation and recovery semantics

- current batch 失败：停止 lookahead admission，清理其 in-flight ownership，向 trainer 传播 current
  root cause。
- lookahead 失败：不污染 current；current 可完成，但 owner 在推进到 failed lookahead 时 fail
  closed，不重新 preview 一个不同 batch 偷换数据。
- trainer 传入与预装 lookahead 不匹配的 prompts：terminal protocol error。
- weight sync partial failure：所有新 admission 保持关闭；不把 lookahead version 标成新版本。
- process restart 不恢复 deque 中的媒体；由 checkpointed sampler 重新 preview，见 recovery sprint。

## 6. Telemetry

```text
continuous.active_batches
continuous.current_batch_ready_groups
continuous.lookahead_generation_groups
continuous.lookahead_unscored_groups
continuous.lookahead_ready_groups
continuous.batch_head_age_s
continuous.lookahead_age_s
continuous.blocked_by_current_batch
continuous.blocked_by_staleness_window
```

## 7. Verification and acceptance gates

- deterministic fake tasks 证明 current reward 被 gate 住时，lookahead generation 已开始/完成。
- lookahead 先 ready 时 consumer 仍等待 current。
- 两批都有 `group_slot=0..N-1` 时无 duplicate false positive，也不会跨批选择。
- current/next prompt mismatch、group size mismatch、future version、too-stale version 全部 fail closed。
- weight sync 前后 item version 不被改写。
- streaming microbatch 和 full-batch recipe 的 sampler 顺序、group ids、seed contract 不变。
- max lookahead 超过 staleness-derived bound 时 config resolve 失败。

## 8. What changes / what stays

### 改变

- active prompt state 从一个 batch 变为有界 deque。
- owner 提前安装 recipe 已提供的 next prompts。
- scheduler/queue validation 按 batch identity 工作。

### 保持

- recipe loop 是 preview/source-of-truth owner。
- 最多 one-version-old 的 GRPO algorithm boundary。
- trainer 一次只消费一个完整 batch。
- strict schedule 无 lookahead queue。

## 9. Non-goals

- 不预取任意多个未来 epoch。
- 不让 controller 自动扩大 staleness。
- 不允许 trainer 跳过慢 current batch 训练 lookahead。
- 不改变 dataset sampling 或 checkpoint RNG。
- 不接入 eval。

## 10. Definition of Done

- [ ] current reward 期间存在可证明的 lookahead generation。
- [ ] 双 batch identity、selection、fairness 和 staleness tests 全绿。
- [ ] prompt sampler 顺序与 restart determinism 不变。
- [ ] host queue bytes 有界。
- [ ] 单 batch/no-next-prompts 路径保持兼容。
- [ ] 下一 sprint 可以从多个 unscored group 形成 reward batch。

## 11. References

- `vrl/scripts/common/online.py`
- `vrl/trainers/online/trainer.py`
- `vrl/rollouts/orchestration/schedule.py`
- `vrl/rollouts/orchestration/continuous/owner.py`
- `vrl/rollouts/orchestration/continuous/producer.py`
- `vrl/rollouts/orchestration/continuous/scheduler.py`
- `vrl/rollouts/orchestration/continuous/consumer.py`
- `tests/rollouts/orchestration/continuous/test_owner.py`
- `tests/rollouts/orchestration/continuous/test_scheduler.py`
