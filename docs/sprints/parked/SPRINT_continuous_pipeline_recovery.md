# SPRINT：Continuous pipeline recovery and artifact cleanup

状态：**parked（2026-07-21）**。等待
[Adaptive backpressure](SPRINT_continuous_adaptive_backpressure.md) 通过 deterministic simulation
和四卡 smoke。

父 program：[Continuous three-stage pipeline](../planned/SPRINT_continuous_three_stage_pipeline_program.md)

## 0. 结论先行

把三段流水线的失败、取消和 restart 语义做成可证明的状态机。大型 unscored/scored media queue
是 ephemeral one-shot state，不写进 checkpoint；process restart 回到最后一个 trainer checkpoint，
由 checkpointed sampler/RNG 重新给出 current + next prompts。恢复的目标是“不丢持久训练状态、
不训练 partial work、不泄漏 artifact”，不是保存每个尚未打分的视频。

eval 不参与 recovery，也不能成为 restart 前置条件。

## 1. Root cause / current behavior

单 finite producer 只有 generation/reward 复合 task，terminal cleanup 相对集中。三段流水线增加：

```text
generation tasks
unscored queue
reward task/request cache
scored ready queue
trainer consumption
lookahead batch state
artifact files
adaptive controller state
```

如果没有统一 ownership，常见错误是：

- reward task 被取消但 service 仍读取 artifact，owner 提前删除文件；
- HTTP retry 用新 request id 重复打分并发布两次 ready item；
- current failure 后 lookahead task 继续写 queue；
- restart 读取旧 metrics counter，把历史 error 当成新 attempt error；
- 为了“恢复队列”把巨大 trajectory/video pickle 进 checkpoint；
- partial weight sync 后 admission 被错误重开。

## 2. Goal and ownership boundary

为每个 logical group 定义单一状态机：

```text
pending_generation
  -> generating
  -> unscored
  -> rewarding
  -> scored
  -> ready
  -> consumed

any non-terminal state
  -> cancelling
  -> cancelled | failed
```

每个 transition 同时指定：

```text
task owner
artifact lease owner
retry counter owner
queue membership
allowed next states
terminal cleanup action
```

状态集合从 typed enum/source 派生 validation，不维护平行 `_ALLOWED_STATES` 常量。

## 3. Correctness and resource invariants

1. 一个 item 同时最多在一个 queue、一个 task owner 下。
2. artifact 只能在最后一个 reader settled 后删除。
3. logical completion/ready publication 只发生一次；retry 只增加 attempt。
4. reward request id + fingerprint 在 retry 中稳定；payload 改变必须新 logical request。
5. current batch terminal failure 关闭所有新 admission，并取消/清理 lookahead。
6. partial weight sync 保持 admission closed，直到整个 owner process 退出/restart。
7. checkpoint 只持久化 trainer/optimizer/sampler/RNG/epoch 等 canonical state，不持久化 queue media。
8. restart 后 batch_id 可使用新 attempt namespace，但 prompt/sample determinism 必须来自 checkpoint。
9. stale/partial ready batch 不可跨 restart 继续训练。
10. supervisor health 读取累计 counters 时使用 attempt 内 delta；稳定的非零累计值不是连续新错误。

## 4. Implementation stages

### T0 — State machine and leases

- 用 typed transition method 统一更新 state、queue、task 和 artifact lease。
- invalid transition fail fast，并输出 batch/group/attempt/root cause。
- lease 支持 generation materialization、reward service validation/inference、batch build 和 debug artifact
  的实际 reader 生命周期。

### T1 — Stage-specific retry

- generation retry 不重复已成功 reward。
- reward retry 不重新 generation，使用稳定 request identity/fingerprint。
- ready publication retry 不重新 reward。
- retryable/non-retryable 分类来自 typed service/runtime errors，不做字符串匹配。
- 每 stage 有独立连续 failure budget；累计 metrics 不直接等同 health streak。

### T2 — Ordered terminal drain

统一 shutdown/failure 顺序：

```text
close admission
cancel/settle generation
cancel/settle reward client + service request
release artifact leases
clear unscored/ready queues
close generation/reward runtimes
flush terminal metrics
return root cause to supervisor
```

cleanup error 与 root cause 都保留，不能让 cleanup exception 覆盖原始训练失败。

### T3 — Checkpoint restart

- supervisor 从最后一个完整 checkpoint 重启。
- metrics 行去重与 attempt lifecycle 分离；新 attempt counters 从零开始。
- sampler preview 重新构造 current/next prompt batches。
- 旧 queue state 一律不加载；旧 attempt artifact 按 manifest/namespace 清理。
- reward service cache 命中只在 artifact/fingerprint 仍有效时使用，否则明确重新提交。

### T4 — Fault injection

覆盖：

```text
generation exception / timeout / cancellation-resistant task
unscored queue cap and publication failure
reward 429 / timeout / parse error / cancellation
reward task success but ready publication failure
weight sync partial failure
trainer exception after ready consume
SIGTERM during each state
supervisor restart with historical nonzero counters
```

## 5. Failure and recovery semantics

| 状态 | process 内 retry | process restart |
|---|---|---|
| pending/generating | 同 identity 有界 retry | 从 checkpointed prompt 重新 generation |
| unscored/rewarding | 同 artifact/request 有界 retry | 删除旧 lease/artifact，重新 generation |
| scored/ready | 不重新 reward；只完成 publish/consume | 丢弃 queue，重新 generation/reward |
| consumed before optimizer checkpoint | 由 trainer step atomicity 决定，不单独提交 queue state | 回到上一个完整 checkpoint 重做 step |
| checkpoint complete | 无 | 从下一 sampler draw 继续 |

## 6. Telemetry

```text
continuous.items_by_state
continuous.invalid_transitions
continuous.generation_attempts
continuous.reward_attempts
continuous.publication_attempts
continuous.cancellations_by_stage
continuous.cleanup_errors
continuous.artifact_live_count
continuous.artifact_live_bytes
continuous.artifact_residual_count
continuous.restart_attempt
continuous.recovered_checkpoint_step
```

## 7. Verification and acceptance gates

- model-based/state-machine tests 随机生成合法/非法 transition，证明 ownership 唯一。
- 每个 fault injection 后 task count、queue items、artifact leases、Ray actors 和 service requests
  回到零或明确定义的 terminal state。
- reward success + publish failure 不产生第二次 reward 调用。
- cancellation 等待 service artifact read settled 后才删除文件。
- restart 不加载旧 queue，sampler prompts/seeds 与从同 checkpoint 的 clean run 一致。
- historical `producer_errors=1` 在新行仍为 1 时不累计 health streak；attempt 内新增 delta 才算新错。
- partial weight sync 后 admission 永不恢复。
- root cause 与 cleanup errors 同时可诊断。

## 8. What changes / what stays

### 改变

- item lifecycle、retry、artifact lease 和 terminal drain 统一成状态机。
- supervisor/metrics 明确 attempt 边界。
- restart 显式丢弃 ephemeral queues。

### 保持

- checkpoint source of truth 和 sampler ownership。
- reward service fingerprint/idempotency contract。
- owner terminal quarantine 和 process supervisor handoff。
- trainer step/checkpoint atomicity。

## 9. Non-goals

- 不把 trajectory/video queue 持久化进 checkpoint。
- 不实现跨主机 distributed durable queue。
- 不在 partial weight sync 后尝试进程内修复未知 fleet。
- 不用 eval 判断是否允许 restart。
- 不保证失败中间生成的视频被复用；正确清理优先。

## 10. Definition of Done

- [ ] 全部 item transition 由单一 typed state machine 执行。
- [ ] stage-specific retry 不重复已成功 work。
- [ ] terminal drain/fault injection 无 task、actor、request、artifact 泄漏。
- [ ] checkpoint restart 与 clean resume prompts/seeds 一致。
- [ ] cumulative health counters 使用 attempt-local delta。
- [ ] queue media 未进入 checkpoint。

## 11. References

- `vrl/rollouts/orchestration/continuous/producer.py`
- `vrl/rollouts/orchestration/continuous/owner.py`
- `vrl/rollouts/orchestration/continuous/types.py`
- `vrl/rollouts/orchestration/prompt_collection.py`
- `vrl/rewards/service/client.py`
- `vrl/rewards/service/server.py`
- `vrl/trainers/checkpointing.py`
- `vrl/scripts/supervise.py`
- `tests/rollouts/orchestration/continuous/test_contracts.py`
- `tests/rewards/service/test_service.py`
- `tests/scripts/test_supervise.py`
