# SPRINT: `vrl/generation/execution/` envelope/stage 死字段清理（planned）

状态：未开始（2026-06-20）。
范围：`ExecutionStage.batch_group_key`（纯死）+ `ChunkExecutionEnvelope.plan_summary` / `capability_summary`（protocol/telemetry，需签字）。
来源：dead-dataclass-hunt + 手动验证。承接 [[SPRINT_segment_signal_dead_field_cleanup]]。

## 0. Core Decision

`execution/` 子系统三个字段无控制流消费，但风险分两档：
- `ExecutionStage.batch_group_key`：`_batch_group_key()` 算出、三处构造写入、**零读取** → 机械删。
- `ChunkExecutionEnvelope.plan_summary` / `capability_summary`：worker 读它们，但 `plan_summary` 只是 `forward_chunk_plan` 协议形参（各实现 `del` 掉）、`capability_summary` 只进 worker telemetry dict → 需签字。

## 1. 现状实锤

### 1.1 `ExecutionStage.batch_group_key` —— 纯死
`vrl/generation/execution/planner.py:64` 定义，`:291,308,361` 三处构造写入（`batch_group_key=self._batch_group_key(axis)` 等），`_batch_group_key()` 方法在 `:376`。`grep -rEn "batch_group_key\b" vrl/` 只命中定义 + 构造 + 那个 producer 方法，**无任何 `.batch_group_key` 读取**。算了存进 stage，没人消费。

### 1.2 `ChunkExecutionEnvelope.plan_summary` / `capability_summary` —— 需签字
`vrl/generation/execution/types.py:45,46`。worker 确有读取：
- `worker.py:276` 把 `envelope.plan_summary` 传给 `forward_chunk_plan(...)` —— 但所有实现（`diffusion/executor.py:441`、`janus_pro/runtime.py:442,807`、`nextstep_1/runtime.py:383`）签名里立即 `del plan_summary`，纯为协议形参一致性，无实际消费。
- `worker.py:299` 把 `dict(envelope.capability_summary)` 塞进 worker 的 metrics/telemetry dict —— 仅遥测。

二者都不进控制流，但 `plan_summary` 删除涉及**改 4 个 `forward_chunk_plan` 实现的签名**（协议边界），不是删字段就完事。

## 2. 落地方案

### A. 删 `ExecutionStage.batch_group_key`（机械）
- 删 `planner.py:64` 字段、`:291,308,361` 构造写入、`:376` 的 `_batch_group_key()` 方法（若无其他调用者）。

### B. `plan_summary`（需签字 + 协议改动）
- 删 `types.py:45` 字段后，同步删 `forward_chunk_plan` 协议形参 + 4 个实现签名里的 `plan_summary` 形参与 `del`。属协议边界收窄，需确认无 future 用途。

### C. `capability_summary`（需签字）
- 仅 telemetry。**默认删** `types.py:46` + `worker.py:299` 的写入；若遥测面板依赖该 key，改标 provenance-only。

## 3. 验证
- `grep -rEn "\.batch_group_key\b" vrl/` 零命中。
- 按 B/C 决策后 `grep -rEn "\.(plan_summary|capability_summary)\b" vrl/` 相应收敛。
- `pytest tests/generation/execution/ -q` 全绿；diffusion/AR chunk 执行冒烟一致。

## 4. 非目标 / Non-Goals
- A 与 B/C 分批：A 可直接落地，B/C 需 owner 确认协议/遥测无依赖。
- 不动 `ExecutionStage` 的活字段（`name`/`stage_id`/`segment`/`axis`/`cache_read`/`cache_write` 等，均在 planner 控制流/summary 中使用）。

## References
- `vrl/generation/execution/planner.py:64,291,308,361,376`
- `vrl/generation/execution/types.py:45,46`、`worker.py:276,299`
- `vrl/generation/diffusion/executor.py:441`、`vrl/models/ar/janus_pro/runtime.py:442,807`、`vrl/models/ar/nextstep_1/runtime.py:383`
- 关联：[[SPRINT_segment_signal_dead_field_cleanup]]
