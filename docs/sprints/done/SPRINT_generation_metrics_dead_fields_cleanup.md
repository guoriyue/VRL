# SPRINT: `GenerationMetrics` 死字段清理（planned）

状态：已完成（commits `f2790b2` queue_wait_s/execution_s + `2468829` num_prompts/num_samples）。A/B 两步全部落地，四字段均已删除。
范围：清理 `vrl/generation/types.py` 的 `GenerationMetrics` 上无 behavior 读者的字段。两个纯死、两个 test-only（需签字）。
来源：dead-dataclass-hunt + 我手动 receiver 消歧验证。承接 [[SPRINT_segment_signal_dead_field_cleanup]]。

## 0. Core Decision

`GenerationMetrics`（`vrl/generation/types.py:14`）的四个字段在 `vrl/generation/` 内**零按值读取**（`grep -rEn "\.(num_prompts|num_samples|queue_wait_s|execution_s)\b" vrl/generation/` 空）。分两档：
- `queue_wait_s` / `execution_s`：从不赋值、从不读 → **纯死，机械删**。
- `num_prompts` / `num_samples`：在 planner/gather 真实赋值，但唯一读者是测试断言 → **display-only，需签字**。

> ⚠️ **同名陷阱（已排除）**：`vrl/trajectory/validation.py:84` 的 `batch.metrics.num_samples != len(...)` 报错串写的是 **`TrajectoryMetrics.num_samples`**，`trajectory/ops.py:111` 的 `data.metrics.num_samples` 同属 `TrajectoryMetrics` —— **另一个 struct**，是活的。本 sprint 只动 `GenerationMetrics`，不碰 `TrajectoryMetrics`。

## 1. 现状实锤

### 1.1 `queue_wait_s` / `execution_s` —— 纯死
`vrl/generation/types.py:24,25`。`grep -rEn "\.(queue_wait_s|execution_s)\b" vrl/` 的命中全是别的东西：`continuous.queue_wait_s` / `reward.queue_wait_s` 是 `stats.as_phase_dict()` 的字符串 phase **key**（`consumer.py:173`、`stats.py:106`），`self.reward_queue_wait_ms` 是别的字段——没有一处读 `GenerationMetrics.queue_wait_s`。这两字段从不被赋值也从不被读。

### 1.2 `num_prompts` / `num_samples` —— test-only（display-only）
`vrl/generation/types.py:21,22`。真实赋值于 `execution/planner.py:414-416`、`diffusion/gather.py:91-93`；但唯一读者是 `tests/generation/execution/test_chunk_gatherer.py:65` 断言。无生产控制流/序列化读取。

> 注意：`vrl/trajectory/validation.py:22-29` 的 `FORBIDDEN_TRAJECTORY_METRICS` denylist 含这些名，是防止 generation 指标名泄漏进 trajectory 的防护，与本字段删除是两回事——删字段后该 denylist 按需保留。

## 2. 落地方案

### A. 删 `queue_wait_s` / `execution_s`（机械）
- 删 `types.py:24,25` 两字段。grep 确认无构造传参（从不赋值）。

### B. `num_prompts` / `num_samples`（需签字）
- **默认删**：删 `types.py:21,22` + `planner.py:414-416`、`gather.py:91-93` 赋值 + `test_chunk_gatherer.py:65` 断言。
- **若保留**：确认有遥测/外部消费后，标注 provenance-only。
- 删前确认 `GenerationMetrics` 实例（`gather.py:91` 构造）的下游不经 `asdict`/序列化读这两字段。

## 3. 验证
- `grep -rEn "\.(queue_wait_s|execution_s)\b" vrl/generation/` 零命中。
- 按 A/B 决策后 `grep -rEn "\.(num_prompts|num_samples)\b" vrl/generation/` 仅剩 `TrajectoryMetrics`（如有）。
- `pytest tests/generation/ -q` 全绿。

## 4. 非目标 / Non-Goals
- 不碰 `TrajectoryMetrics.num_samples`（活，validation.py:84 raise 守卫）。
- 不动 `FORBIDDEN_TRAJECTORY_METRICS` denylist 语义。

## References
- `vrl/generation/types.py:14,21,22,24,25`
- `vrl/generation/execution/planner.py:414-416`、`vrl/generation/diffusion/gather.py:91-93`
- `vrl/trajectory/validation.py:22-29,84`（同名 TrajectoryMetrics，勿删）
- 关联：[[SPRINT_segment_signal_dead_field_cleanup]]
