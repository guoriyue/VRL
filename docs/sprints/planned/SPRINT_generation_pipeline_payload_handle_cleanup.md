# SPRINT: generation pipeline payload/handle 死字段清理（planned）

状态：未开始（2026-06-20）。
范围：`RayPipelineStageHandle.worker_id`（纯死）+ `PipelineStagePayload.sample_identity`（display-only，需签字）。
来源：dead-dataclass-hunt + 手动验证。承接 [[SPRINT_segment_signal_dead_field_cleanup]]。

## 0. Core Decision

generation pipeline 层两个 struct 各一个无控制流消费的字段：`worker_id` 全仓零读者；`sample_identity` 只在 `for_stage()` 拷贝穿透 + 测试断言。

## 1. 现状实锤

### 1.1 `RayPipelineStageHandle.worker_id` —— 纯死
`vrl/generation/ray/pipeline_runner.py:23` 定义 `worker_id: str | None = None`。`grep -rEn "worker_id\b" vrl/generation/pipeline/ vrl/generation/ray/pipeline_runner.py` **仅命中定义行**。全仓其他 `worker_id` 命中均在别的 struct（`RayActorGroup` / `ChunkExecutionResult.worker_id` 等，不同 receiver）。该字段构造时全省略、从不读。

### 1.2 `PipelineStagePayload.sample_identity` —— display-only
`vrl/generation/pipeline/payload.py:17`。唯一「读」是 `for_stage()`（`payload.py:37`）把它拷进下一 stage payload（穿透，非消费）+ 测试断言（`test_pipeline_contracts.py:171,232-242`）。无 Ray 序列化/control-flow 消费。

## 2. 落地方案

### A. 删 `worker_id`（机械）
- 删 `pipeline_runner.py:23` 字段。grep 确认无构造传参。

### B. `sample_identity`（需签字）
- **默认删** `payload.py:17` 字段 + `:37` 的 `for_stage` 拷贝 + 测试断言。
- **若保留**：确认其作 Ray 跨 stage provenance 用途后，标 provenance-only。
- 删前确认 payload 不经 Ray 序列化把 `sample_identity` 传给下游 actor 读取。

## 3. 验证
- `grep -rEn "\.worker_id\b" vrl/generation/` 仅剩别的 struct（如有）。
- 按 B 决策后 `grep -rEn "sample_identity\b" vrl/` 收敛。
- `pytest tests/generation/ -q` 全绿（含 pipeline contract 测试）。

## 4. 非目标 / Non-Goals
- 不碰其他 struct 的同名 `worker_id`（`ChunkExecutionResult.worker_id` 等，活）。
- 不动 `PipelineStagePayload` 的活字段。

## References
- `vrl/generation/ray/pipeline_runner.py:23`、`vrl/generation/pipeline/payload.py:17,37`
- `tests/generation/.../test_pipeline_contracts.py:171,232-242`
- 关联：[[SPRINT_segment_signal_dead_field_cleanup]]
