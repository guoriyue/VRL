# SPRINT: generation pipeline payload/handle 死字段清理（done）

> **Historical correction (2026-07-13).** This document records an incremental
> cleanup of a seam that no longer exists. The remaining topology, payload,
> serial runner, Ray stage adapters, and contract-only tests were later deleted
> after a full audit found zero production consumers. Do not treat the retained
> field list below as a current API.

状态：done（落地 commit `91a086e`；2026-06-21 归档）。
范围：删除 `RayPipelineStageHandle.worker_id` 和 `PipelineStagePayload.sample_identity`。
来源：dead-dataclass-hunt + 手动验证，承接 [[SPRINT_segment_signal_dead_field_cleanup]]。

## 0. Core Decision

generation pipeline 层两个字段都没有非日志/非测试消费者：
- `RayPipelineStageHandle.worker_id` 从未被设置，也从未被读取。`RayPipelineRunner` 路由只依赖 `handle.stage` 和 `handle.actor`。
- `PipelineStagePayload.sample_identity` 只在 `for_stage()` 中穿透拷贝，并由一个 round-trip 测试断言；没有 Ray 序列化、control-flow、日志 provenance 消费。

按派生结构体规则，无行为消费者且未标注 provenance-only 的字段应删除，不保留裸 display 字段。

## 1. 已删除内容

- `vrl/generation/ray/pipeline_runner.py`：删除 `RayPipelineStageHandle.worker_id`。
- `vrl/generation/pipeline/payload.py`：删除 `PipelineStagePayload.sample_identity`。
- `PipelineStagePayload.for_stage()`：删除 `sample_identity` 拷贝。
- `tests/generation/pipeline/test_pipeline_contracts.py`：删除只验证该字段穿透的断言/构造参数。

## 2. 验证

- 落地提交记录：pipeline 相关测试 `45 passed`。
- Review 复核：`rg "\.sample_identity\b|sample_identity" vrl/generation tests/generation -g '*.py'` 无命中；`rg "RayPipelineStageHandle\(|worker_id\b" vrl/generation tests/generation -g '*.py'` 的 `worker_id` 剩余命中属于其他活 struct。
- Review 复核：`pytest tests/models/diffusion tests/generation/pipeline tests/generation/ar tests/rewards/inference tests/rollouts/replay tests/trainers/online/test_step_split.py tests/algorithms -q` → `289 passed`。
- Review 复核：`python -m vrl.config.lint` 通过；`git diff --check origin/main...HEAD` 通过。

## 3. Non-Goals

- 不碰其他 struct 的同名 `worker_id`，例如 `ChunkExecutionResult.worker_id`、`RayActorGroup` / `RayActorJob` 的 placement 字段。
- 不动 `PipelineStagePayload` 的活字段：`request_id`、`stage`、`data`、`policy_version`、`metadata`。
- 不把 `sample_identity` 改塞进 `metadata`；没有行为消费者时换位置仍是死字段。

## References

- `91a086e refactor(pipeline): drop dead stage handle/payload fields`
- `vrl/generation/ray/pipeline_runner.py`
- `vrl/generation/pipeline/payload.py`
- `tests/generation/pipeline/test_pipeline_contracts.py`
- 关联：[[SPRINT_segment_signal_dead_field_cleanup]]
