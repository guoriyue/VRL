# SPRINT：EnginePlan authoritative execution envelope

状态：主路径已完成。single-request local、local batched、Ray chunk 都必须消费 plan-aware envelope；Ray chunk 已删除 legacy `forward_chunk(...)` fallback，capability 也不再有 `generic_family_capability(...)` 兜底。

## 目标

让现有 `EnginePlan` 成为 authoritative execution envelope。

当前代码里已经有 `build_engine_plan(...)`、`EnginePlan`、Ray planner 和 Ray worker 相关路径。问题不是缺少 `EnginePlan` 类型，而是它现在主要还是描述层：能描述 request、workload、micro-batches、execution units、profiler labels，并能挂到 `OutputBatch` 和 Ray chunk 上。single-request local、local batched、Ray chunk 三条路径已经有 plan-aware envelope；剩余工作是把 family runtime 内部 profiler label 和 capability 闭环继续收口。

历史上有两条裸执行入口：

```python
single_request_legacy_call(...)
ray_chunk_legacy_call(...)
```

已修复的新增发现：`GenerationWorker._execute_group(...)` 曾经会先为每个 request 建 plan，但随后直接调用：

```python
outputs = forward_batch(requests, sample_specs_by_request)
```

默认 batch helper 又会合并 prompts 并调用裸 single-request entry：

```python
merged_output = legacy_single_request_call(merged_request, merged_specs)
```

因此只要走 local batching，`forward_plan(...)` 曾经会被绕过。现在 `GenerationWorker` 优先调用 `forward_batch_plan(...)`，默认 merge helper 接收 per-request plans，并在 executor 有 `forward_plan(...)` 时调用 plan-aware merged forward。

这个 sprint 要把已有 plan 变成 local worker 和 Ray rollout 共同消费的 authoritative envelope；engine 主路径不再保留旧 executor fallback。

## 不做的事

- 不实现 Janus KV cache 内核；KV decode 由 `SPRINT_ar_rollout_kv_cache_optimization.md` 负责。
- 不做 paged attention、CUDA graph、token-level distributed scheduler。
- 不把 cache handle、Ray actor、scheduler object 放入 `TrajectoryBatch`。
- 不再新增第二套 distributed chunk plan。

## 当前缺口

1. `EnginePlan` 还不是 authoritative executor 输入，executor 可能自己重新 plan 或忽略 plan。
2. `ExecutionUnit` 还只是 profiler metadata，不是 materialized chunk unit。
3. Ray worker 只收到 chunk 和 profiler label，不知道 plan id、unit、capability。
4. profiler label 一部分来自 plan，一部分 executor 内硬编码。
5. `FamilyCapability` 和 runtime loaded capabilities 没有完整闭环校验。
6. family runtime 内部还有 hardcoded profiler label / capability 细节没有完全由 plan/capability 统一发放。

## 当前实现状态

第一轮已经完成：

- single-request local worker 已接入 strict `forward_plan(...)`，worker 不再回退 legacy `forward(...)`。
- `EnginePlan` 已 materialize per-chunk `ExecutionUnit`，每个 chunk unit 有 stable `unit_id`、`chunk_key`、prompt index、sample range 和 profiler label。
- Ray planner 已为每个 assignment 生成 `RayChunkExecutionEnvelope`，不再只传裸 chunk/profiler label。
- Ray worker 已接收 envelope，优先调用 optional `forward_chunk_plan(...)`，并在 result metrics 写入 `plan_id`、`unit_id`、`unit_name`、`profiler_label`、`chunk_key`。
- Ray gather 后 `OutputBatch.engine_plan` 和 `output.extra["engine_plan"]` 由同一个 authoritative plan summary 写入；chunk metrics 进入 `output.extra["ray_chunk_metrics"]`。
- local batched worker 已接入 strict `forward_batch_plan(...)`。默认 `forward_batch_by_merging_prompts(...)` 必须接收 per-request plans，merged executor 必须实现 `forward_plan(...)`；sliced outputs 会重新 attach 各自 request plan。
- legacy `forward_batch(...)` protocol、worker fallback、AR/diffusion base implementations 已删除。
- Diffusion base executor、Janus-Pro、Janus-Pro-R1、NextStep 现在都有 local `forward_plan(...)` 入口。

仍未做：

- 还没有迁移 Janus、SD3.5、NextStep executor 内部硬编码 profiler label 到 `plan.profiler_label(...)`。
- 还没有编辑 family runtime / registry 来补齐 NextStep runtime capability 闭环。
- 还没有把 AR KV cache 具体执行逻辑接入这些 flags；这继续由 `SPRINT_ar_rollout_kv_cache_optimization.md` 负责。

## 实施阶段

### Phase 1：接入 plan-aware executor protocol

编辑：

```text
vrl/engine/execution/worker.py
vrl/engine/execution/batching.py
vrl/engine/ar/executor_base.py
vrl/engine/diffusion/executor_base.py
```

新增或接入 strict 接口：

```python
forward_plan(request, sample_specs, plan)
forward_chunk_plan(request, chunk, execution_unit, plan_summary)
```

要求：

- `GenerationWorker` 只调用 plan-aware 方法。
- 没有实现 plan-aware 方法的 executor fail fast，不再走 legacy path。
- single-request local worker 只 build 一次 authoritative `EnginePlan`。
- local batched path 不能绕过 plan：新增 `forward_batch_plan(...)`，或让 `forward_batch_by_merging_prompts(...)` 接收 per-request plans / merged plan，并在 executor 有 `forward_plan(...)` 时调用 plan-aware path。
- 新增两条可 batch request 的测试：executor 只实现 `forward_plan(...)` / `forward_batch_plan(...)` 时，batched execution 仍应成功，并且 sliced outputs 的 `engine_plan` / `engine_execution` 对应各自 request。

### Phase 2：materialize chunk-level execution unit

编辑：

```text
vrl/engine/execution/planner.py
vrl/engine/core/capabilities.py
vrl/engine/execution/microbatching.py
```

要求：

- 为每个 `MicroBatchPlan` 生成具体 chunk unit。
- chunk unit 有 stable `unit_id`、sample range、prompt index、profiler label。
- 抽象 unit，例如 `decode_step` / `denoise_step` / `cache_read`，保留为 nested profiler labels。
- chunking 第一轮只覆盖 sample/prompt 轴，不做 token/timestep 级分布式拆分。

### Phase 3：Ray chunk execution envelope

编辑：

```text
vrl/distributed/ray/rollout/planner.py
vrl/distributed/ray/rollout/executor.py
vrl/distributed/ray/rollout/worker.py
```

把：

```python
execute_chunk(request, chunk, profiler_label)
```

升级为 envelope：

```text
request
chunk
plan_id
execution_unit
profiler_label
capability_summary
```

要求：

- Ray assignment 使用具体 chunk unit，不再复用同一个 `primary_chunk_unit`。
- worker result metrics 写入 `plan_id`、`unit_id`、`unit_name`、`profiler_label`、`chunk_key`。
- output gather 后 `OutputBatch.engine_plan` 和 `extra["engine_plan"]` 保持一致。

### Phase 4：profiler labels 由 plan 统一发放

编辑：

```text
vrl/engine/execution/planner.py
vrl/engine/core/profiling.py
vrl/engine/ar/executor_base.py
vrl/engine/diffusion/executor_base.py
vrl/models/families/janus_pro/runtime.py
vrl/models/families/sd3_5/runtime.py
vrl/models/families/nextstep_1/runtime.py
```

要求：

- 新增 plan label helper，例如 `plan.profiler_label("decode_step")`。
- executor 内部 label 先迁移 Janus、SD3.5、NextStep。
- 新测试保证 plan summary、metrics、Ray chunk metrics 三处 label 一致。
- cache labels `engine.cache_read` / `engine.cache_write` 必须出现在 capability / plan 中，不是 runtime 私有字符串。

### Phase 5：capability 闭环

编辑：

```text
vrl/rollouts/family_registry.py
vrl/rollouts/runtime/launch_inputs.py
vrl/models/families/nextstep_1/runtime.py
vrl/distributed/ray/rollout/worker.py
```

要求：

- NextStep 暴露 continuous AR capability。
- Ray worker load executor 后校验 runtime loaded caps 和 runtime spec capability 兼容。
- family registry 继续是 static routing canonical source。
- runtime caps 继续表达 load 后动态能力。

## 测试计划

编辑：

```text
tests/engine/test_engine_planner.py
tests/engine/generation/test_microbatching.py
tests/distributed/ray/test_large_rollout_execution.py
tests/distributed/ray/test_rollout_launcher.py
tests/models/test_nextstep_1_policy.py  # 当前仓库不存在，后续补 NextStep capability 时新增或替换
```

新增断言：

- local worker 和 Ray rollout 都只 build 一次 authoritative `EnginePlan`。
- local batched worker 不再通过 `forward_batch(...) -> executor.forward(...)` 绕过 plan-aware 执行。
- Ray 每个 chunk result 都能追踪到 `engine_plan_id + execution_unit + profiler_label`。
- profiler summary、`OutputBatch.metrics.execution_units`、`output.extra["engine_plan"]` 三者一致。
- capability 禁止 chunking 时，planner 不切 chunk。
- OOM split 后 sample 顺序不乱。

## 完成标准

- single-request executor、local batched executor、Ray chunk executor 都可以收到当前 plan。
- local batching 不能让 executor 退回裸 `forward(...)`。
- Ray worker 通过 envelope 执行 chunk，不再只拿 profiler label。
- `ExecutionUnit` 能表达 chunk-level execution，不只是 profiler 名称。
- cache/read/write labels 可从 `FamilyCapability` / `EnginePlan` 读到。
- AR KV sprint 可以只消费这里暴露的 capability/cache labels，不需要改 planner contract。
- 通过：

```bash
pytest tests/engine/test_engine_planner.py \
  tests/engine/generation/test_microbatching.py \
  tests/distributed/ray
```

## 参考路径

- `/home/mingfeiguo/Desktop/wm-infra/vrl/engine/execution/planner.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/engine/core/capabilities.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/engine/execution/worker.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/engine/execution/batching.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/engine/execution/microbatching.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/engine/ar/executor_base.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/engine/diffusion/executor_base.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/distributed/ray/rollout/planner.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/distributed/ray/rollout/executor.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/distributed/ray/rollout/worker.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/rollouts/family_registry.py`
