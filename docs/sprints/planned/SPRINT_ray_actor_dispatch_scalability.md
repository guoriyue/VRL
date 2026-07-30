# SPRINT: Ray actor dispatch scalability and typed timing（planned）

状态：**planned / CPU-only scheduler implementation**。不启动 Ray cluster。

## 根因与基线

`run_actor_jobs()` 每有 ObjectRef 完成、释放一个 actor slot，就从头扫描全部 pending
jobs：

```python
for _ in range(len(pending)):
    job = pending.popleft()
    if dispatch is None:
        pending.append(job)
```

同步 submit 只会占用 slot，不会释放 slot，因此外层 `while made_progress` 不能创造新
eligibility；它只是扩大重扫。4 个立即完成的 unbound fake workers 的 CPU 基线：

| jobs | wall | 每 job |
|---:|---:|---:|
| 500 | 0.0096s | 19.3µs |
| 1,000 | 0.0326s | 32.6µs |
| 2,000 | 0.1255s | 62.8µs |
| 4,000 | 0.4899s | 122.5µs |
| 8,000 | 1.9401s | 242.5µs |

每 job 成本随 N 近似翻倍，是实测二次增长。

## 设计

新增一个私有 `_ActorJobScheduler`，只拥有 pending priority、stable order、
plan-time binding、dynamic worker capacity 与 inflight count；它不等待 ObjectRef，也不
拥有 telemetry。

### Queue shape

- jobs 只在启动前给定，因此全局 stable sort 一次；
- unbound jobs 是一个按 `(-priority, input_order)` 排序的 deque；
- bound jobs 按 worker 分 deque；
- 一个 lazy-invalidated heap 只保存当前有空位的 bound worker queue head；
- 一个 lazy-invalidated heap 保存 dynamic workers 的
  `(inflight_count, worker_id)`；
- dispatch 比较“最高 priority 的 eligible bound head”与 unbound head，保持当前
  global-highest-eligible 语义；
- completion 只更新一个 worker 的 capacity，并重新发布它的候选。

复杂度从 pending rescans 的 `O(N²)` 降为一次 sort 加每次状态变化的 heap 工作：
`O(N log N + N log W)`；不会用 wall-time assertion 固化机器速度。

### Typed timing

当前 `schedule: list[dict[str, Any]]` 是 closed finite bag，keys 只有：

```text
job_index, worker_id, queue_wait_s, execution_s
```

改为 frozen/slots `ActorJobTiming`。`RayGenerationExecutor` 只在最终
`GenerationOutput.extra["runtime_debug"]` 的外部 primitive/debug 边界转回 dict。
`job_index` 用于行为相关联；其余字段是明确的 display/provenance output，不再是无类型
内部状态。

## 审计裁决

| Suspect | 裁决 | 原因 |
|---|---|---|
| Ray ObjectRef direct await / `asyncio.wait(FIRST_COMPLETED)` | **KEEP** | 正确 async transport boundary |
| per-completion full pending scan | **FIX** | 实测二次增长 |
| outer `while made_progress` | **REMOVE** | submit 不释放 capacity，branch 无新增语义 |
| plan-time binding | **KEEP** | static placement protocol |
| unbound least-inflight + worker-id tie | **KEEP** | dynamic placement protocol |
| stable priority tie | **KEEP** | deterministic gather/debug contract |
| `schedule` dict bag | **FIX** | closed internal schema；typed struct 是单一事实来源 |
| debug payload dict | **KEEP** | user-facing primitive serialization boundary |
| `_ActorJobScheduler` | **ADD** | 拥有真实 scheduling invariant；不是为了减少 LOC 的薄 helper |

## True / false regressions

- bound jobs 不迁移，gather order 仍按 `job_index`；
- unbound jobs 仍由先释放的 worker pull；
- priority 与 equal-priority input order 保持；
- mixed bound/unbound 在 worker capacity 竞争时保持 global-highest-eligible；
- `max_inflight_per_actor > 1` 正确填满并回收 capacity；
- bound remote method 缺失时在 submit 前 fail loud；
- typed timing 一 job 一行，executor debug payload shape 不变；
- deterministic operation-count test 证明 2N jobs 的 pending/candidate 读取不会接近 4 倍；
- 同一 fake benchmark 重新记录 before/after，但测试不做 wall-time threshold。

## 非目标

- 不改变 chunk placement cost model、Ray actor concurrency 或 gather payload；
- 不实现 work stealing、preemption、priority update/cancel；
- 不替换 Ray await 为 `ray.get`；
- 不引入公共 scheduler framework；这是 Ray actor-pool 的私有 invariant owner。

## 验收

```bash
.venv/bin/python -m pytest -q tests/ray/test_chunk_dispatch.py
```

## References

- `vrl/ray/actor_pool.py`
- `vrl/generation/ray/executor.py`
- `tests/ray/test_chunk_dispatch.py`
- https://docs.python.org/3/library/heapq.html#priority-queue-implementation-notes
- https://docs.ray.io/en/latest/ray-core/actors/async_api.html#objectrefs-as-asyncio-futures
