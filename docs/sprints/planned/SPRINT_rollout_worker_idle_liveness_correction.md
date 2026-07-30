# SPRINT: Rollout worker idle-liveness correction（planned）

状态：**planned / CPU-only corrective sprint**。这是
[[SPRINT_rollout_worker_liveness]] 的实现回归修正，不运行 Ray/GPU experiment。

## 根因

public worker adapter 已把 health 放在独立 Ray concurrency group，且实现不碰 model/GPU：

```python
@ray.method(concurrency_group=HEALTH_CONCURRENCY_GROUP)
def health(self) -> str:
    return self.core.worker_id
```

所以 model parked 只改变 residency，不改变 actor process reachability。但 monitor 当前把
两者错误等同：

```python
self._health_monitor.pause()
...
self._health_monitor.resume()
```

这正好漏掉 trainer 使用 GPU 时最长的 idle window。除此之外：

- `health_check_first_wait_s`、`_paused`、`_needs_first_wait` 都只服务这个错误推导；
- `stop()` join timeout 后仍清空 `_thread`，会丢失活线程 ownership 并允许重复线程；
- start/stop 与 `RuntimeLifecycle` 跨 monitor thread/event loop 无锁；
- actor 缺 `health.remote` 时静默跳过真实 protocol violation；
- async teardown 直接 `thread.join()`；
- background probe failure 若发生在最后一个 foreground command 之后，shutdown 可以成功，
  recipe 最终写 success verdict。

## 设计

### 1. Residency 与 reachability 分离

- 删除 `health_check_first_wait_s` 的 schema → resolved config → launcher → runtime 镜像；
- 删除 monitor pause/resume/first-wait state；
- launcher 只在 workers 已创建、policy load/capability 完成后启动 monitor，因此不需要启动
  grace；
- `sleep_workers()` / `wake_workers()` 不再控制探活。

### 2. Thread 与 actor ownership

- monitor 用 lock 串行 `start()` / `stop()`；只在 thread 真退出后清空 handle；
- join timeout 保留 handle，后续 `start()` 返回 false；
- missing actor 或 missing callable `health.remote` 进入同一个 terminal failure path；
- runtime 提供一个锁保护的 worker snapshot/kill seam；monitor failure 与 async shutdown
  共享它，成功 kill 后删除 ownership，失败 handle 保留供 shutdown retry；
- monitor stop/join 与 actor kill 从 event loop 通过 `asyncio.to_thread` 执行。

这里不恢复 ornamental `Manager`/`Handler`。若 worker ownership 方法增长到需要独立
class，名字必须是 concrete `RayGenerationWorkerFleet`，因为它代表 actor inventory
与 cleanup lock；当前 sprint 先以 runtime-owned seam 保持最小 truthful state。

### 3. Background failure handoff

- `RuntimeLifecycle` 用 lock 序列化 terminal transitions，并单独保留 first
  `background_failure`；
- `RayGenerationRuntime.background_failure` 是 read-only capability；on-demand facade 从
  inner runtime 继承并在 teardown 前 snapshot；
- 新 `BackgroundFailureProvider` structural Protocol 让 collector 可选读取，不用
  `getattr` 猜 private state；
- `RolloutCollector.shutdown()` 在释放 runtime 前 snapshot；
- online composition root 最终采用明确优先级：
  1. 已有 training/operation error；
  2. background liveness failure；
  3. cleanup error。

`shutdown()` 继续只负责 resource cleanup，不变成隐式 error-delivery API。

## 审计裁决

| Suspect | 裁决 | 原因 |
|---|---|---|
| parking-derived monitor pause/grace | **REMOVE** | wrong derivation；residency != reachability |
| `health_check_first_wait_s` 三层字段 | **REMOVE** | 只为错误状态提供 producer/consumer |
| health concurrency group/name | **KEEP** | Ray protocol name / out-of-band ownership boundary |
| thin worker `health()` | **KEEP** | 必需 framework adapter |
| monitor OS thread | **KEEP** | trainer event loop 可长期被同步 forward/backward 占用 |
| join grace constant | **KEEP** | bounded cleanup protocol |
| silent missing health branch | **FIX** | launcher 只创建有 endpoint 的 production worker |
| lifecycle/background state | **FIX** | 跨线程 terminal state 与 idle failure delivery |
| transition table | **KEEP** | RUNNING → SHUTTING_DOWN → TERMINATED 是公共生命周期 |

## True / false regressions

- healthy parked actor 持续被 probe；
- parked/idle actor failure 关闭 admission、kill fleet，最终 recipe cleanup 抛
  `RolloutWorkerUnreachable`；
- clean idle shutdown 没有 background failure；
- missing actor/remote health fail loud；
- join timeout 保留 thread handle且禁止第二次 start；正常退出才清空；
- barrier 控制 concurrent start/stop 至多创建一条 thread；
- fake join/kill 阻塞时 asyncio ticker 继续推进；
- monitor/shutdown 同时 kill 时每个成功 handle 只归零一次，失败 handle 保留；
- foreground error 优先于 background，background 优先于 cleanup；
- on-demand inner runtime failure在 facade 清理后仍可读。

## 非目标

- 不添加 business-RPC deadline；健康线程只证明 process endpoint reachability；
- 不在进程内 rebuild actor fleet；
- 不改变 parking/sleep/wake 的 GPU memory ownership；
- 不调整 monitor interval/timeout 默认值；
- 不运行 real-Ray chaos 或 GPU test。

## 验收

```bash
.venv/bin/python -m pytest -q \
  tests/generation/ray/test_health_monitor.py \
  tests/generation/ray/test_lifecycle_fsm.py \
  tests/generation/ray/test_runtime_lease_sleep.py \
  tests/rollouts/collector/test_runtime.py \
  tests/scripts/test_online_lifecycle.py \
  tests/config/test_schema.py \
  tests/generation/ray/test_runtime_config.py
```

## References

- `vrl/generation/ray/worker.py`
- `vrl/generation/ray/health_monitor.py`
- `vrl/generation/ray/runtime.py`
- `vrl/generation/ray/lifecycle_fsm.py`
- `vrl/rollouts/collector/core.py`
- `vrl/scripts/common/online.py`
- [[SPRINT_rollout_worker_liveness]]
