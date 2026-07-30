# SPRINT：Ray rollout operation deadlines

状态：**DONE（2026-07-30）**

## 结论

Ray rollout 现在同时有两条互补的 liveness 边界：

- health monitor 只证明 actor 进程和独立 health concurrency group 可达；
- operation deadline 证明 startup、generation、capability 和 weight ACK 等业务调用不会
  永久等待。

本 sprint 不在进程内恢复 actor。业务调用超过 deadline 后，当前 runtime fail closed，
拒绝 partial output，尽力取消 ObjectRef，直接销毁拥有的 actor；随后由现有
`run_verdict.json` 记录失败，supervisor 再执行 bounded restart policy；获准 retry
时才从最新 complete checkpoint 恢复。

## 为什么是两个配置字段

公共 source of truth 位于 `RolloutWorkerSection`：

```python
worker_rpc_timeout_s: float = 600.0
generation_stall_timeout_s: float = 1800.0
```

`RolloutWorkerConfig` 只是无默认值的 frozen projection。两个边界都拒绝零、负数、
NaN 和正负 infinity。

不能用同一个 timeout 表达两种 SLA：

- startup metadata、capability 和 weight ACK 没有可信的中间进度，使用 600 秒绝对预算；
- generation 已有真实 chunk 进度，使用 1800 秒 stall 预算。

仓库记录过 733 秒的 Cosmos chunk、1719.6 秒的单次 request，以及约 98 分钟的顺序
8-sample generation。600 秒 whole-request deadline 会误杀合法任务；另一方面，让 metadata
或 weight ACK 等两小时才失败也不合理。1800 秒覆盖当前最长单 chunk/request 记录，而长
sequence 只要持续产生 chunk 进度就不会被误判。

证据：

- `docs/sprints/SPRINT_ppo_update_cycle_speedup.md`
- `docs/sprints/done/SPRINT_cosmos_robotic_data_factory_domain_rl.md`
- `vrl/config/schema.py`
- `vrl/generation/ray/config.py`

三个 base Ray rollout preset 不重复写默认值。

## 精确 deadline 语义

| Operation owner | Budget | Reset rule |
|---|---:|---|
| placement-group readiness | fixed 600s | existing placement-owned boundary |
| placement GPU metadata probe | fixed 600s | one probe barrier |
| policy load | `worker_rpc_timeout_s` | fresh startup phase |
| worker metadata | `worker_rpc_timeout_s` | fresh startup phase |
| version-slot capability | `worker_rpc_timeout_s` | fresh capability barrier |
| automatic chunk-size probe | `generation_stall_timeout_s` | one opaque probe barrier |
| standard/dynamic chunk RPC | `generation_stall_timeout_s` | one deadline per submitted ref |
| OOM split child RPC | `generation_stall_timeout_s` | fresh deadline when the child is submitted |
| pipelined generation | `generation_stall_timeout_s` | reset only after completed chunk count grows |
| weight-update ACK barrier | `worker_rpc_timeout_s` | one all-worker transaction |

### Standard and dynamic dispatch

`run_actor_jobs` starts a `RayCallDeadline` immediately after each `.remote()` submission.
A locally queued job does not consume budget before submission. Once submitted, the deadline
includes Ray actor-mailbox starvation as well as execution time; it is not an “actor method body
started” clock.

One worker completing work never extends another worker's deadline. When any ref expires, all
results accumulated for that request are discarded, outstanding refs receive best-effort
`ray.cancel(force=False)`, and the runtime destroys the fleet.

Across concurrent driver calls, Ray's actor mailbox remains the scheduling source of truth.
There is deliberately no second mutable “active deadline” registry on the executor or runtime.

### Pipelined dispatch

Single-worker pipelining uses a driver-side request lock. A later request waits before creating its
stall deadline, because the synchronous worker cannot execute two default-group requests at once.

The worker publishes an immutable `PipelinedRequestProgress` snapshot through the existing health
concurrency group. Only a strict increase in `completed_chunks` resets the stall deadline. Health
success, a missing snapshot, or a repeated valid snapshot do not count as progress. A returned
snapshot for a different request ID is a protocol violation.

Invalid type, request ID, total count, or regressing progress raises
`PipelinedProgressError`, a terminal wire-contract error. Result, progress RPC, and caller
cancellation races all cancel the losing refs. The final gather/teardown after the last chunk must
still finish inside the last stall window.

## Failure and ownership contract

```text
operation exceeds deadline or progress protocol breaks
  -> reject complete and partial output
  -> best-effort cancel submitted ObjectRefs
  -> close runtime admission
  -> cancel a graceful shutdown already waiting on release_policy
  -> skip new release_policy RPCs
  -> ray.kill every owned actor with no_restart=True
  -> preserve the terminal operation error as root cause
  -> stop continuous admission and retry
  -> write failed run verdict
```

`ray.cancel` is not the correctness mechanism. A running synchronous actor method may ignore
best-effort cancellation. Correctness comes from rejecting results and killing the owned actor
fleet.

Startup and placement failures are cleaned by the owner that created the candidate resources.
An active on-demand facade delegates weight restore through the inner runtime's public
`update_weights` boundary, so the actual worker owner receives the terminal error and force-kills
its actors. The same rule applies to a cold activation candidate before publication.

A timeout arriving after graceful shutdown entered its 60-second release wait cancels only that
local release-barrier task. The shared shutdown task then continues into force teardown for every
waiter. The detached `ray.get` thread may finish later, but it does not delay actor destruction or
replace the timeout root.

## Cross-layer terminal protocol

`TerminalRuntimeError` lives in `vrl/runtime_errors.py`, below generation, rollout, and Ray
domains. `RayOperationTimeout` and `PipelinedProgressError` derive from it.

Cleanup wrappers may preserve the original exception in `root_cause` or `__cause__`.
`find_error_cause` and `deepest_error_cause` walk that chain cycle-safely:

- continuous producer detects a nested terminal error and does not retry the prompt slot;
- continuous consumer propagates the cleanup wrapper without adding an opaque retry error;
- verdict writing records the deepest stable error class while keeping the outer cleanup message.

This avoids a forbidden `vrl/ray -> vrl/generation` dependency and keeps rollout orchestration
independent of the Ray transport type.

## 修改内容

- 新增 domain-neutral terminal error/cause-chain boundary。
- 新增共享 Ray monotonic deadline、typed timeout、sync/async wait 和 cancellation adapter。
- 为 placement probe、startup、capability、chunk-size probe、generation 和 weight sync 接入
  owner-local deadline。
- 为 pipelined worker 增加严格 chunk progress 协议和 single-flight admission。
- 为 timeout 增加 force shutdown upgrade、on-demand inner ownership 和 completion gate。
- 让 continuous orchestration 和 verdict 保留 nested terminal root cause。
- 新增 deterministic CPU 和 real-Ray CPU 回归。

## 保持不变

以下薄边界继续保留：

- `RayActorGroup`：Ray actor construction/startup adapter；
- `run_actor_jobs`：standard/dynamic/OOM 共用 dispatch abstraction；
- `RayGenerationExecutor.execute`：pipelined single-flight public API facade；
- `RayGenerationWeightSync`：all-worker transactional ACK boundary；
- `RayGenerationRuntime`：admission、version publication 和 actor ownership boundary；
- `pipelined_progress`：Ray concurrency-group adapter；
- worker adapter / `GenerationWorkerCore` 分层：process/framework 与 model execution 边界。

以下 ALL_CAPS 常量继续保留：

- `_PLACEMENT_READY_TIMEOUT_S`：placement scheduling SLA；
- `HEALTH_CONCURRENCY_GROUP`：health/pipelined-progress decorators 和 actor creation
  共用的 protocol name；
- test fixture constants。

`RayCallDeadline.operation` 和 `context` 是显式的 protocol/provenance-only 字段，用于稳定
错误身份；`timeout_s` 和 expiry 则直接控制等待。

## Non-goals

- 不做 actor restart、fleet rebuild 或 whole-request retry；
- 不做 driver、GCS、raylet、`ray.kill` 或 process watchdog；
- 不增加 reliability manager、timeout matrix、operation taxonomy table；
- 不把 deployment timeout 放入 family capability、request 或 launch-contract wire payload；
- 不修改 health monitor 的 reachability 语义；
- 不重新设计已有 120 秒 sleep/wake residency timeout；
- 不调整 reward HTTP 或 family subprocess timeout；
- 不跑 GPU、训练或长时间 benchmark。

一致的跨 family / framework adapter 形状比减少几行代码更有价值。本 sprint 不 flatten
这些薄边界，也不创建 `DeadlineManager`、`ReliabilityConfig` 或 ALL_CAPS operation table。

## 验证

- affected suite（19 files，含 real-Ray CPU coverage）：567 passed；
- real-Ray CPU hung-business-call case：通过；
- generation/rollout/Ray architecture boundary：通过；
- full repository suite：3894 passed，25 skipped；
- scoped Ruff（39 changed Python files）：check/format 全部通过。

## 参考路径

- `vrl/runtime_errors.py`
- `vrl/ray/operation_deadline.py`
- `vrl/ray/actor_group.py`
- `vrl/ray/actor_pool.py`
- `vrl/ray/placement.py`
- `vrl/generation/ray/executor.py`
- `vrl/generation/ray/runtime.py`
- `vrl/generation/ray/weight_sync.py`
- `vrl/generation/ray/worker.py`
- `vrl/rollouts/orchestration/continuous/producer.py`
- `vrl/rollouts/orchestration/continuous/consumer.py`
- `vrl/scripts/train.py`
