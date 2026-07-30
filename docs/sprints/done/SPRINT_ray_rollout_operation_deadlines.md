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
generation_stall_timeout_s: float = 3600.0
```

`RolloutWorkerConfig` 只是无默认值的 frozen projection。两个边界都拒绝零、负数、
NaN 和正负 infinity。

不能用同一个 timeout 表达两种 SLA：

- startup metadata、capability 和 weight ACK 没有可信的中间进度，使用 600 秒绝对预算；
- generation 已有真实 chunk 进度，使用 3600 秒 stall 预算。

仓库记录过 733 秒的 Cosmos opaque chunk、1719.6 秒的 16-chunk request、约 98 分钟的
顺序 8-sample generation，以及 512p shape 约 30 分钟的 cold compile。600 秒
whole-request deadline 会误杀合法任务；1719.6 秒和 98 分钟不是单个 opaque wait，会随
chunk progress 重置。3600 秒为“30 分钟 compile + 733 秒首 chunk”保留余量，而 metadata
和 weight ACK 仍由独立 600 秒 control budget fail closed。

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
| automatic chunk-size probe | `generation_stall_timeout_s` | single-flight, one deadline per admitted worker probe |
| standard/dynamic chunk RPC | `generation_stall_timeout_s` | one deadline per submitted ref |
| OOM split child RPC | `generation_stall_timeout_s` | fresh deadline when the child is submitted |
| pipelined generation | `generation_stall_timeout_s` | reset only after completed chunk count grows |
| weight-update ACK barrier | `worker_rpc_timeout_s` | one deadline per admitted worker call; all-worker transaction |

### Fleet-wide default-group admission

The launcher creates one fleet-owned `RayActorDispatcher` and injects that exact instance into
generation and weight sync. It gives each synchronous actor one real default-group slot across
standard/dynamic chunks, OOM retries, chunk-size probes, pipelined requests, and weight updates.
It starts a `RayCallDeadline` immediately before each `.remote()` submission. A job waiting for
dispatcher admission does not consume budget and is not pre-queued in the actor mailbox. Once
admitted, the deadline includes driver serialization, Ray transport, and execution time; it is
not an “actor method body started” clock.

Admission is FIFO per compatible worker across concurrent callers. Each worker owns an independent
waiter queue: when a request's pending/active shape changes, retained workers keep their FIFO
position, removed workers drop only that position, and newly compatible workers append at that
worker's tail. A request with more pending chunks therefore cannot reclaim a slot ahead of a
waiting weight update, while unrelated workers remain independently usable. Submission scans one
pending snapshot and stops once no physical slot remains; it does not retry blocked jobs before an
actual slot-state change. This bounds non-draining sync admission without adding a privileged
weight-sync concurrency group.

One worker completing work never extends another worker's deadline. When any ref expires, all
results accumulated for that request are discarded, outstanding refs receive best-effort
`ray.cancel(force=False)`, and the runtime destroys the fleet.

The dispatcher is the single owner of cross-request admission and submitted-ref state. A terminal
timeout, partial/unknown submitted-work cancellation, or actor failure closes it first-error-wins,
cancels all active refs, and wakes every local admission waiter with the same terminal cause. If
all submitted refs completed successfully before cancellation won the driver race, transport has
already linearized: the dispatcher returns the full result set so the caller still performs typed
ACK validation and publishes resulting state. Propagating an ordinary cancellation there would
allow workers to install policy v2 while the runtime remained RUNNING and advertised v1. This
state cannot be request-local: request-local pools would pre-queue concurrent calls into the same
synchronous actor mailbox before their deadlines start.

The run-level automatic chunk-size verdict has a separate runtime single-flight lock, but its
remote probes still enter this same dispatcher. This matters when one first request uses
`samples_per_chunk: auto` while another already carries an explicit integer: neither path can
pre-queue behind the other without driver admission.

Non-draining weight sync skips the higher-level prompt drain only. It does not make a synchronous
actor execute generation and `update_weights` concurrently. An update waits locally for any
already-admitted generation call, then receives its full ACK deadline when it gets the next real
actor slot. The old request remains safe through versioned trainable-state slots.

Health and `pipelined_progress` calls deliberately bypass this dispatcher: both use the dedicated
health concurrency group. Making a progress query wait for the default slot occupied by the
pipelined request it observes would deadlock.

### Pipelined dispatch

Single-worker pipelining uses a driver-side request lock. A later request waits before creating its
stall deadline, because the synchronous worker cannot execute two default-group requests at once.
After that request-level lock, `RayActorDispatcher.run_one` obtains the fleet slot and creates the
initial deadline immediately before submission. Its protocol-specific waiter may replace that
deadline only after verified chunk progress; the dispatcher remains the sole owner of the main
result ObjectRef and fleet terminal state.

The worker publishes an immutable `PipelinedRequestProgress` snapshot through the existing health
concurrency group. Only a strict increase in `completed_chunks` resets the stall deadline. The
reset copies the initial deadline contract so operation, timeout, and diagnostic context remain
the dispatcher's source of truth; only monotonic expiry is renewed. Health success, a missing
snapshot, or a repeated valid snapshot do not count as progress. A returned snapshot for a
different request ID is a protocol violation.

Invalid type, request ID, total count, or regressing progress raises
`PipelinedProgressError`, a terminal wire-contract error. Cancellation before lock/dispatcher
admission remains an ordinary caller cancellation because no actor state changed. Cancellation
after submission raises the original `asyncio.CancelledError` from a
`RayOperationCancelled` terminal marker, cancels losing refs, and makes the owner destroy the
fleet. The final gather/teardown after the last chunk must still finish inside the last stall
window.

For a multi-worker transaction, “after submission” includes the case where one worker has already
completed while another job is still waiting for admission. Cancelling there must fail closed:
the driver cannot publish a policy version after only a subset of workers installed it.

Runtime teardown stops and joins the background health-monitor thread through `asyncio.to_thread`.
The monitor retains its bounded synchronous `stop()` lifecycle owner, while the async runtime loop
remains able to advance concurrent shutdown and terminal-cleanup waiters during that join.

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
`find_error_cause` and `failure_identity_cause` walk that chain cycle-safely:

- continuous producer detects a nested terminal error and does not retry the prompt slot;
- continuous consumer propagates the cleanup wrapper without adding an opaque retry error;
- verdict writing stops at the first domain-owned terminal class instead of exposing a dependency
  exception stored below it, while keeping the outer cleanup message.

This avoids a forbidden `vrl/ray -> vrl/generation` dependency and keeps rollout orchestration
independent of the Ray transport type.

## 修改内容

- 新增 domain-neutral terminal error/cause-chain boundary。
- 新增共享 Ray monotonic deadline、typed timeout、sync wait 和 cancellation adapter；
  async actor waits 统一由 fleet dispatcher 持有。
- 为 placement probe、startup、capability、chunk-size probe、generation 和 weight sync 接入
  owner-local deadline。
- 为 pipelined worker 增加严格 chunk progress 协议、request single-flight 和 fleet admission。
- 让 generation、probe、pipelined request 与 weight sync 共用唯一 fleet dispatcher。
- 为 timeout 增加 force shutdown upgrade、on-demand inner ownership 和 completion gate。
- 让 continuous orchestration 和 verdict 保留 nested terminal root cause。
- 新增 deterministic CPU 和 real-Ray CPU 回归。

## 保持不变

以下薄边界继续保留：

- `RayActorGroup`：Ray actor construction/startup adapter；
- `RayActorDispatcher`：fleet 级 default-group admission 与 submitted-ref owner；
- `run_actor_jobs`：deprecated public API facade；旧的一次调用入口继续可导入，但内部委托
  给 `RayActorDispatcher`，新代码必须让 dispatcher 跟随 fleet 生命周期；
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
旧 `max_inflight_chunks_per_worker` 配置已删除：production worker 是 synchronous actor，
实际并发槽恒为 1，保留这个始终无效的 public knob 只会制造 no-op 配置。continuous
rollout 的 `max_inflight_groups` 是另一层真实 admission policy，保持不变。

## 验证

- affected suite（含 generation/Ray/config/architecture 与 real-Ray CPU coverage）：
  761 passed；
- real-Ray CPU hung-business-call、cross-request admission 与 fleet probe cases：通过；
- generation/rollout/Ray architecture boundary：通过；
- full repository suite：3913 passed，25 skipped；
- scoped Ruff（38 changed Python files）：check/format 全部通过。

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
