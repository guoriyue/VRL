# SPRINT: Runtime time machine —— 用内部因果票据解决迟到事件，不把 fleet generation 变成公共参数

> Historical design, superseded on 2026-07-11 by
> `SPRINT_explicit_rollout_activation.md`. Runtime-issued operation tickets and
> `QUIESCING` were removed; rollout schedules now own admission/draining, while
> the runtime keeps concrete activation/offload/shutdown single-flight tasks.

状态：**in-progress（2026-07-10）**。这是
[`SPRINT_ray_rollout_fault_tolerance.md`](./SPRINT_ray_rollout_fault_tolerance.md)
步骤 3–6 的架构澄清，不是第二套 recovery 实现。P0 的 admission/cleanup truthfulness
和 P1 的 runtime-issued ticket/drain barrier 已落地。outer/inner `_fleet_generation`、
`fleet_generation=` 参数和 write-only expected-kill registry 已删除：在 stale-event
control flow尚未实现时，它们只是会漂移的提前设计。ticket尚未绑定 immutable fleet
identity，single fleet owner仍属于 P2。

## 0. 一句话

异步系统像一台“时间机器”：generation 1 提交的结果、异常和 actor-death 通知，可以在 generation 2
已经上线后才抵达。**不能根据“现在是什么状态”猜“这个事件来自什么时候”**；必须在提交瞬间由 runtime
内部签发不可伪造的因果票据，完成时用同一票据分类。

公共 API 保持：

```python
await runtime.generate(request)
await runtime.update_weights(state, policy_version)
```

不新增：

```python
await runtime.generate(request, fleet_generation=...)
```

调用方不知道、也不应该决定 worker fleet 的代数。内部保留 epoch 概念，但它归唯一 fleet owner，随
immutable `WorkerFleet` / `OperationTicket` 自动传播，不作为到处手传的裸 `int` 参数。

## 1. 根因：现在无法重建过去

真实时间线：

```text
t0  phase=RUNNING, active fleet epoch=1, workers=A/B
t1  request R admitted and submitted to A/B
t2  A dies; recovery starts
t3  replacement fleet epoch=2, workers=C/D, phase returns RUNNING
t4  epoch-1 ObjectRef failure arrives late
```

在 `t4` 只读当前状态：

```text
phase=RUNNING
active fleet epoch=2
```

看不出 failure 属于 epoch 1。若把它当当前 fleet 的新故障，会再次 quarantine C/D；若把它当普通
expected death，也可能吞掉 epoch 2 的真实事故。这就是 time-machine bug：旧因果跨过 await 边界进入现在。

phase 不能解决它：epoch 1 和 epoch 2 都可能是 `RUNNING`。actor name 也不能解决：name 是可读标签，
不是 committed fleet identity。调用方传 `fleet_generation` 更不能解决：调用方可能持有旧值、传错值，
而 runtime 最终仍需保存一个权威当前值来校验它。

## 2. 两个正交状态

runtime 的真实状态是乘积，不是一串无限 enum：

```python
RuntimeState(
    phase=RuntimePhase.RUNNING,
    active_fleet=WorkerFleet(epoch=2, ...),
)
```

- `phase` 回答：现在允许什么行为？
- `fleet epoch` 回答：这个异步操作属于哪一批 worker？
- `policy_version` 回答：这批 worker 当前承载哪一版可训练权重？

三者不能互相替代：

```text
RuntimePhase    finite control protocol
fleet epoch     worker-process incarnation
policy_version  committed trainable-state version
```

不创建 `RUNNING_GEN_1` / `RUNNING_GEN_2`；generation 是状态机的 extended state。它应存在，但不应
从 runtime facade 一层层手传成公共参数。

## 3. 为什么必须有 SHUTTING_DOWN

shutdown 不是瞬时动作：关闭 admission、处理已接收请求、杀 actor、删除 placement group 之间都有 await。
只用 `RUNNING/TERMINATED` 会二选一地撒谎：

- teardown 时继续标 `RUNNING`：新请求会提交给正在退出的 worker；
- teardown 开始就标 `TERMINATED`：重复 shutdown 会提前返回，但资源尚未释放。

正确协议：

```text
RUNNING
  -> SHUTTING_DOWN       close admission atomically
  -> wait/cancel    every admitted OperationTicket reaches terminal state
  -> teardown       record kills, release policy, kill actors, remove PG
  -> TERMINATED        only after teardown succeeds
```

非 terminal 的 phase handoff 使用独立的临时状态：

```text
RUNNING -> QUIESCING -> RUNNING
                    -> SHUTTING_DOWN  # terminal shutdown wins
```

`QUIESCING` 在shared-GPU worker park前关闭admission并等待已有generate/update；resident
runtime或尚未lazy launch inner runtime时，`release()` 仍是真正no-op，不制造barrier。

当前 admission ticket和drain barrier已经落地，因此 `SHUTTING_DOWN` 是行为真实的状态，
不是 teardown log标签。

命名使用服务/executor生命周期术语，不冒充线程调度状态：`QUIESCING` 表示临时拒绝新请求并
等待已接收请求完成；normal handoff使用sleep/wake保留actor，teardown只属于terminal
cleanup或未来failure quarantine。`SHUTTING_DOWN` 表示不可逆的terminal cleanup，
`TERMINATED` 只在cleanup成功后成立。
Ray 自己的 `ALIVE/RESTARTING/DEAD` 仍只描述 actor，不与 runtime facade phase混用。

## 4. 目标对象：不是 fleet 参数，而是 runtime 签发的快照与票据

### 4.1 WorkerFleet：一代 worker 的不可变身份

```python
@dataclass(frozen=True, slots=True)
class WorkerFleet:
    epoch: int
    workers: tuple[DistributedWorkerHandle, ...]
    actor_keys: tuple[str, ...]
```

规则：

- 只有 `WorkerFleetOwner` 能创建；
- epoch 在 publish 新 fleet 时单调递增；sleep/wake 保留同一 actor，不递增；cold launch/recovery
  replacement 才递增；
- executor、weight sync 和 runtime 必须引用同一个 fleet，禁止各自复制 worker list 或重新解析“当前
  fleet”；
- fleet 发布后不可原地换 workers，replacement 必须创建下一 epoch；
- fleet 只承载 worker identity、actor lifetime 和派生索引，不把 executor/weight sync 塞进来；这样避免
  identity object 变成 god object，也避免 executor ↔ fleet 的循环 ownership。

launch/publish 需要原子提交“一批 worker + 对应执行器 + 对应权重同步器”时，owner 内部使用窄的
deployment record：

```python
@dataclass(frozen=True, slots=True)
class FleetDeployment:
    fleet: WorkerFleet
    executor: RayGenerationExecutor
    weight_sync: GenerationWeightSync | None
```

`RayGenerationExecutor` 和 `RayGenerationWeightSync` 各自引用 `deployment.fleet`；它们不拥有第二份 workers
list。`FleetDeployment` 是 owner 的 committed unit，不进入 public `GenerationRuntime` protocol。

### 4.2 OperationTicket：提交瞬间冻结因果

```python
@dataclass(frozen=True, slots=True)
class OperationTicket:
    operation_id: str
    request_id: str
    deployment: FleetDeployment
    policy_version: int | None
```

`generate()` 内部原子完成两件事：关闭竞态窗口并截取 active fleet。

```python
ticket = await lifecycle.admit_generate(
    request_id=request.request_id,
    active_deployment=fleet_owner.active,
    policy_version=current_policy_version,
)
try:
    return await ticket.deployment.executor.execute(request)
finally:
    await lifecycle.complete(ticket)
```

调用方不传 epoch。driver 侧等待 ObjectRef 的闭包持有 `ticket`；worker 在 launch 时已由 fleet 绑定
epoch，并在内部 result metadata 回显。事件抵达时比较 `ticket.deployment.fleet`，不重新引入已删除的裸
generation counter来猜过去。

### 4.3 ExpectedKillRecord 归 fleet，不归 phase

已经删除的旧形状：

```python
lifecycle.record_expected_kill(
    actors,
    fleet_generation=self._fleet_generation,
    reason=reason,
)
```

问题是 lifecycle 不拥有 fleet，调用方必须把 generation 作为参数喂进去；outer lease 与 inner runtime
还可能喂出不同值。当前没有death classifier，因此整个write-only registry也已删除。P2/P3同时落地
WorkerFleet和真实classifier时，目标形状才是：

```python
record = fleet_owner.record_expected_kill(
    fleet=deployment.fleet,
    reason=reason,
)
```

record 从 fleet 自动取得 epoch + actor keys，仍不接受裸 `fleet_generation` 参数。phase 只决定
“SHUTTING_DOWN 下是否允许执行 teardown”；具体死亡是否由本次 teardown 发起，仍以 operation id + actor id +
fleet epoch 匹配，不能仅凭 `phase is SHUTTING_DOWN` 猜。

## 5. 唯一 owner 与原子 publish

新增一个真正有复杂度的 owner，而不是装饰性 manager：

```python
class WorkerFleetOwner:
    active: FleetDeployment | None
    candidate: FleetDeployment | None
    next_epoch: int
    recovery_task: asyncio.Task | None
```

它唯一负责：

1. launch candidate；
2. load base policy；
3. restore last committed trainable state；
4. 收齐 worker `policy_version + digest` ACK；
5. 原子 publish candidate 为 active；
6. retire/quarantine old fleet；
7. 保证同一 active epoch 只有一个 recovery task。

以下双重计数已经删除：

```text
outer lease _fleet_generation = 2
inner RayGenerationRuntime _fleet_generation = 1
```

lease facade、inner runtime、executor、weight sync 不再分别拥有 generation。lease 持有 fleet owner；inner
runtime 使用 owner 发布的 deployment，不重新初始化 epoch。

## 6. 完整状态转换

当前只保留四个有生产消费者的phase：

```text
RUNNING -> QUIESCING -> RUNNING
RUNNING / QUIESCING -> SHUTTING_DOWN -> TERMINATED
```

future recovery落地shared recovery task时，才按真实控制流增加 `RECOVERING`：

```text
(RUNNING, no fleet)
    acquire ticket -> launch/restore/verify epoch 1
        -> (RUNNING, active=1)

(RUNNING, active=1)
    worker-specific failure(ticket.deployment.fleet=1)
        -> (RECOVERING, active=1, candidate=2)
        -> restore committed policy into candidate 2
        -> atomic publish
        -> (RUNNING, active=2)
        -> deterministically replay whole request R on epoch 2

(RUNNING, active=2)
    late failure(ticket.deployment.fleet=1)
        -> obsolete-event classification
        -> no recovery of epoch 2

(RUNNING or RECOVERING, active=N)
    shutdown
        -> SHUTTING_DOWN
        -> close admission
        -> cancel/wait admitted tickets
        -> abort unpublished candidate
        -> expected-kill active fleet N
        -> TERMINATED
```

shutdown 若在 recovery 中到达，`SHUTTING_DOWN` 是更高优先级目标。recovery 在 publish candidate 前必须重新
检查 phase；一旦不是 `RECOVERING`，清理 candidate 并退出，绝不让 worker 在 teardown 后“复活”。

## 7. Failure outcome 与 cleanup phase 必须正交

旧实现把 `FAILED` 同时当运行结果和资源状态，导致运行失败可能跳过cleanup。当前已删除
`FAILED` enum；root cause保存在正交的 `failure` 字段。

当前规则：

- `failure` 保存第一个 root cause；
- failure立即推进到 `SHUTTING_DOWN` 并关闭admission；
- terminal cleanup仍执行；
- 只有 cleanup成功才进入 `TERMINATED`；cleanup失败保持 `SHUTTING_DOWN + failure`，shutdown caller
  取得cleanup error，但lifecycle保存的第一个root cause不被覆盖；
- 重复 shutdown 等待同一个 shutdown task，不重复 kill/remove。

实现可以用单一 `_shutdown_task`，不增加 `cleanup_done`、`shutdown_started`、`is_stopped` 三个互相漂移的
布尔值。

## 8. 不把 epoch 暴露成公共 arg

### 公共边界保持不变

```python
GenerationRuntime.generate(request)
GenerationRuntime.update_weights(state, policy_version)
GenerationRuntime.release()
GenerationRuntime.shutdown()
```

### epoch 只在内部三处可见

1. `WorkerFleet.epoch`：behavior-consumed，决定 stale/obsolete event、recovery publish 和 request replay；
2. worker launch context/result metadata：由 owner 注入，worker 回显，不由用户设置；
3. logs/death records：从 fleet 派生，供 provenance，不是第二 source of truth。

禁止：

- YAML `fleet_generation`；
- public `generate(..., fleet_generation=...)`；
- caller 从 runtime 读 generation 后再传回（TOCTOU）；
- 每个 executor/weight-sync helper 维护自己的 counter；
- 用 actor name 或当前 phase 推导 epoch。

## 9. 实施顺序

### P0 — 修正现有 FSM 的真实性

- **已完成：**`require_running()` 只允许 `RUNNING`；无生产消费者的 `STARTING` 已删除；
- **已完成：**teardown error不进入 `TERMINATED`，保持 `SHUTTING_DOWN + failure`，失败handles
  保留用于retry；重复的 `FAILED` phase已删除；
- **已完成：**shutdown/release single-flight，单个 waiter cancellation 不取消共享 cleanup；
- **已完成：**补失败 cleanup、重复 shutdown、release/shutdown race 测试。

### P1 — admission ticket + 真 drain

- **已完成：**`RuntimeLifecycle` 在无 await 临界区原子签发/完成 `OperationTicket`；
- **已完成：**`SHUTTING_DOWN` 关闭 admission 后等待所有已签发 ticket；
- **已完成：**generate/update_weights 在 ticket 生命周期内运行，release 使用临时
  `QUIESCING` barrier；
- **已完成：**覆盖 admission 后 shutdown、cancelled operation、concurrent shutdown、
  shutdown-during-release竞态；
- **已完成：**lazy cold-launch/wake acquire single-flight；candidate restore 前不 publish，
  acquire task持有独立 ticket，因此 waiter cancellation 不会绕过 terminal drain；
- **已完成：**lease weight update 等待正在进行的 acquire/restore，再推进 facade version并
  推送新 state；`state_ref + policy_version` 作为不可变committed snapshot，在active worker
  ACK成功后才发布，失败保留旧snapshot并terminal cleanup；
- **已完成：**`acquire_task` 优先于active-runtime fast path，wake后restore未完成时并发
  generate继续等待同一个readiness task；
- **P2 待完成：**把 immutable fleet、policy version和request identity冻结进 ticket。

### P2 — 单一 WorkerFleetOwner

- 引入 immutable `WorkerFleet` 和内部 `FleetDeployment`；
- **已完成：**删除 outer/inner 两份 `_fleet_generation`；
- launch、executor、weight sync、teardown 共享同一个 fleet object；
- **已完成：**删除 expected-kill record 的 `fleet_generation=` 参数和整个write-only
  registry；P3有真实death classifier后再从fleet派生identity；
- sleep/wake epoch 不变，cold launch epoch +1。

### P3 — typed failure + single-flight recovery

- failure handler 接收内部 `OperationTicket`，不接裸 epoch；
- old-ticket late event 只分类，不触发 active fleet recovery；
- current-ticket worker failure 复用一个 recovery task；
- cluster failure 直接升级 supervisor，不消耗 actor retry budget。

### P4 — transactional restore + deterministic replay

- candidate 全 worker ACK 后才 publish；
- `policy_version` 与 digest 一致才 commit；
- 失败恢复上一个 committed state；
- replay 完整 request/group，保留 request id/seeds/policy version；
- continuous 请求缺历史 policy slot 时走 `StaleSlotDiscard`，不以 latest 偷换。

## 10. 验收标准

### 纯逻辑/CPU

- `QUIESCING`、`SHUTTING_DOWN`、`TERMINATED` 均拒绝 public admission；
- shutdown 与 generate 在 admission 边界交错，已签发 ticket 被等待/取消，未签发请求 fail-fast；
- teardown失败保持 `SHUTTING_DOWN + failure`，root cause不丢；同一次cleanup single-flight，后续明确的
  shutdown 调用可重试仍被 owner 持有的失败 handles；
- 两个并发 failure ticket 只创建一个 candidate fleet；
- outer lease 与 inner executor 不存在独立 generation counter；
- sleep/wake fleet identity/epoch 不变；cold launch/recovery replacement epoch 单调 +1；
- epoch-1 迟到 result/error 在 epoch-2 RUNNING 时不触发 recovery；
- expected kill 匹配 operation id + actor id + fleet epoch，bystander/新 fleet 不匹配；
- `WorkerFleet.epoch` 至少有 stale-event/recovery control-flow consumer，不是 log-only；
- grep production public signatures 为零个 `fleet_generation` 参数。

### isolated real Ray

- 真 actor READY 后，在一个 in-flight request 中确定性 kill current fleet；只重建一次并完整 replay；
- replacement 上线后注入 old fleet 的迟到 ObjectRef failure，不杀 replacement；
- recovery publish 前并发 shutdown，candidate 被清理，最终无 actor/PG 残留；
- terminal shutdown 中 actor death 有 expected-kill record，不触发 recovery；
- timeout/cancel 后 unknown fleet 被 quarantine，不能返回 partial group；
- executor、weight sync、teardown 的观测 epoch 完全一致。

真实 Ray 测试沿父 sprint 纪律：GPU/共享训练空闲，独立 subprocess + local cluster + namespace/temp dir，
只 `ray.shutdown()`，不执行 `ray stop`。

## 11. 架构卫生

### 应改变

- 已删除 outer/inner `_fleet_generation`；P2只在epoch有stale-event行为消费者时，由
  `WorkerFleetOwner` 创建唯一immutable epoch；
- expected-kill registry等真实death classifier落地时再实现，不保留write-only seam；
- `require_running` check-then-act 改成 ticket admission；
- `SHUTTING_DOWN` 从标签升级为真实 drain barrier；
- failure outcome 与 cleanup phase 解耦，不再复制成 `FAILED` enum。

### 保持不变

- `RuntimePhase` 的 ALL_CAPS enum：它是有限进程生命周期协议，属于合法常量边界；
- `lifecycle_fsm.py` 薄文件：它是 public-operation admission/transition 协议边界；
- `GenerationRuntime` facade：跨 backend 一致的 public API 边界；
- `policy_version`：它是模型状态版本，不与 worker fleet epoch 合并；
- request id/seeds：确定性 replay 的业务身份，不与 operation id 合并。

### 合理的新薄对象

- `WorkerFleet`：immutable identity boundary，保存唯一 worker tuple、epoch 和 actor identity；
- `FleetDeployment`：owner 内部的 atomic publish record，把一个 fleet 与引用它的 executor/weight sync
  绑定成 committed unit，但不让 identity object 变成 god object；
- `OperationTicket`：async causal boundary，把提交时的 fleet/policy/request 冻结到完成；
- `WorkerFleetOwner`：只有在它真正拥有 launch/publish/quarantine/recovery single-flight 后才成立。若只包
  一个 counter，禁止新增。

### 非目标

- 不增加用户可调 fleet 参数；
- 不把 phase × epoch 展开成无限状态枚举；
- 不靠 Ray `max_restarts` 替代 policy restore/commit；
- 不在本 sprint 重写 rollout chunk scheduler、算法 staleness policy 或 weight transport data plane；
- 不为减少 LOC 合并 lifecycle/fleet/request 三种不同身份。

## 参考

- `vrl/generation/ray/lifecycle_fsm.py`
- `vrl/generation/ray/runtime.py`
- `vrl/generation/ray/executor.py`
- `vrl/generation/ray/weight_sync.py`
- `tests/generation/ray/test_lifecycle_fsm.py`
- `docs/sprints/SPRINT_ray_rollout_fault_tolerance.md`
- `docs/sprints/done/SPRINT_shadow_model_weight_sync.md`
- Oracle Database quiescing：
  https://docs.oracle.com/database/121/ADMIN/start.htm
- Java `ExecutorService` shutdown/termination：
  https://docs.oracle.com/en/java/javase/25/docs/api/java.base/java/util/concurrent/ExecutorService.html
- Ray actor states（明确不与runtime phase混用）：
  https://docs.ray.io/en/latest/ray-observability/reference/doc/ray.util.state.common.ActorState.html
