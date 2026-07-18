# SPRINT: Ray rollout worker recovery and lifecycle truthfulness

> Lifecycle correction (2026-07-13): `OperationTicket`、runtime-level `QUIESCING`
> 与 public `RECOVERING` phase 都不是当前或目标 contract。当前 ownership 以
> `SPRINT_explicit_rollout_activation.md` 为准：schedule owns admission/drain；runtime
> owns activation/offload、transactional install 与 terminal cleanup；future recovery
> 是 worker-fleet owner 内部的 concrete single-flight task。

状态：**in-progress（2026-07-10）**。三个 run-level recovery gates 已清
（signal ownership `3bb23f52`、atomic checkpoint `97ac2da7`、repo-owned
supervisor `bb282f7a`），实施顺序 1（lifecycle derivation `b9d768af`）、
2（早期状态机骨架 `e6b0dade`）已由 explicit activation 方案收敛：schedule-owned
drain、runtime activation/offload/shutdown single-flight、terminal lifecycle 与
failed-cleanup retry 已落地，并删除了无行为消费者的 generic ticket/active-operation
状态和 expected-kill registry。on-demand transactional weight commit 也已落地：
`state_ref + policy_version` 作为一个 desired snapshot，在远端
update成功后才发布；失败关闭admission并清理未知worker。剩余 3–7
（worker-fleet owner、typed failure/deadline、digest/version-slot commit、
deterministic replay、observability + real-Ray twin）未开始。true-Ray/GPU
验收按本文纪律等待 GPU 空闲。

lazy cold-launch/wake 使用具体的 `activation_task` single-flight：并发 generate
共享同一次 restore，candidate 在 restore 完成前不 publish；即使唯一 waiter 被 cancel，
terminal shutdown仍会等待 activation task完成并接管 cleanup。startup probe、release-policy、
sleep/wake 的同步 `ray.get` 已移出 event-loop thread；launcher 保留 main-thread Ray init，
已连接后的 actor startup/policy load 通过 async adapter移出 event-loop thread。底层 thread
不被 caller cancellation 强杀，而是由 runtime-owned activation task 保证晚归 candidate
仍被接管。

> 步骤 3–6 的 identity/ownership 细化见
> [`SPRINT_runtime_time_machine_without_fleet_args.md`](./SPRINT_runtime_time_machine_without_fleet_args.md)：
> 裸 `_fleet_generation`、`fleet_generation=` 参数和 write-only expected-kill registry
> 已删除。真正实现 stale-event recovery 时，epoch只能作为 immutable `WorkerFleet`
> 的行为字段，由 fleet owner 在 dispatch/result/recovery publish 时自动检查；不得重新
> 暴露成每层手传参数，也不为它恢复 generic `OperationTicket`。

2026-07-13 native-engine audit 再次确认“剩余 3–7 未完成”是代码事实，不只是计划标签：

- `run_actor_jobs()` 在一个 ObjectRef 失败后取消 asyncio waiter，但明确不取消底层 Ray work；
- worker update ACK 当前只返回整数 policy version，没有 key schema/state digest；
- terminal `RuntimePhase` 只有 `RUNNING/SHUTTING_DOWN/TERMINATED`，当前没有 fleet rebuild
  或 whole-request replay consumer。

因此已完成的 lifecycle truthfulness 不能被描述成 automatic recovery 已存在。

前置：

- [`SPRINT_ray_cluster_ownership_and_shared_host_isolation.md`](./done/SPRINT_ray_cluster_ownership_and_shared_host_isolation.md)
  已完成显式 cluster 选择和有序 teardown。
- `ray.init()` 覆盖 CLI `SIGTERM` handler 的 ownership 缺口已经通过 init 前后
  snapshot/restore 修正；fallback checkpoint 也必须保持原子发布、可验证。actor
  recovery 只能处理 worker-specific failure，不能代替 driver/head failure 后的
  checkpoint resume。
- 当前 `~/.local/bin/run-until-success` 不是仓库资产，也没有可靠的 child process
  group stop/restart contract。actor retry 耗尽后的 run-level fallback 必须由可测试的
  repo-owned supervisor 接管，不能依赖该脚本按退出码猜测故障原因。

## 结论

保留 Ray 作为 rollout 的进程隔离和远程执行后端，但不把“频繁 kill/relaunch”
当作默认优化策略。

本 sprint 的目标不是把所有 `ActorDiedError` 都重试三次，而是建立一个有明确
行为边界的 worker-fleet lifecycle：

1. 正常 GPU 阶段交接优先 `resident` 或 `sleep/wake`；
2. 只有无法安全 sleep、fatal worker quarantine、或最终 shutdown 才
   `kill/relaunch`；当前 canonical handoff没有 normal teardown producer；
3. worker-specific death 在 deadline 和 retry budget 内重建 fleet、恢复最后一个
   committed policy version，并重放整个 generation request；
4. raylet/GCS/head failure 不做 actor-level 空转重试，立即升级为 cluster failure，
   交给进程 supervisor 从完整 checkpoint 恢复。

“actor death 从整个 run 报废降级为丢一个 chunk”不是安全边界。当前
`run_actor_jobs()` 在一个 ObjectRef 失败时不会返回可靠的 partial completion，其他
actor 上的底层任务也不保证被取消。正确的恢复边界是：**丢弃并确定性重放当前
generation request/group**，而不是拼接一次未知部分完成的 chunk 集合。

## 现场证据：Ray 到底慢在哪里

2026-07-10 Cosmos 单卡长跑日志给出以下实测：

| 阶段 | 观测值 | 判断 |
|---|---:|---|
| 本地 cluster startup + connect + PG probe | 约 2.4 秒 | 每个 run 一次，不是 epoch 热点 |
| Ray actor 创建到 worker 开始 load | 约 0.1 秒 | actor 调度/进程创建本身很小 |
| Cosmos generation policy load | 5.7–6.6 秒 | 反复 relaunch 的主要成本 |
| 收到 `ray.kill` 到 worker disconnect | 5–7 毫秒 | force-kill 本身很快 |
| 一次 16-sample Cosmos rollout | 约 1,720 秒 | 本次模型上 reload 约占 0.4% |
| SD3.5 CuMem sleep+wake GPU probe | 845 毫秒 | 比 5,425 ms cold reload 快 6.4× |

样本来源：

- cluster：`13:59:34.283 after_trainer_bundle_build` → `13:59:36.730 PG created`；
- policy load：`09:16:29.163→09:16:35.538`、
  `09:31:07.151→09:31:13.400`、`11:25:04.149→11:25:10.479`、
  `13:18:57.460→13:19:04.037`，以及新 attempt 的
  `13:59:37.717→13:59:43.449`；
- kill：多个 worker 的 `Force kill actor request` 到
  `RayletIpcClient::Disconnect` 为 5–7 ms；
- rollout：`generation wall ... chunks=16 wall_s=1719.601`。
- 历史 Janus 日志分别出现 16、250、53、144、960 次 `before_load_policy`；典型
  cold load 约 5 秒。250/960 次的累计 reload 约为 21/79 分钟，说明短 rollout 或
  高频 phase handoff 下 churn 会成为真实成本；
- 已完成的 SD3.5-medium GPU probe 中，CuMem sleep+wake 为 845 ms，cold reload
  为 5,425 ms；sleep 后只残留约 1.3 GB CUDA context，wake 碎片约 +28 MB。

现有日志能精确量化收到 force-kill 后的 Ray worker 退出，但
`release_policy.remote()` 的模型清理没有独立 phase log，不能把 5–7 ms 误称为完整
release+kill latency。本 sprint 要补齐 `release_policy`、kill、launch、load、sleep、
wake 的分别计时。

这些数字不能外推到短 image rollout：当一次请求只有几秒时，6 秒模型 reload 会
成为显著损耗。即使在 Cosmos 上时间占比不高，频繁重载仍增加模型加载失败、host
memory 波动和“正常 DEAD 被误判为事故”的观察噪声。因此原则是：**不因当前 kill
很快就鼓励 churn；只在 correctness 需要时 kill。**

## 生命周期策略

### 稳态选择

按以下顺序选择：

1. **Dedicated rollout GPU：resident。** worker 和 policy 常驻，不做阶段交接。
2. **Colocated、worker 支持可靠显存 sleep：sleep/wake。** actor 和 placement
   bundle 保留，CUDA physical pages 在 trainer/reward 阶段释放。
3. **未来若 backend 无法安全 sleep 或另一个 Ray role 必须取得 logical bundle：
   显式解析 release strategy。** 当前没有这个生产者，不保留隐式 teardown fallback。
4. **Fatal CUDA / actor unknown state：kill/quarantine + rebuild。** 不复用可能损坏
   的 CUDA context。
5. **Run shutdown：drain 后 kill。** 不做 recovery。

当前 reward 已经是 driver 内的 `InProcessRewardRuntime`，不是 Ray reward actor。
generation actor sleep 后虽然仍持有 Ray logical bundle，但 inline reward 不通过 Ray
申请该 token；因此 canonical `on_demand` rollout统一使用sleep/wake。重复派生的
`sleep_eligible`和missing-topology teardown fallback已删除；没有resolved lifecycle plan
时launcher保持resident，不再由private factory暗中制造另一种on-demand state。

`offload()` 读取 `state.workers_offloaded` 做幂等保护。schedule `finally` 与 terminal
cleanup 即使相邻触达，已 offload 的 worker 也不得再次 sleep。相同地，只有
`workers_offloaded=True` 才需要 wake；成功 restore 并确认 desired policy 后，再提交
`workers_offloaded=False`。

### 当前行为状态机

terminal runtime 只保存被行为消费的状态：

```text
RUNNING -> SHUTTING_DOWN -> TERMINATED
```

`failure: BaseException | None` 与 phase 正交：运行或cleanup failure保存第一个
root cause并推进到 `SHUTTING_DOWN`；cleanup失败继续停在 `SHUTTING_DOWN + failure`，后续
shutdown重试，成功后才进入 `TERMINATED`。不需要重复表达同一事实的 `FAILED` phase。

状态消费者：

- `RUNNING`：runtime terminal admission 仍开放；strict/continuous schedule 在调用
  weight sync、offload 或 shutdown 前负责停止 producer 并 drain；
- `SHUTTING_DOWN`：拒绝新 request；actor death 作为 expected termination，不重拉；
- `TERMINATED`：所有 public runtime operation fail-fast；

`QUIESCING` 属于 schedule 的 pause/drain 行为，不是 runtime phase；`STARTING` 由具体
`activation_task` readiness barrier 表达。future recovery 也不预留 public
`RECOVERING` enum：只有 worker-fleet owner 与 shared recovery task 真正实现后，才增加其
内部状态，而且必须被 stale-result rejection、candidate publish 或 shutdown cleanup 消费。

### 当前 on-demand ownership record

`_OnDemandRuntimeState` 只保存跨 `await` 仍必须存在的启动输入、资源所有权和 policy
提交点：

```python
@dataclass(slots=True)
class _OnDemandRuntimeState:
    config: RayGenerationConfig
    launch_inputs: RayGenerationLaunchInputs
    placement: RolePlacement
    inner_runtime: RayGenerationRuntime | None = None
    activation_task: asyncio.Task[RayGenerationRuntime] | None = None
    desired_policy: _PolicySnapshot | None = None
    active_policy_version: int | None = None
    workers_offloaded: bool = False
```

- `config`、`launch_inputs`、`placement` 是首次lazy launch失败后仍可重试的唯一启动输入；
- `inner_runtime` 是当前actors的所有权，不是fleet generation；
- `activation_task` 同时提供single-flight和readiness barrier，不能由
  `workers_offloaded`替代；
- `desired_policy` 原子绑定state ref与version；active fleet 完成 ACK 后才推进；
- `active_policy_version` 记录当前 fleet 已确认安装的 version，不重复保存 state payload；
- `workers_offloaded` 只描述已有worker是否已经让出physical GPU。

已删除的`last_state`、`sleep_eligible`和`fleet_generation`都不是独立真相。同步能力也不存进
on-demand state：outer runtime的`supports_weight_sync`从`config.sync_trainable_state`派生，resident
runtime则从真实`GenerationWeightSync`派生。这样lazy launch前训练器仍能创建weight syncer，
又不需要`weight_sync=object()`哨兵或新的布尔状态字段。

未来 recovery 必须是 single-flight：同一 immutable fleet identity 上的多个并发 request
同时观察 actor death 时，共享一个 runtime-owned recovery task，不得各自创建 replacement
fleet。shutdown 先把 terminal lifecycle 推进到 `SHUTTING_DOWN`；recovery task 在 publish
candidate 前重新检查 terminal phase 与 fleet identity，若已 shutdown/stale 就清理 candidate，
避免 teardown 后 actor “复活”。这些字段由 fleet owner 持有，不放成 public lifecycle enum
或模糊的 online recipe log field。

这些 ALL_CAPS terminal phase 是进程生命周期协议，属于合理常量边界。每个状态必须至少有一个
非日志行为消费者。

仅凭 `SHUTTING_DOWN` 不能证明具体是谁杀了 actor。它只能证明 driver 已经表达停止意图。
expected termination 还必须关联：

- actor handle/actor id；当前不记录无行为消费者的裸 epoch；
- driver 发起 kill 的 timestamp 和 reason；
- 对应 teardown/recovery operation id。

## 故障分类和升级

### Worker-specific，可在进程内恢复

- `ActorDiedError` / `RayActorError`，且 cluster health probe 成功；
- `ActorUnavailableError` 先进入有界 quarantine/ping；只有确认 actor death 后才
  rebuild，暂时不可达不等于死亡；
- worker process crash/segfault；
- fatal CUDA error，需要 quarantine 该 worker/fleet；
- generation deadline 超时，底层 actor task 无法可靠取消。

恢复流程：

```text
freeze admission
-> cancel pending ObjectRefs best-effort
-> quarantine/kill unknown fleet
-> launch a new fleet generation
-> load base policy
-> restore last committed policy state/version
-> verify worker version/digest ACKs
-> replay the full request with the same request id/seeds
-> return to RUNNING
```

### Cluster-specific，必须升级

- GCS/head/raylet unavailable；
- placement group 已失效；
- attach connection lost and health probe cannot complete；
- node loss导致当前 topology 无法满足 bundles。

这些错误不消耗 actor retry budget。抛出 typed cluster failure，由 supervisor 结束
attempt 并从 latest complete checkpoint 恢复。Ray 官方语义中 head failure 会导致
整个 cluster failure，`max_restarts` 无法跨过这个边界。

### 非重试错误

- user/config/data validation error；
- policy-version contract violation；
- deterministic model build failure；
- `RayTaskError` 或 worker 返回的 user/data application error；
- 相同 request 在新 fleet 上重复触发同一 fatal application error。

## Deadline 和 retry policy

只捕获 `RayActorError` 不足以解决“actor 不死但永久卡住”。统一的 typed reliability
config 应提供：

- actor startup/load deadline；
- generation request deadline；
- continuous drain deadline；
- weight-sync deadline；
- fleet recovery deadline；
- per-request recovery budget；
- per-run recovery circuit breaker。

不要把这些值散成多个 module-level ALL_CAPS timeout；它们应来自一个 typed config
source of truth，并且每个字段必须传到真实 Ray wait/control-flow consumer。

默认 policy：

- 同一 stable failure category 的 worker-specific failure 最多 3 次；
- category 使用 typed enum/class，不用包含 PID/actor id 的完整 message 当 key；
- 每次 recovery 使用递增 fleet generation 和有界 backoff；
- 同时设置 per-request 总 retry、per-run fleet restart 和 wall-clock deadline，避免
  两类错误交替出现时各自永远达不到“同因三次”；
- cluster failure、config error、data error不做三次 actor retry；
- 超预算时异常携带所有 attempt、actor id、node id、policy version、request id 和
  death records。

`ray.cancel` 只做 best-effort。无法确认运行中 actor task 已停止时，必须 kill/rebuild
actor，不能继续向未知状态 CUDA context 发请求。

actor 创建必须显式设置 Ray native `max_restarts=0`、`max_task_retries=0`，证明应用层
worker-fleet owner 是唯一 recovery owner，避免 Ray at-least-once task retry 与应用
request replay 叠加。

## Transactional weight restore

此前 on-demand facade 在远端 update 成功前写入：

```python
state.last_state = state_ref
self.current_policy_version = policy_version
await runtime.update_weights(...)
```

这会让部分失败的版本成为下一次replay source，已经修复。当前 on-demand state 使用不可变
`_PolicySnapshot(state_ref, policy_version)`：active worker全部update返回后才推进
`desired_policy`、`active_policy_version`和`current_policy_version`；失败保留上一snapshot、
关闭terminal admission并cleanup未知worker。worker parked或尚未lazy launch时可以直接接收
snapshot，因为下一次activation必须restore成功后才对generate可见。`activation_task`先于active-runtime fast path，
所以wake已完成但restore仍在进行时，第二个generate也不能穿透readiness barrier。

recovery层仍需继续完成：

1. versioned-slot backend 把 candidate state 安装到 inactive slot；single-slot backend
   必须先 drain admission，再原地安装；
2. 每个 worker 返回 `policy_version + digest` ACK；
3. digest/version全部一致后才允许recovery candidate publish；
4. 任一失败则 quarantine fleet，恢复上一个 committed version；
5. cold-launch/recovery candidate 只有在 load + restore 全成功后才写入 active fleet。

continuous/non-draining 还必须处理旧 request version：若 in-flight request 是 v1，
trainer 已 committed v2，而 replacement fleet 只能恢复当前 v2，则不得拿 v2 重放 v1
并伪装正确。首版选择：若所需 version 不在有界 committed-state history 中，抛
`StaleSlotDiscard`，由 producer 使用当前 policy 重新采样；strict-on-policy 的 request
必须恢复其唯一 committed version 后才能 replay。

不能只设置 Ray `max_restarts`。自动 actor restart 只重跑 constructor，不会自动调用
`load_policy()` 或恢复 version slots；自动 task retry还是 at-least-once，无法替代上述
应用层 commit protocol。

## Observability

每个 log/death record 包含：

- stable `run_id`、`attempt_id`；
- `worker_id`；P2 的 immutable `WorkerFleet` 真正参与 stale guard 后再记录其 epoch；
- Ray `actor_id`、worker PID、node id/IP；
- lifecycle phase；
- request id、policy version；
- expected kill operation id/reason，或 unexpected death category；
- Ray death context、exit type/code、exception class和 traceback；
- recovery attempt/budget 和最终 verdict。

Ray actor name 可作为可读标签，但不是 recovery identity，也不能识别发送 SIGTERM 的
进程。若使用 named actor，名称必须包含 run id + fleet generation，避免旧 DEAD actor
注册尚未清理时阻塞重建。`ray.util.state.list_actors` 只用于 best-effort postmortem，
不得位于恢复关键路径。

## 实施顺序

1. **已完成：**inline reward/trainer handoff统一sleep，不再每轮teardown；删除无生产者的
   `sleep_eligible`和missing-topology fallback。
2. **已完成：**explicit activation + schedule-owned admission/drain；runtime terminal
   lifecycle 只保留 `RUNNING/SHUTTING_DOWN/TERMINATED`，具体 activation/offload/shutdown
   task 各自 single-flight。expected-kill record随步骤4的真实death classifier一起实现，
   不提前保留write-only registry。
3. 把 resident 与 on-demand launch 统一到可重建的 worker-fleet owner；immutable fleet
   identity 由 dispatch/result/recovery publish 自动检查，不恢复 generic ticket 参数链。
4. 引入 typed Ray failure/deadline 和 ObjectRef cancellation/quarantine。
5. **部分完成：**on-demand snapshot在worker ACK后提交，失败terminal cleanup；fleet recovery
   的digest/version-slot transaction仍待实现。
6. 实现 full-request deterministic replay 和 retry circuit breaker。
7. 加 observability 和 isolated real-Ray twin。

## 验收标准

### Fast CPU deterministic，当前即可运行

- lifecycle matrix 证明 dedicated→resident、inline colocated→sleep、fatal→quarantine；
- 同一 phase 连续两次 offload 只 sleep 一次，wake 只在workers-offloaded state执行；
- 正常 `SHUTTING_DOWN` kill 不触发 recovery，且有 matching operation id；
- `RUNNING` actor death 触发一个 internal shared recovery task，重建后恢复最后 committed
  version；不要求新增 public `RECOVERING` phase；
- 两个并发 request 同时发现同一 actor death 只创建一个 replacement fleet；
- recovery task 运行中 shutdown 最终进入 `TERMINATED`，没有 candidate actor 残留；
- `TERMINATED` 后 generate/update_weights fail-fast，重复 shutdown 幂等；
- partial weight update 不推进 committed version；
- request replay 保持相同 request id、sample seeds、policy version和完整 group；
- worker-specific 同因 3 次耗尽 budget；cluster failure 零次 actor retry；
- continuous 旧 version request 在无对应 committed history 时走 `StaleSlotDiscard`，
  不使用 latest weights 偷换版本；
- never-resolving startup/generation/drain/weight-sync refs 都在 deadline 内结束；
- timeout 后剩余 refs 被 best-effort cancel，未知 actor 被 quarantine；
- terminal failure 会唤醒 continuous consumer并保留 root cause；cleanup期间保持
  `SHUTTING_DOWN`，成功后进入 `TERMINATED`；
- state、deadline 和 provenance fields 都有非日志生产消费者；
- 允许 controlled clock、awaitable refs、fake inner runtime 等 deterministic protocol
  doubles 验证 admission、调用顺序和错误注入；不得用 fake 证明 Ray kill/cancel/
  exception semantics，这些必须有 isolated real-Ray twin；
- resident与on-demand sleep state；round-robin、dynamic、single-worker
  pipelined path 都覆盖；OOM split 行为保持不变；
- executor、weight sync、runtime ownership 和 shutdown 观察到同一个 immutable fleet
  identity；该 identity 不是每层手传的裸整数参数。

### Isolated real Ray，GPU 空闲后运行

- 必须在当前 GPU训练结束后，或独立 Unix user/container/CI 中运行；共享主机上的活跃
  训练期间禁止启动这个 true-Ray test；
- 每个 case 放独立 subprocess，使用
  `ray.init(address="local", num_cpus=4, include_dashboard=False,
  _temp_dir=<per-test tmp>, namespace=<unique run id>)`；`try/finally` 中只调用
  `ray.shutdown()`，绝不调用 `ray stop`。temp/namespace 用于测试日志和名字隔离，不
  声称能隔离 host process kill；
- actor 必须先发 READY，再由 test 确定性 `ray.kill(..., no_restart=True)`；真实测试
  必须穿过 production recovery seam，不能只测 toy controller；
- CPU actor 在 request 中被 `ray.kill`：fleet 重建，完整 request replay成功；
- kill 记录被分类为 expected/unexpected正确；
- actor hang 触发 deadline 和 quarantine；
- local raylet/head failure 被分类为 cluster failure，不循环重建 actor；
- 测试结束后无 Ray child、actor 或 placement group 残留。

### GPU gate，资源可用后单独运行

- sleep 后 generation worker 的 physical GPU memory 确实释放，inline reward/trainer
  能安全接管；
- wake 后 policy digest/version不变；
- fatal CUDA fault 使用隔离进程故障注入，不在共享训练 GPU 上执行；
- 对比 resident、sleep/wake、kill/relaunch 的 phase latency 和 peak memory，确认默认
  policy，而不是凭单个 Cosmos 数字推广。

## 应保持不变

- `Runtime -> Executor -> Ray actor adapter -> WorkerCore`：真实协议/进程边界；
- `RayGenerationLauncher`：standalone public launch boundary；
- `vrl/ray/lifecycle.py`：统一 owned-handle cleanup adapter；
- run-level placement owner 和 actor GPU validation；
- generation/reward/build 的阶段边界；
- `_RAY_ADDRESS_ENV`、checkpoint filenames、lifecycle enum values 等真实协议常量。

薄函数/文件不因本 sprint 追求 LOC 而压平。worker-fleet owner 只有在统一 resident 与
on-demand recovery、移除真实重复复杂度时才成立，不能新增装饰性的 manager/handler。

## 非目标

- 不声称 actor 命名能找出发送 SIGTERM 的 OS PID；
- 不用 namespace/temp dir 伪装进程或资源隔离；
- 不恢复 trainer/FSDP rank；训练进程故障仍走 checkpoint；
- 不在本 sprint 构建多进程 controller 或弹性扩缩容；
- 不自动重试 config/data/deterministic application errors；
- 不把自动 `max_restarts/max_task_retries` 当成应用状态恢复；
- 不在当前 GPU被占用时运行 true-Ray/GPU fault injection。

## 参考

- Ray runtime/on-demand state：`vrl/generation/ray/runtime.py`
- actor dispatch：`vrl/ray/actor_pool.py`
- actor construction：`vrl/ray/actor_group.py`
- worker adapter：`vrl/generation/ray/worker.py`
- weight sync：`vrl/generation/ray/weight_sync.py`
- continuous producer：`vrl/rollouts/orchestration/continuous/producer.py`
- teardown：`vrl/scripts/common/online.py:_shutdown_online_recipe_runtime`
- sleep/wake GPU evidence：`docs/sprints/done/SPRINT_frozen_component_preservation.md`
- repeated cold-load evidence：`outputs/janus_smoke/{run,baseline,aesthetic,aesthetic_rbs24,aesthetic_kllow}.log`
- cosmos-rl explicit StopCommand/JobPhase：
  https://github.com/nvidia-cosmos/cosmos-rl/pull/696
- Ray actor fault tolerance：
  https://docs.ray.io/en/latest/ray-core/fault_tolerance/actors.html
- Ray cancellation：
  https://docs.ray.io/en/latest/ray-core/api/doc/ray.cancel.html
- Ray node failure：
  https://docs.ray.io/en/latest/ray-core/fault_tolerance/nodes.html
- Oracle Database quiescing：
  https://docs.oracle.com/database/121/ADMIN/start.htm
- Java `ExecutorService` shutdown/termination：
  https://docs.oracle.com/en/java/javase/25/docs/api/java.base/java/util/concurrent/ExecutorService.html
