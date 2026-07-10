# SPRINT: Ray rollout worker recovery and lifecycle truthfulness

状态：**in-progress（2026-07-10）**。三个 run-level recovery gates 已清
（signal ownership `3bb23f52`、atomic checkpoint `97ac2da7`、repo-owned
supervisor `bb282f7a`），实施顺序 1（lifecycle derivation `b9d768af`）、
2（状态机 + expected-kill records `e6b0dade`）已落地；剩余 3–7
（typed failure/deadline、worker-fleet owner、transactional weight commit、
deterministic replay、observability + real-Ray twin）未开始。true-Ray/GPU
验收按本文纪律等待 GPU 空闲。

前置：

- [`SPRINT_ray_cluster_ownership_and_shared_host_isolation.md`](../done/SPRINT_ray_cluster_ownership_and_shared_host_isolation.md)
  已完成显式 cluster 选择和有序 teardown。
- 在真实长跑验收前，必须先修正 `ray.init()` 覆盖 CLI `SIGTERM` handler 的
  ownership 缺口，并保证 fallback checkpoint 是原子发布、可验证的。actor
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
   `kill/relaunch`；
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
3. **Colocated、sleep 不可用或真实另一个 Ray role 必须取得 bundle：
   kill/relaunch。** 这是兼容性 fallback，不是默认吞吐策略。
4. **Fatal CUDA / actor unknown state：kill/quarantine + rebuild。** 不复用可能损坏
   的 CUDA context。
5. **Run shutdown：drain 后 kill。** 不做 recovery。

当前 reward 已经是 driver 内的 `LocalRewardRuntime`，不是 Ray reward actor。
generation actor sleep 后虽然仍持有 Ray logical bundle，但 inline reward 不通过 Ray
申请该 token；只要 sleep 已证明真正释放 physical GPU memory，保留 bundle 不会阻止
reward/trainer 使用该卡。因此现有：

```python
sleep_eligible = release_before_train and not release_before_reward
```

包含了已删除 remote reward actor 时代的假设，应在本 sprint 中重新推导。目标是
trainer/reward 都是 inline handoff 时优先 sleep/wake；只有存在真实 Ray bundle
handoff 或 backend 无法安全 sleep 时才 teardown。

切换到 sleep 后，`release()` 必须读取 `state.asleep` 做幂等保护。collector 的
reward 前 release 与 schedule `finally` 可能在同一 phase 重复触达 release；第二次
release 不得再次 sleep。相同地，只有 `asleep=True` 才允许 wake，成功后再提交
`asleep=False`。

### 行为状态机

使用行为消费的状态，而不是 log-only 标签：

```text
STARTING -> RUNNING
RUNNING  -> RECOVERING -> RUNNING
RUNNING  -> DRAINING   -> STOPPED
STARTING / RUNNING / RECOVERING / DRAINING -> FAILED
```

状态消费者：

- `RUNNING`：允许提交 request；worker-specific failure 可进入 recovery；
- `RECOVERING`：暂停新 admission/weight sync，只允许 fleet rebuild；
- `DRAINING`：拒绝新 request；actor death 作为 expected termination，不重拉；
- `STOPPED`：所有 public runtime operation fail-fast；
- `FAILED`：保留 root-cause chain，后续调用只重抛 terminal failure。

recovery 必须是 single-flight：同一 fleet generation 上的多个并发 request 同时观察
actor death 时，共享一个 recovery task，不得各自创建 replacement fleet。shutdown
在 `RECOVERING` 中到达时，先把目标状态推进到 `DRAINING`，使 recovery 在 publish 新
fleet 前中止并清理 candidate，避免 teardown 后 actor “复活”。状态机和 transition
由 runtime/fleet owner 持有，不放成模糊的 online recipe log field。

这些 ALL_CAPS 值是进程生命周期协议，属于合理常量边界。每个状态必须至少有一个
非日志行为消费者。

仅凭 `DRAINING` 不能证明具体是谁杀了 actor。它只能证明 driver 已经表达停止意图。
expected termination 还必须关联：

- fleet generation；
- actor handle/actor id；
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

现有 lease 在远端 update 成功前写入：

```python
state.last_state = state_ref
self.current_policy_version = policy_version
await runtime.update_weights(...)
```

这不是 committed state。部分 worker 成功、部分失败时会形成 split-brain，recovery
还会回放一个从未完整提交的版本。

本 sprint 必须改为：

1. workers 把 candidate state 安装到 inactive slot；
2. 每个 worker 返回 `policy_version + digest` ACK；
3. 全部 ACK 一致后，driver 才推进 `last_committed_state` 和
   `current_policy_version`；
4. 任一失败则 quarantine fleet，恢复上一个 committed version；
5. wake/cold-launch candidate 只有在 load + restore 全成功后才写入 lease runtime
   state。

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
- `fleet_generation`、`worker_id`；
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

1. 修正 lifecycle derivation：inline reward handoff 能 sleep 时不再每轮 teardown；为
   resident/sleep/teardown decision 增加纯逻辑测试。
2. 引入状态机和 expected-kill operation record，但暂不自动重试。
3. 引入 typed Ray failure/deadline 和 ObjectRef cancellation/quarantine。
4. 把 resident 与 lease launch 统一到可重建的 worker-fleet owner。
5. 修正 transactional weight commit/restore。
6. 实现 full-request deterministic replay 和 retry circuit breaker。
7. 加 observability 和 isolated real-Ray twin。

## 验收标准

### CPU/fake，当前即可运行

- lifecycle matrix 证明 dedicated→resident、inline colocated→sleep、unsupported→
  teardown、fatal→quarantine；
- 同一 phase 连续两次 release 只 sleep 一次，wake 只在 asleep state 执行；
- 正常 `DRAINING` kill 不触发 recovery，且有 matching operation id；
- `RUNNING` actor death 进入 `RECOVERING`，重建后恢复最后 committed version；
- 两个并发 request 同时发现同一 actor death 只创建一个 replacement fleet；
- recovery 中 shutdown 最终进入 `STOPPED`，没有 candidate actor 残留；
- `STOPPED` 后 generate/update_weights fail-fast，重复 shutdown 幂等；
- partial weight update 不推进 committed version；
- request replay 保持相同 request id、sample seeds、policy version和完整 group；
- worker-specific 同因 3 次耗尽 budget；cluster failure 零次 actor retry；
- continuous 旧 version request 在无对应 committed history 时走 `StaleSlotDiscard`，
  不使用 latest weights 偷换版本；
- never-resolving startup/generation/drain/weight-sync refs 都在 deadline 内结束；
- timeout 后剩余 refs 被 best-effort cancel，未知 actor 被 quarantine；
- terminal `FAILED` 会唤醒 continuous consumer并保留 root cause；
- state、deadline 和 provenance fields 都有非日志生产消费者；
- fake Ray fixture 复用共享 protocol fake，不新增第四份复制体。
- resident、on-demand teardown lease、sleep lease；round-robin、dynamic、single-worker
  pipelined path 都覆盖；OOM split 行为保持不变；
- executor、weight sync、runtime ownership 和 shutdown 观察到同一个 fleet generation。

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
lease recovery、移除真实重复复杂度时才成立，不能新增装饰性的 manager/handler。

## 非目标

- 不声称 actor 命名能找出发送 SIGTERM 的 OS PID；
- 不用 namespace/temp dir 伪装进程或资源隔离；
- 不恢复 trainer/FSDP rank；训练进程故障仍走 checkpoint；
- 不在本 sprint 构建多进程 controller 或弹性扩缩容；
- 不自动重试 config/data/deterministic application errors；
- 不把自动 `max_restarts/max_task_retries` 当成应用状态恢复；
- 不在当前 GPU被占用时运行 true-Ray/GPU fault injection。

## 参考

- Ray runtime/lease：`vrl/generation/ray/runtime.py`
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
