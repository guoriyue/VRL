# INFO：Ray generation engine 当前 ownership 与调用链

状态：**verified against source（2026-07-30）**。

本文只描述当前生产路径。已删除的 `_RuntimeLease`、`RayGenerationWorkerFleet`、
physical-stage adapter、release-per-collect 和带 `stage` 参数的旧调用链不再作为 dormant
capability 保留。

## 1. Ownership 图

```text
Rollout schedule                         admission + normal drain owner
  -> GenerationRuntime                   collector-facing protocol
     -> RayGenerationRuntime              sole lifecycle/policy/phase owner
        -> RayGenerationSession           optional launched resource owner
           -> RayGenerationExecutor       driver plan/dispatch/OOM/gather
           -> optional weight sync
           -> RayGenerationWorker         thin Ray actor adapter
              -> GenerationWorkerCore     build/version/parking/forward owner
                 -> GenerationChunkExecutor
                    -> diffusion or AR family executor
                       -> RuntimeModel / upstream transformer and kernels
```

driver 侧包含 schedule、runtime 和 executor。actor 侧包含 Ray worker、worker core、family
executor 和 model。`RayGenerationRuntime` 是唯一结构化满足 `GenerationRuntime` 的 Ray
实现，也是唯一 `RuntimeLifecycle`、health monitor、activation/offload task、pending policy
与 terminal failure owner。`RayGenerationSession` 只保留一次 launch 产生的 actor handles、
executor、optional weight sync，并实现 sleep/wake/close；它不满足 public runtime protocol，
也没有自己的 lifecycle。dedicated-GPU runtime eager 持有 session，shared-GPU runtime
在 schedule 完成 trainer GPU handoff 后由同一个 runtime 延迟创建 session。仓库里没有
inner runtime、独立 `WorkerFleet` 或 shared runtime base/manager。

wm-infra 原生拥有 lifecycle、policy version、chunk、trajectory 与 replay 语义。底层
transformer/block/kernel 仍可来自 Diffusers、PyTorch 或未来 provider；“native control
plane”不等于“全栈 native model forward”。

## 2. Public runtime contract

`GenerationRuntime` 声明：

```text
current_policy_version
requires_driver_model_offload
activate()
generate(request)
offload()
shutdown()
is_colocated()
```

weight sync 不是所有 runtime 的公共必选方法。训练侧通过
`build_runtime_weight_syncer` 检查 concrete runtime 的 `supports_weight_sync` 和
`update_weights`，再构造独立 control path。

schedule 是 pause/admission/drain 的唯一 owner。runtime 的 `RuntimeLifecycle` 只表达：

```text
RUNNING -> SHUTTING_DOWN -> TERMINATED
```

它不表达 normal drain、GPU residency 或 speculative recovery。

## 3. Composition 与 launch

### 3.1 Source of truth

```text
online recipe
  -> resolve_online_run(...)
  -> resolve_model(..., for_rollout=False)
  -> resolved_run.ray_launch_inputs(replay_model)
     -> ModelFamilyEntry.resolve_model_build(..., for_rollout=True)
     -> ModelFamilyEntry.executor_kwargs(root)
  -> GenerationRuntimeLaunchContract(
       family,
       primitive model_build/executor_kwargs,
       expected_model_identity,
       policy_version,
       profiler config,
       sleep_offload,
       versioned_weight_sync,
     )
  -> RayGenerationLaunchInputs(contract, entry.new_gatherer())
```

`GenerationRuntimeLaunchContract` 不携带 live model、pipeline、callable 或 executor class。
family registry 是 model builder、executor class、gatherer 和 static capability 的唯一
taxonomy source。launch contract 最终只携带 primitive `versioned_weight_sync` fact，
这个 fact 由 `vrl/run.py` 从已解析的 typed trainer schedule、generation config 与
rollout build 一次派生。`RayGenerationLauncher.create_runtime` 只消费 typed
`RayGenerationConfig`、`RayGenerationLaunchInputs` 与 placement，不读取 trainer/model
YAML。它总是返回一个 `RayGenerationRuntime`，resolved
`resources.lifecycle.rollout.mode` 只决定 session 是 eager launch 还是由 activation
延迟创建；旧 free launch-input resolver 与 `launch_from_cfg` layering leak 已删除。

### 3.2 Session construction and eager launch

```text
RayGenerationLauncher.create_runtime(...)
  -> RayGenerationLauncher._launch_session(...)
     -> RayActorGroup.launch(
       startup_method="load_policy",
       worker_rpc_timeout_s=...,
     )
     -> validate worker metadata / GPU assignment
     -> DistributedWorkerHandle[]
     -> one fleet-owned RayActorDispatcher(workers)
     -> RayGenerationExecutor(
       planner,
       workers,
       registry-owned gatherer,
       actor_dispatcher,
       generation_stall_timeout_s=...,
     )
     -> optional RayGenerationWeightSync(
       workers,
       same actor_dispatcher,
       worker_rpc_timeout_s=...,
     )
     -> query all-worker version-slot capability
     -> RayGenerationSession(executor, weight_sync, workers)
  -> RayGenerationRuntime(
       session=session,
       health settings,
     )
  -> start health monitor
```

startup load、metadata 和 capability 各自有 fresh control-plane deadline。任一步失败，
launcher 清理 candidate actors；placement group 仍由 `GlobalRayPlacementOwner` 管理。
`_launch_session` 保留为 private resource-construction seam，是因为 eager creation 与
deferred factory 必须复用完全相同的 actor/executor/weight-sync 组装顺序；它不是第二个
public build API。

### 3.3 Deferred session launch

`RayGenerationLauncher.create_runtime(...)` 在 resolved rollout lifecycle 为
`on_demand` 时不 launch actors，而是把 bound `_launch_session_async` 作为 session factory
交给同一个 `RayGenerationRuntime`。bound factory 保留当前 launcher 的 Ray
address/init policy、typed config、launch inputs 与 placement，不会另建默认 launcher。
runtime 自己持有：

```text
session / deferred session_factory
activation_task / offload_task / shutdown_task
current_policy_version / _installed_policy_version
_pending_install
_session_parked
```

第一次 `activate()` 通过 single-flight `activation_task` 调用
session factory，得到一个拥有 workers/executor/weight sync 但没有 lifecycle 的
`RayGenerationSession`。后续 activation 唤醒同一个 parked session。

pending policy install 由 runtime 调用 active/candidate session 的 typed
`update_weights` resource operation，并在自己的 publication guard 内推进版本。这样 active
session 和未发布 cold candidate 的 weight ACK timeout 都进入同一个 runtime failure
boundary，没有 inner/outer failure 仲裁。完整 CPU state 只在 cold/offloaded worker 尚未
ACK 时由 runtime 暂存；active、cold 或 wake install 成功后立即清空 payload，版本事实由
`current_policy_version / _installed_policy_version` 单独保存。

## 4. Generation 调用链

```text
collector -> runtime.generate(request)
  -> require RUNNING
  -> require an active RayGenerationSession
  -> RayGenerationRuntime:
     -> resolve samples_per_chunk="auto" once when requested
     -> stamp current policy version when absent
     -> RayGenerationSession.executor.execute(request)
        -> build_sample_rows
        -> DistributedExecutionPlanner.plan_with_engine
        -> optional single-worker pipelined request -> RayActorDispatcher.run_one
           OR SampleChunk -> RayActorJob -> RayActorDispatcher.run
        -> correlate request/chunk results
        -> stale-slot whole-request discard
        -> OOM chunk split/retry
        -> policy-version validation
        -> registry-owned ChunkGatherer.gather_chunks
        -> GenerationOutput
  -> terminal completion gate
```

`samples_per_chunk="auto"` 的 remote fleet probe 使用 fleet-owned dispatcher、generation
stall timeout 和 typed ObjectRef cancellation。并发首请求先经过 runtime single-flight，
而显式整数 chunk request 也必须经过同一个 actor slot，因此 probe 不会与 generation
互相预塞进 synchronous actor mailbox；失败不会留下 RUNNING runtime 供 continuous
producer 重试。

### Standard / dynamic chunks

fleet-owned `RayActorDispatcher` 跨 generation、probe、pipelined request、weight sync
和并发 request 维护每个 synchronous actor 的一个真实 default-group slot。job 先通过
driver admission，在 `.remote()` 边界前创建独立 `RayCallDeadline`；
本地 pending job 不计时，也不会预塞进 actor mailbox。submitted ref 的预算包含 Ray
参数序列化、transport 与执行时间。其他 worker 的完成不会延长 hung ref；任一 timeout 关闭整个
dispatcher，丢弃 request 已完成的 partial chunks，并让 owner 销毁 actor fleet。
兼容同一 worker 的 concurrent callers 按 FIFO handoff；已有 request 的 pending chunks
不能在 release 后同步抢回 slot，而不同 worker 仍可独立 admission。

non-draining weight sync 只跳过上层 prompt drain；它仍在同一 dispatcher 里等待已有
generation 释放 actor slot，随后才开始完整 ACK deadline。health 和
`pipelined_progress` 保持在专用 health concurrency group，不进入 default-slot admission。
multi-worker update 一旦任一 job 已提交，caller cancellation 就是 terminal；即使该 job
已完成而另一 worker 仍在等 admission，也不能把 partial install 当作普通本地取消。

### Pipelined request

single-worker pipeline 在 driver 侧 single-flight。worker 通过 health concurrency group
发布 `PipelinedRequestProgress`；只有 `completed_chunks` 严格增长才重置 stall deadline。
该计数只在已 record 的 device produce fence 经非阻塞 `query()` 确认完成后推进，
不会把 host-side CUDA enqueue 误报为完成，也不会为进度逐 chunk synchronize。
request lock 之后仍需通过 shared dispatcher；`run_one` 在真实 slot admission 后、提交前
创建初始 deadline，并持有 main result ref。health success 本身不算业务进度。错误
type/request ID/total/regression 是 terminal wire protocol failure。

worker path：

```text
RayGenerationWorker.execute_chunk(envelope)
  -> GenerationWorkerCore.execute_chunk(envelope)
     -> load/activate request policy slot
     -> family executor.forward_chunk_plan(request, chunk)
     -> typed ChunkExecutionResult
```

diffusion canonical 顺序是 prompt encode → latent prepare → denoise trajectory → VAE decode；
AR path 是 prepare/prefill → request decode loop → VQ decode → typed token trajectory。

## 5. Weight、parking 与 shutdown

### Weight transaction

```text
trainer weight syncer
  -> selected runtime.update_weights(state_ref, policy_version)
     -> stage while no active session, otherwise:
        -> RayGenerationSession.update_weights
           -> RayGenerationWeightSync.push_to_rollout_workers
              -> worker.update_weights
                 -> GenerationWorkerCore.update_weights
```

所有 remote workers 共享一次 `ray.put(state_ref)`。active session update 只在全部 ACK
返回并通过 expected integer version 校验后推进已安装版本。cold/parked deferred update
没有可 ACK 的 active fleet：runtime 立即把
`current_policy_version` 推进为最新 accepted target、暂存 `_pending_install`，同时保持
`_installed_policy_version` 不变。下一次 `activate()` 安装并收到 ACK 后才推进 active
version，并释放 CPU payload。timeout 或 bad ACK 不会 publish candidate active version；
terminal cleanup 也不会保留已无恢复消费者的 pending payload。

### Parking

schedule 必须先 drain，再调用 `offload()`。deferred runtime 调用 session 的
`sleep_workers()` 进入 host-memory parking，下一次 `activate()` 再调用
`wake_workers()`；eager runtime 的 `offload()` 是 protocol-required no-op。health
monitor 在 sleep 前 pause，wake 成功后 resume，并重新应用 first-wait grace。
sleep/wake 的 remote failure、120 秒 timeout、result-validation failure 或 submitted-work
cancellation 都由唯一 runtime 关闭 admission、标记 session force-close，并保留首个
terminal root。重复 `activate()` 经过同一个 lifecycle admission gate，不能把已被 health
monitor 判死的 runtime 误报为 active。offload control task 自己发起失败清理，即使所有
public waiter 已取消也不会放弃 session ownership。清理成功时进入 `TERMINATED`；若连续
teardown 失败，则保留 session handles、首个 operation root 与 `SHUTTING_DOWN`，由后续
显式 `shutdown()` 继续重试。

### Terminal shutdown

唯一 runtime 用一个 shared shutdown task：先 join 自己拥有的 activation/offload task，
再停止 health monitor，并调用 session `close(force=...)`。普通 shutdown 可先发
`release_policy`；terminal distributed error 设置 force mode，session 取消已在等待
release 的 local barrier task、跳过新 graceful RPC，并直接
`ray.kill(no_restart=True)`。

失败 kill 保留 actor handle，供下一次 shutdown retry。timeout root 不会被 cleanup error
替换。

## 6. Reliability 边界

| Boundary | Current implementation |
|---|---|
| process reachability | dedicated health concurrency group + background monitor |
| control-plane business calls | `worker_rpc_timeout_s` absolute barrier |
| generation progress | per-ref / strict pipelined progress `generation_stall_timeout_s` |
| partial result safety | whole-request rejection + actor fleet destruction |
| supervisor handoff | terminal error -> verdict -> bounded restart policy; permitted retry resumes checkpoint |
| in-process recovery | deliberately absent |
| driver/GCS/raylet watchdog | not implemented |

health 与 deadline 缺一不可。一个可达 actor 的 default group 仍可挂住；一个 operation
deadline 也看不到 requests 之间的 idle process death。

## 7. Class roster

| Boundary | Responsibility | Verdict |
|---|---|---|
| `GenerationRuntime` | collector-facing lifecycle protocol | keep thin: public API |
| `RayGenerationRuntime` | sole lifecycle, activation/offload, accepted/pending/active policy state, health and teardown decisions | keep: collector-facing owner |
| `RayGenerationSession` | one launched executor, weight syncer and actor fleet; sleep/wake/close resource operations | keep: resource boundary without lifecycle |
| `RayGenerationExecutor` | plan, dispatch, OOM, correlation, gather | keep: driver scheduler |
| `RayGenerationWorker` | Ray methods and concurrency-group adapters | keep thin: framework boundary |
| `GenerationWorkerCore` | Ray-independent build/version/parking/forward | keep: process core |
| `GenerationChunkExecutor` | family forward/gather shape | keep thin: cross-family protocol |
| `ChunkGatherer` | pure driver-side assembly | keep thin: transport boundary |
| `RayActorGroup` | actor construction/startup/metadata | keep: shared Ray adapter |
| `RayGenerationWeightSync` | all-worker transactional ACK | keep: protocol boundary |
| `GenerationRuntimeLaunchContract` | primitive pickle payload | keep: process boundary |
| `ModelFamilyEntry` / `FAMILY_REGISTRY` | family taxonomy and factories | keep: isolated registry |

`RayGenerationExecutor` 与 `GenerationChunkExecutor` 都叫 executor，但分别处在 driver
scheduler 和 actor family contract 两侧。不要为了少一个名字 flatten 任一层。

eager runtime 的 `activate()` 与 `offload()` 当前只验证 admission 或执行 no-op，但不能
删除：它们是 `GenerationRuntime` 跨 eager/deferred session topology 的统一 public
shape。用同一协议让 schedule 不需要按 topology 分叉，比省掉两个薄方法更重要。

## 8. Architecture hygiene

### Keep

- `HEALTH_CONCURRENCY_GROUP`：health/pipelined-progress decorators 与 actor creation
  共用的 protocol name；
- `_PLACEMENT_READY_TIMEOUT_S`：placement scheduling boundary；
- registry taxonomy constants：单一隔离的 family source of truth；
- `ResolvedOnlineRun.ray_launch_inputs`：resolved online run 拥有 trainer schedule、
  generation config 与 family composition invariant；不恢复 free arg resolver；
- `ModelFamilyEntry.executor_kwargs`：registry entry 拥有 family-specific executor
  projection；不恢复带重复 family identity 的 free resolver；
- launcher `_launch_session` / `_launch_session_async`：前者是 eager/deferred 共用的一次
  resource construction，后者是保留 caller-thread Ray initialization 的 async framework
  adapter；两者都不是 public build API；
- optional `forward_plan_pipelined` seam：真实 single-worker capability，不给 AR family
  增加无行为 stub。

### Do not add

- `WorkerFleetManager`、`DeadlineManager` 或 `RecoveryHandler`；
- 第二个 concrete Ray runtime、shared runtime base、lifecycle manager 或通用 runtime
  util；`RayGenerationSession` 不能实现 `GenerationRuntime` 或拥有自己的 lifecycle；
- duplicated provider/model capability tables；
- ALL_CAPS operation-name/timeout taxonomy；
- family-specific lifecycle or timeout fields；
- 为 LOC reduction flatten public/process/framework adapters。

### 非目标

- 不重命名或合并 `RayGenerationWorker`、`GenerationWorkerCore`、
  `RayGenerationExecutor`、`RolloutRuntimeCoordinator`；这些名字分别标识 Ray adapter、
  process core、driver scheduler 和 rollout schedule owner，边界不同；
- 不删除 deprecated `run_actor_jobs`；它保留旧 public signature、logical concurrency、
  telemetry 与 actor exception passthrough，是显式兼容契约。生产 runtime 继续使用
  fleet-owned `RayActorDispatcher`；
- 不把 eager mode 的薄 `activate()` / `offload()` 展开进 schedule，也不要求 schedule
  识别 concrete runtime 类型。

## References

- `vrl/generation/protocols.py`
- `vrl/generation/launch_contract.py`
- `vrl/generation/ray/config.py`
- `vrl/generation/ray/launcher.py`
- `vrl/generation/ray/runtime.py`
- `vrl/generation/ray/session.py`
- `vrl/generation/ray/executor.py`
- `vrl/generation/ray/worker.py`
- `vrl/generation/ray/weight_sync.py`
- `vrl/generation/execution/worker.py`
- `vrl/families/registry.py`
- `vrl/ray/actor_group.py`
- `vrl/ray/actor_pool.py`
- `vrl/ray/operation_deadline.py`
- `docs/sprints/done/SPRINT_explicit_rollout_activation.md`
- `docs/sprints/done/SPRINT_rollout_worker_liveness.md`
- `docs/sprints/done/SPRINT_ray_rollout_operation_deadlines.md`
