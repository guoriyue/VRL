# INFO：Ray generation engine 当前 ownership 与调用链

状态：**verified against source（2026-07-30）**。

本文只描述当前生产路径。已删除的 `_RuntimeLease`、`RayGenerationWorkerFleet`、
physical-stage adapter、release-per-collect 和带 `stage` 参数的旧调用链不再作为 dormant
capability 保留。

## 1. Ownership 图

```text
Rollout schedule                         admission + normal drain owner
  -> GenerationRuntime                   collector-facing protocol
     -> RayGenerationRuntime             activation/offload/terminal + actor owner
        -> RayGenerationExecutor         driver plan/dispatch/OOM/gather
           -> RayGenerationWorker        thin Ray actor adapter
              -> GenerationWorkerCore    build/version/parking/forward owner
                 -> GenerationChunkExecutor
                    -> diffusion or AR family executor
                       -> RuntimeModel / upstream transformer and kernels
```

driver 侧包含 schedule、runtime 和 executor。actor 侧包含 Ray worker、worker core、family
executor 和 model。`RayGenerationRuntime` 直接拥有 resident actor handles、health monitor
和 optional weight sync；仓库里没有独立 `WorkerFleet` owner。

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
  -> ModelFamilyEntry.resolve_model_build(...)
  -> RayGenerationConfig.from_cfg(...)
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
但当前 `RayGenerationLauncher.launch_from_cfg` 仍直接读取
`trainer.rollout_orchestration.schedule_mode` 并比较字符串 `"continuous"`。它虽然没有
import rollout/trainer enum，仍属于 generation composition 读取 trainer config 的
layering leak；native generation engine Sprint 0 必须把这个 primitive fact 从更高层
composition boundary 传入，不能把现状写成已经解决。

### 3.2 Resident launch

```text
RayGenerationLauncher.launch(...)
  -> RayActorGroup.launch(
       startup_method="load_policy",
       worker_rpc_timeout_s=...,
     )
  -> validate worker metadata / GPU assignment
  -> DistributedWorkerHandle[]
  -> RayGenerationExecutor(
       planner,
       workers,
       registry-owned gatherer,
       generation_stall_timeout_s=...,
     )
  -> optional RayGenerationWeightSync(
       workers,
       worker_rpc_timeout_s=...,
     )
  -> RayGenerationRuntime(
       executor,
       weight_sync,
       owned_workers,
       health settings,
     )
  -> query all-worker version-slot capability
  -> start health monitor
```

startup load、metadata 和 capability 各自有 fresh control-plane deadline。任一步失败，
launcher 清理 candidate actors；placement group 仍由 `GlobalRayPlacementOwner` 管理。

### 3.3 On-demand launch

`with_on_demand_activation` 先保存：

```text
_OnDemandRuntimeState(
  config,
  launch_inputs,
  placement,
  inner_runtime,
  activation_task,
  desired_policy,
  active_policy_version,
  workers_offloaded,
)
```

第一次 `activate()` 通过 single-flight `activation_task` 调用
`RayGenerationLauncher.launch_async`，得到一个真正拥有 workers 的 inner
`RayGenerationRuntime`。后续 activation 唤醒同一个 parked inner runtime。

desired policy restore 通过 inner runtime 的 public `update_weights` failure boundary，
不能直接绕过 owner 调 private installer。这样 active inner 和未发布 cold candidate 在
weight ACK timeout 时都先 force-kill 自己拥有的 actors，outer facade 再关闭自己的
admission。

## 4. Generation 调用链

```text
collector -> runtime.generate(request)
  -> require RUNNING
  -> select resident self or active on-demand inner runtime
  -> resolve samples_per_chunk="auto" once when requested
  -> stamp current policy version when absent
  -> RayGenerationExecutor.execute(request)
     -> build_sample_rows
     -> DistributedExecutionPlanner.plan_with_engine
     -> optional single-worker pipelined request
        OR SampleChunk -> RayActorJob -> RayActorDispatcher.run
     -> correlate request/chunk results
     -> stale-slot whole-request discard
     -> OOM chunk split/retry
     -> policy-version validation
     -> registry-owned ChunkGatherer.gather_chunks
     -> GenerationOutput
  -> terminal completion gate
```

`samples_per_chunk="auto"` 的 remote fleet probe 使用 executor-owned dispatcher、generation
stall timeout 和 typed ObjectRef cancellation。并发首请求先经过 runtime single-flight，
而显式整数 chunk request 也必须经过同一个 actor slot，因此 probe 不会与 generation
互相预塞进 synchronous actor mailbox；失败不会留下 RUNNING runtime 供 continuous
producer 重试。

### Standard / dynamic chunks

executor-owned `RayActorDispatcher` 跨并发 request 维护每个 synchronous actor 的一个真实
slot。job 先通过 driver admission，在 `.remote()` 边界前创建独立 `RayCallDeadline`；
本地 pending job 不计时，也不会预塞进 actor mailbox。submitted ref 的预算包含 Ray
参数序列化、transport 与执行时间。其他 worker 的完成不会延长 hung ref；任一 timeout 关闭整个
dispatcher，丢弃 request 已完成的 partial chunks，并让 owner 销毁 actor fleet。

### Pipelined request

single-worker pipeline 在 driver 侧 single-flight。worker 通过 health concurrency group
发布 `PipelinedRequestProgress`；只有 `completed_chunks` 严格增长才重置 stall deadline。
health success 本身不算业务进度。错误 type/request ID/total/regression 是 terminal wire
protocol failure。

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
  -> RayGenerationRuntime.update_weights(state_ref, policy_version)
     -> RayGenerationWeightSync.push_to_rollout_workers
        -> worker.update_weights
           -> GenerationWorkerCore.update_weights
```

所有 remote workers 共享一次 `ray.put(state_ref)`。只有全部 ACK 返回并通过 expected
integer version 校验后，runtime 才推进 `current_policy_version`。timeout 或 bad ACK
不会 publish candidate version。

### Parking

schedule 必须先 drain，再调用 `offload()`。on-demand inner runtime 进入 sleep/host-memory
parking，下一次 activate wake；resident runtime 的 facade offload 是 no-op。health monitor
在 sleep 前 pause，wake 成功后 resume，并重新应用 first-wait grace。

### Terminal shutdown

`shutdown()` 使用 shared task，join runtime-owned activation/offload task，然后清理真实
owner。普通 shutdown 可先发 `release_policy`；terminal distributed error 设置 force
mode，取消已在等待 release 的 local barrier task、跳过新 graceful RPC，并直接
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
| `RayGenerationRuntime` | lifecycle, on-demand facade, actor ownership, health, teardown | keep: real resource owner |
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

## 8. Architecture hygiene

### Keep

- `HEALTH_CONCURRENCY_GROUP`：health/pipelined-progress decorators 与 actor creation
  共用的 protocol name；
- `_PLACEMENT_READY_TIMEOUT_S`：placement scheduling boundary；
- registry taxonomy constants：单一隔离的 family source of truth；
- optional `forward_plan_pipelined` seam：真实 single-worker capability，不给 AR family
  增加无行为 stub。

### Do not add

- `WorkerFleetManager`、`DeadlineManager` 或 `RecoveryHandler`；
- duplicated provider/model capability tables；
- ALL_CAPS operation-name/timeout taxonomy；
- family-specific lifecycle or timeout fields；
- 为 LOC reduction flatten public/process/framework adapters。

## References

- `vrl/generation/protocols.py`
- `vrl/generation/launch_contract.py`
- `vrl/generation/ray/config.py`
- `vrl/generation/ray/launcher.py`
- `vrl/generation/ray/runtime.py`
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
