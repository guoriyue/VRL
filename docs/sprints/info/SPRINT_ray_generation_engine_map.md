# INFO: Ray generation engine — current ownership and call chains

状态：**verified against source（2026-07-13）**。本文只描述当前生产路径；已删除的
release-per-collect、`_RuntimeLease`、physical stage adapter 与带 `stage` 参数的旧调用链
不再作为 dormant capability 记录。

目的：说明 wm-infra 的 native generation engine 真正拥有哪一层、哪些薄边界必须保留，
以及为什么外部 execution provider 应接在 engine 下面而不是替换 trainer-facing control
plane。

## 1. 一张图

```text
Rollout schedule                          driver: admission + drain owner
  -> GenerationRuntime                    collector-facing transport contract
     -> RayGenerationRuntime              activation/offload/shutdown + policy facade
        -> RayGenerationExecutor          driver-side chunk plan/dispatch/gather
           -> RayGenerationWorker         thin Ray actor adapter
              -> GenerationWorkerCore     model build/version/sleep/forward ownership
                 -> GenerationChunkExecutor
                    -> diffusion or AR family executor
                       -> RuntimeModel / upstream transformer and kernels
```

进程边界：schedule/runtime/executor 位于 driver；Ray worker、worker core、family executor
和 model 位于 actor。它们不能为了减少层数而合并。

ownership 边界：wm-infra 原生拥有中间的 lifecycle、policy version、chunk、trajectory 与
replay 语义；diffusion transformer/block/kernel 仍大量来自 Diffusers/PyTorch。两者同时
成立，不应把“native control plane”误写成“全栈 native model forward”。

## 2. Public runtime contract

`GenerationRuntime` 当前只要求：

```text
current_policy_version
requires_driver_model_offload
activate()
generate(request)
offload()
shutdown()
is_colocated()
```

它不声明 `release()`，也不把 `update_weights()` 强塞给所有 runtime。训练侧通过
`build_runtime_weight_syncer` 检查具体 runtime 是否同时暴露 `supports_weight_sync` 与
`update_weights`，再构造独立的 weight-sync control path。

schedule 是 admission/drain 的唯一 owner：strict schedule 在 collect 前 activate，等待
collect 本身就是 drain，最后 offload；continuous schedule 在不能 non-draining sync 时暂停
producer 并 drain。runtime 不拥有 public `QUIESCING` 或 `RECOVERING` phase，其 terminal
state 只有 `RUNNING -> SHUTTING_DOWN -> TERMINATED`。

## 3. Launch call chain

### 3.1 Composition

```text
online recipe
  -> ModelFamilyEntry.resolve_model_build(...)
  -> RayGenerationConfig.from_cfg(...)
  -> GenerationRuntimeLaunchContract(
       family,
       primitive model_build/executor_kwargs,
       policy_version,
       profiler config,
       sleep_offload,
       versioned_weight_sync,
     )
  -> RayGenerationLaunchInputs(contract, entry.new_gatherer())
```

`GenerationRuntimeLaunchContract` 不携带 live model、pipeline、callable 或 executor class；它
只允许 primitive/list/tuple/dict，并验证 pickle。`executor_cls` 与 rollout builder 的唯一
source 是 worker 重新查到的 `ModelFamilyEntry`。

当前工作树在 `RayGenerationLauncher.launch_from_cfg` 内反向 import
`vrl.rollouts.orchestration.types.RolloutScheduleMode` 来派生 `versioned_weight_sync`，违反
generation boundary test。active native-engine program 已把修复列为 Sprint 0 gate：由中立
composition boundary 解析 schedule-derived bool，再把 primitive fact 传入 generation，不能
让 generation 解释 trainer/rollout enum。

### 3.2 Resident launch

```text
RayGenerationLauncher.launch(config, launch_inputs, placement)
  -> RayActorGroup.launch(
       worker_cls=RayGenerationWorker,
       startup_method="load_policy",
     )
  -> DistributedWorkerHandle[]
  -> RayGenerationExecutor(
       DistributedExecutionPlanner(ChunkPlacementPolicy),
       workers,
       ChunkGatherer,
     )
  -> optional RayGenerationWeightSync(workers)
  -> RayGenerationRuntime(executor, weight_sync, owned_workers)
```

actor startup 内部：

```text
RayGenerationWorker.load_policy()
  -> GenerationWorkerCore.load_policy()
     -> get_model_family_entry(contract.family)
     -> entry.build_rollout(ModelBuild)
     -> import entry.executor_cls
     -> executor_cls(runtime_model, executor_kwargs)
```

worker 构造完成后，launcher 查询每个 worker 是否真的支持 versioned trainable-state slots；
只有全体支持时，runtime 才发布 `supports_non_draining_weight_sync=True`。

### 3.3 On-demand launch

`with_on_demand_activation` 不提前创建 actors，而是保存：

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

第一次 `activate()` 通过 `activation_task` single-flight 调用 `launch_async`；后续 activate
唤醒已 offload workers，并在返回前安装 `desired_policy`。candidate 只有 load + policy
restore 全成功后才写入 `inner_runtime`；caller cancellation 不取消 runtime-owned task。

## 4. Generate call chain

```text
schedule activates runtime
collector -> runtime.generate(request)
  -> require RUNNING + require active on-demand runtime
  -> resolve samples_per_chunk="auto" once when requested
  -> stamp current_policy_version when request has none
  -> RayGenerationExecutor.execute(request)
     -> build_sample_rows(request)
     -> DistributedExecutionPlanner.plan_with_engine(request, workers)
        -> EnginePlan(SampleChunk...)
        -> DeviceAssignment(..., estimated_cost)
     -> single-worker optional per-request pipelined path
        OR per-chunk RayActorJob -> run_actor_jobs
     -> correlate request/chunk results
     -> stale-slot whole-request discard
     -> OOM chunk split/retry
     -> request/result policy-version check
     -> ChunkGatherer.gather_chunks(...)
     -> GenerationOutput
```

worker path：

```text
RayGenerationWorker.execute_chunk(envelope)
  -> GenerationWorkerCore.execute_chunk(envelope)
     -> load_policy()
     -> check/activate request.policy_version slot
     -> executor.forward_chunk_plan(request, chunk)
     -> move output tensors to CPU
     -> ChunkExecutionResult
```

`forward_chunk_plan` 当前只有 `(request, chunk)`，没有 `stage` 参数。diffusion canonical
顺序是 prompt encode → latent prepare → denoise trajectory → VAE decode；AR discrete path
是 prepare/prefill → per-request `ARDecodeLoop` → VQ decode → typed token trajectory。

OOM 时 sample chunk 有序二分；gather 再验证 prompt-major 完整覆盖。任一 stale slot 污染整条
request 时抛 `StaleSlotDiscard`，不会拼出 partial mixed-policy group。

## 5. Weight, offload, and shutdown

### Weight update

```text
trainer weight syncer
  -> RayGenerationRuntime.update_weights(state_ref, policy_version)
     -> RayGenerationWeightSync.push_to_rollout_workers
        -> worker.update_weights
           -> GenerationWorkerCore.update_weights
```

strict path drain 后原地覆盖；允许 versioned slots 的 continuous LoRA path可保留旧 request
所需 slot。当前 ACK 只返回整数 policy version。weight-update ACK 的无界等待归
`SPRINT_ray_rollout_operation_deadlines.md`；schema/digest strengthening 不是当前 native
provider gate，确有 provider payload identity 要求时由该 provider 的 installer contract 验证。

### Offload and shutdown

schedule 必须先 drain，再调用 `offload()`。on-demand runtime 用 `workers_offloaded` 幂等地
sleep actors，下一次 activate wake；resident runtime 的 offload 是 no-op。terminal
`shutdown()` 使用 shared task，先 join activation/offload，再 release policy、kill owned actors，
cleanup 失败保留 ownership 供下一次 shutdown 重试。

## 6. Class roster and keep/delete verdict

| Class/boundary | Current responsibility | Verdict |
|---|---|---|
| `GenerationRuntime` | collector/public transport + explicit lifecycle | keep thin: public protocol boundary |
| `RayGenerationRuntime` | resident/on-demand resource owner, policy facade, offload/shutdown | keep: lifecycle owner |
| `RayGenerationExecutor` | driver chunk plan/dispatch/OOM/version/gather; no model | keep: driver scheduler |
| `RayGenerationWorker` | Ray actor methods + node/GPU metadata delegation | keep thin: framework adapter |
| `GenerationWorkerCore` | Ray-independent build/version slots/memory parking/forward | keep: testable process core |
| `GenerationChunkExecutor` | family `forward_chunk_plan` + gather shape | keep thin: cross-family protocol |
| `ChunkGatherer` | pure driver-side assembly without model ownership | keep thin: transport boundary |
| diffusion/AR base executors | shared per-family production skeletons | keep: cross-family consistency |
| `GenerationRuntimeLaunchContract` | serializable worker construction payload | keep: process/pickle boundary |
| `ModelFamilyEntry` / `FAMILY_REGISTRY` | one family taxonomy and build/executor dispatch source | keep: deliberately isolated registry |

`RayGenerationExecutor` 与 `GenerationChunkExecutor` 都叫 Executor，但分别位于 driver
scheduler 与 actor model-contract 两侧。命名有认知成本，不是死抽象；不要为了少一个名字
删除其中任一层。

single-worker `forward_plan_pipelined`/`execute_request_pipelined` 继续是 optional
diffusion capability，通过明确 guard/getattr 读取。它不属于所有 AR family 的共同协议，不能
为了“接口完整”给 AR 添加无行为空实现。

## 7. Current limitations and decision

| Limitation | Current evidence | Owner |
|---|---|---|
| unbounded Ray worker waits | generation/startup/update ObjectRefs can block attempt exit | active operation-deadline sprint |
| no in-process actor recovery | failed attempts rebuild the runtime only after process exit | accepted boundary: supervisor checkpoint resume |
| version-only ACK | transaction validates the committed integer version | current native contract; stronger provider-local identity only when required |
| no provider selector | one `executor_cls` per family; no provider provenance in launch contract | native-engine N2 + provider/conformance sprints |
| no cross-request forward sharing | EnginePlan and AR decode scheduler are per request | parked step-scheduler sprint |
| no native diffusion blocks | backbone calls upstream transformer | parked native-transformer sprint |
| dense diffusion trajectory capacity | full step×latent observations/actions preallocated | parked paged-trajectory sprint, video-profile gated |

这些事实支持的结论是：保留 wm-infra 作为 RL truth/lifecycle owner，把 native、FlashDreams
和 SGLang 作为其下方不同粒度的 execution implementations。它不证明 native 是最快的
serving engine，也不支持为了“全栈自研”立即重写 transformer/kernel。

## 8. Architecture hygiene

- `GenerationRuntime`、`GenerationChunkExecutor`、`ChunkGatherer`、Ray actor adapter 与
  family builders 都提供真实 public/process/framework/cross-family boundary，保持薄层。
- `FAMILY_REGISTRY` 与 `GENERIC_DIFFUSION_EXECUTOR` 是 taxonomy/import protocol；trajectory
  role/metric 常量是 schema boundary，均可保持 ALL_CAPS。
- 不维护重复 `SUPPORTED_PROVIDERS`/model capability 大表；future provider cases 从 typed
  binding 与真实实现派生。
- `GenerationRequest.priority` 当前无行为消费者，应删除；活的 `RayActorJob.priority` 来自
  `DeviceAssignment.estimated_cost`，二者不能混为一谈。
- 不复活已删除 physical-stage package，不把 provider private scheduler 提升为 wm public API，
  不为 LOC reduction flatten 上述协议。

## References

- `vrl/generation/protocols.py`
- `vrl/generation/launch_contract.py`
- `vrl/generation/ray/{launcher,runtime,executor,worker,weight_sync}.py`
- `vrl/generation/execution/{planner,chunk_placement,worker}.py`
- `vrl/generation/{diffusion,ar}/executor.py`
- `vrl/families/registry.py`
- `vrl/ray/actor_pool.py`
- `docs/sprints/SPRINT_native_generation_engine_program.md`
- `docs/sprints/done/SPRINT_explicit_rollout_activation.md`
- `docs/sprints/SPRINT_ray_rollout_operation_deadlines.md`
