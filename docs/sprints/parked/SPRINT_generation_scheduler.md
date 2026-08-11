# SPRINT: Generation scheduler upgrade

> **Current-state correction (2026-07-13).** Track A is the live chunk-level
> scheduler: `chunk_placement.py` provides the placement policy used when the
> planner assigns whole-model chunks to rollout workers, and multi-worker
> generation is data parallel. Track B remains a proposal. The former topology,
> payload/result, serial-runner, and Ray stage-adapter seams were deleted because
> they had no production constructor or behavioral consumer. A "multi-worker"
> statement in Track A therefore means chunks dispatched to full-model workers,
> not physical stages placed on different workers. If profiling opens Track B,
> its contracts must be rebuilt together with the real scheduler and launcher;
> this document does not describe retained implementation scaffolding.

状态：**Track A implemented（2026-06-11，dormant until multi-GPU）**；
Track B gated 不变。落地内容：

```text
A1  execution/scheduler.py -> chunk_placement.py（原子改名，零残留，
    架构测试清单同步）
T1  vrl/ray/actor_pool.py records queue_wait_s / execution_s. The executor
    includes the schedule only in the propagated runtime_debug payload; the
    former unconditional GenerationOutput.extra["ray_chunk_schedule"] key was
    removed because no production consumer read it.
T2  pull 式动态派发：RayActorJob.worker_id 可为 None，pool 把未绑定 job
    派给当前 inflight 最少的空闲 worker；LPT 经 RayActorJob.priority
    （= estimated_cost）在派发层排序，gather 顺序零接触。
    配置键（实际落点，沿 distributed.rollout.* 平铺模式）：
      distributed.rollout.chunk_placement_strategy: round_robin | dynamic
    默认 round_robin —— 静态路径逐位不变（priority 全 0 + 稳定排序）。
验证  tests/ray/test_chunk_dispatch.py 9 项（fake-ray 控制完成顺序，
    确定性、无真 Ray）：静态绑定不迁移 / pull 让快 worker 多拿活 /
    LPT 提交序 / 未绑定缺 worker_methods 报错 / telemetry 行 /
    planner 两策略 / 成本轴 / 策略词汇表封闭。全量回归 430 passed。
```

收益兑现仍挂在多卡 rollout 之后（单 worker 下两策略等价）——多卡
bring-up 时把 `chunk_placement_strategy: dynamic` 打开即可，T1 telemetry
当场给出 imbalance 对比数据。当前所有实验跑
`ray_rollout_colocated_single_gpu.yaml`（num_workers: 1）——单 worker 下
round-robin 与 dynamic 派发等价、LPT 排序无意义，Track A 收益严格为零。
本文档的价值是把设计坑位提前想清（静态绑定事实、互斥约束、backpressure、
policy barrier），多卡 bring-up 第一周直接照此开工（T1+T2 一起上，telemetry
当场验证收益）。在那之前，单卡的工程时间应花在 cross-model perf sprint 的
Wave 2 transport 项上（uint8 过线 / pinned 拷贝——它们同时是 Track B 未来
relay 的前置测量）。

目标：把当前 generation scheduler 的职责讲清楚，并给出两阶段改进路线：

```text
Track A:
  先改进现有 chunk placement scheduler。
  解决 round-robin / static inflight / observability 不足。

Track B:
  在 physical stage runtime gate 打开后，再实现 per-stage scheduler。
  解决 denoise / decode / reward actors 的队列、batch、backpressure、drain。
```

这两个 track 不能混在一起。当前系统缺的不是一个更大的 `scheduler.py`，而是两个不同层级的 scheduler boundary。

## 0. Core Conclusion

当前 `vrl/generation/execution/chunk_placement.py` 本质上不是 physical stage scheduler runtime，而是 chunk placement policy。

它做三件事：

```text
1. Read request.sampling.samples_per_chunk to size each SampleChunk.
2. Call build_engine_plan(...) to generate chunks.
3. 用 round-robin 把 chunks 分给 rollout workers。
```

关键代码：

```python
max_samples = int(request.sampling.get("samples_per_chunk", request.samples_per_prompt))
engine_plan = build_engine_plan(
    request,
    max_samples_per_chunk=max(1, max_samples),
)
```

worker 分配是：

```python
for idx, chunk in enumerate(engine_plan.chunks):
    worker = workers[idx % len(workers)]
```

Ray 执行层再用 `max_inflight_chunks_per_worker` 做 per-actor inflight 限流：

```python
run_actor_jobs(
    jobs,
    max_inflight_per_actor=self.max_inflight_chunks_per_worker,
)
```

**关键事实（vrl/ray/actor_pool.py:47-49）：chunk→worker 绑定在 plan 时定死，
执行层只做提交节流，永不改派。** 某 worker 满了，它的 pending job 只会回队列
等"自己的" worker，不会换人：

```python
if inflight_by_worker[job.worker_id] >= max_inflight_per_actor:
    pending.append(job)   # 等同一个 worker，不重新分配
```

推论：计划期没有任何 chunk 在跑（inflight 全为零），所以任何"看 inflight 的
计划期策略"都退化成静态成本均衡——真正消除"快 worker 闲等慢 worker"的手段
是把绑定推迟到派发期（pull 式），这决定了 Track A 的解法形态（§4 A2）。

这说明当前 scheduler 只回答：

```text
哪个 SampleChunk 给哪个 rollout worker？
```

它不回答：

```text
一个 chunk 内部的 denoise / decode / reward stage 怎么排队？
stage 怎么 batching？
stage 间怎么 backpressure？
哪个 stage 放在哪张 GPU？
policy_version sync 时怎么 drain in-flight stage payload？
```

## 1. Existing Boundary

### 1.1 Logical work unit: SampleChunk

`SampleChunk` 是 rollout 的逻辑工作单元：

```python
@dataclass(frozen=True, slots=True)
class SampleChunk:
    prompt_index: int
    prompt: str
    sample_start: int
    sample_count: int
```

它保存 prompt/sample range，并支持 OOM retry split：

```python
def split(self) -> tuple[SampleChunk, SampleChunk]:
    ...
```

这个边界必须保留。它承载：

```text
sample identity
group semantics
seed / sample offset
OOM retry unit
chunk gather ordering
Ray worker payload unit
```

### 1.2 Request plan: EnginePlan

`EnginePlan` 负责 request-level metadata：

```text
sample rows
workload signature
family capability
chunks
ExecutionStage profiler labels
```

`ExecutionStage` 当前只是 planner/profiler label，不是 physical actor stage：

```python
@dataclass(frozen=True, slots=True)
class ExecutionStage:
    """One planner-visible execution stage and profiler label."""
```

不要把 placement、queues、actor lifecycle 塞进 `ExecutionStage`。

### 1.3 Current placement: static round-robin

`DistributedExecutionPlanner.plan_with_engine(...)` 现在不看：

```text
worker 当前 inflight
worker 最近执行速度
worker GPU memory pressure
chunk estimated cost
prompt / frame / resolution difference
stage-specific decode cost
reward drain time
```

它只按 chunk index round-robin。这个策略简单、稳定、可测试，但在以下场景会浪费吞吐：

```text
chunks cost 不均匀
不同 workers 速度不同
max_inflight > 1 时某个 worker 已经排队很多
某些 chunks 因 OOM split 变成更多小任务
decode / reward tail 让某个 worker 长时间不释放
```

## 2. Problem Statement

当前 scheduler 的问题不是“没有 pipeline”，而是两层问题叠在一起：

```text
Problem A:
  Chunk placement 太静态。
  它无法根据 cost / inflight / observed duration 做更好的 worker assignment。

Problem B:
  Chunk 内部没有 physical stage scheduler。
  denoise / decode / reward 仍在同一个 worker 内串行执行，不能独立排队、放置、batch。
```

这两个问题需要不同解法：

```text
Problem A -> chunk placement scheduler
Problem B -> physical stage scheduler
```

先改 Problem A 可以马上降低 tail latency 和 worker imbalance；Problem B 只有在 multi-GPU / decode bubble / relay cost 证明后才值得进入。

## 3. What SGLang-Omni Teaches Us

SGLang-Omni 的 scheduler 设计里最有价值的部分不是 AR decode scheduler，而是 stage 与 scheduler 的边界。

### 3.1 Stage is the IO shell; scheduler is compute dispatch

`sglang_omni/pipeline/stage/runtime.py` 里的 `Stage` 负责：

```text
control-plane receive/send
relay IO
input aggregation
stream chunk routing
abort tracking
profiling
downstream routing
```

它不直接做模型计算。计算统一交给 scheduler inbox/outbox：

```text
Stage -> scheduler.inbox.put(IncomingMessage(...))
scheduler -> scheduler.outbox.put(OutgoingMessage(...))
Stage drains outbox -> route result downstream / terminal completion
```

VRL 应该学这个边界：

```text
Future StageActor shell (not present today):
  owns actor lifecycle, routing, relay, metrics, abort/drain.

StageScheduler:
  owns admission, batch, concurrency, backpressure, result/error emission.

model executor / stage handler:
  owns actual denoise/decode/reward compute.
```

不要把 queue、relay、abort、routing 都塞进 model executor，也不要让 scheduler 直接知道 family model internals。

### 3.2 Use a small message contract

SGLang-Omni 的 scheduler message 很小：

```python
@dataclass
class IncomingMessage:
    request_id: str
    type: Literal["new_request", "stream_chunk", "stream_done"]
    data: Any = None

@dataclass
class OutgoingMessage:
    request_id: str
    type: Literal["result", "stream", "error"]
    data: Any = None
    target: str | None = None
    metadata: dict[str, Any] | None = None
```

A future implementation must define a small message contract from measured
requirements. The deleted payload/result classes are not a reusable current API;
the following fields are design requirements, not retained types:

```text
request_id
message type / terminal / error
payload data
target / next stage metadata
scheduler metrics
```

这比让 Ray actor method 返回裸 tensor 更安全，因为 failure、abort、metrics、routing 都有统一 envelope。

### 3.3 Batch and concurrency are scheduler policy, not model policy

`SimpleScheduler` 支持这些 knobs：

```text
batch_compute_fn
max_batch_size
max_batch_wait_ms
request_cost_fn
max_batch_cost
max_concurrency
abort_callback
```

关键约束：

```text
max_concurrency > 1 and batch_compute_fn are mutually exclusive
```

这个约束对 VRL 有用。第一版 stage scheduler 不要同时做：

```text
same-stage dynamic batching
same-stage concurrent per-request execution
```

先二选一：

```text
denoise:
  prefer batch mode, because tensor shape/batch controls GPU efficiency.

reward/artifact CPU-bound stage:
  may prefer max_concurrency, if compute_fn is re-entrant.

decode_latents:
  start with batch mode or serial micro-batch; do not assume concurrency is safe.
```

Track B 将需要显式定义这些 proposed policy fields；当前代码里还没有对应的 policy type：

```text
batch_size
max_inflight
max_batch_wait_ms
max_batch_cost
memory_fraction
```

需要补的是 type、scheduler 与这些字段的真实执行语义；在那之前不应该提前暴露一个无消费者的 YAML 表面。

### 3.4 Abort and error are first-class scheduler behavior

SGLang-Omni 的 simple scheduler 会：

```text
track aborted request ids
call abort_callback for cleanup
drop aborted results before emission
emit per-request error rows into outbox
avoid one failed request silently poisoning other requests
```

VRL stage scheduler 必须把下面这些作为第一版 contract：

```text
abort(request_id)
drain(policy_version barrier)
per-payload error result
cleanup callback for stage-owned buffers / relay slots
bounded retained aborted ids to avoid unbounded memory growth
```

特别是 diffusion RL：失败不能只返回普通 exception。必须知道失败属于哪个 `SampleChunk`、哪个 policy_version、哪个 replay payload，才能正确取消 gather / trainer consumption。

### 3.5 Coordinator tracks terminal completion

SGLang-Omni 的 `Coordinator` 负责：

```text
register stages
submit request to entry stage
track RequestInfo state
wait for one or more terminal stages
broadcast abort
fail pending requests on fatal error
```

VRL 不需要照搬 ZMQ coordinator，但 physical stage scheduler 需要同等职责：

```text
track SampleChunk pipeline state
know entry stage and terminal stages
join terminal result back into ChunkResult
broadcast abort / drain across stage actors
surface fatal scheduler errors to RayGenerationRuntime
```

这件事应该在 scheduler/control-plane 层做，不应该散落在 denoise/decode handler 里。

### 3.6 What not to copy

不要复制这些 SGLang-Omni 细节：

```text
OmniScheduler AR prefill/decode/KV-cache logic
tree cache / prefix cache
streaming scheduler for vocoder-style stream chunks, until VRL has streaming payload need
ZMQ control plane
NIXL/Mooncake relay as default
TP leader/follower fanout
SGLang ServerArgs-specific runtime mapping
```

VRL 可以借鉴的是 scheduler contract：

```text
inbox/outbox
batch wait
request cost
max batch cost
max concurrency
abort callback
central completion tracking
stage shell separate from compute scheduler
```

## 4. Track A: Improve Chunk Placement Scheduler

Track A 保持现有 worker model：

```text
RayGenerationExecutor
  -> assigns ChunkExecutionEnvelope to RayGenerationWorker
  -> worker runs full chunk fused path
```

不引入 denoise/decode actors，不改 replay payload contract。

### A1. Rename boundary in code

当前文件名 `execution/scheduler.py` 容易误导。它不是 runtime scheduler，而是 planner/placement。

建议迁移为：

```text
vrl/generation/execution/chunk_placement.py
```

一次性原子改完所有 import，**不留 deprecated facade**——这是内部代码库，没有
外部消费者；本仓库既有重命名实践（PipelineExecutor→GenerationChunkExecutor、
spec.py→topology.py、stage_pipeline.py→pipeline_runner.py）全部是原子全量改、
零别名。facade 只会制造薄文件和双名并存期。

（family executor class names 已在 2026-06-11 统一为 `*ChunkExecutor`，
此项不再是本 sprint 的工作。）

### A2. Pull-based dynamic dispatch（不是更多计划期策略）

§0 的关键事实排除了"加几个计划期策略"的路线：计划期 inflight 全为零，
`least_inflight` 与 `least_estimated_cost` 在静态模型里是同一个东西（贪心
成本均衡）。真正的改动在派发层：

```text
round_robin（静态，默认/回归基线）:
  现有行为。plan 时绑定 worker，run_actor_jobs 只节流。

dynamic（pull 式）:
  plan 只产出 chunk 顺序，不绑定 worker。
  run_actor_jobs 改为：下一个 pending chunk 给第一个有空位的 worker。
  快 worker 自动多拿活，慢 worker / OOM-split 拖尾不再阻塞同伴。
```

实现位置是 `vrl/ray/actor_pool.py`（`RayActorJob.worker_id` 变为可选，
`_submit_ready` 对未绑定 job 选当前 inflight 最少的 worker），planner 侧
只增加排序（见 A3）。配置入口：

```yaml
rollout:
  chunk_placement:
    strategy: round_robin   # round_robin | dynamic
```

pull 式天然就是 inflight 感知，不需要 work stealing（job 提交后仍不迁移，
取消/replay consistency 语义不变）。

### A3. Add chunk cost model

需要一个可解释的 cost estimate：

```text
diffusion cost ~= sample_count * num_steps * latent_tokens
video decode cost ~= sample_count * num_frames * height * width
AR cost ~= sample_count * max_new_tokens
```

第一版可以是 family-neutral：

```python
estimated_chunk_cost = sample_count * max(1, num_steps or max_new_tokens or 1)
```

The current family-neutral calculation is intentionally inline in
`DistributedExecutionPlanner.plan_with_engine`; there is no standalone
`estimate_chunk_cost` API or capability override. Add an override only when a
measured family-specific scheduler actually consumes it.

**pull 式派发下 cost model 的用途是"先后"不是"去哪"**：placement 由 pull
决定，cost 只用来对 pending 队列做降序排序（经典 LPT——大活先派，避免
最后剩一个大 chunk 拖尾）。这比"cost 决定 placement"更简单也更稳。

注意：cost model 是 scheduler hint，不是 correctness source。不能让它改变
sample identity 或 gather order。OOM split 产生的子 chunk 在 worker 内重试、
不回派发层——cost 估算不跟踪 split，只影响 telemetry 口径，记录即可。

### A4. Record scheduling telemetry

Worker results still carry internal chunk metrics, while the collector propagates
only the explicitly requested `runtime_debug` payload. Driver-side scheduling
metrics are recorded in `vrl/ray/actor_pool.py` because:
`queue_wait_s`（进 pending 到实际 `remote_method` 调用）和提交时刻只有派发
循环自己知道，executor 层补不出来：

```text
chunk_key
assigned_worker
assignment_strategy
estimated_cost
worker_pending_cost_at_assignment
submit_order
queue_wait_s
execution_s
end_to_end_s
```

这些指标会回答：

```text
round-robin 是否真的造成 worker imbalance？
max_inflight_chunks_per_worker 是否过高 / 过低？
chunk cost model 是否预测了真实 duration？
OOM split 是否导致 tail？
```

没有这些 telemetry，不要直接上更复杂策略。

### A5. Preserve current semantics

Track A 必须保持：

```text
SampleChunk identity 不变
EnginePlan gather order 不变
OOM retry 仍在 executor/chunk boundary 内
policy_version mismatch 仍 fail fast
RayGenerationRuntime public contract 不变
```

验收重点不是“新策略一定更快”，而是：

```text
round_robin 行为保持（静态路径回归基线）
dynamic 派发可解释：每个 chunk 的 worker 选择能从 telemetry 还原
metrics 能证明两种派发模式的差异（imbalance / tail）
现有 Ray runtime tests 不回退
```

## 5. Track B: Physical Stage Scheduler

Track B has no retained pipeline contract. If its profiling gate opens, the
minimum proposed concepts are:

```text
PipelineTopology
PipelineStage
PipelineStagePayload
PipelineStageResult
RayPipelineStageWorker
RayPipelineRunner
```

The names above are proposal vocabulary only. They must not be imported or
configured until a launcher, placement plan, and scheduler consume them.

它解决的是 chunk 内部的 physical stage scheduling：

```text
SampleChunk payload
  -> prompt_encode stage
  -> denoise stage
  -> decode_latents stage
  -> reward_artifact stage
```

### B1. Scheduler owns queues, not model logic

目标结构：

```text
StageScheduler
  inbox per stage
  outbox per stage
  batch builder per stage
  max_inflight per stage
  backpressure credits
  drain barrier
```

Stage actor 只执行：

```text
PipelineStagePayload -> PipelineStageResult
```

Scheduler 决定：

```text
when to admit payload
when to batch payloads
where to route result
when upstream must stop producing
when policy sync can drain safely
```

### B2. Per-stage policy

未来 policy 需要的最小字段形状（proposal only）：

```text
batch_size: optional integer
max_inflight: integer, default 1
max_batch_wait_ms: integer, default 0
max_batch_cost: optional integer
memory_fraction: optional float
```

不同 stage 的 policy 应该不同：

```text
denoise:
  larger batch, trainable weights, policy_version barrier

decode_latents:
  smaller micro-batch, VAE memory policy, usually frozen

reward_artifact:
  reward-backend batch / artifact serialization cost
```

### B3. Backpressure is mandatory

没有 backpressure，denoise 可以快速产生 latent/replay payload，把 decode/reward 堆爆。

最小规则：

```text
upstream may submit only if downstream has queue credit
terminal stage completion releases credit
policy sync waits for trainable stage drain
shutdown drains or cancels all stage queues explicitly
```

### B4. Ray actor feasibility

Ray actor 可以做这个，但第一版不要直接追求最终形态。

```text
Level 1:
  driver-routed actors.
  A newly built driver route validates payload and actor boundaries.
  No contract foundation is retained today.

Level 2:
  per-stage actor queues.
  Use bounded Ray/async queues and explicit credits.
  Measure queue + object-store overhead.

Level 3:
  replace validation relay with CUDA IPC / NCCL / shared memory if measured win exists.
```

Level 2 的价值是证明：

```text
overlap gain > queue + relay overhead
```

如果这个不成立，不做 Level 3。

### B5. Policy version barrier

Stage scheduler 必须知道哪些 stage 是 trainable：

```text
denoise: trainable, participates in weight sync
prompt_encode: usually frozen
decode_latents: frozen
reward_artifact: frozen / external
```

Weight sync 流程：

```text
pause admission
drain in-flight trainable stage payloads
sync denoise weights
stamp new policy_version
resume admission
```

不能让一个 `SampleChunk` 的 replay tensors 混入两个 policy versions。

## 6. First Implementation Plan

### T0. Document and naming cleanup

做两件小事：

```text
1. Protocol 层继续使用 GenerationChunkExecutor，不再恢复 PipelineExecutor。
2. 把 scheduler sprint 和 physical stage sprint 互相引用。
```

可选但推荐：

```text
rename execution/scheduler.py -> execution/chunk_placement.py（原子全量改，无 facade，见 A1）
```

### T1. Chunk placement telemetry

打点落在 `vrl/ray/actor_pool.py`（pending/submit/complete 三个时刻），由
`RayGenerationExecutor.execute(...)` 汇总记录：

```text
assignment_strategy
chunk_key
assigned_worker
estimated_cost
submit_index
queue_wait_s
end_to_end_s
```

Emitted only when runtime debug is requested:

```text
GenerationOutput.runtime_debug["chunk_schedule"]
```

There is no unconditional top-level schedule key; the collector did not consume
it. This instrumentation does not change placement behavior.

### T2. Pull-based dynamic dispatch + LPT ordering

两个改动（见 A2/A3）：

```text
1. vrl/ray/actor_pool.py：RayActorJob.worker_id 可选；未绑定 job 派给
   当前 inflight 最少的空闲 worker（pull 式）。
2. planner：strategy=dynamic 时不绑定 worker，只按 estimated_cost
   降序排出提交顺序（LPT）。
```

验收：

```text
round_robin tests 保持当前 assignment（静态路径逐位不变）
dynamic 在不均匀 sample_count / steps 下 imbalance 降低（telemetry 证明）
output gather order 不变（gather 按 chunk index，与派发无关）
```

### T3. Stage scheduler prototype behind explicit flag

新增实验入口，不替换默认 runtime：

```yaml
rollout:
  stage_pipeline:
    enabled: false
    scheduler: bounded_queue
```

首版只支持：

```text
single-node
one denoise stage
one decode_latents stage
bounded queues
driver-owned scheduler
Ray actor stage workers
```

不做 cross-node relay，不做 fan-in，不做 stream routing。

Scheduler contract 对齐 SGLang-Omni 的最小子集：

```text
inbox / outbox
new_request equivalent
result / error envelope
request_cost_fn
max_batch_wait_ms
max_batch_cost
max_concurrency OR batch mode, not both
abort callback
```

### T4. Measurement gate

只有当 profile 证明下面条件成立，才继续：

```text
stage overlap saves more than queue + relay overhead
decode/reward tail is visible in wall time
tensor relay cost is measured, not guessed
policy sync drain remains correct
OOM retry remains correct
```

失败时保留 Track A 的 chunk scheduler improvements，不推进 physical stage runtime。

## 7. What We Should Not Do

```text
不要把当前 scheduler.py 继续扩成 mega scheduler。
不要把 ExecutionStage 改成 physical stage。
不要让 cost model 改变 sample identity / gather order。
不要让 Ray object store relay 成为长期 tensor transport 假设。
不要在没有 Level 2 measurement 前实现 CUDA IPC/NCCL relay。
不要复制 SGLang-Omni 的 AR/KV-cache scheduler 到 diffusion rollout。
不要把 ZMQ control plane 当作 VRL Ray actor pipeline 的默认控制面。
不要把 reward 未完成的 batch 放进 trainer ready queue。
不要绕过 policy_version barrier。
```

## 8. Acceptance Criteria

Track A acceptance:

```text
current round_robin behavior stays available
chunk scheduling metrics visible in GenerationOutput.runtime_debug
dynamic dispatch has deterministic tests (fake actors, controlled completion order)
existing Ray generation runtime tests pass
```

Track B acceptance:

```text
stage scheduler runs only behind explicit flag
bounded queues enforce backpressure
batch mode and max_concurrency are mutually exclusive
per-payload errors are emitted as result envelopes
abort callback cleans up stage-owned payload state
stage metrics include queue wait / execution / relay bytes
policy_version drain barrier has tests
OOM retry behavior is documented for stage failures
```

## 9. References

- `docs/sprints/parked/SPRINT_physical_stage_runtime.md`
- `docs/sprints/parked/SPRINT_diffusion_rollout_stage_pipeline.md`
- `docs/sprints/done/SPRINT_runtime_block_policies.md`
- `vrl/generation/execution/chunk_placement.py`
- `vrl/generation/execution/planner.py`
- `vrl/generation/execution/chunks.py`
- `vrl/generation/ray/executor.py`
- `vrl/ray/actor_pool.py`
- `vrl/generation/pipeline/topology.py`
- `vrl/generation/pipeline/payload.py`
- `vrl/generation/ray/pipeline_runner.py`
- `vrl/generation/ray/stage_worker.py`
- `/home/mingfeiguo/Desktop/sglang-omni/sglang_omni/scheduling/simple_scheduler.py`
- `/home/mingfeiguo/Desktop/sglang-omni/sglang_omni/scheduling/messages.py`
- `/home/mingfeiguo/Desktop/sglang-omni/sglang_omni/pipeline/stage/runtime.py`
- `/home/mingfeiguo/Desktop/sglang-omni/sglang_omni/pipeline/coordinator.py`
