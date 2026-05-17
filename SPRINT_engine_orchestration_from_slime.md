# SPRINT：从 Slime 拆解 RL Rollout Orchestration

状态：implemented。

实现摘要：

```text
vrl/rollouts/orchestration
  strict_on_policy: 默认同步 schedule，保持旧行为。
  one_batch_overlap: 显式开启的 Slime-style shallow async schedule。

OnlineTrainer
  不再直接 collector.collect(...)
  不再直接 weight_syncer.push(...)
  不再直接管理 rollout runtime release / driver offload
```

## 一句话目标

把 `rollout -> reward -> replay/trajectory -> signal -> train -> weight sync` 的节奏从 `OnlineTrainer._step_impl()` 里抽出来，形成 repo 自己的 RL rollout schedule 层，并在本 sprint 内交付一个受保护的 `one_batch_overlap` async schedule。

这不是 vLLM / SGLang 那种底层 inference engine 调度 sprint。底层 generation 继续放在 `vrl/generation`，RL 节奏放在 `vrl/rollouts/orchestration`。

只做代码搬家不算完成。`strict_on_policy` 是行为保持和 debug fallback；真正的新增能力是：

```text
train batch N while rollout batch N+1 is already running,
then block weight sync until in-flight rollout is complete.
```

边界锚点：

```text
target == Slime-style shallow one-batch overlap
target != strict on-policy only
target != full async rollout queue
```

```text
vrl/generation
  owns: GenerationRequest -> OutputBatch
  owns: Ray chunk execution, microbatch execution, worker policy_version guard
  does not own: reward, advantage, replay-to-signal, train loop, stale batch policy

vrl/rollouts/orchestration
  owns: rollout_id, policy_version, collect timing, reward timing, runtime release,
        train-data handoff metadata, future async rollout schedule

vrl/trainers/online
  owns: optimizer, advantage computation, evaluator replay, algorithm loss,
        backward, optimizer step
```

## 为什么不是改 `vrl/generation`

`vrl/generation` 已经有正确的边界雏形：

```python
@dataclass(slots=True)
class GenerationRequest:
    ...
    priority: int = 0
    policy_version: int | None = None
```

`policy_version` 在 generation 里应该只是执行安全线，不应该变成 RL schedule。

Ray runtime 已经会把当前 rollout worker 版本注入 request：

```python
if request.policy_version is None and self.current_policy_version is not None:
    request = replace(request, policy_version=self.current_policy_version)
```

Ray executor 也已经检查 chunk 版本：

```python
if request.policy_version is not None and result.policy_version != request.policy_version:
    raise RuntimeError(...)
```

所以 generation 层做的是：

```text
这个 request 用哪个 policy version 执行
worker 是否真的用了这个 version
chunk 怎么分发和 gather
```

RL orchestration 层才应该做：

```text
什么时候 collect
什么时候 reward
什么时候 train
什么时候 sync weights
是否允许 train batch N 时 generate batch N+1
允许 stale 几个 policy version
```

## Slime 实际做法

Slime 的关键点不是它自己写了低层 inference scheduler，而是它在 RL driver 层管理 rollout/training 两个系统。

同步入口：

```python
rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
actor_model.update_weights()
```

异步入口：

```python
rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)

for rollout_id in range(args.start_rollout_id, args.num_rollout):
    rollout_data_curr_ref = ray.get(rollout_data_next_future)

    if rollout_id + 1 < args.num_rollout:
        rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

    ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))

    if (rollout_id + 1) % args.update_weights_interval == 0:
        rollout_data_curr_ref = ray.get(x) if (x := rollout_data_next_future) is not None else None
        rollout_data_next_future = None
        actor_model.update_weights()
```

这个设计说明三件事：

1. rollout/train overlap 在 RL driver 层。
2. SGLang 只负责 serving / inference execution。
3. 更新权重前必须等 in-flight rollout 结束，避免一个 rollout 中途混 policy。

Slime 的 `RolloutManager.generate(...)` 也是 RL 数据桥：

```text
_get_rollout_data(...)
  -> _convert_samples_to_train_data(...)
  -> _split_train_data_by_dp(...)
```

它不是 KV cache scheduler。KV cache、continuous batching、prefix cache、CUDA graph 都是 SGLang 内部能力。

## 和 Slime 对齐程度

本 sprint 的 `one_batch_overlap` 对齐的是 Slime `train_async.py` 的 shallow async schedule。它不是 strict on-policy，也不是 full async rollout queue。

对齐点：

```text
1. 先启动 rollout future。
2. 当前 batch ready 后，立刻启动下一轮 rollout future。
3. 当前 batch training 和下一轮 rollout 并行。
4. weight update 前等待 in-flight rollout 完成。
5. async 不支持 colocate。
```

Slime 对应代码：

```python
rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)

for rollout_id in range(args.start_rollout_id, args.num_rollout):
    rollout_data_curr_ref = ray.get(rollout_data_next_future)

    if rollout_id + 1 < args.num_rollout:
        rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

    ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))

    if (rollout_id + 1) % args.update_weights_interval == 0:
        rollout_data_curr_ref = ray.get(x) if (x := rollout_data_next_future) is not None else None
        rollout_data_next_future = None
        actor_model.update_weights()
```

差异点：

```text
Slime:
  rollout_data_next_future 是 Ray ObjectRef。
  actor_model.update_weights() 内部同步 Megatron -> SGLang。
  rollout_id 是主版本节奏，weight version 由 updater/engine 管。

wm-infra:
  pending rollout future 会是 asyncio task / runtime collect task。
  weight sync 通过 RayRuntimeWeightSyncer -> GenerationRuntime.update_weights(...)。
  GenerationRequest.policy_version / RayChunkResult.policy_version 已经是一等字段，所以 batch metadata 必须显式记录 rollout_policy_version。
```

不对齐的部分：

```text
Slime examples/fully_async 是另一个模式：
  background worker 常驻
  持续从 data buffer 拉 prompt
  trainer 只 drain completed queue

本 sprint 不实现这个模式。
```

所以判断是：

```text
one_batch_overlap == Slime train_async.py 的 RL driver overlap
one_batch_overlap != strict on-policy train.py fallback
one_batch_overlap != Slime fully_async_rollout.py 的 background producer
```

## 当前 wm-infra 状态

### 已经有的东西

`RolloutCollector` 已经是干净的单次 collect 单元：

```python
output = await self.runtime.generate(collector_request.request)
batch = await self._output_batch_to_rollout_batch(...)
```

它负责：

```text
GenerationRequest build
generation runtime call
reward scoring
OutputBatch -> RolloutBatch
```

`vrl/generation` 已经是独立 runtime：

```text
GenerationRequest
OutputBatch
DistributedGenerationExecutor
RayGenerationRuntime
RayGenerationWorker
GenerationWeightSync
```

Ray worker 已经有 policy version 保护：

```text
request.policy_version
worker.current_policy_version
RayChunkResult.policy_version
```

`ReleasableRayGenerationRuntime` 已经能释放 rollout workers：

```text
generate(...)
update_weights(...)
release_memory(...)
shutdown(...)
```

### 缺的东西

`OnlineTrainer._step_impl()` 现在同时做了太多事：

```text
ensure rollout weights
offload driver model
collect prompts
release rollout runtime
restore driver model
compute advantages
filter/rebatch
evaluate replay
compute loss
backward
optimizer step
sync weights
write profiling events
```

这就是当前最大问题。它不是“trainer 训练模型”，而是一个混合了 RL orchestration、rollout collection、memory lifecycle 和 optimizer loop 的大脚本。

目标不是把这些塞到 `vrl/generation`，而是拆出：

```text
vrl/rollouts/orchestration
```

## 目标调用链

`strict_on_policy` fallback：

```text
OnlineTrainer.step(...)
  -> rollout_schedule.collect(...)
       ensure initial rollout weights
       maybe offload driver model
       collect prompts through RolloutCollector
       release rollout runtime memory
       maybe restore driver model
       return RolloutIteration

  -> OnlineTrainer.train_on_rollout(...)
       compute advantages
       filter/rebatch
       evaluator replay
       algorithm loss
       backward
       optimizer step

  -> rollout_schedule.after_train_step(...)
      sync new trainable weights to rollout runtime
      bump policy_version
```

本 sprint 必须交付的 async path：

```text
OnlineTrainer.step(...)
  -> rollout_schedule.next_iteration(...)
       returns complete RolloutIteration N
       starts RolloutIteration N+1 in background when safe

  -> OnlineTrainer.train_on_rollout(iteration N)
       compute advantages
       evaluator replay
       algorithm loss
       optimizer step

  -> rollout_schedule.after_train_step(...)
       wait for any in-flight rollout if weight sync would mutate workers
       sync new trainable weights
       bump policy_version
```

## 新增目录设计

```text
vrl/rollouts/orchestration/
  __init__.py
  types.py
  schedule.py
  strict_on_policy.py
  one_batch_overlap.py
  lifecycle.py
  prompt_collection.py
```

### `types.py`

定义 RL schedule 层自己的数据结构。

```text
RolloutScheduleMode
  strict_on_policy
  one_batch_overlap

RolloutScheduleState
  rollout_id
  current_policy_version
  initialized
  pending_rollout

RolloutIteration
  rollout_id
  policy_version
  mode
  batches
  prompt_count
  sample_count
  phase_times
  metadata
```

注意：这里的 `policy_version` 是 RL 层 batch metadata。generation runtime 仍然是执行校验的 source of truth。

### `lifecycle.py`

收拢现在散在 trainer 里的 runtime lifecycle：

```text
ensure_initial_weights(...)
maybe_offload_driver_model_for_rollout(...)
release_rollout_runtime_memory(...)
restore_driver_model_after_rollout(...)
sync_weights_after_train(...)
```

这些函数不应该理解 algorithm，也不应该理解 evaluator。

### `prompt_collection.py`

收拢现在 `OnlineTrainer._step_impl()` 里处理 prompts 的逻辑：

```text
plain string prompts
PromptExample
group_size
runtime_debug on first step
group_id remap
split batch by group
```

它应该返回 `list[RolloutBatch]`，不计算 advantage，不训练。

### `strict_on_policy.py`

行为保持 schedule，也是单 GPU / colocate debug fallback。

职责：

```text
collect exactly one on-policy rollout batch set
attach rollout_id / policy_version into RolloutBatch.context
release rollout runtime if configured
expose phase timing
sync weights after train
```

非职责：

```text
no evaluator replay
no advantage computation
no algorithm loss
no optimizer step
no generation chunk planning
```

## Schedule 模式

### 1. `strict_on_policy`

默认安全模式，但不能作为本 sprint 的完成标准。

```text
policy_version N
  collect rollout batch with N
  reward
  build RolloutBatch
  train
  sync weights as N+1
```

特点：

```text
no stale data
no train/rollout overlap
same behavior as current trainer
best debug path
```

完成后，`OnlineTrainer` 不应该再直接调用：

```text
collector.collect(...)
_collector_runtime_requires_driver_model_offload(...)
_release_collector_runtime_memory(...)
_move_model_to_device(..., "cpu")
weight_syncer.push(...)
```

这些都应通过 schedule/lifecycle。

### 2. `one_batch_overlap`

本 sprint 的 functional target。默认关闭，但必须实现并能通过 fail-fast guard 显式打开。

```text
generate batch N
start generate batch N+1
train batch N
wait batch N+1 before weight update barrier
sync weights
```

限制：

```text
requires separate rollout/train GPU ownership
disabled when runtime.requires_driver_model_offload is true
max_stale_policy_versions = 1
one RolloutIteration must contain exactly one rollout_policy_version
weight sync cannot run while a rollout future is in flight
```

这个模式等价于 Slime `train_async.py` 的第一层 async，不等价于 fully async replay buffer。

### 3. `bounded_stale_replay`

暂时只写设计，不实现。

```text
background rollout producer
bounded queue
trainer consumes completed batches
batch may be stale by <= K policy versions
algorithm must explicitly accept stale policy correction
```

当前不要做。没有 versioned replay queue、old policy correction 和 clear metrics 前，做它会让训练语义变模糊。

## Phase 1：边界、类型和 schedule interface

新增：

```text
vrl/rollouts/orchestration/__init__.py
vrl/rollouts/orchestration/types.py
vrl/rollouts/orchestration/schedule.py
```

内容：

```text
RolloutScheduleMode
RolloutScheduleState
RolloutIteration
RolloutSchedule protocol/base class
```

要求：

- 不 import `vrl.generation.ray` 具体实现。
- 不 import algorithm。
- 不 import evaluator。
- 只依赖 `RolloutBatch` 和通用 typing。

完成标准：

- `RolloutIteration` 能完整表达一个 trainer step 消费的 rollout 数据。
- `RolloutIteration.metadata` 至少包含：

```text
rollout_id
rollout_policy_version
schedule_mode
prompt_count
sample_count
```

## Phase 2：抽出 strict collect lifecycle

新增：

```text
vrl/rollouts/orchestration/lifecycle.py
vrl/rollouts/orchestration/prompt_collection.py
vrl/rollouts/orchestration/strict_on_policy.py
```

从 `OnlineTrainer._step_impl()` 移出：

```text
_ensure_rollout_weights_initialized()
offload_driver_model_for_rollout
pending_prompts / pending_indices / flush_pending_prompts
PromptExample collect path
release_rollout
restore_driver_model
```

保留在 `OnlineTrainer`：

```text
advantage computation
zero-advantage filtering
evaluator replay
algorithm loss
backward
optimizer step
EMA
metrics aggregation
debug first-step logprob parity
```

完成标准：

- `OnlineTrainer._step_impl()` 里的 collect 阶段变成一个 schedule 调用。
- 训练行为不变。
- `RolloutBatch.context` 里能看到 `rollout_id` 和 `rollout_policy_version`。
- 这一步只是 async 的底座，不是 sprint 终点。

## Phase 3：把 post-train weight sync 归入 schedule lifecycle

当前 trainer 末尾直接做：

```python
if self.weight_syncer is not None:
    await self.weight_syncer.push(self.sync_state_getter())
```

这必须先变成：

```text
rollout_schedule.after_train_step(...)
```

原因：

```text
strict_on_policy: train 完立刻 sync
one_batch_overlap: sync 前必须先等 in-flight rollout future 完成
future stale replay: sync cadence 可能不是每 step 一次
```

完成标准：

- `OnlineTrainer` 不直接调用 `weight_syncer.push(...)`。
- schedule lifecycle 是唯一 train -> rollout weight sync 入口。
- sync 后能从 runtime 读到新的 `current_policy_version`。

## Phase 4：policy version 明确化

目标：RL 层显式记录 batch 版本，generation 层继续做执行保护。

规则：

```text
initial rollout weight sync creates policy_version 1
collect uses current runtime policy_version
RolloutIteration.policy_version == GenerationRequest.policy_version
every RolloutBatch in one RolloutIteration must have same rollout_policy_version
post-train sync increments runtime policy_version
```

注意当前 `RayRuntimeWeightSyncer` 已经内部维护 next policy version：

```text
push(state) -> runtime.update_weights(state, policy_version)
```

所以第一版不要新造另一个 version counter。schedule 读取 runtime/current batch metadata，避免两个 source of truth。

完成标准：

- batch metadata 能回答：

```text
这个 batch 是第几轮 rollout
这个 batch 用哪个 policy version 生成
这个 batch 是 strict_on_policy 还是 overlap
```

- generation 仍然不理解 reward / advantage / trainer。

## Phase 5：实现 `one_batch_overlap`

这是本 sprint 的 functional target，不是后续优化项。这阶段只实现受保护的 overlap，不做 fully async。

新增：

```text
RolloutScheduleMode.ONE_BATCH_OVERLAP
vrl/rollouts/orchestration/one_batch_overlap.py
OneBatchOverlapRolloutSchedule
```

Config 形状：

```yaml
rollout_orchestration:
  mode: one_batch_overlap        # strict_on_policy | one_batch_overlap
  max_pending_rollouts: 1        # this sprint only supports 1
  require_separate_gpus: true
  weight_sync_barrier: before_sync
```

`max_pending_rollouts > 1` 属于 bounded stale replay，不在本 sprint 实现。

Guard：

```text
if runtime.requires_driver_model_offload:
    fail fast
if rollout and trainer share the same GPU ownership:
    fail fast
if reward_fn is known blocking on trainer GPU:
    fail fast
```

第一版 overlap 只支持 rollout GPU 和 trainer GPU 独立。单 GPU / colocate debug 继续走 `strict_on_policy`。

Schedule：

```text
prepare first future
for step:
  current = await pending_future
  pending_future = create next collect future
  train current
  before weight sync:
    await pending_future if sync cadence requires new weights now
    sync weights
```

完成标准：

- overlap mode 必须显式打开。
- default remains `strict_on_policy` for single-GPU and colocate debug.
- 一个 `RolloutIteration` 内禁止混多个 policy version。
- weight sync 时没有 in-flight rollout 使用旧 runtime state。
- at least one smoke path can prove rollout collect and train overlap in wall-clock trace.

## Phase 6：metrics 和 trace

Schedule 层需要输出稳定 phase metrics：

```text
rollout.schedule_wait_s
rollout.collect_s
rollout.reward_s
rollout.release_runtime_s
rollout.restore_driver_s
rollout.weight_sync_s
rollout.policy_version
rollout.sample_count
rollout.prompt_count
```

这些和 generation metrics 分开：

```text
generation.queue_wait_s
generation.execution_s
generation.micro_batches
generation.execution_stages
generation.engine_counters
```

完成标准：

- `TrainStepMetrics.phase_times` 能同时看到 rollout schedule 和 train phase。
- 不把 reward/advantage metrics 写进 `GenerationMetrics`。
- debug JSONL 里能看到 rollout_id / policy_version。

## Phase 7：清理 compatibility 命名

状态：done。

`rollout runtime` 旧路径已经删除，runtime 命名收敛到 generation runtime：

```text
generation runtime config/build stays in vrl/generation/runtime
RL schedule stays in vrl/rollouts/orchestration
rollout family wiring stays in vrl/rollouts/families
```

完成标准：

- 新代码不再从 `vrl.rollouts.runtime` 引入 generation runtime。
- 不保留 `vrl.rollouts.runtime` compatibility export。
- 不保留 `RolloutBackendConfig` / `build_rollout_backend_from_cfg` runtime alias。

## 明确非目标

本 sprint 不做：

```text
不实现 KV cache allocator
不实现 vLLM/SGLang continuous batching
不把 reward/advantage 放进 vrl/generation
不做 fully async replay buffer
不允许一个 rollout batch 混多个 policy version
不重写 diffusion denoise scheduler
不重写 AR token scheduler
```

## 架构边界 gate

这些 gate 用来防止边界再次混乱。

`vrl/generation` 不应该 import RL orchestration：

```bash
rg "vrl\\.rollouts\\.orchestration|vrl\\.trainers|AlgorithmInput|RewardScorer" vrl/generation
```

`OnlineTrainer` 完成 Phase 2 后不应该直接管 collect lifecycle：

```bash
rg "collector\\.collect|_release_collector_runtime_memory|_collector_runtime_requires_driver_model_offload|_move_model_to_device\\(.*cpu" vrl/trainers/online/trainer.py
```

Phase 3 后不应该直接 sync rollout weights：

```bash
rg "weight_syncer\\.push" vrl/trainers/online/trainer.py
```

允许的关系：

```text
OnlineTrainer -> RolloutSchedule
RolloutSchedule -> RolloutCollector
RolloutCollector -> GenerationRuntime
GenerationRuntime -> generation workers
```

禁止的关系：

```text
GenerationRuntime -> RolloutSchedule
GenerationRuntime -> Algorithm
GenerationRuntime -> RewardScorer
GenerationRuntime -> Trainer
```

## 最小落地顺序

第一批实现可以分成两个 commits，但不要把 async 推到另一个 sprint。最小完整交付是：

```text
1. add vrl/rollouts/orchestration/types.py and schedule.py
2. add lifecycle.py and prompt_collection.py
3. add StrictOnPolicyRolloutSchedule as fallback
4. move collect/offload/release/restore out of OnlineTrainer
5. move post-train weight sync into schedule lifecycle
6. attach rollout_id / rollout_policy_version to RolloutBatch.context
7. add OneBatchOverlapRolloutSchedule behind fail-fast guards
8. expose schedule mode config
9. prove strict path keeps behavior identical
10. prove overlap path launches next rollout before current train finishes
```

可以拆到后续 sprint 的只有：

```text
bounded stale replay
background rollout producer
policy correction for stale batches
opportunistic free-GPU rollout scheduling
```

## 判断标准

完成这个 sprint 后，repo 应该能清楚回答：

```text
谁决定什么时候 rollout？
谁决定用哪个 policy version rollout？
谁保证 train 不吃混合 policy batch？
谁决定什么时候 sync weights？
谁决定 rollout/train 是否 overlap？
谁只负责 generation execution？
```

期望答案：

```text
RL schedule decides rollout timing.
Generation runtime executes a request with a pinned policy_version.
Collector converts generation output into rewarded RolloutBatch.
Trainer computes advantages, replay signals, loss, and optimizer update.
Schedule syncs trainable weights after trainer update.
OneBatchOverlapSchedule decides whether rollout/train overlap is allowed.
```

## 参考路径

Slime：

```text
/home/mingfeiguo/Desktop/slime/train.py
/home/mingfeiguo/Desktop/slime/train_async.py
/home/mingfeiguo/Desktop/slime/slime/ray/placement_group.py
/home/mingfeiguo/Desktop/slime/slime/ray/rollout.py
/home/mingfeiguo/Desktop/slime/slime/ray/actor_group.py
/home/mingfeiguo/Desktop/slime/slime/backends/megatron_utils/actor.py
/home/mingfeiguo/Desktop/slime/slime/rollout/sglang_rollout.py
```

wm-infra：

```text
/home/mingfeiguo/Desktop/wm-infra/vrl/trainers/online/trainer.py
/home/mingfeiguo/Desktop/wm-infra/vrl/trainers/online/collection.py
/home/mingfeiguo/Desktop/wm-infra/vrl/rollouts/collector/core.py
/home/mingfeiguo/Desktop/wm-infra/vrl/rollouts/collector/requests.py
/home/mingfeiguo/Desktop/wm-infra/vrl/rollouts/batch/__init__.py
/home/mingfeiguo/Desktop/wm-infra/vrl/rollouts/batch/ops.py
/home/mingfeiguo/Desktop/wm-infra/vrl/rollouts/families/registry.py
/home/mingfeiguo/Desktop/wm-infra/vrl/generation/types.py
/home/mingfeiguo/Desktop/wm-infra/vrl/generation/ray/runtime.py
/home/mingfeiguo/Desktop/wm-infra/vrl/generation/ray/executor.py
/home/mingfeiguo/Desktop/wm-infra/vrl/generation/ray/worker.py
/home/mingfeiguo/Desktop/wm-infra/vrl/trainers/weight_sync.py
```
