# SPRINT：清理 vrl.engine legacy 边界

状态：implemented。

## 背景

`vrl.engine` 现在同时承载三类东西：

- 当前主路径：`GenerationRequest` / `OutputBatch` / capability / engine plan / Ray chunk execution / trajectory schema。
- family-neutral helper：AR KV decode、diffusion denoise chunk、trajectory build/resolve/validation。
- 历史遗留 facade：旧 in-process registry worker runtime、过宽 package re-export、已经不作为入口的 Protocol 名称。

这轮 sprint 的目标不是重写 engine，而是删除或降级 legacy surface，让 engine 的 public contract 和当前 rollout runtime 一致。

当前主路径应该是：

```text
RolloutCollector
  -> RolloutBackend.generate(GenerationRequest)
  -> Ray rollout runtime / family runtime executor
  -> OutputBatch + TrajectoryBatch
  -> collector batch_builder / evaluator
```

不再把 `vrl.engine` 当成一个可注册 executor 并本地执行的通用 facade。

## 核心结论

应该保留的边界：

```text
vrl/engine/core/types.py
vrl/engine/core/capabilities.py
vrl/engine/core/runtime_spec.py
vrl/engine/execution/planner.py
vrl/engine/execution/microbatching.py
vrl/engine/execution/gather.py
vrl/engine/execution/batching.py
vrl/engine/ar/*
vrl/engine/diffusion/*
vrl/engine/trajectory/*
```

需要清理的 legacy 边界：

```text
vrl/engine/core/protocols.py
  CapabilityAwareFamilyPipelineExecutor
  PlanAwareFamilyPipelineExecutor
  PlanAwareBatchedFamilyPipelineExecutor

vrl/engine/core/registry.py
  ExecutorKey
  FamilyPipelineRegistry

vrl/engine/execution/worker.py
  GenerationWorker
  private local execution helpers

vrl/engine/execution/runtime.py
  GenerationRuntime
```

需要保留但可能搬家的部分：

```text
vrl/engine/execution/worker.py
  GenerationIdFactory
```

`GenerationIdFactory` 仍被 Ray rollout executor、chunk gather tests、Janus-R1 tests 使用。它不是 legacy；legacy 是它所在文件里夹带的 local registry worker。

## 当前引用事实

### `core/protocols.py`

仍在用：

```text
PipelineChunkResult
ChunkedFamilyPipelineExecutor
FamilyPipelineExecutor
```

其中 `PipelineChunkResult` 是 family chunk payload 的 typing contract；`ChunkedFamilyPipelineExecutor` 是 Ray worker / gather path 的最低分布式 chunk contract。

legacy-ish：

```text
CapabilityAwareFamilyPipelineExecutor
PlanAwareFamilyPipelineExecutor
PlanAwareBatchedFamilyPipelineExecutor
```

这些名字现在主要只是 re-export。实际调用点没有用它们做 `isinstance` 或正式接口入口，而是直接按方法 duck-type：

```python
plan_method = getattr(executor, "plan", None)
forward_plan = getattr(executor, "forward_plan", None)
forward_batch_plan = getattr(executor, "forward_batch_plan", None)
```

所以这些 Protocol 现在增加了“看起来是 public contract，但运行时并不依赖”的噪音。

### `core/registry.py` + `execution/worker.py` + `execution/runtime.py`

这是一组旧 local in-process runtime：

```text
FamilyPipelineRegistry
  -> GenerationWorker
  -> GenerationRuntime
```

当前 production rollout path 走 `RolloutBackend`，Ray path 走 `vrl.distributed.ray.rollout.*`，family runtime 由 `RuntimeBuildSpec` / `RuntimeBundle` builder 构建。没有看到 production code 构造 `GenerationWorker(...)` 或 `FamilyPipelineRegistry(...)`。

需要拆掉这条链，但不能误删 `GenerationIdFactory`，因为它仍用于生成 `GenerationSampleSpec`。

### `execution/batching.py`

保留。`forward_batch_by_merging_prompts(...)` 被 AR 和 diffusion executor 的 `forward_batch_plan(...)` 使用：

```text
vrl/engine/ar/executor_base.py
vrl/engine/diffusion/executor_base.py
```

它不是 registry worker legacy。

### `execution/gather.py`

保留。Ray rollout executor 和 diffusion family registry 使用：

```text
gather_pipeline_chunks(...)
DiffusionChunkGatherer
require_chunk_gatherer(...)
```

它是分布式 chunk path 的主路径。

### `execution/microbatching.py` 和 `execution/planner.py`

保留。planner、Ray planner、family executors 都依赖：

```text
MicroBatchPlan
EnginePlan
ExecutionUnit
build_engine_plan(...)
attach_engine_plan(...)
```

这组是当前 engine plan contract。

### `ar/*`

保留。当前 AR rollout KV cache path 依赖：

```text
run_kv_decode(...)
TokenScheduler
ActiveSequence
ARGenerationSpec
ARPipelineExecutorBase
ar_split_rows(...)
ar_concat_rows(...)
```

这组不是 legacy。

### `diffusion/*`

保留。`VideoGenerationRequest` 看起来名字偏旧，但仍是 diffusion model-facing request adapter，被 SD3/Wan/Cosmos model 和 runtime 使用。不要在本 sprint 里删除。

后续如果要改名，应该单独做 “diffusion model request naming cleanup”，因为它跨 family model method signatures。

### `trajectory/*`

保留。它是 rollout source-of-truth schema 和 replay/evaluator resolver 的主路径：

```text
TrajectoryBatch
TrajectorySegment
TrajectoryTensor
ReplayInput
RewardView
TrainingView
build_*_trajectory(...)
trajectory_replay_tensor_dict(...)
require_output_trajectory(...)
validate_trajectory_batch(...)
```

不要在这个 sprint 里动 trajectory schema。

## Phase 1：收窄 public exports

目标：先把 legacy 从 public surface 上拿掉，再删实现。

编辑：

```text
vrl/engine/__init__.py
vrl/engine/core/__init__.py
vrl/engine/execution/__init__.py
```

移除 re-export：

```text
CapabilityAwareFamilyPipelineExecutor
PlanAwareFamilyPipelineExecutor
PlanAwareBatchedFamilyPipelineExecutor
ExecutorKey
FamilyPipelineRegistry
GenerationWorker
GenerationRuntime
```

保留 re-export：

```text
GenerationIdFactory
RolloutBackend
GenerationRequest
GenerationSampleSpec
OutputBatch
GenerationMetrics
GenerationRuntimeSpec
FamilyCapability
PipelineChunkResult
ChunkedFamilyPipelineExecutor
build_engine_plan
attach_engine_plan
gather_pipeline_chunks
forward_batch_by_merging_prompts
```

要求：

- 先改 imports，不删实现。
- 跑引用扫描，确认新代码没有从 `vrl.engine` 顶层拿 legacy 名字。

扫描 gate：

```text
rg 'from vrl\.engine import .*GenerationWorker|from vrl\.engine import .*GenerationRuntime|from vrl\.engine import .*FamilyPipelineRegistry|from vrl\.engine import .*PlanAware|from vrl\.engine\.core import .*PlanAware|from vrl\.engine\.core import .*FamilyPipelineRegistry' vrl tests
```

## Phase 2：拆出 `GenerationIdFactory`

目标：从 `execution/worker.py` 里保留有用部分，隔离 legacy worker。

新增：

```text
vrl/engine/execution/ids.py
```

移动：

```text
GenerationIdFactory
```

更新引用：

```text
vrl/distributed/ray/rollout/executor.py
vrl/engine/__init__.py
vrl/engine/execution/__init__.py
tests/engine/generation/test_chunk_gatherer.py
tests/engine/generation/test_generation_contracts.py
tests/models/test_janus_r1_model.py
```

临时兼容策略：

- 第一小步可以在 `execution/worker.py` re-export `GenerationIdFactory`，减少 blast radius。
- 第二小步删除这个兼容 re-export。

完成后 gate：

```text
rg 'vrl\.engine\.execution\.worker import GenerationIdFactory|from vrl\.engine\.execution\.worker import' vrl tests
```

预期：无 production 命中。测试也改到 `vrl.engine.execution.ids` 或 `vrl.engine` 顶层。

## Phase 3：删除 local registry worker runtime

目标：删除旧的 in-process executor registry path。

删除：

```text
vrl/engine/core/registry.py
vrl/engine/execution/worker.py
vrl/engine/execution/runtime.py
```

同步删除或迁移：

```text
ExecutorKey
FamilyPipelineRegistry
GenerationWorker
GenerationRuntime
```

保留：

```text
RolloutBackend
```

`RolloutBackend` 是 collector-facing backend protocol，仍被 rollout runtime 使用。删除 `execution/runtime.py` 前要把它搬到更准确的位置：

推荐新增：

```text
vrl/engine/core/backend.py
```

或：

```text
vrl/rollouts/runtime/protocol.py
```

更推荐 `vrl/engine/core/backend.py`，因为 `RolloutBackend.generate(GenerationRequest) -> OutputBatch` 仍是 engine contract。

迁移引用：

```text
vrl/rollouts/collector/core.py
vrl/rollouts/collector/factory.py
vrl/rollouts/runtime/backend.py
vrl/distributed/ray/rollout/runtime.py
```

删除后 gate：

```text
rg 'GenerationWorker|GenerationRuntime|FamilyPipelineRegistry|ExecutorKey|vrl\.engine\.execution\.worker|vrl\.engine\.execution\.runtime|vrl\.engine\.core\.registry' vrl tests
```

允许保留的命中：

```text
SPRINT_engine_legacy_cleanup.md
```

其他命中必须解释或删除。

## Phase 4：删除 unused Protocol facade

目标：让 `core/protocols.py` 只保留真实 contract。

编辑：

```text
vrl/engine/core/protocols.py
vrl/engine/core/__init__.py
vrl/engine/__init__.py
```

删除：

```text
CapabilityAwareFamilyPipelineExecutor
PlanAwareFamilyPipelineExecutor
PlanAwareBatchedFamilyPipelineExecutor
```

保留：

```text
PipelineChunkResult
FamilyPipelineExecutor
ChunkedFamilyPipelineExecutor
```

注意：

- `FamilyPipelineExecutor` 目前只被 registry 和 Protocol inheritance 使用。删除 registry 后它可能也会变成无实际 consumer。
- 如果 Phase 3 后 `FamilyPipelineExecutor` 只剩 `ChunkedFamilyPipelineExecutor` inheritance 使用，可以进一步把 `family/task/workload_signature` 直接写进 `ChunkedFamilyPipelineExecutor`，再删 `FamilyPipelineExecutor`。
- 这一步先按扫描结果决定，不预设硬删。

Gate：

```text
rg 'CapabilityAwareFamilyPipelineExecutor|PlanAwareFamilyPipelineExecutor|PlanAwareBatchedFamilyPipelineExecutor' vrl tests
```

预期：无命中。

## Phase 5：收窄 package-level exports

目标：减少 `vrl.engine` 顶层“什么都能 import”的错觉。

保留顶层导出：

```text
GenerationRequest
GenerationSampleSpec
GenerationMetrics
OutputBatch
GenerationRuntimeSpec
RolloutBackend
GenerationIdFactory
```

考虑保留但不鼓励：

```text
FamilyCapability
PipelineChunkResult
ChunkedFamilyPipelineExecutor
```

建议改为从具体模块导入：

```text
vrl.engine.core.capabilities
vrl.engine.core.protocols
vrl.engine.execution.planner
vrl.engine.execution.gather
vrl.engine.trajectory
vrl.engine.ar
vrl.engine.diffusion
```

移除顶层导出：

```text
AxisCapability
ExecutionUnitCapability
build_engine_plan
attach_engine_plan
forward_batch_by_merging_prompts
gather_pipeline_chunks
profiler_label_for_unit
resolve_executor_capability
```

理由：

这些是 engine internals / planning helpers，不该从 `vrl.engine` 顶层暴露成 public app API。

这一步风险较高，因为测试和部分 code 可能依赖顶层 convenience import。先做扫描，再逐个迁到具体模块。

Gate：

```text
rg 'from vrl\.engine import' vrl tests
```

所有保留下来的顶层 import 必须只拿 core dataclass / backend protocol / id factory。

## Phase 6：删除本地 pycache 和死文件残留

目标：清理工作树里的非源码缓存，不影响 git history。

本地删除：

```text
find vrl/engine -type d -name __pycache__ -prune -exec rm -rf {} +
find vrl/engine -type f -name '*.pyc' -delete
```

注意：

- 这些通常不在 git 里。
- 不作为主要代码 commit 的核心内容。
- 如果 git 没 track，不需要单独提交。

## 不做的事

本 sprint 不做：

- 不改 `TrajectoryBatch` schema。
- 不改 AR KV decode algorithm。
- 不改 diffusion denoise math。
- 不改 Ray rollout actor lifecycle。
- 不把 `VideoGenerationRequest` 改名。
- 不改 `RuntimeBuildSpec` / `RuntimeBundle`。
- 不改 collector/evaluator/algorithm contract。
- 不删除 `forward_batch_by_merging_prompts(...)`。
- 不删除 `DiffusionChunkGatherer` / `gather_pipeline_chunks(...)`。

## 验收标准

完成后应该满足：

- `GenerationWorker` / `GenerationRuntime` / `FamilyPipelineRegistry` 不再出现在 production code。
- `GenerationIdFactory` 有独立模块，不依赖 legacy worker 文件。
- `RolloutBackend` 仍可被 collector 和 rollout runtime 使用。
- `PlanAware*` / `CapabilityAware*` Protocol 删除。
- 顶层 `vrl.engine` re-export 缩小到稳定 public dataclass/protocol。
- Ray chunk execution、family executors、collector runtime、trajectory resolver 测试保持通过。
- legacy 扫描无命中，除 sprint 文档自身。

## 验证计划

至少运行：

```text
pytest tests/engine/generation tests/engine/ar
pytest tests/distributed/ray/test_rollout_launcher.py tests/distributed/ray/test_ray_resident_session.py
pytest tests/rollouts
pytest tests/models/test_janus_replay.py tests/models/test_janus_r1_model.py tests/models/test_diffusion_model_base.py
pytest tests/trainers/test_online.py tests/trainers/test_weight_sync.py
python -m compileall vrl/engine vrl/rollouts vrl/models tests/engine tests/rollouts
git diff --check
```

Legacy scan：

```text
rg 'GenerationWorker|GenerationRuntime|FamilyPipelineRegistry|ExecutorKey|CapabilityAwareFamilyPipelineExecutor|PlanAwareFamilyPipelineExecutor|PlanAwareBatchedFamilyPipelineExecutor' vrl tests
rg 'vrl\.engine\.execution\.worker|vrl\.engine\.execution\.runtime|vrl\.engine\.core\.registry' vrl tests
```

Public import scan：

```text
rg 'from vrl\.engine import' vrl tests
```

## 参考文件

Current main path:

```text
vrl/engine/core/types.py
vrl/engine/core/capabilities.py
vrl/engine/core/runtime_spec.py
vrl/engine/execution/planner.py
vrl/engine/execution/microbatching.py
vrl/engine/execution/gather.py
vrl/engine/execution/batching.py
vrl/engine/ar/decode_loop.py
vrl/engine/diffusion/executor_base.py
vrl/engine/trajectory/types.py
vrl/engine/trajectory/resolver.py
```

Legacy target:

```text
vrl/engine/core/protocols.py
vrl/engine/core/registry.py
vrl/engine/execution/worker.py
vrl/engine/execution/runtime.py
vrl/engine/__init__.py
vrl/engine/core/__init__.py
vrl/engine/execution/__init__.py
```
