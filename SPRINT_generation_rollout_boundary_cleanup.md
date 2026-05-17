# SPRINT：Generation Runtime / RL Rollout 边界清理

状态：implemented。

## 核心结论

当前 repo 最大的命名混乱是把两件事都叫成 rollout / engine：

```text
RL rollout:
  prompts -> generation -> reward -> trainer RolloutBatch

Generation runtime:
  GenerationRequest -> family/Ray generation executor -> GenerationOutput
```

这两个东西必须拆开。以后规则是：

```text
rollout 只表示 RL data collection lifecycle
generation 只表示 pure model generation / serving execution runtime
engine 不再作为 package 或主架构名
```

目标不是重写模型，也不是接入外部 backend。外部 serving 系统只作为分层参考，不写进本 sprint 的目标目录、代码注释或实现计划。

本 sprint 的目标是先把边界清干净，让我们自己的 generation runtime 有明确位置：

```text
vrl/generation/
```

而不是继续塞进：

```text
vrl/rollouts/
vrl/engine/
vrl/distributed/ray/rollout/
```

## 当前问题

现在这些路径混在一起：

```text
vrl/engine/core/types.py
  GenerationRequest
  OutputBatch
  GenerationMetrics

vrl/engine/core/protocols.py
  RolloutBackend
  FamilyPipelineExecutor
  ChunkedFamilyPipelineExecutor

vrl/engine/ar/
vrl/engine/diffusion/
  family-specific generation executors

vrl/engine/trajectory/
  trainer replay record / training view

vrl/rollouts/runtime/
  rollout backend config/factory, but actually builds generation runtime

vrl/distributed/ray/rollout/
  Ray worker/runtime/executor/launcher, but actually executes generation chunks

vrl/rollouts/collector/
  actual RL rollout collector
```

最大的问题是这个 protocol：

```python
class RolloutBackend(Protocol):
    """Generation backend consumed by rollout collectors."""

    async def generate(self, request: GenerationRequest) -> OutputBatch: ...
```

这个不是 rollout backend。它是 generation runtime。collector consume 它来做 RL rollout。

## 目标目录

最终目标结构：

```text
vrl/
  generation/
    __init__.py
    types.py
    protocols.py
    capabilities.py
    launch_contract.py

    runtime/
      factory.py
      config.py
      launch_inputs.py

    execution/
      ids.py
      microbatching.py
      planner.py
      request_batch.py

    ar/
      executor.py
      layout.py
      token_loop/

    diffusion/
      executor.py
      gather.py
      layout.py

    ray/
      dependencies.py
      launcher.py
      runtime.py
      executor.py
      planner.py
      worker.py
      types.py
      weight_sync.py

  runtime/
    resources.py

  trajectory/
    __init__.py
    types.py
    builders.py
    views.py
    resolver.py
    ops.py
    validation.py

  rollouts/
    collector/
    evaluators/
    batch.py
    family_registry.py
    settings.py

```

最终清理后不再保留这些旧入口：

```text
vrl/engine/
vrl/engine/core/
vrl/engine/execution/
vrl/engine/ar/
vrl/engine/diffusion/
vrl/engine/trajectory/
vrl/distributed/
vrl/distributed/ray/rollout/
```

`vrl/rollouts/runtime/` 暂时仍保留 rollout-facing wrapper；`vrl/engine/` 和
`vrl/distributed/` 已删除。

## 依赖方向

必须变成单向依赖：

```text
rollouts.collector -> generation
rollouts.collector -> rewards
rollouts.collector -> trajectory

trainers/evaluators -> trajectory
trainers/evaluators -> models.interfaces.replay

generation -> models
generation -> utils
generation -> distributed.ray.dependencies

trajectory -> engine/generation-neutral typed data only
```

禁止：

```text
vrl/generation imports vrl.rollouts
vrl/generation imports vrl.rewards
vrl/generation imports vrl.trainers
vrl/generation imports vrl.algorithms

vrl/trajectory imports vrl.rollouts
vrl/trajectory imports vrl.rewards
vrl/trajectory imports vrl.trainers
vrl/trajectory imports vrl.generation.ar
vrl/trajectory imports vrl.generation.diffusion

vrl/distributed/ray imports vrl.rollouts
```

## 命名规则

### Keep `rollout` for RL collection only

保留：

```text
RolloutCollector
RolloutBatch
RolloutSettings
RewardRollout
rollouts/collector
rollouts/batch.py
```

这些确实属于 RL rollout。

### Rename generation runtime types

目标命名：

```text
OutputBatch -> GenerationOutput
RolloutBackend -> GenerationRuntime
RayDistributedRuntime -> RayGenerationRuntime
DistributedRolloutExecutor -> DistributedGenerationExecutor
RayRolloutWorker -> RayGenerationWorker
RayRolloutLauncher -> RayGenerationLauncher
RolloutWeightSync -> GenerationWeightSync
RolloutEngineRequestBuilder -> GenerationRequestBuilder
```

兼容期 alias：

```python
OutputBatch = GenerationOutput
RolloutBackend = GenerationRuntime
```

但新文件、新测试、新文档只能使用新名字。

### Avoid generic `engine`

`engine` 这个词只在这些地方可接受：

```text
engine_plan_id / legacy metrics field
compatibility import path
```

主包名不用 `engine` / `inference`，用：

```text
generation
execution
trajectory
```

## Phase 1：建立 `vrl.generation` 新入口

新增：

```text
vrl/generation/__init__.py
vrl/generation/types.py
vrl/generation/protocols.py
vrl/generation/capabilities.py
vrl/generation/launch_contract.py
```

第一阶段先建立 `vrl.generation`，迁移后由新路径承载真实实现。最终清理后 `vrl.engine` 已删除：

```python
# vrl/engine/core/types.py
from vrl.generation.types import GenerationRequest, GenerationOutput, OutputBatch
```

```python
# vrl/engine/core/protocols.py
from vrl.generation.protocols import GenerationRuntime, RolloutBackend
```

更新代码 import 到 `vrl.generation.*`，并删除旧路径。

验收：

- `from vrl.generation import GenerationRequest, GenerationOutput, GenerationRuntime` 可用。
- tests 不再依赖 `vrl.engine`。
- 新代码不再新增 `from vrl.engine.core...`。

## Phase 2：把 collector 入口改成 generation runtime

编辑：

```text
vrl/rollouts/collector/core.py
vrl/rollouts/collector/requests.py
vrl/rollouts/collector/batch_builder.py
vrl/rollouts/family_registry.py
```

目标：

```python
from vrl.generation import GenerationOutput, GenerationRuntime
```

替换：

```text
RolloutBackend -> GenerationRuntime
OutputBatch -> GenerationOutput
RolloutEngineRequestBuilder -> GenerationRequestBuilder
```

语义边界：

```text
RolloutCollector:
  owns RL rollout lifecycle
  calls runtime.generate(...)
  calls reward scorer
  builds trainer RolloutBatch

GenerationRuntime:
  owns generation lifecycle
  returns GenerationOutput
  does not know rewards / GRPO / trainer batch
```

验收：

- `RolloutCollector` docstring 不再说 runtime is rollout backend。
- `GenerationRequestBuilder` 只负责 generation request，不负责 reward/group semantics。
- collector tests 仍通过。

## Phase 3：把 `rollouts/runtime` 迁到 `generation/runtime`

当前：

```text
vrl/rollouts/runtime/backend.py
vrl/rollouts/runtime/config.py
vrl/rollouts/runtime/launch_inputs.py
```

当前 workspace 没有 `vrl/engine/core/backend.py`。如果其他分支或后续改动恢复这个文件，它也不能留在 `engine/core`，应并入：

```text
vrl/generation/runtime/factory.py
```

目标：

```text
vrl/generation/runtime/factory.py
vrl/generation/runtime/config.py
vrl/generation/runtime/launch_inputs.py
```

命名：

```text
RolloutBackendConfig -> GenerationRuntimeConfig
RolloutRuntimeInputs -> GenerationRuntimeInputs
build_rollout_backend_from_cfg -> build_generation_runtime_from_cfg
validate_rollout_backend_config -> validate_generation_runtime_config
```

rollout-facing wrapper：

```text
vrl/rollouts/runtime/*.py
  import and re-export from vrl.generation.runtime.*
```

验收：

- runtime factory 不 import collector。
- runtime config 不出现 reward / advantage / trainer batch 字段。
- `vrl/engine/core` 不保留 backend factory；`vrl/engine` 已删除。
- repo 内不再存在 `vrl.engine` import。

## Phase 4：Ray generation backend 改名迁移

当前：

```text
vrl/distributed/ray/rollout/launcher.py
vrl/distributed/ray/rollout/runtime.py
vrl/distributed/ray/rollout/executor.py
vrl/distributed/ray/rollout/planner.py
vrl/distributed/ray/rollout/worker.py
vrl/distributed/ray/rollout/types.py
vrl/distributed/ray/rollout/weight_sync.py
```

目标：

```text
vrl/generation/ray/launcher.py
vrl/generation/ray/runtime.py
vrl/generation/ray/executor.py
vrl/generation/ray/planner.py
vrl/generation/ray/worker.py
vrl/generation/ray/types.py
vrl/generation/ray/weight_sync.py
```

类名替换：

```text
RayRolloutLauncher -> RayGenerationLauncher
RayDistributedRuntime -> RayGenerationRuntime
DistributedRolloutExecutor -> DistributedGenerationExecutor
RayRolloutWorker -> RayGenerationWorker
RolloutWeightSync -> GenerationWeightSync
RayChunkExecutionEnvelope stays acceptable
RayChunkResult stays acceptable
```

旧路径已删除：

```text
vrl/distributed/ray/rollout/
  deleted
```

`vrl/distributed/` 已删除；Ray dependency helpers 移到
`vrl/generation/ray/dependencies.py`。

验收：

- `vrl.generation.ray` 不 import `vrl.rollouts`。
- `vrl.distributed` 已删除。
- Ray tests 覆盖新路径。

## Phase 5：迁移 family generation executors

当前：

```text
vrl/engine/ar/
vrl/engine/diffusion/
vrl/engine/execution/
```

目标：

```text
vrl/generation/execution/
vrl/generation/ar/
vrl/generation/diffusion/
```

迁移规则：

- `vrl/generation/execution/` 只放通用 scheduling / batching / planning。
- `vrl/generation/ar/` 和 `vrl/generation/diffusion/` 放 family-specific generation executors。
- family model 仍在 `vrl/models/ar`、`vrl/models/diffusion`。
- executor 可以 import model interfaces。
- executor 不 import rollout collector / rewards / trainer algorithms。

旧路径已删除：

```text
vrl/engine/ar/
vrl/engine/diffusion/
vrl/engine/execution/
```

不再保留 re-export。

验收：

- model runtime files import from `vrl.generation.ar.*` / `vrl.generation.diffusion.*`。
- new code imports from `vrl.generation.ar.*` / `vrl.generation.diffusion.*`。
- tests import from `vrl.generation.ar.*` / `vrl.generation.diffusion.*`。
- old `vrl.engine.ar` / `vrl.engine.diffusion` imports remain available temporarily。

## Phase 6：把 trajectory 移出 engine

当前：

```text
vrl/engine/trajectory/
```

目标：

```text
vrl/trajectory/
```

理由：

trajectory 是训练记录，不是 generation runtime。

迁移：

```text
vrl/trajectory/types.py
vrl/trajectory/builders.py
vrl/trajectory/views.py
vrl/trajectory/resolver.py
vrl/trajectory/ops.py
vrl/trajectory/validation.py
```

旧路径已删除：

```text
vrl/engine/trajectory/
  deleted
```

依赖规则：

```text
vrl/trajectory can import generation types only for request/sample row typed IDs if needed.
vrl/trajectory cannot import generation family executors or runtime backends.
vrl/trajectory cannot import rewards/trainers/algorithms.
```

验收：

- evaluator / batch builder / trainer imports use `vrl.trajectory`。
- `TrajectoryBatch` validation still rejects runtime-only state。
- trajectory tests pass under new path。

## Phase 7：记录外部 backend 非目标

不新增 serving package。本 repo 当前不承诺未来一定会接外部 generation backend。

外部 backend 只作为架构参考：它们说明低层 generation execution、RL rollout orchestration、训练权重同步之间应该分层，但本 sprint 不把这种分层固化成 `vrl/generation/serving/` 目录。

当前本 repo 仍然优先做自己的 family / Ray generation runtime：

```text
vrl/generation/ar/
vrl/generation/diffusion/
vrl/generation/ray/
```

如果未来真的需要外部 backend adapter，再单独开 sprint 设计。当时仍需满足这些边界：

```text
external backend adapter does not import RolloutCollector
external backend adapter does not import RewardScorer
external backend adapter does not construct RolloutBatch
external backend adapter does not compute advantages
```

验收：

- no `vrl/generation/serving/` package is added。
- sprint clearly says external backends are references, not implementation targets。
- current implementation scope remains `vrl/generation/ar/`, `vrl/generation/diffusion/`, and `vrl/generation/ray/`。

## Phase 8：加入架构边界测试

新增：

```text
tests/architecture/test_generation_rollout_boundaries.py
```

不引入新依赖，直接用 `ast` 扫 import。

规则：

```text
vrl/generation/** cannot import:
  vrl.rollouts
  vrl.rewards
  vrl.trainers
  vrl.algorithms

vrl/trajectory/** cannot import:
  vrl.rollouts
  vrl.rewards
  vrl.trainers
  vrl.algorithms
  vrl.generation.ar
  vrl.generation.diffusion

vrl/distributed/** cannot exist
```

旧 alias 目录不能存在：

```text
vrl/engine/**
vrl/distributed/**
```

`vrl/rollouts/runtime/**` 仍可作为 rollout-facing wrapper，但不能重新引入
`vrl.engine` 或 `vrl.distributed.ray.rollout`。

测试实现必须显式断言旧目录不存在：

```python
assert not Path("vrl/engine").exists()
assert not Path("vrl/distributed").exists()
```

验收：

- architecture test fail-fast。
- 新 package dependency direction 被固定住。
- old package removal is explicit in the test, not implicit in comments。

## 非目标

本 sprint 不做：

- 不实现外部 generation backend。
- 不新增 `vrl/generation/serving/`。
- 不重写 diffusion denoise loop。
- 不改变 reward / advantage / GRPO / DPO 算法语义。
- 不改变 `TrajectoryBatch` 数据语义。
- 不改变 model family layout under `vrl/models`。
- 不保留 `vrl.engine` / `vrl.distributed.ray.rollout` 旧 import path。
- 不引入新 import boundary dependency 工具。

## 迁移顺序

推荐顺序：

```text
1. Add vrl.generation package.
2. Rename protocol/type usage in collector-facing code.
3. Move rollouts/runtime -> generation/runtime.
4. Move distributed/ray/rollout -> generation/ray.
5. Move engine/execution -> generation/execution.
6. Move engine/ar and engine/diffusion -> generation/ar and generation/diffusion.
7. Move engine/trajectory -> vrl/trajectory.
8. Add architecture import boundary tests.
9. Remove repo-internal old-path imports.
10. Delete old-path aliases after repo imports are migrated.
```

每一步都必须能单独测试通过，不能一次性大搬。

## 验证命令

每个 phase 至少跑：

```bash
ruff check vrl tests
pytest tests/rollouts tests/engine tests/generation tests/models
```

Ray 路径迁移后额外跑：

```bash
pytest tests/generation/ray
pytest tests/rollouts/test_runtime_inputs.py
```

trajectory 迁移后额外跑：

```bash
pytest tests/trainers tests/rollouts/evaluators tests/engine/generation
```

如果 AR/diffusion generation executor 迁移，额外跑：

```bash
pytest tests/engine/ar tests/engine/diffusion tests/models
```

## 完成标准

完成后 repo 里应该能一句话说清楚：

```text
vrl.generation:
  generation runtime and execution

vrl.rollouts:
  RL data collection around generation output

vrl.trajectory:
  trainer replay record and train views

vrl.generation.resources:
  trainer/generation resource ownership planning
```

并且代码上满足：

- `RolloutCollector` consume `GenerationRuntime`，不再 consume `RolloutBackend`。
- `GenerationOutput` 是 generation output，`RolloutBatch` 是 trainer-ready RL batch。
- repo 不新增 `vrl/generation/serving`；外部 backend adapter 以后需要时另开 sprint。
- `vrl/distributed` 已删除。
- `vrl/engine` 已删除。
- architecture test 固定禁止反向依赖。

## 实施记录

已完成：

- `vrl/generation`、`vrl/generation/runtime`、`vrl/generation/execution`、`vrl/generation/ar`、`vrl/generation/diffusion`、`vrl/generation/ray` 已建立。
- `vrl/generation/resources.py` 已承载 trainer/generation GPU ownership planning。
- `vrl/trajectory` 已建立，repo 内生产代码不再 import `vrl.engine.trajectory`。
- `vrl/engine/**` 和 `vrl/distributed/**` 已删除。
- `vrl/rollouts/runtime/**` 保留 rollout-facing wrapper。
- `vrl/generation/ray/dependencies.py` 承载 Ray lazy import / actor metadata helpers。
- `RolloutCollector` 已改为消费 `GenerationRuntime` / `GenerationOutput`。
- architecture boundary test 已加入 `tests/architecture/test_generation_rollout_boundaries.py`。
- architecture test 已固定 `vrl/engine` 和 `vrl/distributed` 不再存在。

已验证：

```bash
ruff check vrl tests
pytest -q
```

迁移过程中也单独跑过：

```bash
pytest -q tests/architecture/test_generation_rollout_boundaries.py \
  tests/engine/generation/test_runtime_factory.py \
  tests/generation/ray/test_rollout_launcher.py \
  tests/generation/ray/test_ray_resident_session.py \
  tests/rollouts/test_runtime_inputs.py \
  tests/engine/ar/test_cache.py \
  tests/engine/ar/test_decode_loop.py \
  tests/engine/generation/test_ar_token_scheduler.py \
  tests/engine/diffusion/test_layout.py \
  tests/engine/generation/test_chunk_gatherer.py \
  tests/rollouts/test_engine_requests.py
```

## 参考路径

已迁移并删除的旧路径：

```text
/home/mingfeiguo/Desktop/wm-infra/vrl/engine/core/types.py
/home/mingfeiguo/Desktop/wm-infra/vrl/engine/core/protocols.py
/home/mingfeiguo/Desktop/wm-infra/vrl/engine/core/capabilities.py
/home/mingfeiguo/Desktop/wm-infra/vrl/engine/core/launch_contract.py
/home/mingfeiguo/Desktop/wm-infra/vrl/engine/execution
/home/mingfeiguo/Desktop/wm-infra/vrl/engine/ar
/home/mingfeiguo/Desktop/wm-infra/vrl/engine/diffusion
/home/mingfeiguo/Desktop/wm-infra/vrl/engine/trajectory
/home/mingfeiguo/Desktop/wm-infra/vrl/rollouts/collector/core.py
/home/mingfeiguo/Desktop/wm-infra/vrl/rollouts/collector/requests.py
/home/mingfeiguo/Desktop/wm-infra/vrl/rollouts/collector/batch_builder.py
/home/mingfeiguo/Desktop/wm-infra/vrl/rollouts/runtime
/home/mingfeiguo/Desktop/wm-infra/vrl/distributed/ray/rollout
```
