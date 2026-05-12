# SPRINT：Trajectory 与 Engine 统一化

这份 sprint 的目标不是继续把更多 family 写进同一个 repo，而是把现在“写在一起”的 diffusion / AR / video / R1 pipeline 收敛成一个可迁移的 engine + trajectory contract。只有 trajectory 记录、engine plan、view builder 的边界清楚，后续 KV cache、batching、profile、compile、Ray rollout、reward 并发这些优化才有机会一次改动、多 family 受益。

当前第一组可执行 sprint 不是全 repo 主路径迁移。baseline gate + Sprint A + Sprint B 的完成范围是：

```text
TrajectoryBatch contract
+ SD3.5 diffusion trajectory emission
+ one AR family trajectory emission
+ generic TrajectoryRolloutPacker parity
+ SD3.5 OCR baseline 不破坏
```

`OnlineTrainer`、evaluator、algorithm 的 strict 主路径迁移放到 Sprint C/F 之后的 gate。现在如果直接要求所有 trainer/evaluator/algorithm 主路径切换，会把风险集中到 SD3.5 OCR 这种已经工作的 recipe 上。

## 1. 现状判断

当前 repo 更准确的定位是：

```text
multi-family visual RL codebase with early shared trainer/runtime pieces
```

还不能 claim：

```text
unified online RL framework
```

已经共享的部分：

- `OnlineTrainer` 训练循环。
- reward interface。
- `GenerationRequest -> OutputBatch` engine 边界。
- Ray rollout worker / runtime。
- family packer 把 `OutputBatch` 转成 `RolloutBatch`。
- GRPO / TokenGRPO / MultiSegmentTokenGRPO / DiffusionNFT 等算法模块。

还没有真正统一的地方：

- family train script 仍然分散：Janus、NextStep、SD3.5、Wan、Cosmos 各有自己的 glue code。
- trajectory 表达没有统一：diffusion 是 denoising timestep，Janus 是 discrete image token，NextStep 是 continuous image token，R1 是 multi-segment text/image 混合。
- advantage / logprob / mask 的 axis 语义不统一：有的是 `[B, T]`，有的是 `[B, steps]`，有的是 segment dict。
- packer 现在像 adapter collection，不是统一的 trajectory contract。
- metrics 没有统一 schema：不同 family 的 `num_steps`、reward、old logprob 含义不可直接比较。
- algorithm 不是 family-agnostic，而是靠多个 algorithm class 分别处理。

## 2. 核心 proposal

引入一个显式的核心抽象：

```text
TrajectoryBatch
```

它不强行把所有 family flatten 成 token，而是显式表达 axis、segment、distribution、replay inputs、reward view。

目标是把现在的隐式 contract：

```text
OutputBatch.extra + family packer + evaluator assumptions
```

升级为显式 contract：

```text
GenerationRequest
  -> EnginePlan
  -> FamilyRuntime / PipelineExecutor
  -> OutputBatch(trajectory=TrajectoryBatch)
  -> RewardView / TrainingView
  -> Evaluator
  -> Algorithm
```

统一不是指所有 family 的 tensor 形状一样，而是指每个 tensor 都声明自己的 axis 和 role。算法和 engine 优化可以读这些 metadata，而不是靠 family-specific key 名称猜。

### 2.1 命名结论

不要叫 `VisualTrajectoryBatch`。repo 本身已经是 visual model RL infra，在核心类型里继续加 `Visual` 会制造噪音，也会让后续 diffusion/video/AR 之外的视觉 family 看起来像接入了另一套 visual-specific adapter。

采用这些名字：

```text
TrajectoryBatch
TrajectorySegment
TrajectoryTensor
TrajectorySignalBatch
AlgorithmInput
TrajectoryRolloutPacker
```

避免这些名字：

```text
VisualTrajectoryBatch
VisualRLRecipe
VisualEngineRuntime
EngineResult
```

`EngineResult` 也不要新增。当前 `OutputBatch` 已经承担 engine envelope 角色，新增 `EngineResult` 会和它重复。正确边界是：

```text
OutputBatch = engine envelope，负责 request_id/family/task/decoded output/error/metrics
TrajectoryBatch = serializable trajectory record，负责 axis/segment/action/logprob/mask/reward/replay inputs
RewardView = 从 TrajectoryBatch 派生的 reward 输入视图
TrainingView = 从 TrajectoryBatch 派生的 loss/replay 视图
RolloutBatch = trainer 兼容容器，迁移期保留
```

`RewardView` 和 `TrainingView` 不能变成第二套 batch。它们应该是轻量 view，引用 `TrajectoryBatch` 中的 tensor 和 metadata，只表达“reward 看什么”和“loss 迭代什么”。

### 2.2 架构去重结论

当前架构里已经有三个重要容器：

```text
GenerationRequest
OutputBatch
RolloutBatch
```

新的架构不能再加一个平行的 `EngineResult` 或另一个 trainer batch。正确关系是：

```text
GenerationRequest
  -> EnginePlan
  -> OutputBatch
       - output/decoded artifacts/error/metrics
       - trajectory: TrajectoryBatch
  -> RewardView(output-derived, no tensor copy)
  -> TrainingView(trajectory-derived, no tensor copy)
  -> RolloutBatch(legacy compatibility only)
```

最终收敛方向：

- `GenerationRequest` 保留，负责输入请求。
- `OutputBatch` 保留，负责 engine envelope。
- `TrajectoryBatch` 新增，负责可序列化的训练/奖励轨迹记录。
- `RolloutBatch` 迁移期保留，长期可以变薄或只作为 trainer compatibility layer。
- `RewardView` / `TrainingView` 不持有事实，只是 view builder 的返回值。

所以这不是四套 batch 并存，而是：

```text
request envelope + output envelope + trajectory record + compatibility adapter
```

### 2.3 反 legacy 规则

这个 refactor 最大风险不是类型不够，而是为了兼容旧路径又散落一堆新 legacy。迁移必须遵守这些规则：

- 兼容代码只能集中在 `vrl/engine/trajectory/compat.py`、`vrl/rollouts/packers/trajectory.py` 和少量 family packer adapter 中。
- `OnlineTrainer`、algorithm、evaluator 主路径不能新增 family-specific `if family == ...` 分支。
- `OutputBatch.extra["trajectory"]` 只能作为短期 bridge；新增代码必须优先读 `OutputBatch.trajectory`。
- 不新增新的 loose key 作为训练主语义，例如新的 `extra["new_log_probs"]`、`extra["segment_masks_v2"]`。
- 每个 legacy bridge 必须有对应 strict-mode 测试，证明 `extra` 缺失时新路径仍能跑。
- 不删除正在保护真实 recipe 的旧测试。只有当旧测试断言的是被明确移除的 legacy key，并且已有等价 trajectory 测试覆盖时，才能删。
- 不把旧 packer 改成半新半旧的长期状态。每个 family 只能处在三种状态之一：

```text
legacy_only  -> 旧路径，未迁移
dual_path    -> 旧路径和 trajectory 路径做 parity
strict       -> 主路径只读 TrajectoryBatch
```

`dual_path` 不能无限期存在。进入 `dual_path` 的同一个 sprint 必须定义进入 `strict` 的测试门槛。

### 2.4 SD3.5 OCR 是硬回归基线

当前 `sd3_5_ocr_grpo` 已经工作，所以它是这个大 refactor 的保护对象，不是实验对象。

涉及当前入口：

```text
configs/experiment/sd3_5_ocr_grpo.yaml
vrl/scripts/sd3_5/train.py
vrl/models/families/sd3_5/runtime.py
vrl/models/families/sd3_5/policy.py
vrl/rollouts/packers/diffusion.py
vrl/rollouts/evaluators/diffusion/flow_matching.py
```

SD3.5 OCR 的迁移策略：

- baseline gate 先锁当前行为。
- Sprint A/B 只增加 `TrajectoryBatch` emission，不改变 `DiffusionRolloutPacker` 的旧输出。
- Sprint B 做 old packer 和 trajectory packer parity，确认 `observations/actions/log_probs/timesteps/kl/reward_before_kl/videos/prompts` 一致。
- Sprint C 迁移 evaluator/algorithm input 时，SD3.5 OCR 必须同时跑旧 GRPO path 和新 `AlgorithmInput` adapter parity 测试。
- 在 SD3.5 OCR strict path 通过前，不删除旧 diffusion packer/evaluator 的行为。

### 2.5 重复设计审计结论

这个 sprint 不能新增第三套事实源。当前 repo 已经有这些容易重复的边界：

```text
GenerationRequest / OutputBatch
RolloutBatch
SignalBatch
RuntimeBundle.runtime_caps
FAMILY_REGISTRY / RolloutFamilyEntry
FamilyPipelineRegistry
ExecutionPlan / MicroBatchPlan / DistributedExecutionPlanner
family packers
```

新设计必须按下面规则落地：

| 新名字 | 允许新增吗 | 不能重复谁 | 正确边界 |
| --- | --- | --- | --- |
| `TrajectoryBatch` | 允许 | `OutputBatch` / `RolloutBatch` | 只做可序列化训练轨迹记录；`OutputBatch` 仍是 engine envelope，`RolloutBatch` 只是 trainer 兼容容器。 |
| `RewardView` | 允许 | `RolloutPacker.reward_outputs()` | 只声明 reward 读哪些 artifact/tensor；不复制 image/video/text，不再次打包 reward payload。 |
| `TrainingView` | 允许 | `RolloutBatch.extras` | 只声明 loss unit 和 tensor refs；不复制 action/logprob/mask。 |
| `TrajectorySignalBatch` | 允许 | `SignalBatch` | 作为长期 signal schema；`SignalBatch` 只能作为 legacy adapter 输入/输出，不能长期双主路径。 |
| `AlgorithmInput` | 允许 | algorithm class / objective | 只是 adapter 输入；不新增新的 GRPO objective，不把现有 algorithm 合成一个大类。 |
| `EnginePlan` | 有条件允许 | `ExecutionPlan` / `MicroBatchPlan` / `DistributedExecutionPlanner` | 必须复用或替代现有 planning 类型，不能并行维护两套 chunk 语义。 |
| `FamilyCapability` | 有条件允许 | `RuntimeBundle.runtime_caps` / `RolloutFamilyEntry` | 不能做第三个 registry；静态路由来自 family registry，动态能力来自 runtime bundle，resolver 合并成一个 view。 |
| `RuntimeSession` | 有条件允许 | `GenerationRuntimeSpec` / `RayWorkerHandle` | 只表示 worker 内 live state；不能进入 request/spec/trajectory 的可序列化 payload。 |

如果实现时发现同一个字段同时出现在两处，必须先判断哪个是事实源：

```text
request identity / decoded artifact / error / metrics -> OutputBatch
action / old_logprob / mask / replay input -> TrajectoryBatch
trainer legacy tensors -> RolloutBatch compatibility fields
live runtime state -> RuntimeSession / Ray worker
static family routing -> RolloutFamilyEntry
dynamic runtime capability -> RuntimeBundle.runtime_caps
```

不允许新增这些长期重复层：

- `EngineResult`。
- 独立的 `FamilyCapabilityRegistry`。
- 独立的 `TrajectoryRegistry`。
- 第二套 artifact store，和 `OutputBatch.output` / `trajectory_decoded` 并行长期存在。
- 第二套 algorithm hierarchy，只为了消费 `AlgorithmInput`。

## 3. 设计原则

- 不把 diffusion timestep、AR token、R1 segment 强行合成一个假 token axis。
- 不马上删除现有 `OutputBatch` / `RolloutBatch`，先加兼容层。
- family runtime 内的 pipeline executor 仍然负责真实生成逻辑。
- shared engine 负责调度、batching、profiling、weight sync、shape/capability metadata。
- shared algorithm input 只消费标准 logprob / mask / advantage scope，不关心图像是怎么生成的。
- packer 逐步从 family adapter 变成 view builder。
- 优化必须挂在 engine/trajectory contract 上，不能只写在某个 family script 里。
- `TrajectoryBatch` 是可序列化的训练/奖励轨迹记录，不持有 runtime cache handle、scheduler state、Ray actor state。
- `EnginePlan` / `ExecutionUnit` / `FamilyCapability` 才表达 cache、scheduler、resident state、batching 能力。
- weak compatibility 可以作为迁移手段，但不能作为最终完成标准。

## 4. 新核心类型

新增模块：

```text
vrl/engine/trajectory/
```

建议新增文件：

```text
vrl/engine/trajectory/axes.py
vrl/engine/trajectory/types.py
vrl/engine/trajectory/views.py
vrl/engine/trajectory/validation.py
vrl/engine/trajectory/compat.py
```

核心 dataclass 形态：

```python
AxisSpec(
    name="token" | "timestep" | "segment" | "frame" | "sample",
    kind="discrete_token" | "continuous_token" | "denoise_step" | "text_token" | "media",
    length=int | None,
)

TrajectoryTensor(
    name=str,
    value=Any,
    axes=tuple[str, ...],
    role="observation" | "action" | "old_logprob" | "mask" | "replay_input" | "media",
)

TrajectorySegment(
    name=str,
    modality="image" | "video" | "text" | "latent" | "mixed",
    trainable=bool,
    distribution="categorical" | "gaussian" | "flow_matching" | "deterministic",
    tensors=dict[str, TrajectoryTensor],
    reward_view=str | None,
    advantage_scope="sample" | "segment" | "axis",
)

TrajectoryBatch(
    request_id=str,
    family=str,
    task=str,
    sample_specs=list[GenerationSampleSpec],
    group_ids=Any,
    axes=dict[str, AxisSpec],
    segments=dict[str, TrajectorySegment],
    reward_views=dict[str, Any],
    metrics=TrajectoryMetrics,
    context=dict[str, Any],
)
```

这几个字段要成为统一 contract：

- `sample_specs`：prompt/sample 顺序。
- `group_ids`：group-relative normalization / preference grouping / per-prompt statistics 需要。
- `segments`：R1、diffusion、AR 都可以表达。
- `old_logprob`：采样时的 old policy logprob。
- `mask`：训练有效位置。
- `replay_input`：evaluator 重新计算 current logprob 所需的可序列化输入。
- `reward_views`：reward 看到的是 image/video/text，不一定等于训练 action。
- `metrics`：采样产生的可序列化指标，例如 token 数、timestep 数、旧 logprob 统计。

`TrajectoryBatch` 不能包含这些 runtime-only 内容：

```text
KV cache handle
CUDA graph handle
Ray actor/session handle
Python generator/stateful scheduler object
model module reference
open file handle
HTTP client/session
```

replay 信息可以记录，但必须是可序列化的 replay input，不是 runtime method handle：

```python
ReplayInput(
    name=str,
    tensor_refs=tuple[str, ...],
    context_refs=tuple[str, ...],
    signal_kind="logprob" | "kl_intermediates" | "prediction",
)
```

cache、decode、scheduler、resident state 放在 engine 层：

```text
FamilyCapability -> declares supported optimization surface
EnginePlan       -> chooses execution strategy for this request
ExecutionUnit    -> names prefill/decode/denoise/vq_decode/profile units
RuntimeSession   -> owns live cache/session state inside a rollout worker
```

### 4.1 View 类型只做派生，不持有第二份语义

`RewardView` 和 `TrainingView` 放在 `vrl/engine/trajectory/views.py`，但它们不能复制 trajectory tensor：

```python
RewardView(
    name=str,
    modality="image" | "video" | "text" | "mixed",
    tensor_refs=tuple[str, ...],
    prompt_refs=tuple[str, ...],
    target_refs=tuple[str, ...],
    metadata=dict[str, Any],
)

TrainingView(
    loss_units=tuple[LossUnit, ...],
    primary_segment=str | None,
    algorithm_family="policy_gradient" | "supervised" | "preference" | "custom",
    legacy_fields=dict[str, Any],
)

LossUnit(
    segment=str,
    axis=str,
    axis_index=int | None,
    action_ref=str,
    old_logprob_ref=str,
    mask_ref=str,
    advantage_scope="sample" | "segment" | "axis",
    signal_requirements=tuple[str, ...],
    replay_input_refs=tuple[str, ...],
)
```

边界规则：

- `TrajectoryBatch` 持有事实。
- `RewardView` 只声明 reward function 应该读取哪些事实，不复制图像/视频/text tensor。
- `TrainingView` 只声明 trainer/algorithm 应该按什么 loss unit 迭代，不绑定具体算法类。
- `RolloutBatch` 只是迁移期 trainer 容器，不再定义语义。

`AlgorithmInput` 是算法统一入口，但不是新算法：

```python
AlgorithmInput(
    trajectory=TrajectoryBatch,
    training_view=TrainingView,
    signals=TrajectorySignalBatch | None,
    rewards=Any,
    group_ids=Any,
    advantages=Any | None,
    metadata=dict[str, Any],
)
```

现有 `GRPO`、`TokenGRPO`、`MultiSegmentTokenGRPO`、`DiffusionNFT`、`DPO` 都可以通过 adapter 消费 `AlgorithmInput`。本 sprint 不要求合并这些 algorithm。

## 5. Engine v2 边界

当前 engine 边界：

```text
GenerationRequest -> OutputBatch
```

短期保留，但增加：

```text
OutputBatch.trajectory: TrajectoryBatch | None
OutputBatch.engine_plan: EnginePlan | None
```

如果暂时不想改 `OutputBatch` 字段，可以先放在：

```text
OutputBatch.extra["trajectory"]
```

最终目标不是新增 `EngineResult`，而是让 `OutputBatch` 成为唯一 engine envelope：

```text
GenerationRequest -> OutputBatch
```

其中：

```python
OutputBatch(
    request_id=str,
    family=str,
    task=str,
    output=Any,
    trajectory=TrajectoryBatch | None,
    engine_plan=EnginePlan | None,
    metrics=GenerationMetrics | None,
    trajectory_decoded=list[Any] | None,
    extra=dict[str, Any],
    error=str | None,
)
```

`output` / `trajectory_decoded` 是 decoded image/video/text 结果；`trajectory` 是训练和优化主语义。不要在 Sprint A/B 新增平行的 `artifacts` dict，除非它明确替代现有 decoded artifact 字段。也不要把 `reward_views` 放进 `OutputBatch` 顶层，reward view 应该由 `TrajectoryBatch.reward_views` 或 `vrl/engine/trajectory/views.py` 动态构建。

### 5.1 EnginePlan

新增 engine planning 层：

```text
vrl/engine/core/planner.py
```

这不是第二套 `ExecutionPlan`。当前已有：

```text
vrl/engine/microbatching.py::ExecutionPlan
vrl/engine/microbatching.py::MicroBatchPlan
vrl/distributed/ray/rollout/planner.py::DistributedExecutionPlanner
```

`EnginePlan` 必须复用 `MicroBatchPlan`，并逐步成为 local worker 和 Ray rollout 的同一份 planning envelope。`DistributedExecutionPlanner` 只能负责 device/worker assignment，不能再定义另一套 chunk 语义。

用途：

- 根据 `GenerationRequest` 和 family capability 生成 `EnginePlan`。
- 统一 batch/microbatch/chunk 切分。
- 统一 seed/prompt-major order。
- 统一 profile label 和 metric scope。

建议类型：

```python
EnginePlan(
    request=GenerationRequest,
    sample_specs=list[GenerationSampleSpec],
    workload=WorkloadSignature,
    trajectory_kind="diffusion" | "ar_discrete" | "ar_continuous" | "multisegment",
    expected_axes=dict[str, AxisSpec],
    chunks=list[MicroBatchPlan],
    execution_units=list[ExecutionUnit],
)
```

`ExecutionUnit` 是 engine 真正执行和优化的单位：

```python
ExecutionUnit(
    name="prefill" | "decode_step" | "denoise_step" | "vq_decode" | "reward_artifact",
    segment=str | None,
    axis=str | None,
    axis_index=int | None,
    batch_group_key=tuple[Any, ...],
    cache_read=bool,
    cache_write=bool,
    profiler_name=str,
)
```

这比现在的 `MicroBatchPlan(prompt_index, sample_start, sample_count)` 更强。`MicroBatchPlan` 只能表达 sample chunk，不能表达：

- Janus prompt prefill 和 token decode 分离。
- R1 initial image / selfcheck text / final image 三段的 replay 和 loss。
- diffusion denoise timestep 的 same-step batching。
- Ray worker 里哪些 state 可以常驻，哪些要回传 CPU。

短期可以保留 `MicroBatchPlan`，但 `EnginePlan.execution_units` 必须成为优化入口。

### 5.2 FamilyCapability

新增：

```text
vrl/engine/core/capabilities.py
```

这不是新的 registry。当前已有两个相关事实源：

```text
vrl/rollouts/families/specs.py::FAMILY_REGISTRY
vrl/models/runtime.py::RuntimeBundle.runtime_caps
```

规则：

- static routing/import/config metadata 仍然来自 `RolloutFamilyEntry`。
- dynamic runtime flags 仍然来自 `RuntimeBundle.runtime_caps`。
- `FamilyCapability` 是 resolver 生成的 typed view，不是第三份手写列表。
- 如果 capability 字段是静态的，优先挂到 family registry entry 或 family runtime 旁边的 entry 定义。
- 如果 capability 字段依赖实际 backend/load result，保留在 `RuntimeBundle.runtime_caps`。

每个 family runtime 声明 capability：

```python
FamilyCapability(
    family="janus_pro",
    trajectory_kind="ar_discrete",
    supports_batched_requests=True,
    supports_chunked_execution=True,
    supports_kv_decode=True,
    supports_prefill_decode_split=True,
    supports_resident_rollout_state=True,
    batchable_axes=("token_position",),
    cache_kinds=("kv_cache", "prompt_embed_cache"),
    supports_cuda_graph=False,
    supports_torch_compile=False,
    trainable_segments=("image_tokens",),
    reward_views=("image",),
)
```

这让 engine 优化可以读 capability，而不是在 trainer 或 script 里 hardcode family 名字。

diffusion family 的 capability 不能假装支持 KV cache，它应该明确声明不同的优化面：

```python
FamilyCapability(
    family="sd3_5",
    trajectory_kind="diffusion",
    supports_batched_requests=True,
    supports_chunked_execution=True,
    supports_kv_decode=False,
    supports_prefill_decode_split=False,
    supports_resident_rollout_state=True,
    batchable_axes=("timestep", "sample"),
    cache_kinds=("prompt_embed_cache", "latent_cache"),
    trainable_segments=("denoise",),
    reward_views=("image",),
)
```

统一的价值不是所有 family 都有同一种优化，而是所有 family 都用同一种方式声明“我能被怎么优化”。

### 5.3 Executor output contract

当前 family pipeline executor 返回 `OutputBatch`，里面把训练需要的内容塞到 `extra`。

迁移后 pipeline executor 应该返回：

```text
TrajectoryBatch + decoded output
```

兼容期：

```text
pipeline_executor.forward(...) -> OutputBatch
OutputBatch.extra["trajectory"] = TrajectoryBatch
```

## 6. Packer 迁移方案

当前 packer：

```text
DiffusionRolloutPacker
ARDiscreteRolloutPacker
ARContinuousRolloutPacker
ARR1RolloutPacker
```

短期不删。先新增：

```text
vrl/rollouts/packers/trajectory.py
```

新增：

```python
TrajectoryRolloutPacker
```

这不是第五个长期 packer。它是 generic compatibility packer，用来替代当前 family packer 里重复的 action/logprob/mask/extras 组装逻辑。

它根据 `TrajectoryBatch` 生成现有 `RolloutBatch`，并保留 trainer 兼容字段：

```text
RolloutBatch.observations
RolloutBatch.actions
RolloutBatch.rewards
RolloutBatch.group_ids
RolloutBatch.extras["log_probs"]
RolloutBatch.extras["token_mask" / "step_mask" / "segment_masks"]
RolloutBatch.context
RolloutBatch.videos
RolloutBatch.prompts
```

family packer 逐步变薄：

```text
family OutputBatch -> TrajectoryBatch -> TrajectoryRolloutPacker -> RolloutBatch
```

完成后，packers 不再是主要 contract，只是 legacy adapter。

进入 strict path 后，同一个 family 不能同时长期维护：

```text
family packer reads OutputBatch.extra
TrajectoryRolloutPacker reads TrajectoryBatch
```

只能保留前者作为 parity test fixture 或短期 bridge。

## 7. Evaluator v2

当前 evaluator 输出：

```python
SignalBatch(log_prob, ref_log_prob, aux=...)
```

问题是 `log_prob` 的 axis 语义不明确。

新增：

```text
vrl/rollouts/evaluators/trajectory.py
```

建议类型：

```python
TrajectorySignalBatch(
    segments=dict[str, SegmentSignal],
    group_ids=Any,
    context=dict[str, Any],
)

SegmentSignal(
    name=str,
    log_prob=Any,
    old_log_prob=Any,
    ref_log_prob=Any | None,
    mask=Any,
    axes=tuple[str, ...],
    dist_family=str,
    aux=dict[str, Any],
)
```

Evaluator v2 contract：

```text
evaluate(model, TrajectoryBatch, SignalRequest) -> TrajectorySignalBatch
```

兼容期：

- diffusion evaluator 继续返回 `SignalBatch`。
- AR evaluator 继续返回 `SignalBatch`。
- 新 adapter 把旧 `SignalBatch` 包成 `TrajectorySignalBatch`。

长期只能有一个主 signal schema。`SignalBatch` 不能继续通过 `aux` 承载 segment schema；strict path 通过后，R1/multisegment 的主路径必须读 `TrajectorySignalBatch.segment_signals`。

## 8. Algorithm input v2

当前 algorithm 分裂是事实，但 Sprint C 不应该强行合并算法：

- `GRPO`：diffusion / continuous。
- `TokenGRPO`：AR token。
- `MultiSegmentTokenGRPO`：R1 segment。
- `DiffusionNFT`：特殊训练目标。
- `DPO`：preference / offline objective。

新增：

```text
vrl/algorithms/trajectory.py
```

建议：

```python
AlgorithmInput(
    trajectory=TrajectoryBatch,
    training_view=TrainingView,
    signals=TrajectorySignalBatch | None,
    rewards=Any,
    group_ids=Any,
    advantages=Any | None,
    metadata=dict[str, Any],
)

AlgorithmAdapter.compute_loss(
    algorithm=Algorithm,
    inputs=AlgorithmInput,
) -> tuple[Any, TrainStepMetrics]
```

这不是新的 objective hierarchy。`AlgorithmInput` 只负责把 trajectory/view/signal/reward 拼成现有 algorithm 能消费的输入。

关键改变：

- 统一的是 algorithm 输入，不是强行统一 algorithm 实现。
- advantage scope、mask、axis、segment weight 由 `TrainingView` / `AlgorithmInput` 显式传递。
- algorithm adapter 负责把标准输入投影到现有 `GRPO` / `TokenGRPO` / `MultiSegmentTokenGRPO` / `DiffusionNFT` / `DPO`。
- 现有 algorithm 类可以继续存在；迁移目标是让它们不再直接猜 family-specific shape 或 loose extras。
- R1 可以自然表达 per-segment reward / per-segment advantage，但不要求所有 algorithm 合并成同一个 GRPO 实现。

共享 GRPO 实现可以作为后续清理任务，但不是本 sprint 的核心完成条件。核心条件是：所有 algorithm 都能通过统一 input/view 入口接入未来的 engine/trajectory 优化。

## 9. Train Script 统一方案

当前 family train scripts 仍然负责太多 glue：

```text
vrl/scripts/janus_pro/train.py
vrl/scripts/nextstep_1/train.py
vrl/scripts/sd3_5/train.py
vrl/scripts/wan_2_1/train.py
vrl/scripts/cosmos/train.py
```

新增：

```text
vrl/training/recipe.py
vrl/training/factory.py
vrl/training/loop.py
```

目标 contract：

```python
build_online_rl_recipe(cfg) -> OnlineRLRecipe
```

```python
OnlineRLRecipe(
    bundle=RuntimeBundle,
    collector=RolloutCollector,
    evaluator=Evaluator,
    algorithm=Algorithm,
    trainer_config=TrainerConfig,
    prompts=list[PromptExample | str],
    checkpoint_hooks=CheckpointHooks,
)
```

family script 最终变成薄 wrapper：

```python
async def train_janus_pro_r1_codex_qa_grpo(cfg):
    recipe = build_online_rl_recipe(cfg)
    await run_online_recipe(recipe)
```

不要求在 Sprint A/B 完成全部脚本统一。后续 Sprint F 再把共同 glue 抽走：

- config build。
- distributed resource resolve。
- checkpoint resume/save。
- reward construction。
- rollout runtime setup。
- trainer construction。
- prompt manifest loading。

## 10. Optimization 如何跨 family 复用

统一 contract 后，优化位置应该移动到 engine 层：

### 10.1 Request batching

位置：

```text
vrl/engine/core/worker.py
vrl/engine/core/planner.py
```

收益：

- diffusion 多 prompt 同 shape batching。
- Janus/NextStep 多 prompt x group batching。
- R1 segment batching。

要求：

- `WorkloadSignature` 必须包含 axis/capability 信息。
- seed 和 prompt-major order 必须稳定。

### 10.2 Chunk/microbatch planning

位置：

```text
vrl/engine/microbatching.py
vrl/distributed/ray/rollout/planner.py
```

收益：

- Ray 多 worker 规划从 family-specific chunk 变成 trajectory-aware chunk。
- 多 GPU rollout 可以按 sample、segment、timestep/token 切分。

### 10.3 KV cache / AR decode

位置：

```text
vrl/engine/ar/
vrl/models/families/janus_pro/policy.py
vrl/models/families/nextstep_1/policy.py
```

收益：

- Janus、Janus-R1、NextStep 共享 AR scheduler 和 KV helper。
- profiler metrics 用统一 `decode_steps` / `prefill_steps` 记录。

### 10.4 Diffusion timestep replay

位置：

```text
vrl/engine/diffusion/
vrl/rollouts/evaluators/diffusion/
```

收益：

- SD3.5、Wan、Cosmos 共享 timestep selection、SDE logprob、KL replay metadata。
- `TrajectoryTensor(role="old_logprob", axes=("sample", "timestep"))` 能被统一 algorithm 消费。

### 10.5 Profiling

位置：

```text
vrl/trainers/profiling.py
vrl/distributed/ray/rollout/worker.py
vrl/engine/core/planner.py
```

收益：

- trace label 用 `family/task/segment/axis`。
- Janus token decode、diffusion denoise step、R1 segment 都能进同一个 metric schema。

### 10.6 Torch compile / CUDA graph

位置：

```text
vrl/engine/core/capabilities.py
vrl/engine/core/planner.py
vrl/models/families/*/runtime.py
vrl/models/families/*/policy.py
```

收益：

- 只有 shape/axis stable 的 workload 才启用 compile/graph。
- 避免在每个 family script 里手写 compile 开关。

## 11. Slime 对照：要学边界，不要照搬文本 serving

`/home/mingfeiguo/Desktop/slime` 的关键价值不是 “SGLang 很快所以直接用 SGLang”，而是它把系统切成稳定边界：

```text
Ray ServerGroup / RolloutServer
  -> long-lived rollout engine
  -> standard rollout samples
  -> train actor group
  -> weight update / offload / onload
```

对应到本 repo，要形成类似边界：

```text
GenerationRuntime
  -> EnginePlan
  -> long-lived FamilyExecutionSession
  -> OutputBatch(trajectory=TrajectoryBatch)
  -> TrainingView / RewardView
  -> OnlineTrainer
```

需要借鉴的点：

- rollout worker 常驻，不在每轮训练里反复构造/销毁大量模型状态。
- weight sync 是 runtime 能力，不是 family train script 的私有逻辑。
- offload/onload/release memory 是 engine lifecycle，不是 collector 的临时补丁。
- rollout execution 应该有 server/session 级别的状态；Janus KV cache、prompt embedding cache、diffusion prompt conditioning cache 都属于这里。
- profiling label 应该从 engine plan 生成，而不是每个 pipeline executor 自己拼字符串。

不能照搬的点：

- SGLang/vLLM 的 serving IR 基本是 text token request；本 repo 还要表达 diffusion timestep、video frame、continuous token、R1 multi-segment。
- HTTP async scheduler 不是第一优先级；本 repo 的第一优先级是把 generation/replay/loss 的 axis 语义统一。
- Paged attention/radix cache 对 Janus AR 可能有用，但对 diffusion/video 不是同一个优化面。统一层应该表达 capability，而不是强行假设所有 family 都是 LLM decode。

## 12. 当前代码需要改变的具体位置

这一节是按当前代码状态写的改动矩阵。目标是避免只写抽象，不知道落在哪些文件。

### 12.1 `OutputBatch`：从 loose extras 变成 trajectory carrier

当前文件：

```text
vrl/engine/core/types.py
```

当前问题：

```python
extra: dict[str, Any] = field(default_factory=dict)
```

`token_ids`、`token_log_probs`、`token_mask`、`prompt_input_ids`、`segments`、`denoising_env` 等训练主语义都靠 `extra` 或 family-specific trajectory data 传递。

需要改成：

```text
OutputBatch.trajectory: TrajectoryBatch | None
OutputBatch.engine_plan: EnginePlan | None
```

兼容期规则：

- `OutputBatch.trajectory` 是主路径。
- `OutputBatch.extra["trajectory"]` 只作为老代码读取的 bridge。
- `extra` 不能再成为 action/logprob/mask/replay 的主来源。

### 12.2 Executor protocol：需要 capability 和 plan-aware forward

当前文件：

```text
vrl/engine/core/protocols.py
```

当前 protocol 只有：

```python
def forward(...)
def forward_chunk(...)
def gather_chunks(...)
```

需要新增：

```python
def capability(self) -> FamilyCapability: ...
def plan(self, request: GenerationRequest, sample_specs: list[GenerationSampleSpec]) -> EnginePlan: ...
```

或者由 registry 提供 capability，worker 统一调用 planner：

```text
GenerationRequest + FamilyCapability -> EnginePlan
```

只能选一个事实源，不能两边各维护一份 capability。如果 protocol method 存在，它必须委托到同一个 capability resolver；如果 registry 提供 capability，executor 不再手写另一套 capability dict。

完成标准不是“能 forward”，而是 engine 在 forward 之前已经知道：

- 会产生哪些 trajectory axes。
- 哪些 segment 可训练。
- 哪些 execution unit 能 batch。
- 哪些 state 能 cache。
- 哪些 profiler region 应该出现。

### 12.3 `GenerationWorker`：不能只按 request key batch

当前文件：

```text
vrl/engine/core/worker.py
```

当前 batching key：

```python
return (
    request.family,
    request.task,
    request.samples_per_prompt,
    request.policy_version,
    tuple(sorted(request.return_artifacts)),
    _freeze(request.sampling),
)
```

这个 key 只能做 request-level batching，不能做 token position / timestep / segment 级别调度。

需要改成：

```text
GenerationWorker
  -> resolve FamilyCapability
  -> build EnginePlan
  -> execute EnginePlan
```

短期保留现有 grouping，但新增 planner path，先让 metrics/profiler/trajectory 都能看到同一个 `EnginePlan`。

### 12.4 Janus runtime：不能只吐 token extras

当前文件：

```text
vrl/models/families/janus_pro/runtime.py
```

当前 Janus 输出：

```python
extra = {
    "token_ids": token_ids,
    "token_log_probs": token_log_probs,
    "token_mask": token_mask,
    "prompt_input_ids": prompt_ids,
    "prompt_attention_mask": prompt_mask,
    "uncond_input_ids": uncond_ids,
    "uncond_attention_mask": uncond_mask,
}
```

需要改成原生 emit：

```text
TrajectoryBatch
  segment=image_generation
  axis=token
  action=token_ids
  old_logprob=token_log_probs
  mask=token_mask
  prompt_context=prompt_ids/prompt_mask/uncond_ids/uncond_mask
  replay_inputs=prompt ids/masks + token_ids
```

这一步是 Janus KV cache 能不能变成 shared optimization 的关键。如果只是继续在 Janus runtime 里私有地 `use_cache=True`，统一层没有收益。

### 12.5 Janus R1 runtime：segments 要成为 first-class trajectory

当前文件：

```text
vrl/models/families/janus_pro/runtime.py
```

当前 R1 输出：

```python
extra={
    "initial_image": result.initial_image,
    "final_image": result.final_image,
    "selfcheck": result.selfcheck,
    "segments": segment_extra,
}
```

需要改成：

```text
TrajectoryBatch.segments:
  initial_image:
    axis=token
    modality=image
    reward_view=initial_image | None
    advantage_scope=segment
  selfcheck_text:
    axis=text_token
    modality=text
    reward_view=selfcheck_text | None
    advantage_scope=segment
  final_image:
    axis=token
    modality=image
    reward_view=final_image
    advantage_scope=segment
```

这样才能解决当前 `MultiSegmentTokenGRPO` 对每个 segment 复用同一份 advantage 的架构缺口。

### 12.6 Diffusion runtime：denoise trajectory 要进同一个 contract

当前文件：

```text
vrl/engine/diffusion/executor_base.py
vrl/engine/gather.py
```

当前 diffusion 通过：

```text
rollout_trajectory_data
dit_trajectory.latents
dit_trajectory.timesteps
denoising_env.extra
```

传给 packer/evaluator。

需要改成：

```text
TrajectoryBatch
  segment=denoise
  axis=timestep
  observation=latents
  action=next_latents / prev_sample
  old_logprob=rollout_log_probs
  mask=step_mask
  replay_inputs=latents/timesteps/conditioning refs
```

这证明 `TrajectoryBatch` 不是 token-only schema。

### 12.7 Packer：从 adapter collection 改成 view builder

当前文件：

```text
vrl/rollouts/packers/diffusion.py
vrl/rollouts/packers/ar/discrete.py
vrl/rollouts/packers/ar/continuous.py
vrl/rollouts/packers/ar/r1.py
```

当前 AR packer 直接读：

```python
token_ids = output.extra["token_ids"]
token_log_probs = output.extra["token_log_probs"]
```

统一后：

```text
TrajectoryBatch -> TrainingView -> RolloutBatch
```

family packer 最多只负责：

- reward output 选择。
- image/video/text artifact rescale。
- legacy key bridge。

训练 tensor 不再从 family `extra` 直接读。

### 12.8 `RolloutBatch`：要携带 trajectory 或 training view

当前文件：

```text
vrl/rollouts/batch.py
```

当前 `RolloutBatch` 没有 trajectory 字段，trainer 只能从：

```python
observations
actions
extras
```

猜语义。

需要新增：

```text
RolloutBatch.trajectory: TrajectoryBatch | None
RolloutBatch.training_view: TrainingView | None
```

同时改：

```text
stack_batches
_select_batch
_move_training_batch_to_device
```

让 trajectory/training view 可以被 cat、select、move。

### 12.9 Evaluator：`SignalBatch.aux` 不能继续承载 segment schema

当前文件：

```text
vrl/rollouts/evaluators/types.py
vrl/rollouts/evaluators/ar/token_logprob.py
vrl/rollouts/evaluators/ar/continuous_token_logprob.py
vrl/rollouts/evaluators/ar/multi_segment_token_logprob.py
vrl/rollouts/evaluators/diffusion/flow_matching.py
```

当前 R1 segment signals 被塞进：

```python
SignalBatch(aux={"segments": segment_signals})
```

需要改成：

```text
TrajectorySignalBatch.segment_signals[name]
```

每个 signal 必须显式携带：

- `axis`
- `segment`
- `distribution`
- `new_logprob`
- `old_logprob`
- `ref_logprob`
- `mask`
- `replay_metrics`

### 12.10 `OnlineTrainer`：不能从 tensor shape 推断训练轴

当前文件：

```text
vrl/trainers/online.py
```

当前核心假设：

```python
num_timesteps = filtered_batches[0].observations.shape[1]
old_lp = b.extras.get("log_probs")
old_lp_j = old_lp[:, j] if old_lp.ndim > 1 else old_lp
```

这会把 diffusion timestep、AR token、R1 primary segment 都压成同一个 loop。

需要改成：

```text
for unit in batch.training_view.loss_units:
    signals = evaluator.evaluate(model, batch.trajectory, unit)
    loss = algorithm.compute_trajectory_loss(signals, advantages, unit)
```

trainer 不再知道 `j` 是 timestep 还是 token。这个语义由 `TrainingView.loss_units` 提供。

### 12.11 Algorithm：统一输入，不统一算法实现

当前文件：

```text
vrl/algorithms/grpo/continuous.py
vrl/algorithms/grpo/token.py
vrl/algorithms/grpo/multisegment.py
vrl/algorithms/diffusion_nft.py
```

需要新增：

```text
vrl/algorithms/trajectory.py
```

Sprint C 的目标不是合并这些算法，而是给它们统一输入：

- `AlgorithmInput`
- `AlgorithmAdapter`
- `TrainingView`
- `TrajectorySignalBatch`

adapter 覆盖：

- `GRPO`
- `TokenGRPO`
- `MultiSegmentTokenGRPO`
- `DiffusionNFT`
- `DPO`

原因：这些不是同一种 objective。统一框架应该让它们消费同一种 trajectory/training/signal view，而不是强行把 loss 合并成一个类。

### 12.12 Ray rollout：从 chunk forward 改成 session/plan forward

当前文件：

```text
vrl/distributed/ray/rollout/worker.py
vrl/distributed/ray/rollout/executor.py
vrl/distributed/ray/rollout/runtime.py
vrl/rollouts/runtime/backend.py
```

当前 Ray worker 核心是：

```python
return self.executor.forward_chunk(request, chunk)
```

需要新增 session-aware path：

```text
RayRolloutWorker.load_policy()
RayRolloutWorker.prepare_plan(EnginePlan)
RayRolloutWorker.execute_unit(ExecutionUnit)
RayRolloutWorker.finalize_plan()
```

Sprint D/E 不必把所有 family 都改成 unit-level remote call，但 Janus KV cache 不能只在 local runtime 内部实现；至少要让 Ray worker 持有：

- prompt embeddings。
- KV cache state。
- sampled token buffer。
- decode metrics。
- VQ decode artifact。

### 12.13 Family registry：collector kind 不够，需要 capability registry

当前文件：

```text
vrl/rollouts/families/specs.py
```

当前 registry 有：

```text
collector.kind
executor_cls
runtime_builder
runtime_spec_extractor
gatherer
```

需要新增：

```text
capability_cls 或 capability metadata
trajectory_schema
default_engine_plan
supported_optimizations
```

新增 family 时，必须实现：

```text
runtime builder
runtime capability
trajectory emission
reward views
replay inputs
```

不能再要求新增 family 重写 trainer/packer/algorithm。

### 12.14 Train scripts：只能做 recipe wrapper

当前文件：

```text
vrl/scripts/janus_pro/train.py
vrl/scripts/nextstep_1/train.py
vrl/scripts/sd3_5/train.py
vrl/scripts/wan_2_1/train.py
vrl/scripts/cosmos/train.py
```

需要抽到：

```text
vrl/training/recipe.py
vrl/training/factory.py
vrl/training/loop.py
```

脚本里不应该继续承载：

- rollout grouping。
- weight sync 策略。
- reward construction 重复逻辑。
- evaluator/algorithm selection。
- checkpoint/eval loop 重复逻辑。
- profiling region 命名。

## 13. Sprint 切分

这份计划拆成 A-F，不再把 contract、packer、algorithm、engine、KV cache、recipe 收敛混成一个巨大完成标准。

依赖关系：

```text
baseline gate
  -> Sprint A
  -> Sprint B
  -> Sprint C

Sprint A
  -> Sprint D
  -> Sprint E

Sprint F 在 Sprint C/D 之后做，避免先把 train script 抽象成另一层 legacy。
```

### 13.1 Baseline gate：锁住 SD3.5 OCR

这个 gate 在任何重构前先过。它不是新架构的一部分，而是防止已经工作的 SD3.5 OCR recipe 被 trajectory 重构破坏。

必须满足：

- `experiment/sd3_5_ocr_grpo` config 能加载。
- entrypoint 仍然是 `vrl.scripts.sd3_5.train:train_sd3_5_grpo`。
- family registry 仍然解析到 `sd3_5` / `t2i` / `DiffusionChunkGatherer`。
- diffusion chunk gatherer 仍然产出旧 `OutputBatch` 字段。
- OCR reward 行为不变。
- SD3 resume/checkpoint 逻辑不受影响。
- 当前 SD3.5 packer 输出 contract 被测试记录，后续 trajectory packer 必须做 parity。

### 13.2 Sprint A：Trajectory contract

目标：只定义事实记录和 view contract，不改变任何 recipe 行为。

新增：

```text
vrl/engine/trajectory/__init__.py
vrl/engine/trajectory/axes.py
vrl/engine/trajectory/types.py
vrl/engine/trajectory/views.py
vrl/engine/trajectory/validation.py
vrl/engine/trajectory/compat.py
tests/engine/trajectory/test_trajectory_types.py
```

必须满足：

- 定义 `TrajectoryBatch` / `TrajectorySegment` / `TrajectoryTensor`。
- 定义最小版 `RewardView` / `TrainingView` / `LossUnit`。
- 定义 `AlgorithmInput` 需要引用的 view-facing fields，但不实现 algorithm adapter。
- 支持 diffusion、AR discrete、AR continuous、R1 的最小 schema。
- validator 能检查 batch dim、axis 名称、mask/logprob shape。
- `TrajectoryBatch` 不包含 KV cache handle、CUDA graph handle、Ray actor/session、scheduler object、model module、HTTP client。
- 没有任何现有 recipe 行为变化。

不能算完成：

- 只定义 `TrajectoryBatch`，但没有 `TrainingView` / `RewardView`。
- 把 `TrainingView` 做成第二套 batch，复制 action/logprob/mask tensor。
- 在 contract 里放 runtime-only state。

### 13.3 Sprint B：Generic trajectory packer + SD3.5 / Janus-Pro 接入

目标：让一个 diffusion family 和一个 AR family 真实产出或兼容 `TrajectoryBatch`，并证明 generic packer 能生成旧 `RolloutBatch`。

新增：

```text
vrl/rollouts/packers/trajectory.py
tests/rollouts/test_sd3_5_ocr_trajectory_parity.py
tests/rollouts/test_trajectory_packer.py
```

编辑：

```text
vrl/engine/core/types.py
vrl/rollouts/batch.py
vrl/rollouts/packers/base.py
vrl/rollouts/packers/diffusion.py
vrl/rollouts/packers/ar/discrete.py
vrl/models/families/janus_pro/runtime.py
vrl/models/families/sd3_5/runtime.py
```

必须满足：

- `OutputBatch.trajectory` 字段存在。
- `OutputBatch.extra["trajectory"]` 只作为 bridge，新增主路径优先读 `OutputBatch.trajectory`。
- SD3.5 runtime 写入 diffusion timestep `TrajectoryBatch`。
- Janus-Pro discrete image-token path 写入 AR token `TrajectoryBatch`。
- 现有 SD3.5 OCR packer 输出完全不变。
- `TrajectoryRolloutPacker` 可以从 SD3.5 和 Janus-Pro 的 `TrajectoryBatch` 生成旧 `RolloutBatch`。
- SD3.5 diffusion old packer 和 `TrajectoryRolloutPacker` 对同一个 `OutputBatch` 产生 parity：
  - `observations`
  - `actions`
  - `rewards`
  - `group_ids`
  - `extras["log_probs"]`
  - `extras["timesteps"]`
  - `extras["kl"]`
  - `extras["reward_before_kl"]`
  - `videos`
  - `prompts`
- Janus-Pro old AR packer 和 `TrajectoryRolloutPacker` 对 token ids、old logprobs、mask、prompt replay inputs 做 parity。
- parity 通过前，`DiffusionRolloutPacker` 和 `ARDiscreteRolloutPacker` 不切到 strict。

本 sprint 不强制迁移：

```text
vrl/models/families/nextstep_1/runtime.py
vrl/models/families/wan_2_1/runtime.py
vrl/models/families/cosmos/predict2/runtime.py
vrl/models/families/cosmos/predict2_5/runtime.py
```

这些 family 后续按同一 contract 接，不在第一组实现里一起碰。

### 13.4 Sprint C：TrajectorySignalBatch + AlgorithmInput adapters

目标：统一 algorithm 输入，不强行合并现有 algorithm。

新增：

```text
vrl/rollouts/evaluators/trajectory.py
vrl/algorithms/trajectory.py
tests/rollouts/evaluators/test_trajectory_signals.py
tests/algorithms/test_algorithm_input_views.py
```

编辑：

```text
vrl/rollouts/evaluators/types.py
vrl/rollouts/evaluators/ar/token_logprob.py
vrl/rollouts/evaluators/ar/continuous_token_logprob.py
vrl/rollouts/evaluators/ar/multi_segment_token_logprob.py
vrl/rollouts/evaluators/diffusion/flow_matching.py
vrl/algorithms/base.py
vrl/algorithms/grpo/continuous.py
vrl/algorithms/grpo/token.py
vrl/algorithms/grpo/multisegment.py
vrl/algorithms/diffusion_nft.py
```

必须满足：

- 定义 `TrajectorySignalBatch` / `SegmentSignal`。
- 旧 `SignalBatch` 可以适配成 `TrajectorySignalBatch`。
- diffusion / AR / R1 signals 都有 explicit `segment`、`axis`、`distribution`、`mask`。
- 定义 `AlgorithmInput` 和 `AlgorithmAdapter`。
- 现有 `GRPO` / `TokenGRPO` / `MultiSegmentTokenGRPO` / `DiffusionNFT` / `DPO` 可以通过 adapter 消费 `AlgorithmInput`。
- adapter 不改变现有 algorithm 数值行为。
- SD3.5 OCR 现有 GRPO loss 和 `AlgorithmInput` adapter 在同一批 fake/recorded diffusion signals 上数值 parity。
- R1 可以表达 per-segment reward / per-segment advantage；不再要求每个 segment 复用同一份 advantage。

不能算完成：

- 把所有 objective 强行合并成一个新的 GRPO 类。
- `SignalBatch.aux["segments"]` 继续作为 R1 segment signal 主路径。
- algorithm adapter 继续猜 family-specific loose extras。

### 13.5 Sprint D：EnginePlan / FamilyCapability / profiler labels

目标：把优化入口放到 engine contract，而不是 family script 或私有 runtime 分支。

新增：

```text
vrl/engine/core/planner.py
vrl/engine/core/capabilities.py
tests/engine/test_engine_planner.py
```

编辑：

```text
vrl/engine/core/protocols.py
vrl/engine/core/worker.py
vrl/engine/core/runtime.py
vrl/engine/microbatching.py
vrl/distributed/ray/rollout/planner.py
vrl/distributed/ray/rollout/executor.py
vrl/distributed/ray/rollout/worker.py
vrl/rollouts/families/specs.py
vrl/trainers/profiling.py
vrl/models/families/janus_pro/runtime.py
vrl/models/families/sd3_5/runtime.py
```

必须满足：

- 每个 migrated family 能声明 `FamilyCapability`。
- `EnginePlan` 能表达 `trajectory_kind`、expected axes、microbatch chunks、execution units。
- request batching 和 chunk planning 至少能读取 capability，不再只靠 request key。
- profiler label 从 `EnginePlan.execution_units` 生成。
- torch profiler trace 能看到统一命名空间：

```text
engine.plan
engine.prefill
engine.decode_step
engine.denoise_step
engine.vq_decode
engine.cache_read
engine.cache_write
collector.reward_score
trainer.replay
trainer.loss
```

不能算完成：

- planner 只是包装当前 `forward_chunk`，没有 axis/segment/execution-unit 信息。
- Janus KV cache 或 diffusion prompt cache 写成 runtime 私有逻辑，capability 看不到。
- profiler 只能看到 `forward_chunk`。

### 13.6 Sprint E：Janus KV decode + Ray resident session

目标：把 Janus AR sampling 改成真正的 `prefill once + decode step`，并让 Ray rollout worker 可以持有 resident state。

新增或编辑：

```text
vrl/engine/ar/
vrl/models/families/janus_pro/policy.py
vrl/models/families/janus_pro/runtime.py
vrl/distributed/ray/rollout/worker.py
vrl/distributed/ray/rollout/executor.py
vrl/distributed/ray/rollout/runtime.py
tests/models/test_janus_wrapper.py
tests/models/test_janus_r1_policy.py
tests/engine/generation/test_ar_token_scheduler.py
tests/distributed/ray/test_ray_resident_session.py
```

必须满足：

- Janus AR sampling 先 prompt prefill，一步 decode 只喂新增 image token embedding。
- `use_cache=False` 不再是主路径默认。
- 同一 position 的 prompt groups 能合 batch。
- cond/uncond batch 维度稳定，不在每个 token step 动态重建大 tensor。
- 避免 per-token `.item()`、Python list cat、动态 `torch.cat` 重建上下文。
- Ray rollout worker 至少能保留一种 resident state，例如 prompt embeddings、KV cache、sampled token buffer、VQ decode helper。
- KV/cache metrics 写入 engine metrics；`TrajectoryBatch.metrics` 只记录轨迹本身的可序列化统计。
- correctness parity 通过：相同 seed/config 下，旧 decode 和 KV decode 的 token/logprob/mask contract 一致，允许浮点容差。

不能算完成：

- 只在 local Janus runtime 里私有启用 `use_cache=True`。
- Ray worker 每个 rollout step 仍然重新构造/释放所有 sampling state。
- optimization 没有挂到 `FamilyCapability` / `EnginePlan`。

### 13.7 Sprint F：Common recipe / train script 收敛

目标：最后再抽 train script glue，避免先抽象出一层还没吃到 trajectory/engine contract 的 legacy。

新增：

```text
vrl/training/recipe.py
vrl/training/factory.py
vrl/training/loop.py
tests/training/test_online_rl_recipe_factory.py
```

编辑：

```text
vrl/scripts/janus_pro/train.py
vrl/scripts/nextstep_1/train.py
vrl/scripts/sd3_5/train.py
vrl/scripts/wan_2_1/train.py
vrl/scripts/cosmos/train.py
```

必须满足：

- 至少 Janus-Pro 和 SD3.5 使用 common recipe runner。
- family script 只保留 family-specific builder 和 entrypoint。
- checkpoint/save/eval/prompt loop 逻辑减少重复。
- reward construction、runtime setup、trainer construction 使用 shared factory。
- SD3.5 OCR 入口和行为保持兼容。

不能算完成：

- 只是把旧 script 复制到 `factory.py`。
- common recipe 内部继续按 family hardcode rollout/evaluator/algorithm glue。

## 14. 总完成标准

### 14.1 当前可执行完成标准

先完成 baseline gate + Sprint A + Sprint B。达到这里后，repo 还不能 claim fully unified，但已经有了可落地的 trajectory contract 和 generic packer。

必须满足：

1. SD3.5 OCR baseline gate 通过。
2. `TrajectoryBatch` 是可序列化的训练/奖励轨迹记录，不包含 runtime state。
3. `RewardView` / `TrainingView` 有具体类型定义，且只做 view，不复制 tensor。
4. `OutputBatch.trajectory` 字段存在；`OutputBatch.extra["trajectory"]` 只是 bridge。
5. SD3.5 原生 emit `TrajectoryBatch`，且旧 packer 输出完全不变。
6. Janus-Pro discrete image-token path 原生 emit `TrajectoryBatch`。
7. `TrajectoryRolloutPacker` 可以从 SD3.5 和 Janus-Pro 的 `TrajectoryBatch` 生成旧 `RolloutBatch`。
8. SD3.5 diffusion old packer 和 `TrajectoryRolloutPacker` parity 通过。
9. Janus-Pro old packer 和 `TrajectoryRolloutPacker` parity 通过。
10. 所有新增 legacy bridge 都有删除门槛和 strict-mode 测试。

### 14.2 Fully unified gate

这些必须在 README / paper intro claim unified 之前完成：

1. 每个 family runtime 都原生 emit `TrajectoryBatch`。
2. `OutputBatch.extra` 不再是 action/logprob/mask/replay 的主来源。
3. `TrainingView` 覆盖 diffusion timestep、discrete token、continuous token、multi-segment loss units。
4. `TrajectorySignalBatch` 取代 `SignalBatch.aux["segments"]` 作为 segment signal 主路径。
5. 所有 algorithm 都能通过 `AlgorithmInput` adapter 接入，不再直接依赖 family-specific loose extras。
6. `EnginePlan` / `FamilyCapability` 能为 Janus 生成 prefill/decode plan。
7. Janus AR rollout 至少有一版真实 `prefill once + decode step` 优化通过 shared engine path 接入。
8. Ray rollout worker 能在 plan/session 级别保留至少一种 resident state。
9. 新增 family 时，只需要补 capability + trajectory emission + reward/training view，不需要新增 trainer 主循环或 algorithm class。
10. `dual_path` 只能用于 parity，不允许成为长期默认路径。

不能算完成的情况：

- 只在 `OutputBatch.extra["trajectory"]` 放一个对象，但 packer/trainer/evaluator 仍然读旧 extras。
- 把 cache handle、scheduler object、Ray actor/session state 放进 `TrajectoryBatch`。
- Janus KV cache 写成 Janus runtime 私有优化，`EnginePlan` 和 `FamilyCapability` 看不到。
- R1 segment 仍然通过 `SignalBatch.aux["segments"]` 传递，algorithm 仍然复用同一份 advantage。
- profiler 只能看到 `forward_chunk`，看不到 prefill/decode/cache/timestep/segment。
- train script 仍然写 family-specific rollout/evaluator/algorithm glue。
- 为了让新 trajectory 路径通过而删掉 SD3.5 OCR 的 config/runtime/packer/evaluator 回归测试。
- 新增的 compatibility key 比旧 extras 更多，导致 trainer/algorithm 继续依赖 loose dict。

## 15. 验收命令

Baseline gate：

```bash
pytest tests/config/test_load_all_experiments.py \
  tests/rollouts/test_runtime_inputs.py \
  tests/rollouts/test_family_registry.py \
  tests/engine/generation/test_chunk_gatherer.py \
  tests/rewards/test_ocr.py \
  tests/scripts/test_sd3_resume.py
```

Sprint A：

```bash
pytest tests/engine/trajectory/test_trajectory_types.py
```

Sprint B：

```bash
pytest tests/rollouts/test_sd3_5_ocr_trajectory_parity.py \
  tests/rollouts/test_trajectory_packer.py
```

Sprint C：

```bash
pytest tests/rollouts/evaluators/test_trajectory_signals.py \
  tests/algorithms/test_algorithm_input_views.py
```

Sprint D：

```bash
pytest tests/engine/test_engine_planner.py
```

Sprint E：

```bash
pytest tests/engine/generation/test_ar_token_scheduler.py \
  tests/models/test_janus_wrapper.py \
  tests/models/test_janus_r1_policy.py \
  tests/distributed/ray/test_ray_resident_session.py
```

Sprint F：

```bash
pytest tests/training/test_online_rl_recipe_factory.py
```

现有 family 回归：

```bash
pytest tests/engine/generation \
  tests/models/test_janus_wrapper.py \
  tests/models/test_janus_r1_policy.py \
  tests/models/test_nextstep_1_policy.py \
  tests/rollouts \
  tests/trainers
```

profile 回归：

```bash
python -m vrl.scripts.train --config profile/janus_pro_r1_codex_qa_1epoch
```

## 16. 何时才能 claim unified

只有同时满足这些条件，README / paper intro 才能说 unified：

- 每个 family runtime 都产出 `TrajectoryBatch`。
- trainer 不再需要通过 family-specific shape 猜 logprob/mask 语义。
- 至少一个 generic trajectory packer 被 diffusion 和 AR 同时使用。
- 至少一个 generic `AlgorithmInput` adapter 同时覆盖 diffusion-style axis 和 AR-style axis。
- engine planner 能基于 capability 做 batching/chunk/profile，而不是 hardcode family。
- 新增一个优化时，不需要同时改 Janus、NextStep、SD3.5、Wan、Cosmos 五套 glue code。

在此之前，更诚实的描述是：

```text
This repo is moving toward a unified online RL stack for generation.
The current gap is trajectory unification across denoising steps, image tokens,
continuous tokens, and multi-segment regeneration.
```

## 17. 和 AR KV sprint 的关系

`SPRINT_ar_rollout_kv_cache_optimization.md` 是性能 sprint，解决 Janus/NextStep 这类 AR generation 太慢的问题。

本 sprint 是架构 sprint，解决为什么性能优化不能自然惠及所有 family。

推荐执行顺序：

1. 先完成 baseline gate + Sprint A + Sprint B：trajectory contract、SD3.5 / Janus-Pro trajectory emission、generic packer parity。
2. 并行推进 Sprint E 的 Janus KV decode，但不要把 KV cache handle 塞进 `TrajectoryBatch`。
3. 当 AR KV decode 完成后，把 `prefill_steps` / `decode_steps` / `cache_hit` 写入 `OutputBatch.metrics` 或 engine metrics；`TrajectoryBatch.metrics` 只记录和轨迹本身相关的可序列化统计。
4. 再完成 Sprint D，把 AR 和 diffusion 的 batching/profiling 统一到 `EnginePlan` / `FamilyCapability`。
