# SPRINT：VisualTrajectory 与 Engine 统一化

这份 proposal 的目标不是继续把更多 family 写进同一个 repo，而是把现在“写在一起”的 diffusion / AR / video / R1 pipeline 收敛成一个真正可优化的 engine + trajectory contract。只有这样，后续 KV cache、batching、profile、compile、Ray rollout、reward 并发这些优化才会一次改动、多 family 受益。

## 1. 现状判断

当前 repo 更准确的定位是：

```text
multi-family visual RL codebase with early shared trainer/runtime pieces
```

还不能 claim：

```text
unified visual RL framework
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
VisualTrajectoryBatch
```

它不强行把所有 family flatten 成 token，而是显式表达 axis、segment、distribution、replay context、reward view。

目标是把现在的隐式 contract：

```text
OutputBatch.extra + family packer + evaluator assumptions
```

升级为显式 contract：

```text
GenerationRequest
  -> EnginePlan
  -> FamilyExecutor
  -> VisualTrajectoryBatch
  -> RewardView
  -> TrainingView
  -> Evaluator
  -> Algorithm
```

统一不是指所有 family 的 tensor 形状一样，而是指每个 tensor 都声明自己的 axis 和 role。算法和 engine 优化可以读这些 metadata，而不是靠 family-specific key 名称猜。

## 3. 设计原则

- 不把 diffusion timestep、AR token、R1 segment 强行合成一个假 token axis。
- 不马上删除现有 `OutputBatch` / `RolloutBatch`，先加兼容层。
- family-specific executor 仍然负责真实生成逻辑。
- shared engine 负责调度、batching、profiling、weight sync、shape/capability metadata。
- shared algorithm 只消费标准 logprob / mask / advantage scope，不关心图像是怎么生成的。
- packer 逐步从 family adapter 变成 view builder。
- 优化必须挂在 engine/trajectory contract 上，不能只写在某个 family script 里。

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
    role="observation" | "action" | "old_logprob" | "mask" | "replay_context" | "media",
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

VisualTrajectoryBatch(
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
- `group_ids`：GRPO per-prompt normalization 需要。
- `segments`：R1、diffusion、AR 都可以表达。
- `old_logprob`：采样时的 old policy logprob。
- `mask`：训练有效位置。
- `replay_context`：evaluator 重新计算 current logprob 所需的全部上下文。
- `reward_views`：reward 看到的是 image/video/text，不一定等于训练 action。
- `metrics`：engine 和 trajectory 的统一指标。

## 5. Engine v2 边界

当前 engine 边界：

```text
GenerationRequest -> OutputBatch
```

短期保留，但增加：

```text
OutputBatch.trajectory: VisualTrajectoryBatch | None
```

如果暂时不想改 `OutputBatch` 字段，可以先放在：

```text
OutputBatch.extra["visual_trajectory"]
```

最终目标：

```text
GenerationRequest -> EngineResult
```

其中：

```python
EngineResult(
    request_id=str,
    trajectory=VisualTrajectoryBatch,
    reward_views=dict[str, Any],
    metrics=GenerationMetrics,
    artifacts=dict[str, Any],
)
```

### 5.1 EnginePlan

新增 engine planning 层：

```text
vrl/engine/core/planner.py
```

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
)
```

### 5.2 FamilyCapability

新增：

```text
vrl/engine/core/capabilities.py
```

每个 family builder 声明 capability：

```python
FamilyCapability(
    family="janus_pro",
    trajectory_kind="ar_discrete",
    supports_batched_requests=True,
    supports_chunked_execution=True,
    supports_kv_decode=True,
    supports_cuda_graph=False,
    supports_torch_compile=False,
    trainable_segments=("image_tokens",),
    reward_views=("image",),
)
```

这让 engine 优化可以读 capability，而不是在 trainer 或 script 里 hardcode family 名字。

### 5.3 Executor output contract

当前 family executor 返回 `OutputBatch`，里面把训练需要的内容塞到 `extra`。

迁移后 executor 应该返回：

```text
VisualTrajectoryBatch + artifacts
```

兼容期：

```text
executor.forward(...) -> OutputBatch
OutputBatch.extra["visual_trajectory"] = VisualTrajectoryBatch
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
VisualTrajectoryRolloutPacker
```

它根据 `VisualTrajectoryBatch` 生成现有 `RolloutBatch`，并保留 trainer 兼容字段：

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
family OutputBatch -> VisualTrajectoryBatch -> VisualTrajectoryRolloutPacker -> RolloutBatch
```

完成后，packers 不再是主要 contract，只是 legacy adapter。

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
evaluate(model, VisualTrajectoryBatch, SignalRequest) -> TrajectorySignalBatch
```

兼容期：

- diffusion evaluator 继续返回 `SignalBatch`。
- AR evaluator 继续返回 `SignalBatch`。
- 新 adapter 把旧 `SignalBatch` 包成 `TrajectorySignalBatch`。

## 8. Algorithm v2

当前 algorithm 分裂：

- `GRPO`：diffusion / continuous。
- `TokenGRPO`：AR token。
- `MultiSegmentTokenGRPO`：R1 segment。
- `DiffusionNFT`：特殊训练目标。

新增：

```text
vrl/algorithms/trajectory_grpo.py
```

建议：

```python
TrajectoryGRPOConfig(
    eps_clip=0.2,
    init_kl_coef=0.0,
    advantage_scope="sample" | "segment" | "axis",
    segment_weights=dict[str, float],
    train_segments=dict[str, bool],
    normalize_by="mask" | "sample" | "segment",
)

TrajectoryGRPO.compute_loss(
    signals: TrajectorySignalBatch,
    rewards: Tensor[B] | dict[str, Tensor],
    group_ids: Tensor[B],
)
```

关键改变：

- advantage scope 显式化。
- segment weight 显式化。
- mask 和 axis 显式化。
- algorithm 不再通过 `old_lp.ndim` 猜 shape。
- R1 可以自然表达 per-segment reward / per-segment advantage。

`DiffusionNFT` 可以暂时不并入，因为它不是 standard logprob policy gradient；但它也应该消费 `VisualTrajectoryBatch` 的 replay context，而不是靠 loose extras。

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
build_visual_rl_recipe(cfg) -> VisualRLRecipe
```

```python
VisualRLRecipe(
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
    recipe = build_visual_rl_recipe(cfg)
    await run_online_recipe(recipe)
```

不要求第一阶段完成全部脚本统一。先把共同 glue 抽走：

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
vrl/models/families/*/executor.py
```

收益：

- 只有 shape/axis stable 的 workload 才启用 compile/graph。
- 避免在每个 family script 里手写 compile 开关。

## 11. 实施阶段

### Phase 1：Contract only，不改变行为

新增：

```text
vrl/engine/trajectory/axes.py
vrl/engine/trajectory/types.py
vrl/engine/trajectory/validation.py
vrl/engine/trajectory/compat.py
tests/engine/trajectory/test_visual_trajectory_types.py
```

完成标准：

- 定义 `VisualTrajectoryBatch` / `TrajectorySegment` / `TrajectoryTensor`。
- 支持 diffusion、AR discrete、AR continuous、R1 的最小 schema。
- validator 能检查 batch dim、axis 名称、mask/logprob shape。
- 没有任何现有 recipe 行为变化。

### Phase 2：OutputBatch 兼容接入

编辑：

```text
vrl/engine/core/types.py
vrl/models/families/janus_pro/executor.py
vrl/models/families/janus_pro/r1_executor.py
vrl/models/families/nextstep_1/executor.py
vrl/models/families/sd3_5/executor.py
vrl/models/families/wan_2_1/executor.py
vrl/models/families/cosmos/executor.py
```

完成标准：

- 每个 executor 在 `OutputBatch.extra["visual_trajectory"]` 写入 trajectory。
- 现有 packer 仍然走旧字段。
- 单元测试确认 trajectory 和旧字段内容一致。

### Phase 3：Generic trajectory packer

新增：

```text
vrl/rollouts/packers/trajectory.py
tests/rollouts/test_visual_trajectory_packer.py
```

编辑：

```text
vrl/rollouts/packers/diffusion.py
vrl/rollouts/packers/ar/discrete.py
vrl/rollouts/packers/ar/continuous.py
vrl/rollouts/packers/ar/r1.py
```

完成标准：

- 新 packer 可以从 `VisualTrajectoryBatch` 生成旧 `RolloutBatch`。
- family packer 逐步代理到 generic packer。
- `RolloutBatch.extras["trajectory"]` 保留原始 trajectory 引用或 lightweight view。

### Phase 4：Evaluator v2

新增：

```text
vrl/rollouts/evaluators/trajectory.py
tests/rollouts/evaluators/test_trajectory_signals.py
```

完成标准：

- 旧 evaluator 输出可适配成 `TrajectorySignalBatch`。
- diffusion / AR / R1 signals 都有 explicit axes。
- trainer debug logprob parity 不再用 loose shape guess。

### Phase 5：TrajectoryGRPO

新增：

```text
vrl/algorithms/trajectory_grpo.py
tests/algorithms/test_trajectory_grpo.py
```

完成标准：

- 统一处理 sample-level、axis-level、segment-level loss。
- 支持 per-segment weights。
- 支持 per-segment advantage。
- TokenGRPO / MultiSegmentTokenGRPO 可以作为 wrapper 或 deprecated path 保留。

### Phase 6：Engine planner/capability

新增：

```text
vrl/engine/core/planner.py
vrl/engine/core/capabilities.py
tests/engine/test_engine_planner.py
```

编辑：

```text
vrl/engine/core/worker.py
vrl/distributed/ray/rollout/planner.py
vrl/distributed/ray/rollout/executor.py
vrl/distributed/ray/rollout/worker.py
```

完成标准：

- request batching 和 chunk planning 读 capability。
- Ray rollout chunk metrics 写入 unified trajectory metrics。
- engine-level profiling label 包含 trajectory kind / segment / axis。

### Phase 7：Recipe factory

新增：

```text
vrl/training/recipe.py
vrl/training/factory.py
vrl/training/loop.py
tests/training/test_visual_rl_recipe_factory.py
```

编辑：

```text
vrl/scripts/janus_pro/train.py
vrl/scripts/nextstep_1/train.py
vrl/scripts/sd3_5/train.py
vrl/scripts/wan_2_1/train.py
vrl/scripts/cosmos/train.py
```

完成标准：

- 至少 Janus-Pro 和 SD3.5 先迁移到 common recipe runner。
- family script 只保留 family-specific builder。
- checkpoint/save/eval/prompt loop 逻辑减少重复。

### Phase 8：Migration gate

完成标准：

- 所有现有 profile recipe 仍能跑。
- 所有 existing packer/evaluator tests 通过。
- 新 trajectory tests 覆盖四类 trajectory：
  - diffusion timestep
  - discrete image token
  - continuous image token
  - multi-segment R1
- README 中只 claim “unified” after this phase。

## 12. 文件清单

新增：

```text
vrl/engine/trajectory/__init__.py
vrl/engine/trajectory/axes.py
vrl/engine/trajectory/types.py
vrl/engine/trajectory/views.py
vrl/engine/trajectory/validation.py
vrl/engine/trajectory/compat.py
vrl/engine/core/planner.py
vrl/engine/core/capabilities.py
vrl/rollouts/packers/trajectory.py
vrl/rollouts/evaluators/trajectory.py
vrl/algorithms/trajectory_grpo.py
vrl/training/recipe.py
vrl/training/factory.py
vrl/training/loop.py
tests/engine/trajectory/test_visual_trajectory_types.py
tests/engine/test_engine_planner.py
tests/rollouts/test_visual_trajectory_packer.py
tests/rollouts/evaluators/test_trajectory_signals.py
tests/algorithms/test_trajectory_grpo.py
tests/training/test_visual_rl_recipe_factory.py
```

编辑：

```text
vrl/engine/core/types.py
vrl/engine/core/worker.py
vrl/engine/core/runtime.py
vrl/engine/microbatching.py
vrl/distributed/ray/rollout/planner.py
vrl/distributed/ray/rollout/executor.py
vrl/distributed/ray/rollout/worker.py
vrl/rollouts/batch.py
vrl/rollouts/packers/base.py
vrl/rollouts/packers/diffusion.py
vrl/rollouts/packers/ar/discrete.py
vrl/rollouts/packers/ar/continuous.py
vrl/rollouts/packers/ar/r1.py
vrl/rollouts/evaluators/types.py
vrl/rollouts/evaluators/ar.py
vrl/rollouts/evaluators/diffusion/flow_matching.py
vrl/algorithms/base.py
vrl/algorithms/grpo/continuous.py
vrl/algorithms/grpo/token.py
vrl/algorithms/grpo/multisegment.py
vrl/trainers/online.py
vrl/scripts/janus_pro/train.py
vrl/scripts/nextstep_1/train.py
vrl/scripts/sd3_5/train.py
vrl/scripts/wan_2_1/train.py
vrl/scripts/cosmos/train.py
```

## 13. 验收命令

类型和 contract 测试：

```bash
pytest tests/engine/trajectory/test_visual_trajectory_types.py \
  tests/engine/test_engine_planner.py \
  tests/rollouts/test_visual_trajectory_packer.py \
  tests/rollouts/evaluators/test_trajectory_signals.py \
  tests/algorithms/test_trajectory_grpo.py
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

## 14. 何时才能 claim unified

只有同时满足这些条件，README / paper intro 才能说 unified：

- 每个 family executor 都产出 `VisualTrajectoryBatch`。
- trainer 不再需要通过 family-specific shape 猜 logprob/mask 语义。
- 至少一个 generic trajectory packer 被 diffusion 和 AR 同时使用。
- 至少一个 generic trajectory algorithm 处理 diffusion-style axis 和 AR-style axis。
- engine planner 能基于 capability 做 batching/chunk/profile，而不是 hardcode family。
- 新增一个优化时，不需要同时改 Janus、NextStep、SD3.5、Wan、Cosmos 五套 glue code。

在此之前，更诚实的描述是：

```text
This repo is moving toward a unified online RL stack for visual generation.
The current gap is trajectory unification across denoising steps, image tokens,
continuous tokens, and multi-segment regeneration.
```

## 15. 和 AR KV sprint 的关系

`SPRINT_ar_rollout_kv_cache_optimization.md` 是性能 sprint，解决 Janus/NextStep 这类 AR generation 太慢的问题。

本 sprint 是架构 sprint，解决为什么性能优化不能自然惠及所有 family。

推荐执行顺序：

1. 先落地本 sprint 的 Phase 1-3：trajectory contract 和 generic packer 兼容层。
2. 并行推进 AR KV sprint 的 Janus KV decode。
3. 当 AR KV decode 完成后，把 `prefill_steps` / `decode_steps` / `cache_hit` 写入 `VisualTrajectoryBatch.metrics`。
4. 再做 engine planner/capability，把 AR 和 diffusion 的 batching/profiling 统一起来。
