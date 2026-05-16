# SPRINT：Trajectory Runtime 性能与内存优化

状态：proposed。

## 核心结论

这份 sprint 解决所有 RL trajectory family 都会遇到的 runtime 性能问题：

```text
trajectory 存储策略
CPU/GPU 搬运
reward artifact 生命周期
profiler counters
trainer replay slice access
```

这些不是 diffusion 专属。AR token trajectory、Janus-R1 multi-segment trajectory、NextStep continuous-token trajectory、diffusion timestep trajectory 都会受益。

diffusion 只是第一批落地对象，因为它的 tensor 最大、内存压力最明显。

## 当前代码状态

关键路径：

```text
vrl/engine/core/types.py
  GenerationMetrics
  OutputBatch

vrl/engine/trajectory/
  TrajectoryBatch
  build_*_trajectory(...)
  resolver / ops / validation

vrl/rollouts/collector/core.py
  OutputBatch -> reward scoring -> RolloutBatch

vrl/rollouts/collector/batch_builder.py
  trajectory-backed OutputBatch -> trainer RolloutBatch

vrl/rollouts/batch.py
  trainer-side RolloutBatch

vrl/rollouts/evaluators/
  ReplayModel output -> TrajectorySignalBatch
```

当前主要问题：

- `GenerationMetrics.engine_counters` 字段已经存在，但还没有统一 counter schema，无法比较 trajectory bytes / reward artifact bytes / replay tensor bytes。
- `TrajectoryBatch` 已经禁止 runtime-only state，但还没有 storage policy。
- reward artifact 和 trainer replay tensor 的生命周期没有明确 contract。
- `RolloutBatch` 可能重复持有大 tensor。
- evaluator 通常只需要当前 loss slice，但 trajectory 里保存的是整条 generation record。
- 不同 family 的 metrics schema 还不够可比。

## 非目标

本 sprint 不做：

- 不改变 `TrajectoryBatch` 基本 schema。
- 不改变 reward、advantage、GRPO / DPO 算法语义。
- 不把 KV cache、scheduler state、pipeline handle 放进 trajectory。
- 不默认 CPU offload。
- 不删除当前 SD3.5 OCR / Janus / NextStep 可跑路径。
- 不处理 diffusion denoise loop preallocation；那放到 diffusion-specific sprint。

## 设计原则

### 1. Trajectory 是可序列化训练记录

`TrajectoryBatch` 只能保存训练/奖励/replay 需要的可序列化事实：

```text
action
old_log_prob
mask
reward view
replay input tensors
metadata / context
```

不能保存：

```text
KV cache handle
diffusers pipeline object
scheduler mutable state object
Ray actor handle
CUDA graph handle
```

### 2. Reward artifact 和 trainer replay 分离

目标语义：

```text
OutputBatch.output
  reward artifact：image/video/text，给 reward scoring 用。

TrajectoryBatch
  trainer replay record：actions / old_log_prob / mask / replay inputs。

RolloutBatch
  trainer-facing batch view：尽量引用 trajectory，不复制大 tensor。
```

reward 完成后，如果 trainer replay 不需要 decoded output，就不应继续长期持有它。

### 3. Storage policy 是 runtime policy，不是 algorithm policy

新增 storage policy 只决定 tensor 存放位置和 dtype：

```text
device: preserve | cpu
dtype: preserve | float32 | float16 | bfloat16
```

它不能改变：

```text
old_log_prob 数值语义
mask 语义
reward scope
advantage scope
distribution
axis
```

### 4. Reward artifact lifecycle 是 collector policy

`OutputBatch.output` 是 reward artifact，不是 trajectory tensor。

因此 reward artifact 的生命周期不能放进 `TrajectoryStoragePolicy`。它必须由 collector artifact helper 或 rollout collector config 管：

```text
trajectory storage policy
  controls TrajectoryBatch tensor device / dtype.

reward artifact policy
  controls whether decoded image/video/text survives after reward scoring.
```

### 5. Profiler counters 只做观测，不做 source of truth

`GenerationMetrics.engine_counters` 是 debug/profile 输出，不参与算法。

示例：

```text
trajectory_total_bytes
trajectory_action_bytes
trajectory_old_logprob_bytes
trajectory_replay_tensor_bytes
reward_artifact_bytes
storage_policy_device
storage_policy_dtype
reward_artifact_released
```

不要实现 duplicate byte counter。判断 `RolloutBatch` 是否和 `TrajectoryBatch` 重复持有同一份大 tensor，靠 identity check 或 byte-level comparison 都不稳定，也会把 profiling 做重。这里应该用结构约束和测试保证：

```text
RolloutBatch does not retain decoded image/video after reward scoring when policy says release.
RolloutBatch training fields resolve from TrajectoryBatch / TrainingView, not copied reward artifacts.
```

## 目标接口

### `GenerationMetrics.engine_counters`

编辑：

```text
vrl/engine/core/types.py
```

当前字段已经存在：

```python
engine_counters: dict[str, Any] = field(default_factory=dict)
```

要求：

- 本 sprint 不再新增字段，只统一 counter schema 和保留路径。
- counter values 必须可 JSON 序列化。
- counters 不写进 `TrajectoryBatch.context`。
- local executor、Ray worker、request batching、family gather 都要保留 counters。
- `vrl/engine/execution/gather.py` 只保留 generic gather contract；family-specific counter 聚合由对应 family gatherer 负责，例如 diffusion 放在 `vrl/engine/diffusion/gather.py`。

### `TrajectoryStoragePolicy`

新增或放入：

```text
vrl/engine/trajectory/storage.py
```

建议 schema：

```python
@dataclass(frozen=True, slots=True)
class TrajectoryStoragePolicy:
    device: str = "preserve"      # preserve | cpu
    dtype: str = "preserve"       # preserve | float32 | float16 | bfloat16
```

解析 helper：

```text
trajectory_storage_policy_from_cfg(...)
apply_trajectory_storage_policy(...)
trajectory_tensor_bytes(...)
```

### Reward artifact lifetime helper

新增或放入：

```text
vrl/rollouts/collector/artifacts.py
```

目标：

- 明确 reward artifact 从 `OutputBatch.output` 进入 reward scoring。
- reward 后构建 `RolloutBatch` 时避免重复拷贝大 tensor。
- family 不需要各自手写 artifact lifetime 逻辑。

最小 API：

```python
@dataclass(frozen=True, slots=True)
class RewardArtifactPolicy:
    keep_after_reward: bool = True


def extract_reward_artifact(output_batch: OutputBatch) -> Any:
    ...


def reward_artifact_bytes(artifact: Any) -> int:
    ...


def release_reward_artifact(rollout_batch: RolloutBatch) -> None:
    ...
```

语义：

- `extract_reward_artifact(...)` 只从 `OutputBatch.output` 取 reward 输入，不从 `TrajectoryBatch.context` 或 `RolloutBatch.extras` 猜。
- `reward_artifact_bytes(...)` 只用于 `GenerationMetrics.engine_counters["reward_artifact_bytes"]`。
- `release_reward_artifact(...)` 只清理 reward-only decoded output，不删除 trainer replay 需要的 trajectory tensor。
- 如果 `RewardArtifactPolicy.keep_after_reward=True`，helper 保持当前行为，便于 debug / visual inspection。

## 需要编辑的文件

### Core runtime

```text
vrl/engine/core/types.py
vrl/engine/execution/batching.py
vrl/engine/execution/planner.py
vrl/distributed/ray/rollout/worker.py
vrl/distributed/ray/rollout/executor.py
vrl/distributed/ray/rollout/types.py
```

目标：

- 标准化 `GenerationMetrics.engine_counters` key 命名和值类型。
- local batching / Ray chunk execution 不丢 counters。
- error output 也保留已知 counters。
- family gatherer 能聚合 family-specific counters。

### Trajectory

```text
vrl/engine/trajectory/storage.py
vrl/engine/trajectory/__init__.py
vrl/engine/trajectory/builders.py
vrl/engine/trajectory/ops.py
vrl/engine/trajectory/resolver.py
vrl/engine/trajectory/validation.py
```

目标：

- storage policy helper 可被所有 family builder 复用。
- `move_trajectory_batch(...)` 或现有 ops 不做不必要的整条 GPU copy。
- validation 继续拒绝 runtime-only state。
- byte counters 能按 segment / role / tensor 名统计。

### Collector / RolloutBatch

```text
vrl/rollouts/collector/core.py
vrl/rollouts/collector/batch_builder.py
vrl/rollouts/collector/artifacts.py
vrl/rollouts/batch.py
```

目标：

- `RolloutBatch` 不再无意识复制 reward artifact。
- reward-only decoded output 生命周期明确。
- trainer replay 优先从 `TrajectoryBatch` 解析。

### Evaluators

```text
vrl/rollouts/evaluators/trajectory.py
vrl/rollouts/evaluators/ar/token_logprob.py
vrl/rollouts/evaluators/ar/continuous_token_logprob.py
vrl/rollouts/evaluators/ar/multi_segment_token_logprob.py
vrl/rollouts/evaluators/diffusion/flow_matching.py
```

目标：

- evaluator 不直接依赖 duplicated `RolloutBatch.extras`。
- step-level signal 尽量只 resolve 当前需要的 tensor slice。
- `SegmentSignal` source of truth 保持 `TrajectoryBatch` / `TrainingView`。

### Configs

```text
configs/base/rollout/*.yaml
configs/profile/torch_profiler.yaml
README.md
```

新增可选配置：

```yaml
rollout:
  trajectory_storage:
    device: preserve   # preserve | cpu
    dtype: preserve    # preserve | float32 | float16 | bfloat16

  reward_artifact:
    keep_after_reward: true
```

默认保持当前行为：

```text
device=preserve
dtype=preserve
reward_artifact.keep_after_reward=true
```

## Tests

新增：

```text
tests/engine/trajectory/test_storage_policy.py
tests/engine/trajectory/test_trajectory_byte_counters.py
tests/rollouts/test_reward_artifact_lifetime.py
tests/rollouts/test_rollout_batch_no_duplicate_artifacts.py
tests/rollouts/test_evaluator_slice_access.py
```

测试要求：

- storage policy 不改变 `TrajectoryBatch` axis / segment / role / distribution。
- CPU policy 只在显式配置时发生。
- dtype policy 有 numerical tolerance test。
- reward artifact 不进 `TrajectoryBatch.context`。
- reward artifact lifecycle policy 不属于 `TrajectoryStoragePolicy`。
- `RolloutBatch` 不保留 reward-only decoded output 时，用结构断言测试，不做 byte-level duplicate detection。
- counters 可 JSON 序列化。
- evaluator 仍输出 `TrajectorySignalBatch`。

## 实施阶段

### Phase 1：metrics counters contract

完成标准：

- `GenerationMetrics.engine_counters` 使用统一 key schema；不重复新增字段。
- local batching / Ray output / family gather 不丢 counters。
- counters 不进入 `TrajectoryBatch.context`。
- 基础 tests 通过。

### Phase 2：TrajectoryStoragePolicy helper

完成标准：

- 新增 storage policy dataclass 和 parser。
- 新增 byte counting helper。
- 默认 preserve 行为不改变现有 tests。
- validation 继续拒绝 runtime-only state。

### Phase 3：reward artifact lifetime

完成标准：

- reward artifact 和 trainer replay record 有明确 helper/API。
- `extract_reward_artifact(...)` / `reward_artifact_bytes(...)` / `release_reward_artifact(...)` 有单测。
- `RolloutBatch` 不重复保存不必要的大 tensor。
- `keep_after_reward` 配置不在 `TrajectoryStoragePolicy` 里。
- SD3.5 OCR / Janus image reward / R1 reward path 不退化。

### Phase 4：evaluator slice access

完成标准：

- evaluator helper 能只 resolve 当前 loss slice。
- 不回退到 legacy extras。
- `TrajectorySignalBatch` contract 不变。

### Phase 5：family first adoption

完成标准：

- 至少一个 family 端到端采用 `TrajectoryStoragePolicy` 和 reward artifact helper，并通过原有 recipe 回归。
- 建议第一批采用 SD3.5 OCR，因为 tensor 最大、最容易暴露内存问题。
- adopted family 的 `GenerationMetrics.engine_counters` 能输出 trajectory bytes、reward artifact bytes、storage policy device/dtype。
- `RolloutBatch` 对 adopted family 不再把 decoded reward artifact 当 trainer replay source of truth。
- README 记录 profile 输出位置和 counters 解释。

## 验收命令

```bash
pytest tests/engine/trajectory/test_storage_policy.py \
  tests/engine/trajectory/test_trajectory_byte_counters.py \
  tests/rollouts/test_reward_artifact_lifetime.py \
  tests/rollouts/test_rollout_batch_no_duplicate_artifacts.py \
  tests/rollouts/test_evaluator_slice_access.py

pytest tests/models tests/engine tests/rollouts tests/trainers/test_online.py
python -m compileall vrl tests
git diff --check
```

## 风险与处理

- CPU offload 可能降低速度：默认 preserve，只在显式 config 下启用。
- dtype policy 可能影响 replay logprob：必须有 tolerance test，默认 preserve。
- artifact lifetime 清理可能影响 reward：先从 tests 固化 SD3.5 OCR / Janus reward contract。
- counters 可能被误当训练输入：只放 `GenerationMetrics`，不放 algorithm input。

## 最终完成标准

完成后必须能回答：

- 每个 rollout batch 的 trajectory / reward artifact / replay tensor 大概占多少内存。
- storage policy 是否能跨 diffusion、AR、R1、NextStep 复用。
- reward artifact 是否和 trainer replay record 解耦。
- evaluator 是否仍严格从 `TrajectoryBatch` / `TrainingView` 取 old logprob、mask、distribution。
