# SPRINT：Trajectory / Reward Artifact 内存边界与观测

状态：deferred，但仍然有效。

## 核心结论

这个 sprint 仍然有意义，但它不再是旧 `engine` sprint。

当前边界应该是：

```text
vrl/generation
  负责 GenerationRequest -> GenerationOutput、runtime metrics、chunk/ray 保留 counters。

vrl/trajectory
  负责可序列化训练记录、axis/view/resolver、storage policy。

vrl/rollouts/collector
  负责 reward artifact 生命周期和 GenerationOutput -> RolloutBatch。

vrl/rollouts/evaluators
  负责从 TrajectoryBatch / TrainingView 解析训练信号。
```

本 sprint 的目标是把跨 family 的内存边界和 profile 观测做实，不解决 diffusion denoise loop 本身的计算路径。diffusion 可以作为第一批 adoption family，但通用 policy 不应写进 diffusion-only 模块。

## 当前代码事实

- `GenerationMetrics.engine_counters` 已经存在于 `vrl/generation/types.py`，但 key schema 还不统一。
- `vrl/generation/diffusion/gather.py` 目前只聚合 `stage_durations_s`，还不能回答 trajectory / reward artifact / replay tensor 占多少。
- `vrl/trajectory/validation.py` 已经拒绝 runtime-only state，这个方向正确。
- `vrl/trajectory/storage.py` 已经提供 `TrajectoryStoragePolicy`、config parser、apply helper 和 trajectory byte counter。
- `vrl/trajectory/resolver.py` 已经支持 `LossUnit.axis_index`，可以先 slice 当前 loss unit，再返回 resolved tensor。
- `vrl/rollouts/collector/artifacts.py` 已经提供 reward artifact lifecycle policy；collector 可在 reward scoring 后显式释放 reward-only decoded output。

## 非目标

本 sprint 不做：

- 不改变 `TrajectoryBatch` schema。
- 不改变 reward、advantage、GRPO / DPO / DiffusionNFT 算法语义。
- 不把 KV cache、scheduler state、pipeline handle、Ray actor handle、CUDA graph handle 放进 trajectory。
- 不默认 CPU offload。
- 不做 diffusion denoise preallocation；那属于 diffusion-specific performance sprint。
- 不让 `engine_counters` 变成训练输入或 source of truth。

## 设计原则

### 1. Trajectory 是训练记录

`TrajectoryBatch` 只能保存训练、奖励、replay 需要的可序列化事实：

```text
actions
old_log_prob
mask
reward view metadata
replay input tensors
serializable context
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
GenerationOutput.output
  reward artifact：image / video / text，给 reward scoring 用。

TrajectoryBatch
  trainer replay record：actions / old_log_prob / mask / replay inputs。

RolloutBatch
  trainer-facing batch view：尽量引用 trajectory，不复制 decoded reward artifact。
```

reward 完成后，如果 trainer replay 不需要 decoded output，就不应继续长期持有它。

### 3. Storage policy 是 runtime policy

Storage policy 只决定 trajectory tensor 存放位置和 dtype：

```yaml
rollout:
  trajectory_storage:
    device: preserve   # preserve | cpu
    dtype: preserve    # preserve | float32 | float16 | bfloat16
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

### 4. Reward artifact lifecycle 属于 collector

reward artifact policy 不属于 `TrajectoryStoragePolicy`。

```yaml
rollout:
  reward_artifact:
    keep_after_reward: true
```

默认保持当前行为，方便 debug / visual inspection。只有显式配置时才释放 reward-only decoded output。

## 需要编辑的文件

### Generation metrics / counter preservation

```text
vrl/generation/types.py
vrl/generation/execution/request_batch.py
vrl/generation/execution/planner.py
vrl/generation/ray/executor.py
vrl/generation/ray/worker.py
vrl/generation/diffusion/gather.py
vrl/models/ar/janus_pro/runtime.py
vrl/models/ar/nextstep_1/runtime.py
```

目标：

- 标准化 `GenerationMetrics.engine_counters` key 命名和值类型。
- in-process batching、Ray worker、family gather 不丢 counters。
- error output 也保留已知 counters。
- counters 必须 JSON-serializable。
- counters 不写入 `TrajectoryBatch.context`。

建议 key namespace：

```text
generation_*
trajectory_*
reward_artifact_*
diffusion_*
ar_*
```

### Evaluator slice access

编辑：

```text
vrl/trajectory/views.py
vrl/trajectory/resolver.py
vrl/rollouts/evaluators/trajectory.py
vrl/rollouts/evaluators/ar/token_logprob.py
vrl/rollouts/evaluators/ar/continuous_token_logprob.py
vrl/rollouts/evaluators/ar/multi_segment_token_logprob.py
vrl/rollouts/evaluators/diffusion/sde_logprob.py
```

目标：

- evaluator 不回退 legacy extras。
- step-level signal 优先通过 `TrainingView` / `LossUnit.axis_index` 解析当前 slice。
- `TrajectorySignalBatch` contract 不变。
- 对已经在目标 device 上的 tensor，slice 应保持 view 行为，不引入整条 copy。

## 实施阶段

### Phase 1：metrics counters contract

完成标准：

- `GenerationMetrics.engine_counters` key schema 明确。
- in-process / Ray / family gatherer 不丢 counters。
- counters 可 JSON 序列化。
- counters 不进入 trajectory context。

### Phase 2：evaluator slice access

完成标准：

- resolver 能稳定返回当前 loss slice。
- diffusion timestep、AR token、R1 segment 都不依赖 duplicated extras。
- 对 tensor view / copy 行为有 regression guard。

### Phase 3：family first adoption

完成标准：

- 至少一个 family 的 counters 能输出 trajectory bytes、reward artifact bytes、storage policy device/dtype。
- 建议第一批接入 diffusion OCR，因为 tensor 最大、最容易暴露内存问题。

## Tests

新增或编辑：

```text
tests/rollouts/test_evaluator_slice_access.py
tests/generation/ray/test_ray_resident_session.py
tests/engine/generation/test_chunk_gatherer.py
tests/trainers/test_memory_guards.py
```

测试要求：

- counters 可 JSON 序列化。
- evaluator 仍输出 `TrajectorySignalBatch`。

## 验收命令

```bash
pytest tests/rollouts/test_evaluator_slice_access.py \
  tests/engine/generation/test_chunk_gatherer.py \
  tests/generation/ray/test_ray_resident_session.py

pytest tests/models tests/rollouts tests/trainers/test_memory_guards.py
python -m compileall vrl tests
git diff --check
```

## 最终完成标准

完成后必须能回答：

- 每个 rollout batch 的 trajectory / reward artifact / replay tensor 大概占多少内存。
- storage policy 是否能跨 diffusion、AR、R1、NextStep 复用。
- evaluator 是否严格从 `TrajectoryBatch` / `TrainingView` 取 old logprob、mask、distribution。
