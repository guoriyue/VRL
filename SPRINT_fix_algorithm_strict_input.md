# SPRINT：Algorithm strict AlgorithmInput

状态：已完成。`SignalBatch` 类型、strict-to-legacy signal bridge、`trajectory_signals_to_signal_batch(...)` 和 `old_log_probs_from_trajectory_signals(...)` 已删除；algorithm 主路径原生消费 `AlgorithmInput` / `TrajectorySignalBatch`。

## 目标

让 algorithm 原生消费 `AlgorithmInput` / `TrajectorySignalBatch` / `TrainingView`，删除 algorithm 主路径里把 strict signals 反向压回 legacy `SignalBatch` 的适配。

历史上 `AlgorithmInput.signals` 已经禁止 legacy `SignalBatch`，但 adapter 仍在 loss 计算前做回退：

```python
legacy_signals = trajectory_signals_to_signal_batch(
    signals,
    mask_key=_mask_key(algorithm),
)
old_log_probs = old_log_probs_from_trajectory_signals(signals)
advantages = self.compute_advantages(algorithm, inputs)
return algorithm.compute_signal_loss(legacy_signals, advantages, old_log_probs)
```

这个 sprint 的核心是删掉这条反向回退；当前代码已经完成。

## 不做的事

- 不强行合并 GRPO、TokenGRPO、MultiSegmentTokenGRPO、DiffusionNFT、DPO 成一个大算法类。
- 不先解决 replay inputs 完全脱离 `RolloutBatch.extras` 的问题；这由 `SPRINT_fix_rollout_extras_thinning.md` 推进。
- 不改变 advantage 数学公式。
- 不把 evaluator strict migration 和 algorithm strict migration 混成一个 PR。

## 实施阶段

### Phase 1：定义 native algorithm API

编辑：

```text
vrl/algorithms/base.py
vrl/algorithms/trajectory.py
```

要求：

- 新增或固化 `compute_loss(inputs: AlgorithmInput)` 作为长期入口。
- `compute_advantages_from_tensors(...)` 保留，因为 trainer 仍需要跨 batch/group 计算 advantage。
- `AlgorithmAdapter.compute_loss(...)` 不再调用 `trajectory_signals_to_signal_batch(...)`。
- `vrl/algorithms/base.py` 不再 import `SignalBatch`。
- `old_log_probs_from_trajectory_signals(...)` 不能再被 algorithm/trainer 主路径使用。

### Phase 2：迁移 GRPO / TokenGRPO

编辑：

```text
vrl/algorithms/grpo/continuous.py
vrl/algorithms/grpo/token.py
```

要求：

- `GRPO` 从 `inputs.signals.primary` 读取 `log_prob`、`old_log_prob`、`ref_log_prob`、`distribution`、flow matching intermediates。
- legacy `dist_family` 替换为 `SegmentSignal.distribution`。
- `TokenGRPO` 使用 `SegmentSignal.mask`，不再读 `SignalBatch.aux["token_mask"]`。
- KL 计算继续读 `segment.ref_log_prob`。
- shape/device/dtype guard 保留，错误信息改成 segment-aware。

### Phase 3：迁移 MultiSegmentTokenGRPO

编辑：

```text
vrl/algorithms/grpo/multisegment.py
```

要求：

- 多段来源为 `inputs.signals.segments`。
- segment 顺序优先由 `inputs.training_view.loss_units` 决定。
- `TrainingView` 定义可训练 loss units，config 的 `segment_weights` / `train_segments` 只做过滤和权重。
- 每段直接读取 `SegmentSignal.log_prob`、`old_log_prob`、`mask`、`ref_log_prob`。
- 删除对 `SignalBatch.aux["segments"]` 和 `SignalBatch.aux["old_log_probs"]` 的依赖。

### Phase 4：迁移 DiffusionNFT direct path

编辑：

```text
vrl/algorithms/diffusion_nft.py
vrl/trainers/online.py
```

要求：

- `DiffusionNFT` 提供 native `compute_loss(inputs: AlgorithmInput)`。
- trainer 不再直接调用 `compute_batch_timestep_loss(model, batch, j, adv_b)`。
- 短期允许 `AlgorithmInput.metadata` 携带 `model`、`rollout_batch`、`timestep_index`。
- 完成标准是 trainer 只调用 algorithm native loss，不是 replay payload 已经完全 trajectory-pure。

### Phase 5：收口 DPO 特殊分支

编辑：

```text
vrl/algorithms/dpo.py
vrl/algorithms/trajectory.py
```

要求：

- `DiffusionDPOConfig` 可以保留现有行为，但应通过显式 native method 进入。
- adapter 不再靠 algorithm 类型做越来越多的特殊分支。

## 测试计划

编辑：

```text
tests/algorithms/test_algorithm_input_views.py
tests/algorithms/test_grpo.py
tests/algorithms/test_grpo_token.py
tests/algorithms/test_multisegment_token_grpo.py
tests/algorithms/test_diffusion_nft.py
tests/trainers/test_online.py
```

新增断言：

- algorithm path 不调用 `trajectory_signals_to_signal_batch(...)`。
- native `AlgorithmInput` loss 和旧公式的数值期望一致。
- TokenGRPO partial mask、全 0 mask、KL k1/k3 都从 `SegmentSignal.mask` 测。
- MultiSegmentTokenGRPO 使用 `TrainingView.loss_units` 控制 segment 顺序和选择。
- DiffusionNFT 从 `AlgorithmInput` 进入，不由 trainer 特殊调用 batch/timestep API。

## 完成标准

- `rg "trajectory_signals_to_signal_batch|old_log_probs_from_trajectory_signals|compute_signal_loss\\(" vrl/algorithms vrl/trainers/online.py` 不再显示 algorithm/trainer 主路径回退。
- `vrl/algorithms/base.py` 不再 import `SignalBatch`。
- `GRPO`、`TokenGRPO`、`MultiSegmentTokenGRPO`、`DiffusionNFT` 都有 native `AlgorithmInput` loss 入口。
- `OnlineTrainer` 只构造 `AlgorithmInput`，不把 `TrajectorySignalBatch` 降回 `SignalBatch`。
- 通过：

```bash
pytest tests/algorithms \
  tests/rollouts/evaluators/test_trajectory_signals.py \
  tests/trainers/test_online.py
```

## 参考路径

- `/home/mingfeiguo/Desktop/wm-infra/vrl/algorithms/base.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/algorithms/trajectory.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/algorithms/grpo/continuous.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/algorithms/grpo/token.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/algorithms/grpo/multisegment.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/algorithms/diffusion_nft.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/algorithms/dpo.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/trainers/online.py`
