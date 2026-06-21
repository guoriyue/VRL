# SPRINT: 跨子系统散点死字段收口（done）

状态：done（落地 commit `efb1924`；2026-06-21 归档）。
范围：删除 4 个分散在 rollouts/orchestration、algorithms、models/interfaces、rewards 的死字段：`ContinuousRolloutItem.submitted_at`、`TrainStepMetrics.lr`、`ReplayResult.context`、`RewardInferenceResult.raw_response`。
来源：dead-dataclass-hunt + 手动验证，承接 [[SPRINT_segment_signal_dead_field_cleanup]]。

## 0. Core Decision

四个字段都没有行为消费者：
- `ContinuousRolloutItem.submitted_at` 是一条死管道：producer submit 时打时间戳，经 `_collect_group` 和 result dict 传回 item，但 item 的 `.submitted_at` 从不读取。下游实际使用的是 `completed_at` 和 `age_s`。
- `TrainStepMetrics.lr` 从不设置、从不读取；`lr=optim.lr` 命中属于 optimizer 构造，不是 metrics 字段。
- `ReplayResult.context` 既无生产构造点写入，也无读取。测试里的 `context=` 构造的是 `Trajectory` / `RolloutBatch`，不是 `ReplayResult`。
- `RewardInferenceResult.raw_response` 没有生产写入，也没有读取；不具备 provenance-only 保留条件。

`TrainingBatch.adv_saturation` 已明确剔除：它是活字段，经 `vrl/scripts/common/online.py` 的微批加权聚合进入 optimizer update，不属于本 sprint。

## 1. 已删除内容

- `vrl/rollouts/orchestration/continuous/types.py`：删除 `ContinuousRolloutItem.submitted_at`。
- `vrl/rollouts/orchestration/continuous/producer.py`：删除 submit timestamp → `_collect_group` 参数 → result dict → item 构造的整条死链。
- `vrl/algorithms/types.py`：删除 `TrainStepMetrics.lr`。
- `vrl/models/interfaces/replay.py`：删除 `ReplayResult.context`。
- `vrl/rewards/inference.py`：删除 `RewardInferenceResult.raw_response`。
- tests：删除对应构造参数和死字段断言。

## 2. 验证

- 落地提交记录：相关回归 `502 passed`。
- Review 复核：`rg "\.submitted_at\b|\.raw_response\b|\.lr\b|ReplayResult\([^\\n]*context=|RewardInferenceResult\([^\\n]*raw_response=|TrainStepMetrics\([^\\n]*lr=" vrl tests -g '*.py'` 无目标字段生产使用；剩余 `lr` 命中属于 optimizer/config。
- Review 复核：`pytest tests/models/diffusion tests/generation/pipeline tests/generation/ar tests/rewards/inference tests/rollouts/replay tests/trainers/online/test_step_split.py tests/algorithms -q` → `289 passed`。
- Review 复核：`python -m vrl.config.lint` 通过；`git diff --check origin/main...HEAD` 通过。

## 3. Non-Goals

- 不删 `TrainingBatch.adv_saturation`：它是活的训练统计聚合字段。
- 不碰 optimizer learning rate 配置或日志，例如 `actor.optim.lr` / `cfg.lr`。
- 不删 `Trajectory.context` / `RolloutBatch.context` / reward result `metadata`；这些是不同 receiver 上的活 payload。
- 不把 `raw_response` 改名进 `metadata`；没有写入和消费时迁移位置没有价值。

## References

- `efb1924 refactor: drop dead replay/reward/trainer/rollout fields`
- `vrl/rollouts/orchestration/continuous/types.py`
- `vrl/rollouts/orchestration/continuous/producer.py`
- `vrl/algorithms/types.py`
- `vrl/models/interfaces/replay.py`
- `vrl/rewards/inference.py`
- `vrl/scripts/common/online.py`
- 关联：[[SPRINT_segment_signal_dead_field_cleanup]]
