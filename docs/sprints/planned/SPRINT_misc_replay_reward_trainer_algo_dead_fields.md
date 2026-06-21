# SPRINT: 跨子系统散点死字段收口（planned）

状态：未开始（2026-06-20）。
范围：4 个分散在 models/interfaces、rollouts/orchestration、rewards、algorithms 的孤立死/display 字段——各自所在 struct 无兄弟死字段，不够单独成 sprint，合并收口。
来源：dead-dataclass-hunt + 手动验证（已剔除一个被误判的活字段，见 §0）。承接 [[SPRINT_segment_signal_dead_field_cleanup]]。

## 0. Core Decision + 一处自动审计修正

四个字段分两档：机械删（`submitted_at`、`lr`）、display-only 需签字（`ReplayResult.context`、`RewardInferenceResult.raw_response`）。

> ⚠️ **已剔除 `TrainingBatch.adv_saturation`**：自动审计把它列为 display-only，但手动验证证伪——`vrl/scripts/common/online.py:352` `adv_sat_w += batch.adv_saturation * weight` 是微批**加权聚合**（与 `group_size`/`trained_prompt_num`/`pre_filter_*` 等已知活字段同模式），结果传给 `finish_optimizer_update`。`adv_saturation` 是**活字段**，不在本 sprint。这条修正同时提醒：`TrainingBatch` 的统计字段几乎都走 `online.py:348-355` 聚合块，删任何一个前必须对照该块。

## 1. 现状实锤

### 1.1 机械删
- **`ContinuousRolloutItem.submitted_at`**（`vrl/rollouts/orchestration/continuous/types.py:60`）：`producer.py:307` 构造写入，`grep -rEn "\.submitted_at\b" vrl/` 全仓**仅**定义+构造，零读取。
- **`TrainStepMetrics.lr`**（`vrl/algorithms/types.py:38`，默认 `0.0`）：所有算法构造 `TrainStepMetrics` 均省略它（`grpo/token.py:122`、`continuous.py:172`、`multisegment.py:130`、`diffusion_nft.py:282`）；trainer 聚合（`trainer.py:1228`）、日志（`online.py:435-468` + CSV header）、测试（`test_step_split.py:96` 列 10 字段）全不含 `lr`。从不赋值、从不读、从不记。

### 1.2 display-only（需签字）
- **`ReplayResult.context`**（`vrl/models/interfaces/replay.py:82`）：所有构造省略（默认空 dict），唯一「读」经 `ordered_ar_chunks[0].context` 之类间接路径——确认无真实消费后删，或标 provenance-only。
- **`RewardInferenceResult.raw_response`**（`vrl/rewards/inference.py:164`）：仅经 `asdict()` 序列化进 `base.py:339` 的 JSON 日志。无控制流读。删 or 标 provenance-only（若调试需要原始响应留存）。

## 2. 落地方案

### A. 删 `submitted_at` / `lr`（机械）
- 删 `continuous/types.py:60` + `producer.py:307` 构造传参。
- 删 `algorithms/types.py:38` 的 `lr` 字段。

### B. `context` / `raw_response`（需签字）
- 默认删字段 + 构造点；若团队要留作调试 provenance，改标注，不留无消费者裸字段。
- 删 `raw_response` 前确认 `base.py:339` 的 JSON 日志不被外部工具按该 key 解析。

## 3. 验证
- `grep -rEn "\.(submitted_at|lr)\b" vrl/`：`submitted_at` 零命中；`lr` 仅剩无关命中（如 optimizer lr，不同语义）。
- 按 B 决策后 `grep -rEn "\.(context|raw_response)\b"` 在对应 struct 上收敛。
- `pytest tests/ -q` 全绿；**重点回归**：删 `submitted_at`/`lr` 不影响 `online.py` 聚合（这两个不在聚合块内）。

## 4. 非目标 / Non-Goals
- **不删 `TrainingBatch.adv_saturation`**（活，`online.py:352` 聚合）及 `TrainingBatch` 其余聚合字段（`group_size`/`trained_prompt_num`/`adv_zero_rate`/`pre_filter_reward_mean`/`pre_filter_reward_std`/`pre_filter_adv_mean`，全 OVERTURNED）。
- `lr` 仅指 `TrainStepMetrics.lr`，不碰 optimizer 的 learning rate。

## References
- `vrl/rollouts/orchestration/continuous/types.py:60`、`vrl/rollouts/orchestration/continuous/producer.py:307`
- `vrl/algorithms/types.py:38`、`vrl/trainers/online/trainer.py:1228`、`vrl/scripts/common/online.py:348-355,435-468`
- `vrl/models/interfaces/replay.py:82`、`vrl/rewards/inference.py:164`、`vrl/rewards/base.py:339`
- 关联：[[SPRINT_segment_signal_dead_field_cleanup]]
