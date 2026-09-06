# SPRINT: `vrl/trajectory/` views/types 死字段清理（done）

状态：done（2026-06-20）。逐字段 receiver 消歧后，删除 **6 个**死字段（不止预估的 2 个），保留 2 个被
`_reject_runtime_state` 不变量消费的 metadata。验证：`ruff` 全绿，`tests/trajectory/ tests/rollouts/ tests/models/ar/janus_pro/` **161 passed**。
范围：`trajectory/views.py` + `types.py` 上的死/display 字段，经全仓 receiver 消歧确认。
来源：dead-dataclass-hunt（report 多条）。trajectory 层 `metadata`/`modality`/`*_refs` 字段名在
`RewardView`/`TrainingView`/`LossUnit`/`TrajectorySegment`/`TrajectoryTensor`/`TrajectoryAxis` 间大量复用，
自动审计 grep 被严重污染——本 sprint 已逐字段手动消歧。承接 [[SPRINT_segment_signal_dead_field_cleanup]]。

## 0. Core Decision

这是误删高风险区：trajectory 多个 struct 共享同名字段，`grep ".metadata"` / `".modality"` 命中混杂。
落地规则：只删 **receiver 消歧后零行为消费** 的字段；被「能 raise 的校验」（`_reject_runtime_state`）消费的字段
按 AGENTS.md「a validation that can raise 是合法 consumer」一律保留——这点**推翻**了原计划把
`ReplayInput.metadata`/`TrajectoryTensor.metadata` 当 display-only 删除候选的判断（见 §1.4）。

## 1. 落地结果（实锤）

### 1.1 机械删（receiver 消歧后全仓零命中）— 已删
- **`ReplayInput.signal_kind`**（`types.py:88`）：`grep -rEn "\.signal_kind\b" vrl/ tests/` 全仓零命中。
  4 处 `ReplayInput` 构造全用默认值。→ 已删，连带孤儿别名 `ReplaySignalKind`（`types.py:46` + `__all__` + `__init__.py` 再导出）。
- **`TrainingView.algorithm_family`**（`views.py:80`）：`grep -rEn "\.algorithm_family\b"` 全仓零命中。
  `validate_training_view` 只校验 `loss_units`/`primary_segment`。→ 已删，连带孤儿别名 `AlgorithmFamily`。

### 1.2 逐字段消歧 → 确认死 — 已删（原计划标「需消歧」，消歧后实锤为死）
- **`RewardView.modality`**（`views.py:26`）：全部 `.modality` reader 的 receiver 都是 `TrajectorySegment`
  （`ops.py:138`、`janus_pro/model.py:491-492`、`multi_segment_token_logprob.py:160-163`）。
  `validate_reward_view`（`validation.py:124-132`）**不读** `view.modality`，`RewardView.__post_init__` 也不校验它。
  → RewardView receiver 零读取，已删；连带孤儿别名 `RewardModality`；5 处 RewardView 构造的 `modality=` kwarg 一并清掉
  （`builders.py` ×4 + `tests/rollouts/collector/test_runtime.py` ×1）。
- **`RewardView.prompt_refs` / `target_refs`**（`views.py:28,29`）：唯一「reader」是各自 `__post_init__` 里的
  `validate_string_tuple` 自校验；全仓无任何构造传入它们（始终默认 `()`），无下游消费者。→ 自指型死字段，已删（连带自校验行）。
- **`LossUnit.metadata`**（`views.py:59`）：全仓零 reader，`validate_loss_unit` 也**不**校验它，构造（`views.py:110`）从不传值。
  → 真死，已删。

### 1.3 勿删——已确认活（保留）
- **`RewardView.tensor_refs` / `name` / `metadata`**：`validate_reward_view` 真读 `tensor_refs`/`name`；
  `metadata` 除 `_reject_runtime_state` 外还有功能性 reader `batch_builder.py:244` `view.metadata.get("output_ref")`（控制流）。
- **`TrainingView.metadata`**：`validation.py:145` `_reject_runtime_state(view.metadata)` 真校验。
- **`TrajectoryAxis.metadata`**：`ops.py:246` frozen dataclass `__eq__` 全字段比较（含 metadata）驱动控制流。
- **`LossUnit.replay_input_refs`**：`validation.py:181` `for replay_ref in unit.replay_input_refs` 控制流。
- **`TrajectorySegment.modality` / `metadata`**：`janus_pro`/`multi_segment` 的 `.get(...)` 功能性 reader。

### 1.4 推翻原计划：保留（原 §1.2 标为 display-only 删除候选）
- **`ReplayInput.metadata`**（`types.py:89`）、**`TrajectoryTensor.metadata`**（`types.py:73`）：被
  `_reject_runtime_state`（`validation.py:265,285`）消费——这是「能 raise 的不变量校验」，按 AGENTS.md 即合法 consumer，
  与 §1.3 保留 `TrainingView.metadata` 同理，**不能区别对待**。且 `TrajectoryTensor.metadata` 有真实 producer 写入
  （`collector/artifacts.py:93`）。→ 二者均**保留**，未删。

## 2. 验证（实锤）
- `grep -rEn "ReplaySignalKind|AlgorithmFamily|RewardModality|signal_kind|algorithm_family|prompt_refs|target_refs" vrl/ tests/ configs/` 零残留。
- `uv run ruff check vrl/trajectory/ …` → All checks passed。
- `uv run pytest tests/trajectory/ tests/rollouts/ tests/models/ar/janus_pro/ -q` → **161 passed**。
- `map_tensor_tree`（`device.py:33`）按 `dataclasses.fields` 泛型重建，删字段不改其行为。

## 3. 改动文件
- `vrl/trajectory/views.py`：删 `RewardModality`/`AlgorithmFamily` 别名、`RewardView.modality/prompt_refs/target_refs`、
  `LossUnit.metadata`、`TrainingView.algorithm_family` + 对应 `__post_init__` 校验 + `__all__`。
- `vrl/trajectory/types.py`：删 `ReplaySignalKind` 别名、`ReplayInput.signal_kind` + `__all__`。
- `vrl/trajectory/__init__.py`：删 3 个孤儿别名的 import + `__all__` 再导出。
- `vrl/trajectory/builders.py`：4 处 RewardView 构造去掉 `modality=`。
- `tests/rollouts/collector/test_runtime.py`：1 处 RewardView 构造去掉 `modality=`。

## 4. Non-Goals（遵守）
- 未按自动审计 grep 机械删 §1.4 metadata——同名碰撞 + 不变量消费使其为活字段。
- 未删 §1.3 活字段。

## References
- `vrl/trajectory/types.py`、`views.py`、`__init__.py`、`builders.py`
- `vrl/trajectory/validation.py:124-132,145,181,265,285`、`ops.py:138,246`、`device.py:33`
- `vrl/rollouts/collector/batch_builder.py:244`、`collector/artifacts.py:93`
- 关联：[[SPRINT_segment_signal_dead_field_cleanup]]
