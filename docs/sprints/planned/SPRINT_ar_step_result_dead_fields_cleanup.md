# SPRINT: `ARStepResult` 死字段清理 + 传递链评估（planned）

状态：未开始（2026-06-20）。
范围：清理 `ARStepResult` 上「两个 runner 构造、decode_loop 只读 `debug_counters`」的死字段，并评估随之失活的 `ARStepBatch.sequences → sequence_ids property → ARStepResult` 传递链。
来源：dead-dataclass-hunt confirmed（4 字段 dead + 1 display-only），但含一条**传递链**，落地非纯机械。承接 [[SPRINT_segment_signal_dead_field_cleanup]]。

## 0. Core Decision

`ARStepResult` 定义（`vrl/generation/ar/decode_loop.py:160-167`）：

```python
class ARStepResult:
    """One scheduled AR token step."""
    sequence_ids: list[str]
    positions: list[int]
    token: Any
    log_prob: Any
    replay_extras: dict[str, Any] = field(default_factory=dict)
    debug_counters: dict[str, Any] = field(default_factory=dict)
```

两个 runner 完整构造它（`vrl/models/ar/nextstep_1/runner.py:157-161`、`vrl/models/ar/janus_pro/runner.py:138-141`），但 decode_loop 消费侧**只读 `debug_counters`**（`decode_loop.py:367` 经 `getattr(result, "debug_counters")`）。`sequence_ids` / `positions` / `token` / `log_prob` 构造后零读取；`replay_extras` 仅构造拷贝，无控制流。

## 1. 现状实锤

### 1.1 四个死字段
`grep -rEn "result\.(sequence_ids|positions|token|log_prob)\b"` 在 `decode_loop.py` 及消费侧零命中。runner 里出现的 `batch.sequence_ids` / `batch.positions`（`nextstep_1/runner.py:157-158,198`）是读 **`ARStepBatch`**（另一 struct）的字段来**构造** `ARStepResult`，不是读 `ARStepResult` 自身。

### 1.2 传递链（落地前必须评估）
验证阶段把 `ARStepBatch.sequences` 判为**活**（理由：`decode_loop.py:194-195` 的 `sequence_ids` property 读 `self.sequences`，property 又在 runner 构造 `ARStepResult.sequence_ids` 时被调用）。但这条「活」是**以 `ARStepResult.sequence_ids` 为终点**的：

```
ARStepBatch.sequences → property sequence_ids (decode_loop.py:194-195) → ARStepResult.sequence_ids (runner.py:157)
```

若 `ARStepResult.sequence_ids` 确为死字段（§1.1），则整条链是 transitive-dead —— `sequences` 的「活」是假阳性。**这是本 sprint 的核心判断点**，不能机械删一端、留另一端。

## 2. 落地方案

分两步，第二步需在第一步确证后进行：

### A. 删 `ARStepResult` 四死字段 + replay_extras（机械）
- 删 `decode_loop.py:163-166` 的 `sequence_ids` / `positions` / `token` / `log_prob`、`:167` 的 `replay_extras`。
- 删两 runner 构造传参：`nextstep_1/runner.py:157-161`、`janus_pro/runner.py:138-141`。
- 保留 `debug_counters`（唯一活字段，`decode_loop.py:367` 读）。

### B. 评估并清理传递链（需确证）
- 删 `ARStepResult.sequence_ids` 后，`decode_loop.py:194-195` 的 `sequence_ids` property 失去唯一消费者 → 评估是否连带删 property。
- property 删除后，`ARStepBatch.sequences` 是否还有别的读者？若无 → `sequences` 也成死字段，连带清理。grep `\.sequences\b`（receiver=ARStepBatch）确认。
- **若发现 `sequences`/property 另有活读者，则停在 A，不动链** —— 把发现记进本 doc。

## 3. 验证

- `grep -rEn "result\.(sequence_ids|positions|token|log_prob|replay_extras)\b" vrl/` 零命中。
- `pytest tests/generation/ tests/models/ar/ -q` 全绿。
- AR rollout 冒烟（janus_pro / nextstep_1 各一）token 序列与删前一致。

## 4. 非目标 / Non-Goals
- 不动 `debug_counters`（活）。
- 不在未确证传递链终点死活前删 `ARStepBatch.sequences` / `sequence_ids` property —— 避免误删假阳性「活」字段。

## References
- `vrl/generation/ar/decode_loop.py:160-167,194-195,366-369`
- `vrl/models/ar/nextstep_1/runner.py:157-161,198`
- `vrl/models/ar/janus_pro/runner.py:138-141`
- 关联：[[SPRINT_segment_signal_dead_field_cleanup]]
