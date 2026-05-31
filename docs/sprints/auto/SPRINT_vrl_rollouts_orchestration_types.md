# SPRINT(auto): vrl/rollouts/orchestration/types.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/rollouts/orchestration/types.py` (105 LOC)
角色判定: interface-boundary
结论: improve

## 0. 一句话
文件整体是合理的共享类型边界，但 `RolloutScheduleState` 上有两个死字段（`current_policy_version`、`pending_rollout`），应删掉以免误导后来者以为 state 里存了策略版本/挂起 rollout。

## 1. 现状（读代码得出）
这是 orchestration 包的共享类型层，被 `strict_on_policy.py`、`continuous/schedule.py`、`continuous/consumer.py`、`continuous/producer.py` 共同依赖。包含：

- `RolloutScheduleMode` enum（strict_on_policy / continuous）— 真协议边界，被 `build_rollout_schedule` 用于选 schedule。
- `RolloutIteration` dataclass + `build_rollout_iteration` 工厂 + `annotate_batch_context` — 跨 strict/continuous 两条管线统一的 trainer 交付契约，有 `__post_init__` 校验，是真共享抽象。
- `RolloutScheduleState` dataclass：

```python
@dataclass(slots=True)
class RolloutScheduleState:
    rollout_id: int = 0
    current_policy_version: int | None = None   # line 25
    initialized: bool = False
    pending_rollout: Any | None = None           # line 27
```

## 2. 质疑点 / 改进机会
死字段（grep 全仓确认）：

- `RolloutScheduleState.current_policy_version`（types.py:25）：全仓无任何 `state.current_policy_version` 的读或写。策略版本实际从 `lifecycle.current_policy_version()` 取（strict_on_policy.py:38、continuous/schedule.py:92、continuous/producer.py:166），即从 runtime 查，不经过这个字段。该字段与 `RolloutIteration.policy_version` / lifecycle 的取数路径概念重叠，留着会让人误以为 state 缓存了版本号。
- `RolloutScheduleState.pending_rollout`（types.py:27）：全仓只在两处 `reset()` 里被赋值为 `None`（strict_on_policy.py:73、continuous/schedule.py:130），从无读取、从无赋非 None 值。是纯死字段——名字暗示"挂起的 rollout"语义，但根本没人往里放东西。

证据（grep `state.current_policy_version` / `pending_rollout` 非 `= None` 的结果均为空）：无外部读，无非 None 写。

实际被使用的字段只有 `rollout_id`（strict/continuous 自增）和 `initialized`（被 lifecycle 的 `weights_initialized` 回调间接体现 / reset 复位）。

## 3. 建议动作
- 删除 `RolloutScheduleState.current_policy_version`（types.py:25）和 `pending_rollout`（types.py:27）两个字段。
- 同步删掉两处 `reset()` 里的 `self.state.pending_rollout = None`（strict_on_policy.py:73、continuous/schedule.py:130）。
- `RolloutScheduleState` 瘦身后剩 `rollout_id` + `initialized`，仍是合理的 per-schedule 可变状态容器，保留。
- 不要把 enum / `RolloutIteration` / `build_rollout_iteration` / `annotate_batch_context` 动掉。

## 4. 不动什么 / 为什么不是过度清理
- `RolloutScheduleMode` 是真协议名 taxonomy，符合 AGENTS.md "协议名" 例外，保留。
- `build_rollout_iteration` + `annotate_batch_context` 是 strict 与 continuous 两家族共用的统一交付形状（grepability / 一致性），即使看起来薄也属 keep-justified，不要拍平进各自 schedule。
- `RolloutIteration.__post_init__` 校验是真不变量，保留。
- 这里只删确证无引用的死字段，不是为省 LOC 拍结构。

## 5. 验证
- grep 确认零引用后删除：`grep -rn "current_policy_version" vrl/rollouts/orchestration/`（应只剩 lifecycle/runtime 的方法调用，无 state 字段访问）、`grep -rn "pending_rollout" vrl tests`（删后应无命中）。
- 跑 `pytest tests/rollouts/test_orchestration.py tests/rollouts/orchestration/continuous/test_schedule.py`。
- `ruff check vrl/rollouts/orchestration/`。
