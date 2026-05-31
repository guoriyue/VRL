# SPRINT(auto): vrl/generation/execution/scheduler.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/generation/execution/scheduler.py` (118 LOC)
角色判定: core
结论: improve

## 0. 一句话
`DistributedExecutionPlanner` 本体是活的核心，但它对外暴露了一个无人调用的 `plan()` 薄 wrapper，以及 `DeviceAssignment.execution_unit` 这个无人读的别名属性，两者应删除。

## 1. 现状（读代码得出）
活路径：ray/executor.py:44 和 ray/launcher.py:130 只用 `plan_with_engine(...)`，消费 `generation_plan.assignments` 和 `.engine_plan`（ray/executor.py:49-50）。

存在两处冗余转发：
```python
def plan(self, request, workers) -> list[DeviceAssignment]:   # scheduler.py:48
    return list(self.plan_with_engine(request, workers).assignments)
```
```python
@property
def execution_unit(self) -> ExecutionStage | None:            # scheduler.py:29
    return self.execution_stage
```

## 2. 质疑点 / 改进机会
- 薄 wrapper 死代码：`DistributedExecutionPlanner.plan()` 只是 `plan_with_engine(...).assignments` 的丢信息版本（丢掉了 engine_plan）。`grep -rn "planner\.plan(" . --include=*.py` 无命中；其它 `.plan(` 命中都是 diffusion/executor 和 janus_pro runtime 上的同名但无关方法，不是这里的 `DistributedExecutionPlanner.plan`。它没有提供任何边界价值（不是协议方法、不是跨家族统一形状），属于纯转发。
- 别名属性死代码：`DeviceAssignment.execution_unit`（scheduler.py:29-31）只是返回 `self.execution_stage`。`grep -rn execution_unit . --include=*.py` 只命中定义本身，无任何读取方。这是命名遗留（旧叫 execution_unit，现统一为 execution_stage），别名没人用。

## 3. 建议动作
- 删除 `DistributedExecutionPlanner.plan()`（scheduler.py:48-53）。
- 删除 `DeviceAssignment.execution_unit` 属性（scheduler.py:29-31）。
- 两处删除均有 grep 佐证无调用方/读取方。

## 4. 不动什么 / 为什么不是过度清理
- `plan_with_engine`、`DistributedGenerationPlan`、`DistributedExecutionPlanner.__init__` 全保留，是活路径。
- `DeviceAssignment` 的 `execution_stage` 字段保留（被 `ChunkExecutionEnvelope` 构造和 worker 链路使用）。
- 这不是为省 LOC 拍平统一形状：`execution_unit` 不是跨家族一致命名（家族里统一叫 execution_stage），删别名反而提升 grepability。

## 5. 验证
- 删除后 `grep -rn "execution_unit\|\.plan(" vrl/generation/` 中应不再出现 scheduler 的这两个符号。
- 跑 `pytest tests/generation/execution -q` 和 ray 相关测试（若有 `tests/generation` 下覆盖 launcher/executor 的用例）。
- `ruff check vrl/generation/execution/scheduler.py`。
