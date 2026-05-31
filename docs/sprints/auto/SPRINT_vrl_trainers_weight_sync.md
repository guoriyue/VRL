# SPRINT(auto): vrl/trainers/weight_sync.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/trainers/weight_sync.py` (184 LOC)
角色判定: interface-boundary
结论: improve

## 0. 一句话
文件整体是合理的 trainer↔rollout 权重同步边界，但 `InMemoryWeightSyncer` 是一个全仓只被 `__init__` re-export、从未被实例化的死类，应删除。

## 1. 现状（读代码得出）
`WeightSyncer(ABC)` 定义 `push`/`pull` 协议，`RayRuntimeWeightSyncer` 是生产实现，`build_runtime_weight_syncer` / `build_trainable_state_sync_getter` / `flatten_trainable_module_state` 是工厂与状态扁平化逻辑——这些都被 `vrl/scripts/common/online.py:45-47` 与 `vrl/trainers/online/trainer.py:39` 真实使用，且有 `tests/trainers/test_weight_sync.py` 覆盖。

`InMemoryWeightSyncer` 声称用于单卡开发：

```python
class InMemoryWeightSyncer(WeightSyncer):
    """Simple in-process syncer for single-GPU development."""
    async def push(self, state_dict): self._state = dict(state_dict)
    async def pull(self): return dict(self._state)
```

## 2. 质疑点 / 改进机会
- `InMemoryWeightSyncer` 全仓无实例化点。grep 结果只有 3 处，全是定义/导出，无任何调用方：

  ```
  grep -rn "InMemoryWeightSyncer" . --include=*.py
  vrl/trainers/__init__.py:13:    InMemoryWeightSyncer,      # import
  vrl/trainers/__init__.py:24:    "InMemoryWeightSyncer",     # __all__
  vrl/trainers/weight_sync.py:26:class InMemoryWeightSyncer    # 定义
  ```
  连测试都不 new 它。这是「export 给一个不存在的用户」的死代码 stub。
- 单卡开发路径实际不需要 syncer：`build_runtime_weight_syncer`（line 69）在 runtime 不支持 `update_weights` 或 `weight_sync` 为 None 时直接返回 `None`，下游 `lifecycle.py` 对 `weight_syncer is None` 已有完整短路（`vrl/rollouts/orchestration/lifecycle.py:49,61`）。所以「in-process 占位 syncer」是多余的概念。

其余符号均为 justified 边界，不质疑。

## 3. 建议动作
- 删除 `InMemoryWeightSyncer` 类（line 26-36）。
- 同步从 `vrl/trainers/__init__.py` 的 import（line 13）和 `__all__`（line 24）移除 `InMemoryWeightSyncer`。
- grep 确认无引用结果见上。

## 4. 不动什么 / 为什么不是过度清理
保留 `WeightSyncer` ABC、`RayRuntimeWeightSyncer`、`build_runtime_weight_syncer`、`build_trainable_state_sync_getter`、`flatten_trainable_module_state` 以及私有 helper（`_resolve_next_policy_version` / `_require_trainable_modules` / `_trainable_parameter_names` / `_cpu_state_dict` / `_to_cpu`）。它们是真实使用且测试覆盖的同步边界，符合 AGENTS.md 的 interface-boundary 保留条件。本 sprint 只删一个无人用的占位类，不动协议形状。

## 5. 验证
- 删除后 `grep -rn "InMemoryWeightSyncer" --include=*.py .` 应为空。
- `pytest tests/trainers/test_weight_sync.py -q` 全绿。
- `ruff check vrl/trainers/__init__.py vrl/trainers/weight_sync.py` 无 unused-import / undefined-export。
