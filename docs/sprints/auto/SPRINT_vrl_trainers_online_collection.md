# SPRINT(auto): vrl/trainers/online/collection.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/trainers/online/collection.py` (37 LOC)
角色判定: dead
结论: delete

## 0. 一句话
这个文件的三个 helper 全是死代码——没有任何模块导入它们，而它们实现的逻辑已经在 `vrl/rollouts/orchestration/lifecycle.py` 里被重新实现。

## 1. 现状（读代码得出）
文件导出三个 collector/runtime helper：

```python
async def _release_collector_runtime_memory(collector: Any) -> None: ...
def _collector_runtime_requires_driver_model_offload(collector: Any) -> bool:
    runtime = collector.runtime
    return bool(getattr(runtime, "requires_driver_model_offload", False))
def _move_model_to_device(model, device) -> None: ...
```

## 2. 质疑点 / 改进机会
- 全树 grep 三个符号的唯一引用是本文件 + `vrl/trainers/online/__init__.py:4-6,12-14` 的 re-export，没有任何业务模块/测试 import 它们（`grep -rn` 全树结果只有 collection.py 自己和 __init__.py）。
- `OnlineTrainer` 现在把 collect/offload/release 全部委托给 `self.rollout_schedule`（trainer.py:190-199, 295）；trainer.py 里没有任何对 collection 符号的引用（grep 确认 `collection` 只命中标准库 `from collections import defaultdict`）。
- 同样的判定逻辑已经在 `vrl/rollouts/orchestration/lifecycle.py:75-77` 里近乎逐字重写：
  ```python
  def requires_driver_model_offload(self) -> bool:
      return bool(getattr(runtime, "requires_driver_model_offload", False))
  ```
  说明 collection.py 是迁移到 rollout orchestration 之后遗留的孤儿文件。

## 3. 建议动作
- 删除 `vrl/trainers/online/collection.py` 整个文件。
- 同步从 `vrl/trainers/online/__init__.py` 删除对应的 3 行 import 和 3 个 `__all__` 条目（见该文件的独立 sprint）。
- grep 证据：`grep -rn "_release_collector_runtime_memory\|_collector_runtime_requires_driver_model_offload\|_move_model_to_device" --include="*.py" .` 只命中 collection.py 与 online/__init__.py，无外部消费者。

## 4. 不动什么 / 为什么不是过度清理
- `lifecycle.py` 里的 `requires_driver_model_offload` 是现役实现，不动。
- 这不是"为省几行拍平 thin function"——是删除已无调用方、且逻辑已被别处取代的死代码，符合 AGENTS.md 死代码规则。

## 5. 验证
- `grep -rn "online.collection\|online import.*collector\|_move_model_to_device" --include="*.py" .` 删除后应为空。
- `ruff check vrl/trainers/online/` 无未使用 import 报错。
- `pytest tests/trainers/test_online.py` 仍通过（这些 helper 本就不在测试路径上）。
