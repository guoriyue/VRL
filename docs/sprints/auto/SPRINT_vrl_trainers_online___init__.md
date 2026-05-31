# SPRINT(auto): vrl/trainers/online/__init__.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/trainers/online/__init__.py` (17 LOC)
角色判定: package-init
结论: improve

## 0. 一句话
package init 把死代码 collection helper 和私有 `_validate_ema_state_shapes` 提升进了 public `__all__`，但外部只消费 `OnlineTrainer`——应收窄到只导出 `OnlineTrainer`。

## 1. 现状（读代码得出）
```python
from vrl.trainers.online.collection import (
    _collector_runtime_requires_driver_model_offload,
    _move_model_to_device,
    _release_collector_runtime_memory,
)
from vrl.trainers.online.trainer import OnlineTrainer, _validate_ema_state_shapes
__all__ = ["OnlineTrainer", "_collector_runtime_...", "_move_model_to_device",
           "_release_collector_runtime_memory", "_validate_ema_state_shapes"]
```

## 2. 质疑点 / 改进机会
- 实际通过 `from vrl.trainers.online import X` 消费的只有 `OnlineTrainer`（`vrl/trainers/__init__.py:10`、`vrl/scripts/common/online.py:43`、`tests/...`）。grep `from vrl.trainers.online import` 全树没有任何条目导入其余 4 个符号。
- 三个 collection 符号本身是死代码（见 SPRINT_vrl_trainers_online_collection），把死代码 re-export 进 `__all__` 让它们看起来像 public API，掩盖了"无人使用"的事实。
- `_validate_ema_state_shapes` 是 trainer.py 内部 helper，仅被 `trainer.load_state_dict` 调用（trainer.py:846）；它带下划线前缀却被列进 `__all__`，语义矛盾，且无外部消费者。

## 3. 建议动作
- 把整个文件收窄为：
  ```python
  """Online training loop package."""
  from vrl.trainers.online.trainer import OnlineTrainer
  __all__ = ["OnlineTrainer"]
  ```
- 删除对 `collection` 的 import（配合删除 collection.py）和对 `_validate_ema_state_shapes` 的 re-export。

## 4. 不动什么 / 为什么不是过度清理
- 保留 `OnlineTrainer` 的 re-export——这是真正的 package public facade，被 `vrl/trainers/__init__.py` 和脚本依赖。
- 不引入新抽象，只移除虚假 public surface。

## 5. 验证
- `grep -rn "from vrl.trainers.online import" --include="*.py" . | grep -vE "import OnlineTrainer"` 删除后应为空。
- `python -c "from vrl.trainers.online import OnlineTrainer"` 成功。
- `pytest tests/trainers/test_online.py` 通过。
