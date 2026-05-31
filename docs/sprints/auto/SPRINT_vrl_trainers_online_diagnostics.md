# SPRINT(auto): vrl/trainers/online/diagnostics.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/trainers/online/diagnostics.py` (16 LOC)
角色判定: thin-wrapper
结论: question

## 0. 一句话
这是一个纯转发的 compat shim，把 `vrl.utils.model_diagnostics` 的四个符号原样 re-export；它没有提供边界价值，且与并行调用方的导入方式不一致。

## 1. 现状（读代码得出）
```python
"""Compatibility exports for trainer diagnostics."""
from vrl.utils.model_diagnostics import (
    parameter_state_summary, tensor_stats, trainable_state_digest, write_jsonl,
)
__all__ = [...]
```
四个符号全部来自 canonical 源 `vrl/utils/model_diagnostics.py`（确认存在，6.2KB）。

## 2. 质疑点 / 改进机会
- 纯转发，无任何转换/适配/lazy 边界，文件 docstring 自述 "Compatibility exports"——典型 thin wrapper。
- 消费方不一致：`trainer.py:31` 和 `tests/e2e/test_real_checkpoint_rl.py:32` 走这个 shim，但 `vrl/generation/execution/worker.py:99` 直接 `from vrl.utils.model_diagnostics import (...)`。同一份诊断函数有两条 import 路径，grepability 反而变差。
- 不确定点：docstring 写 "Compatibility" 暗示曾有历史路径迁移。如果确认没有外部包/旧 checkpoint 脚本依赖 `vrl.trainers.online.diagnostics` 这个路径，那它纯属冗余；如果有跨包稳定路径契约，则属于 keep-justified 的 facade。审计范围内未发现 vrl/ 之外的消费者，故判 question。

## 3. 建议动作
- 倾向删除：把 `trainer.py:31` 和 `tests/e2e/test_real_checkpoint_rl.py:32` 改成直接 `from vrl.utils.model_diagnostics import ...`，与 `worker.py` 统一，然后删除本文件。
- 删除前确认：`grep -rn "trainers.online.diagnostics" .`（当前只命中 trainer.py + 该 test），无 vrl/ 外部消费者即可删。

## 4. 不动什么 / 为什么不是过度清理
- 不动 `vrl/utils/model_diagnostics.py` 这个 canonical 源。
- 若后续发现这是被外部脚本/文档承诺的稳定导入路径，则保留并加注释说明它是 public path facade——此时按 AGENTS.md "consistency over cleanup" 判 keep-justified，不要误删。本 sprint 的前提是"未发现外部消费者"。

## 5. 验证
- `grep -rn "trainers.online.diagnostics" --include="*.py" .` 改完应为空。
- `pytest tests/e2e/test_real_checkpoint_rl.py::* -k digest` 与 `pytest tests/trainers/test_online.py` 通过。
