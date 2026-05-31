# SPRINT(auto): vrl/trainers/fsdp.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/trainers/fsdp.py` (184 LOC)
角色判定: dead
结论: delete

## 0. 一句话
整个文件（FSDPConfig / fsdp_wrapper / save_fsdp_checkpoint / OptimizerOffloadHook / register_optimizer_offload_hooks / init_distributed）从 flow_grpo 移植后，repo 内没有任何地方 import 或调用它，是死代码。

## 1. 现状（读代码得出）
文件头自述是移植品：

```python
"""FSDP utilities for distributed training.

Ported from flow_grpo/fsdp_utils.py.  Provides a config + wrapper for
PyTorch FSDP, activation checkpointing, and optimizer CPU offloading.
"""
```

提供 6 个对外符号：`FSDPConfig`（dataclass）、`fsdp_wrapper`、`save_fsdp_checkpoint`、`OptimizerOffloadHook`、`register_optimizer_offload_hooks`、`init_distributed`。文件没有 `__all__`，也没有被 `vrl/trainers/__init__.py` re-export。

## 2. 质疑点 / 改进机会
- 全仓零引用。对每个符号 grep（含 tests/docs/scripts，排除文件自身）结果为空：

  ```
  grep -rn "fsdp_wrapper|FSDPConfig|save_fsdp_checkpoint|OptimizerOffloadHook|register_optimizer_offload_hooks|init_distributed|trainers.fsdp|from vrl.trainers.fsdp" --include=*.py .
  # 仅匹配 vrl/trainers/fsdp.py 本身，无其它命中
  ```
- `vrl/trainers/__init__.py:1-36` 的 `__all__` 不含任何 FSDP 符号，说明它连 package facade 都没接入。
- 当前训练实际走的是 Ray + `RuntimeBundle.trainable_modules` + `weight_sync.py` 的 LoRA 路径（见 `vrl/scripts/common/online.py`、`vrl/trainers/online/trainer.py`），与这套 FSDP full-shard 工具栈无关。这是移植时一并搬进来、但 RL post-training 走 LoRA 路线后未再使用的遗留。
- 它不是「保留以备将来」的合理 stub：没有测试、没有文档引用、没有 entrypoint，符合 AGENTS.md「死代码 -> delete」。

## 3. 建议动作
删除整个 `vrl/trainers/fsdp.py`。

删除前的引用确认（已执行）：
- `grep -rn "trainers.fsdp" --include=*.py .` → 无命中
- 6 个符号逐个 grep（含 tests/docs）→ 无命中
- `grep -rn "fsdp" vrl/trainers/__init__.py` → 无命中

若团队确实计划未来引入 full-shard 训练，正确做法是删除现状、需要时从 flow_grpo 重新移植并立即接线 + 加测试，而不是把未接线的移植件长期挂在 import graph 边缘腐烂。

## 4. 不动什么 / 为什么不是过度清理
本 sprint 只针对 `fsdp.py`。同目录的 `weight_sync.py`、`checkpointing.py`、`frozen_module.py`、`precision.py` 都有真实 import 与测试，均不动。删除 FSDP 不影响现有 LoRA + Ray 训练路径（它们从不引用本文件）。

## 5. 验证
- 删除后 `grep -rn "trainers.fsdp\|from vrl.trainers.fsdp" --include=*.py .` 应为空。
- 跑 `pytest tests/trainers/ -q` 应全绿（这些测试从不 import fsdp）。
- `ruff check vrl/trainers/` 无新增 unused-import（无人 import 它，故无连带）。
