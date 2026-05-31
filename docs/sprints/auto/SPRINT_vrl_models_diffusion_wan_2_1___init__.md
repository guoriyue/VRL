# SPRINT(auto): vrl/models/diffusion/wan_2_1/__init__.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/models/diffusion/wan_2_1/__init__.py` (0 LOC)
角色判定: package-init
结论: improve

## 0. 一句话
这是整个 diffusion family 目录里唯一一个 0 字节的 `__init__.py`，连家族描述 docstring 都没有，破坏了跨家族一致性，建议补一行家族 docstring（不需要补 re-export）。

## 1. 现状（读代码得出）
文件完全为空：

```
$ wc -c vrl/models/diffusion/wan_2_1/__init__.py
0 vrl/models/diffusion/wan_2_1/__init__.py
```

对比同级家族：
- `sd3_5/__init__.py` — 1 行 docstring：`"""SD3 family: Stable Diffusion 3.5-Medium image model for Flow-GRPO RL training."""`
- `cosmos/predict2/__init__.py` — docstring + 显式 re-export（`CosmosPredict2Model` / `CosmosPipelineExecutor` / builders）
- `cosmos/predict2_5`、`cosmos/anima` — 同样是 docstring + re-export
- `wan_2_1/__init__.py` — 0 字节，什么都没有

## 2. 质疑点 / 改进机会
- 一致性缺口（非功能性 bug）：`wan_2_1` 的 family 符号全部通过 fully-qualified 字符串路径被引用，因此空 init 在运行时没问题：
  - `vrl/rollouts/families/registry.py:149-151` 用 `vrl.models.diffusion.wan_2_1.runtime:Wan_2_1PipelineExecutor` 等字符串路径
  - 测试与脚本均 `from vrl.models.diffusion.wan_2_1.model import ...` / `.runtime import ...` 直接打到子模块
  所以这里 **不需要** 像 cosmos 那样补 re-export。
- 但每个兄弟家族 init 至少有一行"这是哪个 model family"的 docstring，唯独 wan 没有。grep/阅读目录时缺这一行会让人误以为文件被截断或漏写。属于 AGENTS.md "跨家族一致性"语境下的轻微 flag。

## 3. 建议动作
补一行与 sd3_5 风格一致的家族 docstring，例如：

```python
"""Wan family: Wan 2.1 1.3B text-to-video diffusion model for RL post-training."""
```

不要补 re-export。本仓库 wan 的引用方均走完整子模块路径，硬加 re-export 反而会引入 eager import diffusers 的副作用（runtime.py 刻意把 backend import 放进 `from_spec` 内部以避免 eager 加载），与现有 lazy-import 设计冲突。

## 4. 不动什么 / 为什么不是过度清理
- 不引入 re-export：见上，会破坏 lazy backend import 边界。
- 不动 registry 的字符串路径写法：那是刻意的 lazy registration 边界。
- 这是纯一致性补全，不是 LOC 增减博弈；改动仅一行 docstring。

## 5. 验证
- `python -c "import vrl.models.diffusion.wan_2_1"` 仍能 import 且不触发 diffusers 加载。
- `ruff check vrl/models/diffusion/wan_2_1/__init__.py`
- grep 确认无任何 `from vrl.models.diffusion.wan_2_1 import X`（即依赖 init re-export）的用法：当前为 0，故只补 docstring 安全。
