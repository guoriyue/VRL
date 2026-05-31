# SPRINT(auto): vrl/nn/layers/attention/__init__.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/nn/layers/attention/__init__.py` (30 LOC)
角色判定: package-init
结论: improve

## 0. 一句话
包门面 re-export 没有任何消费者（所有 import 都直接走子模块），且其中一项 `AttentionCacheView` 是死代码，应至少删除死项。

## 1. 现状（读代码得出）
文件把三个子模块的符号 re-export 到包顶层：

```python
# __init__.py:3-5
from vrl.nn.layers.attention.base import AttentionCacheView
from vrl.nn.layers.attention.joint import SD3JointAttentionProcessor
from vrl.nn.layers.attention.paged import (ARPagedAttentionBackend, ...)
```

## 2. 质疑点 / 改进机会
- 门面无人使用：grep `from vrl.nn.layers.attention import` / `import vrl.nn.layers.attention` 在仓内（排除本目录）0 命中。所有真实消费者都直接 import 子模块：
  - `vrl/models/diffusion/sd3_5/runner.py:13` -> `...attention.joint`
  - `vrl/nn/kernels/attention/vllm_paged.py:11`、`vrl/nn/modules/ar_decoder.py:13`、`vrl/models/ar/{janus_pro,nextstep_1}/runner.py` -> `...attention.paged`
  即这个 package facade 当前没提供实际边界价值。
- 其中 `AttentionCacheView`（line 3 / line 27）指向的 `base.py` 是死代码（见 `SPRINT_vrl_nn_layers_attention_base`），这一项必须删。

## 3. 建议动作
- 删除对 `AttentionCacheView` 的 import（line 3）与 `__all__` 条目（line 27），随 `base.py` 一并清理。
- `SD3JointAttentionProcessor` 与 `ARPaged*` 的 re-export 可保留作为包公共 API 门面（grepability / 未来稳定入口），属于可接受的 package-init facade；不强制删除。这是本文件唯一的"改"——把死项摘掉即可，不需要推倒门面。

## 4. 不动什么 / 为什么不是过度清理
- 保留 `joint` 与 `paged` 的 re-export：它们对应有真实消费者的符号，作为 `vrl.nn.layers.attention` 的稳定公共入口是合理的 facade（AGENTS.md 规则 2：public API facade 可保留）。即便当前各 caller 直连子模块，统一门面的 grepability 价值高于 LOC 缩减，不应误删。
- 不要把这个 `__init__` 整体清空——只摘除死掉的 `AttentionCacheView`。

## 5. 验证
- `grep -rn "AttentionCacheView" --include=*.py` 删除后 0 命中。
- `python -c "from vrl.nn.layers.attention import SD3JointAttentionProcessor, ARPagedAttentionBackend"` 仍可用。
- `ruff check vrl/nn/layers/attention/__init__.py` 无未用 import 警告。
