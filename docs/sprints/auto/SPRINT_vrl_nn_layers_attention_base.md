# SPRINT(auto): vrl/nn/layers/attention/base.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/nn/layers/attention/base.py` (20 LOC)
角色判定: dead
结论: delete

## 0. 一句话
整个文件只定义了一个 `AttentionCacheView` dataclass，全代码库无任何消费者，是死代码。

## 1. 现状（读代码得出）
文件全部内容就是一个 frozen dataclass：

```python
# base.py:11-17
@dataclass(frozen=True, slots=True)
class AttentionCacheView:
    """Backend-neutral view of cache metadata consumed by attention layers."""
    block_table: torch.Tensor | None = None
    slot_mapping: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

文档字符串声称它是 "consumed by attention layers"，但实际没有任何 attention layer 消费它。

## 2. 质疑点 / 改进机会
- 死代码（AGENTS.md 规则 6）。grep 全仓 `AttentionCacheView`，除自身定义外唯一出现是 `__init__.py` 的 re-export：
  - `vrl/nn/layers/attention/base.py:12` 定义
  - `vrl/nn/layers/attention/base.py:20` `__all__`
  - `vrl/nn/layers/attention/__init__.py:3` re-export
  - `vrl/nn/layers/attention/__init__.py:27` `__all__`
  没有任何 runner / kernel / module / test 引用它。
- paged-attention 的真实 cache 元数据走的是 `paged.py` 里的 `ARPagedAttentionStepInput.sequence_states` / `block_table` 等专用结构，`AttentionCacheView` 这个"通用视图"从未被接进任何路径，属于提前设计但未落地的抽象。

## 3. 建议动作
- 删除 `vrl/nn/layers/attention/base.py`。
- 同步删除 `__init__.py` 中对 `AttentionCacheView` 的 import（line 3）与 `__all__` 条目（line 27）。
- grep 证据：`grep -rn "AttentionCacheView" --include=*.py` 仅命中 base.py 与 __init__.py 自身，无外部 import/调用，删除安全。

## 4. 不动什么 / 为什么不是过度清理
- 同目录的 `paged.py`（AR paged-attention 契约）和 `joint.py`（SD3 processor）都有真实消费者，均保留。
- 这不是"为省几行拍平 thin function"——它是一个完全无人消费的预留抽象，删除符合 AGENTS.md 规则 6（死代码）。若将来确有跨 backend 的 cache view 需求，应在有第一个消费者时再引入。

## 5. 验证
- `grep -rn "AttentionCacheView" --include=*.py` 删除后应只剩 0 命中（或确认无剩余）。
- `python -c "import vrl.nn.layers.attention"` 不报错。
- `ruff check vrl/nn/layers/attention/` 通过（无未用 import）。
- 跑 `pytest tests/nn/layers/` 确认 attention 相关测试仍通过。
