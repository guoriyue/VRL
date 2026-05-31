# SPRINT(auto): vrl/nn/kernels/attention/torch.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/nn/kernels/attention/torch.py` (45 LOC)
角色判定: dead
结论: delete

## 0. 一句话
`TorchAttentionKernel` 是死代码：除了被 `__init__.py` re-export 之外，全仓库没有任何地方实例化或调用它，docstring 声称的 "parity and debug paths" 用途从未落地。

## 1. 现状（读代码得出）
文件只定义一个类，内部其实也是直接转发给 `F.scaled_dot_product_attention`：

```python
class TorchAttentionKernel:
    """Small scaled dot-product attention reference implementation."""
    def __call__(self, query, key, value, *, causal=True, scale=None):
        ...
        return F.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=scale)
```
（torch.py:11-41）

它和 `sdpa.py:TorchSDPAAttentionKernel` 高度重叠，都是 SDPA 的薄封装，唯一区别：
- 强制 3D `[B, T, H]` 输入（torch.py:23），而 sdpa.py 允许 3D/4D（sdpa.py:23-24）；
- 自己手算 `tril` causal mask（torch.py:31-34），而 sdpa.py 用 `is_causal` 标志（sdpa.py:32）。

## 2. 质疑点 / 改进机会
- 死代码（AGENTS.md 规则 6）：grep 结果显示 `TorchAttentionKernel` 仅出现在
  - `vrl/nn/kernels/attention/torch.py`（自身定义 + `__all__`）
  - `vrl/nn/kernels/attention/__init__.py:4,6`（re-export）

  没有任何 production 代码、test、脚本实例化它。命令：
  `grep -rn "TorchAttentionKernel" --include=*.py .`（排除 `TorchSDPAAttentionKernel` 后）只剩这三处。
- docstring 写 "reference implementation for parity and debug paths"（torch.py:1,12），但没有任何 parity test 引用它——属于"声称的用途没兑现"，正是规则 4/6 要 flag 的情况。
- 与 `sdpa.py` 职责重复：两个 SDPA 薄封装并存，留着只会让人困惑"该用哪个"。

## 3. 建议动作
- delete 整个 `vrl/nn/kernels/attention/torch.py`。
- 同步从 `vrl/nn/kernels/attention/__init__.py` 移除 `from ... torch import TorchAttentionKernel` 与 `__all__` 中的 `"TorchAttentionKernel"`（init.py:4,6）。
- grep 证据：删除前已确认无 production/test 引用（见第 2 节命令与结果）。
- 备选（仅当确有 parity 需求时）：把它真正接入一个 numerical-parity 测试，对比 `TorchSDPAAttentionKernel` 输出，并改名 `*_reference` 以表明是参考实现而非生产 kernel。但当前没有该需求，应直接删。

## 4. 不动什么 / 为什么不是过度清理
- 不动 `sdpa.py:TorchSDPAAttentionKernel`：它被 `vrl/nn/layers/attention/joint.py:9,15-16` 作为可注入 kernel 实际使用（`kernel or TorchSDPAAttentionKernel()`），是真实的依赖注入 seam，保留。
- 这不是为省 LOC 而拍平 thin function——是删除一个完全无引用、且与现有 kernel 重复的类。符合"fix root cause"而非"consistency over cleanup"误伤：这里根本没有跨家族一致性可言（不存在其它 `*AttentionKernel` 家族成员需要它对齐）。

## 5. 验证
- 删除后跑 `grep -rn "TorchAttentionKernel" --include=*.py .` 应只剩 0 处（连定义都没了）。
- `ruff check vrl/nn/kernels/attention/` 确认无未用 import 残留。
- `python -c "import vrl.nn.kernels.attention as a; print(a.__all__)"` 确认包仍可导入且 `__all__` 只剩 `TorchSDPAAttentionKernel`。
- 跑相关 test：`pytest tests/nn/ -q` 确认无回归（本类无 test，删除不影响）。
