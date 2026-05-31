# SPRINT(auto): vrl/rewards/functions/ocr.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/rewards/functions/ocr.py` (98 LOC)
角色判定: thin-wrapper
结论: question

## 0. 一句话
reward wrapper 本体 justified，但文件顶部 36 行的 `_normalize_text` / `_normalized_edit_distance` / `_edit_distance` 三个 helper 既不被 reward 类用、也不被 model 用，仅被 test import——它们是放错位置的纯测试工具，混进了长期 import graph 的 functions 模块。

## 1. 现状（读代码得出）
文件分两部分。下半部分 `OCRReward` 是标准薄 wrapper（构造 `OCRRewardModel` + `LocalRewardRuntime`），上半部分是三个 module-level 文本距离 helper：

```python
def _normalize_text(text: str) -> str:
    """Normalize text for helper-level OCR edit-distance tests."""
    ...
def _normalized_edit_distance(a: str, b: str) -> float:
    """Return Levenshtein distance normalized by the longer input length."""
    ...
def _edit_distance(a: str, b: str) -> int:
    """Small dependency-free Levenshtein implementation for tests."""
    ...
```

docstring 自己写明 "for ... tests" / "for tests"。grep 确认调用者：`_normalize_text` / `_normalized_edit_distance` 仅 `tests/rewards/test_ocr.py:7-11` import；`models/ocr.py` 里没有这三个符号（实际 OCR 评分用的是 paddleocr，距离逻辑不在此处）。即 reward 类本体（`OCRReward`，`:64-95`）从不调用这三个 helper。

此外 `OCRReward._engine` property/setter（`:89-95`）也是纯转发到 `self._model._engine`，唯一用途是让 `tests/rewards/test_ocr.py:88,104` 注入 fake paddleocr 引擎。

## 2. 质疑点 / 改进机会
- 测试工具混入长期模块：三个 `_*edit_distance`/`_normalize_text` helper 是 self-described "for tests"，但放在 production import graph 的 `vrl/rewards/functions/ocr.py` 顶层。按 AGENTS.md "one-shot vs long-term"，一次性/测试专属逻辑不该混进长期代码模块。证据：`ocr.py:26-61` 定义，唯一消费者 `tests/rewards/test_ocr.py:7`。
- 这三个 helper 与 model 的真实评分路径不共享代码（`models/ocr.py` 不含任何 edit-distance 实现），所以它们也不是 "reward 与 model 共用的抽象"——它们就是独立的一份测试参考实现。若意图是测 model 的距离逻辑，应直接测 model 暴露的函数；若 model 没有可测的距离函数，则这三个 helper 测的是它们自己，是空转测试。
- `_engine` property/setter 是 test-only 注入口（`:89-95`），把 model 私有 `_engine` 提升到 reward 表面，与 nsfw_safety 的 passthrough 同类问题。
- 不确定点（故判 question）：需确认这三个 helper 的设计意图——是 (a) 准备给 model 复用但还没接线，还是 (b) 纯粹 test fixture 误放。证据不足以直接 delete。

## 3. 建议动作
- 若是 (b)：把 `_normalize_text` / `_normalized_edit_distance` / `_edit_distance` 移到 `tests/rewards/test_ocr.py`（或 `tests/rewards/_ocr_helpers.py`），functions/ocr.py 只保留 `OCRReward`，使其与同目录其它 reward 文件形状一致。
- 若是 (a)（希望 model 复用归一化/距离）：把这三个 helper 下沉到 `vrl/rewards/models/ocr.py` 并让 model 真正调用它们，functions 层不再持有，test 改为 import model 层符号。这样消除 "测一份没人用的参考实现" 的问题。
- `_engine` property/setter：保留可接受（它是 model 的 framework-adapter 注入口，跨 reward 调试常用），但应与 nsfw_safety sprint 统一裁决——若测试改为直接打 model，则此 property 也可删。

## 4. 不动什么 / 为什么不是过度清理
- `OCRReward` 类主体保留：registry builtin（`registry.py:37`），eager build + `debug_dir` reward-hacking audit（`:75-77` 注释）是有意行为，属 keep-justified。
- 不要为省 LOC 把 wrapper 拍平进 model：functions/models 分层受 architecture test 约束。
- 若最终选 (a) 方案，注意 model 复用的归一化语义必须与 flow_grpo 的 `OcrScorer` 对齐（文件 docstring `:11-13` 引用），不能借机改算法。

## 5. 验证
- `pytest tests/rewards/test_ocr.py` 全绿（无论 helper 移到 test 还是下沉 model，断言数值不变）。
- `grep -rn "_normalize_text\|_normalized_edit_distance\|_edit_distance" vrl tests` 确认 helper 只剩在一处（test 或 model），functions/ocr.py 不再持有。
- `ruff check vrl/rewards/functions/ocr.py`。
