# SPRINT(auto): vrl/rewards/functions/claude_image_qa.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/rewards/functions/claude_image_qa.py` (74 LOC)
角色判定: thin-wrapper
结论: improve

## 0. 一句话
RewardFunction wrapper 本体是 justified 的薄边界，但 `__all__` 里 re-export 的 4 个符号（`DEFAULT_COMMAND` / `DEFAULT_PROMPT_TEMPLATE` / `_extract_score_from_text` / `_render_prompt_template`）是死 re-export，全仓没有任何模块从这里 import 它们。

## 1. 现状（读代码得出）
文件从 model 层 import 了 4 个符号再原样 re-export：

```python
from vrl.rewards.models.claude_image_qa import (
    DEFAULT_COMMAND,
    DEFAULT_PROMPT_TEMPLATE,
    ClaudeImageQARewardModel,
    _extract_score_from_text,
    _render_prompt_template,
)
...
__all__ = [
    "DEFAULT_COMMAND",
    "DEFAULT_PROMPT_TEMPLATE",
    "ClaudeImageQAReward",
    "_extract_score_from_text",
    "_render_prompt_template",
]
```

类本体（`ClaudeImageQAReward.__init__`）只实际用到了 `DEFAULT_COMMAND`（line 42）和 `ClaudeImageQARewardModel`（line 55）。其余三个（`DEFAULT_PROMPT_TEMPLATE` / `_extract_score_from_text` / `_render_prompt_template`）在本文件 body 里完全没被引用，只是为了塞进 `__all__`。

## 2. 质疑点 / 改进机会
- 死 re-export：grep 全仓 `vrl/`+`tests/`，没有任何文件从 `vrl.rewards.functions.claude_image_qa` import `DEFAULT_COMMAND` / `DEFAULT_PROMPT_TEMPLATE` / `_extract_score_from_text` / `_render_prompt_template`。真正消费这些符号的测试是直接从 model 层 import 的（见 `tests/rewards/`），functions 层这份 re-export 没有任何消费者。证据：`claude_image_qa.py:11-17` import，`:67-73` re-export；grep 结果显示这些符号名只在本文件和 `vrl/rewards/models/claude_image_qa.py` 出现。
- re-export 私有符号（前缀 `_`）本身违反封装意图：`_extract_score_from_text` / `_render_prompt_template` 是 model 层的内部实现，functions 层把它们提到 `__all__` 等于把别人的私有 API 当公共 API 转发，没有边界价值。

## 3. 建议动作
- 收窄 import 到实际使用的两个符号：

  ```python
  from vrl.rewards.models.claude_image_qa import (
      DEFAULT_COMMAND,
      ClaudeImageQARewardModel,
  )
  ```
- `__all__` 收窄为 `["ClaudeImageQAReward"]`（与 `aesthetic.py` / `nsfw_safety.py` / `geneval.py` / `ocr.py` 的单符号 `__all__` 形状一致）。
- 如果将来确实有外部代码需要 default 模板，应直接 `from vrl.rewards.models.claude_image_qa import DEFAULT_PROMPT_TEMPLATE`，不要经 functions 层中转。

## 4. 不动什么 / 为什么不是过度清理
- `ClaudeImageQAReward` 类本身保留：它是 registry（`registry.py::_register_builtins`）注册的 builtin reward，提供 config→RewardFunction 的构造边界（template file > string > default 的 precedence 逻辑在 `:49-54`），是跨家族一致的 reward 入口，属 keep-justified 的薄边界。
- 不要把这个 wrapper 合并进 model：functions 层与 models 层的分层是 architecture test 强约束的（`tests/architecture/test_generation_rollout_boundaries.py:123-144` 锁定 functions/ 目录文件集），分层本身是 justified 的。

## 5. 验证
- 改完跑 `grep -rn "from vrl.rewards.functions.claude_image_qa import" vrl tests` 确认只有 registry 在 import 该模块，且只取 `ClaudeImageQAReward`。
- `python -c "import vrl.rewards.functions.claude_image_qa"` 不报错。
- `pytest tests/rewards/test_multi.py -k claude` 通过（覆盖 claude_anatomy config → claude_image_qa backend）。
- `ruff check vrl/rewards/functions/claude_image_qa.py`（确认无 F401 unused import）。
