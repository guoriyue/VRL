# SPRINT(auto): vrl/rewards/functions/codex_image_qa.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/rewards/functions/codex_image_qa.py` (59 LOC)
角色判定: thin-wrapper
结论: improve

## 0. 一句话
RewardFunction wrapper 本体 justified，但文件 import 了 `_extract_score_from_text` / `_render_prompt_template` 却在 body 里一次都没用，仅为塞进 `__all__` 做死 re-export——无任何外部消费者。

## 1. 现状（读代码得出）
```python
from vrl.rewards.models.codex_image_qa import (
    DEFAULT_PROMPT_TEMPLATE,
    CodexImageQARewardModel,
    _extract_score_from_text,
    _render_prompt_template,
)
...
__all__ = [
    "DEFAULT_PROMPT_TEMPLATE",
    "CodexImageQAReward",
    "_extract_score_from_text",
    "_render_prompt_template",
]
```

body 里只用到 `DEFAULT_PROMPT_TEMPLATE`（line 28，作为默认参数）和 `CodexImageQARewardModel`（line 34）。`_extract_score_from_text` 和 `_render_prompt_template` 既没在 body 引用，也没有任何外部模块从本文件 import。

## 2. 质疑点 / 改进机会
- 死 re-export + 未使用 import：grep `vrl/`+`tests/`，没有文件从 `vrl.rewards.functions.codex_image_qa` 取 `_extract_score_from_text` / `_render_prompt_template` / `DEFAULT_PROMPT_TEMPLATE`。`tests/rewards/test_multi.py:10` 只 import `CodexImageQAReward`。证据：`codex_image_qa.py:6-11` import、`:54-59` re-export；grep 显示这两个私有符号名只出现在本文件和 `vrl/rewards/models/codex_image_qa.py`。
- 与 `claude_image_qa.py` 同病：把 model 层私有符号（前缀 `_`）提升到 functions 层 `__all__`，没有边界价值，且会被 ruff 标为 F401（import 未使用）。

## 3. 建议动作
- import 收窄为：

  ```python
  from vrl.rewards.models.codex_image_qa import (
      DEFAULT_PROMPT_TEMPLATE,
      CodexImageQARewardModel,
  )
  ```
- `__all__` 收窄为 `["CodexImageQAReward"]`，与同目录其它 reward 文件一致。
- 与 `SPRINT_vrl_rewards_functions_claude_image_qa.md` 一起改，保持两个 image-QA wrapper 形状对称。

## 4. 不动什么 / 为什么不是过度清理
- `CodexImageQAReward` 类保留：registry builtin（`registry.py:34`），且 `:33` 注释说明 eager build 是为了让 "command required" 校验立刻触发，这是有意的构造时行为边界，不是冗余。
- `DEFAULT_PROMPT_TEMPLATE` 作为构造默认值（line 28）必须 import，保留。
- functions/models 分层不动（architecture test 强约束）。

## 5. 验证
- `ruff check vrl/rewards/functions/codex_image_qa.py` 不再报 F401。
- `pytest tests/rewards/test_multi.py -k codex` 通过（含 `test_codex_image_qa_reward_config_builds_primary_entry_without_legacy_alias`）。
- `grep -rn "functions.codex_image_qa import" vrl tests` 确认只有 registry + test 取 `CodexImageQAReward`。
