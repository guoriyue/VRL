# SPRINT(auto): vrl/rewards/models/kling_video_reward.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/rewards/models/kling_video_reward.py` (776 LOC)
角色判定: core
结论: improve

## 0. 一句话
这是一个合法的、自有的 Kling VideoReward model adapter（从 VideoAlign 收窄移植，署名清晰），整体应保留；唯一值得改的是它自带的 `_torch_dtype` 与 `base.py:resolve_dtype` 重复了一套 dtype 解析，且文件偏大可考虑把 ALL_CAPS prompt 模板表外移。

## 1. 现状（读代码得出）
文件做四件相关的事，都围绕同一个模型：
- prompt 模板 taxonomy（`_SIMPLE_PROMPT`/`_VIDEOSCORE_QUERY_PROMPT`/`_DETAILED_PROMPT*`，行 59-143）+ `_build_video_reward_prompt` 选模板（463-491）。
- 模型类 `KlingVideoRewardModel`（181-349）实现 `RewardModel` 协议，`__call__` 打分一个视频 artifact。
- Qwen2-VL backbone + reward head `KlingQwen2VLRewardModel`（352-452）。
- checkpoint 加载/state_dict remap/config 解析（494-769）。

ALL_CAPS 常量基本都是真边界：
- `_SPECIAL_TOKENS`（28）= 模型新增的特殊 token，协议/架构边界。
- `_DEFAULT_REWARD_MODEL`/`_DEFAULT_REVISION`/`_DEFAULT_MODEL_SUBDIR`（29-31）= checkpoint repo/文件名边界。
- `_SCORE_KEY_MAP`（32-37）= model→public score key 映射，刻意隔离的 taxonomy。
- `_DIMENSION_DESCRIPTIONS`（39-57）+ 各 prompt 模板 = 从 VideoAlign 收窄的 prompt taxonomy。

`_from_dataclass`（708-710）已经正确用 `{field.name for field in fields(cls)}` 从 dataclass derive 字段白名单——符合 AGENTS.md「从源头 derive」的要求，是正面例子。

## 2. 质疑点 / 改进机会
- **dtype 解析重复**：本文件 `_torch_dtype`（729-743）把 `"bf16"/"fp16"/"fp32"/"auto"` 映射到 torch dtype；`vrl/rewards/models/base.py:resolve_dtype`（base:18-21）用 `getattr(torch, name, float32)` 做同样的事。两者语义略有差异（kling 支持别名 `bf16` 且对未知值 raise，base 静默回退 float32）。这是两套并存的 dtype 解析，值得统一到一个 util（如 `vrl/utils` 或 reward 公共模块），让 base 与 kling 共用，kling 保留它「未知值 raise + 别名」的更严格行为作为该 util 的默认。
- **god-file 边缘**：776 LOC 把 prompt 模板表（约 90 行纯字符串）、backbone 定义、checkpoint loader 塞在一个文件。这些确实都属于同一个模型家族、cohesive，不构成「多条不相关管线」，所以不强判 consolidate。但 prompt 模板 taxonomy（59-143 + `_DIMENSION_DESCRIPTIONS`）是纯数据，可外移到 `kling_video_reward_prompts.py` 让主文件聚焦模型/加载逻辑，提升可读性。这是可选改进，不是必须。

## 3. 建议动作
1. 抽一个共享 dtype util（如 `vrl/utils/torch_dtype.py` 的 `resolve_torch_dtype(name, *, strict, fallback)`），让 `base.py:resolve_dtype` 和本文件 `_torch_dtype` 都改调它；保留 strict 模式的别名与 raise 行为。
2. （可选）把 prompt 模板常量 + `_DIMENSION_DESCRIPTIONS` + `_build_video_reward_prompt` 外移到 `kling_video_reward_prompts.py`，主文件 import。

## 4. 不动什么 / 为什么不是过度清理
- 不要拆模型类、backbone、checkpoint loader 到多个文件——它们紧耦合（remap key、特殊 token、reward head 维度必须一致），分散反而损害可读性与调试。AGENTS.md「consistency over cleanup」适用。
- 所有 ALL_CAPS 常量保留：它们是 checkpoint 文件名 / 特殊 token / score-key 映射 / 上游 prompt taxonomy，全是真边界，不是手抄的 typed 结构。
- `_from_dataclass` 的 derive 写法是正确范例，保留。
- `preflight_kling_video_reward_backend`（455-461）被 `scripts/common/online.py:283` 调用做依赖预检，是真用途，保留。

## 5. 验证
- 统一 dtype util 后：`grep -rn "resolve_dtype\|_torch_dtype\|resolve_torch_dtype" vrl/` 确认无遗漏旧调用。
- 跑 reward 单测确认 Kling 打分路径仍工作（需 weights 的用例可 skip，至少跑 `preflight` 与 config 解析）。
- `ruff check vrl/rewards/models/kling_video_reward.py`。
