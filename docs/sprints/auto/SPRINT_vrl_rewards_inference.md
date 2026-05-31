# SPRINT(auto): vrl/rewards/inference.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/rewards/inference.py` (321 LOC)
角色判定: interface-boundary
结论: improve

## 0. 一句话
文件整体是合理的 reward inference 契约边界（dataclass + Protocol + 纯函数），但 `media_type` 合法值集合 `{"image","video","tensor"}` 被硬抄在两个文件里，应抽到单一来源；其余保持不动。

## 1. 现状（读代码得出）
本文件定义 reward 推理的稳定契约：`RewardInferenceArtifact` / `RewardInferenceRequest` / `RewardInferenceResult` 三个 frozen dataclass，`RewardInferenceRuntime` Protocol，以及 `shard_reward_request` / `validate_reward_results` / `score_artifacts_with_model` 三个被 local + ray 两条 transport 共享的纯函数。

`media_type` 的合法值在本文件 `__post_init__` 校验：

```python
if self.media_type not in {"image", "video", "tensor"}:  # inference.py:34
    raise ValueError("RewardInferenceArtifact.media_type must be image, video, or tensor")
```

同一个字面量集合在 `vrl/rewards/artifacts.py:28` 又出现一次：

```python
if media_type not in {"image", "video", "tensor"}:  # artifacts.py:28
    raise ValueError("media_type must be image, video, or tensor")
```

## 2. 质疑点 / 改进机会
- `{"image","video","tensor"}` 是协议级合法值（artifact 的 media kind），属于真边界——但它被手抄在两处（`inference.py:34` 与 `artifacts.py:28`）。任意一处新增/改名（如加 `"audio"`）会与另一处悄悄不一致。按 AGENTS.md 第 1 条，应从单一来源导出而非两份手维护。
- `artifacts.py` 已经 `from vrl.rewards.inference import RewardInferenceArtifact`（`artifacts.py:12`），共享一个常量没有额外依赖成本。
- 这是本文件唯一的 hygiene 问题。`score_aggregation` 目前只支持 `"sum"`（多处 `if != "sum": raise`），这是有意收窄的契约不是坏味道，不 flag。

## 3. 建议动作
- 在 `inference.py` 顶部定义单一来源，例如 `MEDIA_TYPES = frozenset({"image", "video", "tensor"})`（这是协议合法值表，属于 AGENTS.md 允许保留的"刻意隔离的 taxonomy 表"，放在契约文件正确）。
- `RewardInferenceArtifact.__post_init__` 改用 `if self.media_type not in MEDIA_TYPES`。
- `artifacts.py` import 该常量并复用，删掉自己那份字面量集合。
- 可加入 `__all__` 以便其它 transport（ray worker）复用。

## 4. 不动什么 / 为什么不是过度清理
- 三个 dataclass + `_ScoreSelection` mixin + Protocol 是真契约边界，被 base/runtime/ray 三方依赖，不动。
- `shard_reward_request` / `validate_reward_results` / `score_artifacts_with_model` 是 local 与 ray 两条 transport 共享的逻辑（`runtime.py:50`、`ray/runtime.py:59`、`ray/worker.py:61`），正是 AGENTS.md 认可的"移除真实复杂度的共享抽象 / 跨家族一致性"，绝不拆散。
- 不要把 `_ScoreSelection` 内联进两个 dataclass——它正是让 request/result 共享 `select_score` 校验语义的小抽象。

## 5. 验证
- 改完 `grep -rn '"image", "video", "tensor"'` 应只剩 `inference.py` 定义处一行。
- 跑 `pytest tests/rewards/test_reward_inference_runtime.py tests/rewards/test_video_reward_artifacts.py -q` 全绿（覆盖 media_type 校验路径）。
- `ruff check vrl/rewards/inference.py vrl/rewards/artifacts.py`。
