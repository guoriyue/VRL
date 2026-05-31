# SPRINT(auto): vrl/models/ar/janus_pro/model.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/models/ar/janus_pro/model.py` (1357 LOC)
角色判定: core
结论: improve

## 0. 一句话
这是 Janus-Pro 真核心 model wrapper，整体 justified；但有一处死字段 `_frame_constants` 和一处被 runtime.py 手抄的 taxonomy tuple（`JANUS_R1_SEGMENTS`），应清理/统一来源。

## 1. 现状（读代码得出）
文件把所有 Janus-specific 细节（image-token vocab 范围、`gen_head` 投影、CFG 采样、VQ decode、R1 三段 refine）封装在 `JanusProModel` 后面，供通用 GRPO trainer 调用。ALL_CAPS 常量是真边界（架构维度/checkpoint 配置）：

```python
JANUS_IMAGE_TOKEN_NUM = 576           # 24 x 24 latent grid per image
JANUS_IMAGE_VOCAB_SIZE = 16_384       # gen_vision_model codebook size
JANUS_IMAGE_PATCH_SIZE = 16
JANUS_IMAGE_PIXEL_SIZE = 384
```

R1 segment 名集合也是常量：

```python
JANUS_R1_SEGMENTS = ("initial_image", "selfcheck_text", "final_image")  # line 64
```

`JanusProConfig` 有一个字段从未被使用：

```python
# line 117-118
# Cached references — populated at __post_init__ time by the wrapper
_frame_constants: dict[str, int] = field(default_factory=dict)
```

## 2. 质疑点 / 改进机会
- **死字段 `_frame_constants` (model.py:118)**：注释说 "populated at __post_init__ time by the wrapper"，但 `JanusProConfig` 是 `@dataclass(slots=True)` 没有 `__post_init__`，全仓 grep 仅此一处定义、零读零写。注释描述的填充逻辑根本不存在 —— 这是腐烂的占位字段，应删。
  ```
  $ grep -rn "_frame_constants" vrl/ --include=*.py
  vrl/models/ar/janus_pro/model.py:118:    _frame_constants: dict[str, int] = field(default_factory=dict)
  ```
- **taxonomy tuple 被手抄 (runtime.py:51)**：`JANUS_R1_SEGMENTS` 是 R1 段名的唯一真源，model.py 内部用它做默认值和校验（line 581/598）。但 runtime.py 没引用它，而是手抄了一份：
  ```python
  # runtime.py:47-52
  JANUS_PRO_R1_FAMILY_CAPABILITY = ar_discrete_family_capability(
      "janus_pro_r1", "ar_t2i_r1",
      trajectory_kind="multisegment",
      trainable_segments=("initial_image", "selfcheck_text", "final_image"),  # ← 手抄 JANUS_R1_SEGMENTS
  )
  ```
  这正是 AGENTS.md 点名的"手抄一份 typed 结构"反模式：将来给 R1 加一个 segment（改 model.py:64），`trainable_segments` 会悄悄漏改而不报错。runtime.py 已经 import 了 `JANUS_R1_SEGMENTS`（line 27）却没用在这里。

## 3. 建议动作
- 删除 `JanusProConfig._frame_constants` 字段及其误导性注释（model.py:117-118）。确认无引用（已 grep）。
- 把 runtime.py:51 的 `trainable_segments=("initial_image", "selfcheck_text", "final_image")` 改为 `trainable_segments=JANUS_R1_SEGMENTS`（该符号已在 runtime.py:27 import）。让 model.py 成为段名的唯一真源。
- 其余不动。

## 4. 不动什么 / 为什么不是过度清理
- 4 个 `JANUS_IMAGE_*` ALL_CAPS 是真架构/checkpoint 常量（codebook size、latent grid、patch/pixel size），AGENTS.md 明确允许保留，不要 derive 或下沉。
- `JANUS_R1_*_PROMPT` 是 R1 协议 prompt 字符串（协议常量），保留。
- `_base()` / `_lm_trunk()` / `image_token_logits_from_hidden` 这些薄 accessor 不是无意义转发：它们封装了 PEFT 包裹层的 unwrap 逻辑（注释解释了为何不能用 `hasattr(base_model)`），是移除真实复杂度的边界，保留。
- `JanusProReplayCore` / `JanusProReplayModel` 不是死代码：被 runtime.py 的 `build_janus_pro_replay_runtime_bundle` 使用，后者又被 `vrl/scripts/ar/janus_pro/train.py:88` 调用。保留。

## 5. 验证
- 删字段后：`grep -rn "_frame_constants" vrl/` 应为空；`python -c "from vrl.models.ar.janus_pro.model import JanusProConfig; JanusProConfig()"` 正常。
- 改 tuple 后：`python -c "from vrl.models.ar.janus_pro.runtime import JANUS_PRO_R1_FAMILY_CAPABILITY as c; assert c.trainable_segments == ('initial_image','selfcheck_text','final_image')"`。
- `ruff check vrl/models/ar/janus_pro/model.py vrl/models/ar/janus_pro/runtime.py`。
- 跑 Janus 相关测试：`pytest -k janus`。
