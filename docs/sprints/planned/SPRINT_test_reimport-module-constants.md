# SPRINT: 测试重抄模块级常量/协议面，应改为 import 派生（planned）

状态：未开始（2026-06-21）。
范围：把一批**已经是可 import 的模块常量 / 协议 `__protocol_attrs__` / registry 默认值**、却被测试用字面量手抄一遍的断言，改成「import 源头符号 + 从它派生期望值」。源头是 single source of truth，手抄副本会在任何 rename/新增字段时静默漂移：源头一处更新，冻结的副本要么对新值误报失败、要么继续验证陈旧的接口面。优先级 medium。**不**改动与字面量同处的真实行为断言（leak-check、override pass-through、shape 校验等），那些是契约本体。

> 与已落地的 canonical 反例对齐：`registered_rollout_families() == tuple(FAMILY_REGISTRY)` —— registry 是源头，任何「手写一份 family/alias/常量清单再 `==` 比对」的测试就是 wan_2_2-missing 那类 bug 的同构体。本 sprint 是把这类「重抄」逐条收口到 import-and-derive。

## 0. Core Decision（先看这一段）

裁决标准只有一条：**测试断言里出现的那个值，是不是源码里一个可 import 的常量 / 协议属性集 / dataclass 或 registry 默认值？** 是 → 测试必须 import 它并派生期望，而不是把字面量再敲一遍。

派生模式（canonical pattern，全 sprint 统一）：

- 协议必需方法面 → `tuple(sorted(ReplayModel.__protocol_attrs__))`，不要手写方法名元组。
- 别名/分数映射 → `set(scores) == set(_SCORE_KEY_MAP)`，逐键值用 `_SCORE_KEY_MAP[k]` 索引原始键。
- taxonomy（segment 名）/ upsample 因子 → 迭代 `JANUS_R1_SEGMENTS`、乘 `JANUS_IMAGE_PATCH_SIZE`。
- provenance 串 / dotted-path / registry 默认 tuple → import `DOMAIN`/`TEMPLATE_ID`/`DECODE_METHOD`/`_KLING_VIDEO_REWARD_MODEL`/`_default_return_artifacts` 直接比对。
- 函数默认值（`cache_dtype="auto"`）→ 从源函数签名默认派生，不再硬抄 `"auto"`。

**只动重抄那一行**：每条 finding 都有同处的真实行为断言（如 kling 的 raw-key leak-check、AR 的 `block_size==32` override pass-through、videophy 的 `source_frame_size`），那些原样保留 —— 它们断言的是行为，不是重抄源头。

所有 10 条均已开文件核实：测试行号/snippet 与源头常量 import 路径逐一对照确认可派生（下文逐条标注源头 file:line）。无一条在核查中被证伪。

## 1. 现状实锤

### 1.1 `ReplayModel` / `RuntimeModel` 协议面手抄（成对）

`tests/models/interfaces/test_replay_model_contract.py:19-21`：

```python
# ReplayModel's required surface. Mirrors ``require_replay_model`` so a method
# rename in the protocol guard surfaces here too.
_REPLAY_MODEL_METHODS = ("replay_forward", "disable_adapter")
```

`tests/models/interfaces/test_runtime_model_contract.py:24-26`：

```python
# RuntimeModel's required surface. Mirrors ``require_runtime_model`` so a method
# rename in the protocol guard surfaces here too.
_RUNTIME_MODEL_METHODS = ("replay_forward", "disable_adapter", "load_trainable_state")
```

注释自己承认「Mirrors `require_*`」—— 即明知是副本。源头是 `vrl/models/interfaces/replay.py:108-132` 两个 `@runtime_checkable` Protocol。已实测：

```
ReplayModel.__protocol_attrs__  -> {'disable_adapter', 'replay_forward'}
RuntimeModel.__protocol_attrs__ -> {'disable_adapter', 'load_trainable_state', 'replay_forward'}
```

两个 Protocol 都已在测试文件顶部 import（`replay.py` 行 11-17、`runtime.py` 行 15-22）。`RuntimeModel(ReplayModel, Protocol)` 继承 —— 一旦在 `ReplayModel` 上加共享方法，runtime 元组也得同步，但手抄不会，冻结副本继续验证陈旧面。

### 1.2 kling `_SCORE_KEY_MAP` 别名集手抄

`tests/rewards/kling_video_reward/test_model_loading.py:314-319`：

```python
assert scores == {
    "visual_quality": 1.0,
    "motion_quality": 2.0,
    "text_alignment": 3.0,
    "overall_reward": 6.0,
}
```

源头 `vrl/rewards/models/kling_video_reward.py:31-36`（注意方向是 `public -> raw`）：

```python
_SCORE_KEY_MAP = {
    "overall_reward": "Overall",
    "visual_quality": "VQ",
    "motion_quality": "MQ",
    "text_alignment": "TA",
}
```

`_normalize_scores` 纯靠迭代 `_SCORE_KEY_MAP.items()` 产出。测试把全套 public 别名键手写成字面 dict 再 `==`。真正有价值的是紧随其后的 `:320-321` raw-key leak-check（`not ({"VQ","MQ","TA","Overall"} & set(scores))`）—— 保留。

### 1.3 kling `_KLING_VIDEO_REWARD_MODEL` dotted-path 手抄

`tests/rewards/inference/test_runtime_factory.py:74-80`：

```python
assert reward._actor_runtime.worker_config == {
    "model_path": "",
    "dtype": "bfloat16",
    "model_factory": "vrl.rewards.models.kling_video_reward:KlingVideoRewardModel",
    "reward_model_name": "KlingTeam/VideoReward@main",
    "reward_model_version": "KlingTeam/VideoReward@main",
}
```

`model_factory` 串是 `vrl/rewards/functions/kling_video_reward.py:18` 的 `_KLING_VIDEO_REWARD_MODEL` 常量（行 29 注入 `model_factory=_KLING_VIDEO_REWARD_MODEL`）的手抄。其余键（`model_path`/`dtype` passthrough、`reward_model_name`/`version` 从 `reward_name` 派生）是真 wiring 行为 —— 只该把 factory 串改为 import 派生。

### 1.4 JANUS R1 segment taxonomy 手抄（三处）

`tests/models/ar/janus_pro/test_r1_model.py:270`：

```python
assert set(out["segments"]) == {"initial_image", "selfcheck_text", "final_image"}
```

`:399-403`：

```python
assert set(out.trajectory.segments) >= {
    "initial_image",
    "selfcheck_text",
    "final_image",
}
```

stub 构造 `:357-360`：

```python
for idx, name in enumerate(
    ("initial_image", "selfcheck_text", "final_image"),
    start=1,
):
```

源头 `vrl/models/ar/janus_pro/__init__.py:13`：`JANUS_R1_SEGMENTS = ("initial_image", "selfcheck_text", "final_image")`，已 export（`__init__.py:30`），生产侧 `model.py:532` 作 `task_stages` 默认、`:549` 作校验集。加/改 refine 段，生产自动跟随，三处字面集静默分叉。

注：`:297` 的 `set(result.segments) == {"selfcheck_text", "final_image"}` 是 `ReplayRequest(segment_names=(...))` 入参的回显，应派生为 `JANUS_R1_SEGMENTS` 的切片（去掉 `initial_image`）。

### 1.5 JANUS upsample 因子 16 手抄

`tests/models/ar/fixtures.py:84-87`：

```python
def decode_code(self, ids: torch.Tensor, shape: list[int]) -> torch.Tensor:
    del ids
    batch, _, height, width = shape
    return torch.zeros(batch, 3, height * 16, width * 16)
```

`16` 正是 `vrl/models/ar/janus_pro/model.py:59` 的 `JANUS_IMAGE_PATCH_SIZE = 16  # decoder upsample factor → 384 px`，已 export（`model.py:1277`）。stub 重烤了源码已命名的 decoder upsample 因子。

### 1.6 danbooru `DOMAIN` / `TEMPLATE_ID` provenance 串手抄

`tests/data/test_danbooru.py:58-59`：

```python
assert {row["metadata"]["domain"] for row in rows} == {"anime"}
assert {row["metadata"]["template_id"] for row in rows} == {"anime_anatomy_v1"}
```

源头 `vrl/scripts/data/danbooru.py:72-73`：`TEMPLATE_ID = "anime_anatomy_v1"`、`DOMAIN = "anime"`，写入行在 `:584/:591`。template 版本一升（`anime_anatomy_v2`）行构造行为不变，字面副本却失败。

### 1.7 videophy `DECODE_METHOD` 手抄

`tests/data/test_videophy_i2v.py:107`：

```python
assert train_rows[0]["metadata"]["decode_method"] == "imageio_ffmpeg_first_frame"
```

源头 `vrl/scripts/data/videophy_i2v.py:40`：`DECODE_METHOD = "imageio_ffmpeg_first_frame"`，写入 `:170`。

注（核查修正）：同段 `:109` 的 `image_size == {"width": 832, "height": 480}` 不是 `DECODE_METHOD` 那类常量，而是 `prepare_videophy_i2v_dataset(width=832, height=480)`（`videophy_i2v.py:112-113`）的**函数默认入参**回显；`:108` 的 `source_frame_size == {"width": 4, "height": 4}` 是测试自己喂的输入。本 sprint 只改 `decode_method` 一行为 import 派生；`image_size` 若要去字面化，应显式把 width/height 传进函数再断言同值（让测试拥有它控制的输入），不在本 sprint 强制范围。

### 1.8 family registry `_default_return_artifacts` tuple 手抄

`tests/rollouts/runtime/test_family_registry.py:158-162`：

```python
for family in FAMILY_REGISTRY:
    assert FAMILY_REGISTRY[family].collector.return_artifacts == (
        "output",
        "trajectory",
    )
```

源头 `vrl/rollouts/families/registry.py:29` `_default_return_artifacts = ("output", "trajectory")`，每个 entry 经 `:112/:271/:293/:320` `return_artifacts=_default_return_artifacts` 接入。测试文件已 import `FAMILY_REGISTRY`（`:10-11`）。默认加第三件 artifact，源头一处改，字面 tuple 把所有 entry 误判为失败。

### 1.9 AR paged cache_dtype `"auto"` 默认手抄（成对）

`tests/generation/ar/test_janus_paged_attention_one_step.py:163` 与 `tests/generation/ar/test_nextstep_vllm_paged_attention_backend.py:94`：

```python
assert cache_dtype == "auto"
```

两个测试的 request 都**没设** `ar_paged_cache_dtype`，故验证的是默认解析路径。源头默认在 `vrl/nn/modules/ar_attention_backends.py:76` `cache_dtype: str = "auto"`（`build_vllm_attention_backend` 签名默认），executor `vrl/generation/ar/executor.py:64` `cache_dtype=str(sampling.get("ar_paged_cache_dtype", "auto"))` 回显同一默认。两份字面 `"auto"` 一起腐烂。

注：同处 `block_size == 32` 不是默认 —— 是 request 自带的 `ar_paged_block_size: 32` 回显，正确测的是 override pass-through，保留。

## 2. 落地方案

canonical 派生模式：**import 源头符号 → 用它构造期望**。下文逐条给 BEFORE/AFTER。

### A. 协议面 → `__protocol_attrs__`

`test_replay_model_contract.py:19-21`：

```python
# BEFORE
_REPLAY_MODEL_METHODS = ("replay_forward", "disable_adapter")
# AFTER（ReplayModel 已 import）
_REPLAY_MODEL_METHODS = tuple(sorted(ReplayModel.__protocol_attrs__))
```

`test_runtime_model_contract.py:24-26`：

```python
# BEFORE
_RUNTIME_MODEL_METHODS = ("replay_forward", "disable_adapter", "load_trainable_state")
# AFTER（RuntimeModel 已 import，自动含继承自 ReplayModel 的方法）
_RUNTIME_MODEL_METHODS = tuple(sorted(RuntimeModel.__protocol_attrs__))
```

注释里「Mirrors …」一句改写为「Derived from the protocol's `__protocol_attrs__`, so a method add/rename auto-widens the contract check.」。`test_require_*_reports_missing_*`（两文件末尾）原样保留 —— 那是 guard 报错行为断言。

### B. kling `_SCORE_KEY_MAP` 派生

`test_model_loading.py:314-319`：

```python
# BEFORE
assert scores == {"visual_quality": 1.0, "motion_quality": 2.0,
                  "text_alignment": 3.0, "overall_reward": 6.0}
# AFTER
from vrl.rewards.models.kling_video_reward import _SCORE_KEY_MAP, _normalize_scores
raw = {"VQ": 1.0, "MQ": 2.0, "TA": 3.0, "Overall": 6.0}
scores = _normalize_scores(raw)
assert set(scores) == set(_SCORE_KEY_MAP)
for public_key, raw_key in _SCORE_KEY_MAP.items():
    assert scores[public_key] == raw[raw_key]
```

`:320-321` raw-key leak-check 原样保留。

### C. kling factory dotted-path 派生

`test_runtime_factory.py:74-80`：把 `"model_factory"` 那行从字面改为引用常量，其余键不动：

```python
# AFTER
from vrl.rewards.functions.kling_video_reward import _KLING_VIDEO_REWARD_MODEL
assert reward._actor_runtime.worker_config == {
    "model_path": "",
    "dtype": "bfloat16",
    "model_factory": _KLING_VIDEO_REWARD_MODEL,
    "reward_model_name": "KlingTeam/VideoReward@main",
    "reward_model_version": "KlingTeam/VideoReward@main",
}
```

### D. JANUS segment taxonomy 派生

`test_r1_model.py` 顶部 import `JANUS_R1_SEGMENTS`，然后：

```python
# :270  BEFORE -> AFTER
assert set(out["segments"]) == set(JANUS_R1_SEGMENTS)

# :399-403 BEFORE -> AFTER
assert set(out.trajectory.segments) >= set(JANUS_R1_SEGMENTS)

# :357-360 stub BEFORE -> AFTER
for idx, name in enumerate(JANUS_R1_SEGMENTS, start=1):

# :297 ReplayRequest 子集（去掉首段 initial_image）BEFORE -> AFTER
assert set(result.segments) == set(JANUS_R1_SEGMENTS[1:])
```

`:271-272` / `:298-299` 的逐段 shape 断言与 `:293` 的 `segment_names=(...)` 入参（亦可改用 `JANUS_R1_SEGMENTS[1:]` 保持一致）属行为断言，保留。

### E. JANUS upsample 因子派生

`fixtures.py:84-87`：

```python
# AFTER（import JANUS_IMAGE_PATCH_SIZE）
from vrl.models.ar.janus_pro.model import JANUS_IMAGE_PATCH_SIZE
...
return torch.zeros(
    batch, 3, height * JANUS_IMAGE_PATCH_SIZE, width * JANUS_IMAGE_PATCH_SIZE
)
```

### F. danbooru / videophy provenance 串派生

`test_danbooru.py:58-59`：

```python
# AFTER（import 模块或常量）
from vrl.scripts.data import danbooru
assert {row["metadata"]["domain"] for row in rows} == {danbooru.DOMAIN}
assert {row["metadata"]["template_id"] for row in rows} == {danbooru.TEMPLATE_ID}
```

`test_videophy_i2v.py:107`：

```python
# AFTER
from vrl.scripts.data.videophy_i2v import DECODE_METHOD
assert train_rows[0]["metadata"]["decode_method"] == DECODE_METHOD
```

`:109` image_size 不改（见 §1.7 注）。

### G. family registry 默认 tuple 派生

`test_family_registry.py:158-162`：

```python
# AFTER（从 registry 模块 import 默认）
from vrl.rollouts.families.registry import _default_return_artifacts
for family in FAMILY_REGISTRY:
    assert FAMILY_REGISTRY[family].collector.return_artifacts == _default_return_artifacts
```

### H. AR cache_dtype 默认派生

两个 AR paged 测试 `:163` / `:94`：从源函数签名默认派生，不再硬抄 `"auto"`：

```python
# AFTER
import inspect
from vrl.nn.modules.ar_attention_backends import build_vllm_attention_backend
_DEFAULT_CACHE_DTYPE = (
    inspect.signature(build_vllm_attention_backend).parameters["cache_dtype"].default
)
...
assert cache_dtype == _DEFAULT_CACHE_DTYPE
```

同处 `block_size == 32`（override 回显）原样保留。

## 3. 验证（finishing criteria）

- 重抄字面量在改动文件内归零（仅允许真实行为断言/输入回显残留）：
  - `grep -n '("replay_forward"' tests/models/interfaces/test_*_contract.py` 不再命中手写方法元组定义。
  - `grep -n '"visual_quality": 1.0' tests/rewards/kling_video_reward/test_model_loading.py` 零命中。
  - `grep -n 'vrl.rewards.models.kling_video_reward:KlingVideoRewardModel' tests/rewards/inference/test_runtime_factory.py` 零命中（已改为常量引用）。
  - `grep -n '"initial_image", "selfcheck_text", "final_image"' tests/models/ar/janus_pro/test_r1_model.py` 零命中。
  - `grep -n 'height \* 16' tests/models/ar/fixtures.py` 零命中。
  - `grep -n '"anime_anatomy_v1"\|== {"anime"}' tests/data/test_danbooru.py` 零命中。
  - `grep -n 'imageio_ffmpeg_first_frame' tests/data/test_videophy_i2v.py` 零命中。
  - `grep -n '"output",' tests/rollouts/runtime/test_family_registry.py` 在 return_artifacts 断言处零命中。
  - `grep -rn 'cache_dtype == "auto"' tests/generation/ar/` 零命中。
- 派生确实引用了源头 import：`grep -rn '__protocol_attrs__\|_SCORE_KEY_MAP\|_KLING_VIDEO_REWARD_MODEL\|JANUS_R1_SEGMENTS\|JANUS_IMAGE_PATCH_SIZE\|danbooru.DOMAIN\|DECODE_METHOD\|_default_return_artifacts\|build_vllm_attention_backend' tests/` 命中改动文件。
- pytest 全绿：
  - `pytest tests/models/interfaces/ tests/rewards/kling_video_reward/ tests/rewards/inference/ -q`
  - `pytest tests/models/ar/janus_pro/test_r1_model.py tests/data/test_danbooru.py tests/data/test_videophy_i2v.py -q`
  - `pytest tests/rollouts/runtime/test_family_registry.py tests/generation/ar/ -q`
- 反向哨兵（可选）：在某源头常量临时加一个新成员（如给 `_SCORE_KEY_MAP` 加一键、给 `JANUS_R1_SEGMENTS` 加一段），改后的派生测试应**自动跟随**（不因新增而误失败、也不漏覆盖），随后还原。这验证「副本不再冻结」。

## 4. 非目标 / Non-Goals

- **不动同处的真实行为断言**：kling raw-key leak-check、AR `block_size==32` override pass-through、JANUS 逐段 shape、videophy `source_frame_size`/`image_size`、各 `require_*` 报错断言 —— 这些断言的是行为/输入回显，不是重抄源头，保留。
- **不去字面化「测试自己拥有的输入」**：被测函数的入参回显（如 `block_size==32`、`image_size 832/480`、`source_frame_size 4/4`）是测试控制的输入，不是源头常量，不在本 sprint 范围（videophy image_size 的可选改法见 §1.7 注）。
- **不碰真正的外部固定契约**：上游 HF repo id + commit-hash revision（`tests/models/diffusion/registry.py` 的 tiny-pipe 固定 pin）是外部可复现性 pin，不是仓库内可派生值，不动。
- **不扩展到本主题之外的 finding**：同次审计里的 literal_config_assertion / frozen_snapshot（directory-listing、AlgorithmConfig.kind Literal、`_FAMILY_SCHEDULERS` 等）属其他主题，归各自 sprint，本 sprint 只收口「重抄可 import 模块常量/协议面」这 10 条。

## References

- `tests/models/interfaces/test_replay_model_contract.py:19-21`、`test_runtime_model_contract.py:24-26`
- `vrl/models/interfaces/replay.py:108-132,135-158`
- `tests/rewards/kling_video_reward/test_model_loading.py:306-321`
- `vrl/rewards/models/kling_video_reward.py:31-36`（`_SCORE_KEY_MAP`）、`:679-683`（`_normalize_scores`）
- `tests/rewards/inference/test_runtime_factory.py:74-80`
- `vrl/rewards/functions/kling_video_reward.py:18,29`（`_KLING_VIDEO_REWARD_MODEL`）
- `tests/models/ar/janus_pro/test_r1_model.py:270,293,297,357-360,399-403`
- `vrl/models/ar/janus_pro/__init__.py:13,30`（`JANUS_R1_SEGMENTS`）、`model.py:59,532,549,1277`（`JANUS_IMAGE_PATCH_SIZE`）
- `tests/models/ar/fixtures.py:84-87`
- `tests/data/test_danbooru.py:58-59`、`vrl/scripts/data/danbooru.py:72-73,584,591`（`DOMAIN`/`TEMPLATE_ID`）
- `tests/data/test_videophy_i2v.py:107-109`、`vrl/scripts/data/videophy_i2v.py:40,112-113,170,179`（`DECODE_METHOD`）
- `tests/rollouts/runtime/test_family_registry.py:158-162`、`vrl/rollouts/families/registry.py:29,112,271,293,320`（`_default_return_artifacts`）
- `tests/generation/ar/test_janus_paged_attention_one_step.py:163`、`test_nextstep_vllm_paged_attention_backend.py:94`
- `vrl/nn/modules/ar_attention_backends.py:71-86`（`build_vllm_attention_backend`, `cache_dtype` 默认）、`vrl/generation/ar/executor.py:64`
- 关联：[[SPRINT_segment_signal_dead_field_cleanup]]（派生结构体死字段，同一「源头是 single source of truth、副本会腐烂」原则的字段级版本）
