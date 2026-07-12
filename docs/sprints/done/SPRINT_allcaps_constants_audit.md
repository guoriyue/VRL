# SPRINT: vrl/ 模块级 ALL_CAPS 常量审计与整理（planned）

状态：DONE（全部 P0–P3 + 附录 B 已落地 main）。P0/P2 danbooru 词表外置+空串全局改参 (1254093)；P1 DEFAULT_ARTIFACT_FIELDS 从 PromptExample field(metadata={"artifact":True}) 派生 (45084f9)；附录 B 改写为 source+derived+改名：_SAFETY_RATING_SPELLINGS→SAFETY_RATING_ALIASES、_CANONICAL_STRING→_DTYPE_NAME_BY_INPUT(_DTYPE_SPELLINGS)、_ARTIFACT_DIRS→_INIT_DIRS_BY_DATASET、R1 prompts KEEP+注释、DEFAULT_NEGATIVE_PROMPT→_DEFAULT_NEGATIVE_PROMPT (a819878, e833d95)。唯一余项 _resolve_lora_block 统一已被本文显式划为单独 follow-up。
findings + 路径 + 整改逻辑都在本文。按 P0→P3 分批做，每批独立 PR，互不依赖。

> 方法：用 8 个并行 agent 扫了 `vrl/` 下全部 **131 个**模块级 ALL_CAPS 赋值（含
> 子拆分共 134 条 finding），逐个真实读源码 + grep 验证，再由我独立复核了所有
> DERIVE/CONSOLIDATE 候选。复核**修正了汇总的 3 处误判**（见 §8 校验记录），本文
> 是修正后的定稿。

---

## 0. 实施状态（2026-06-13 → `~/Desktop/VRL`，独立 clone，已验证）

应用户要求，把可安全落地的修复实现在独立 clone `~/Desktop/VRL`（与 wm-infra 工作区分开 review）。
该 clone 改前工作区干净、目标常量行号与本仓一致。**全部改动已验证**：`ruff` 通过、`tests/data`
30 passed、Group A 领域（trajectory/artifact）33 passed、18 个 danbooru 常量改前/改后逐一值等价。

**已落地：**
- **danbooru 词表 → `datasets/danbooru/config.yaml`**：18 个 tag/权重常量抽到 YAML，import 时
  `yaml.safe_load` 加载，**保留同名常量与运行时类型**（set/tuple/dict）。danbooru.py 净减 ~263 行。
- **danbooru 可变全局 → 显式参数**：删 `METADATA_PATH`/`HF_CACHE_DIR`/`DOWNLOAD_DANBOORU_METADATA`
  （全仓无外部引用），`build_default_manifests` 改 keyword 参数，默认值行为等价。
- **`SUBJECT_PROMPT_TAGS = tuple(SUBJECT_TAGS)`**：派生，消除手抄。
- **`DEFAULT_ARTIFACT_FIELDS`**：给 `PromptExample` 三个字段加 `field(metadata={"artifact": True})`，
  从字段元数据派生（单一真相源）。
- **`FORBIDDEN_TRAJECTORY_METRICS` / provenance 三处**：加澄清 + 交叉引用注释（理由见下）。

**实施期细读源码后修正的判断（证据驱动，避免过度设计）：**
- **`_ARTIFACT_DIRS`（setup.py）→ 改判 KEEP**（原 P3 CONSOLIDATE）：细读 setup.py 后确认其 key
  （"anime" 等）并非 `COMMAND_NAME` 的重复（danbooru 根本没有单一 "anime" 命令），dirs
  （`data_root/danbooru/images`）与 danbooru `OUTPUT_DIR`（`datasets/danbooru`）是不同路径、不同用途；
  模块 docstring 明写 "No generic artifact-manifest framework lives here"。打散到各模块反而违背该设计意图。
- **Janus R1 prompts → 改判 KEEP**（原 P3 MOVE）：prompt 含 `<begin_of_image>` 等特殊 token，是模型
  R1 协议文本而非可调业务词表；搬 yaml 有破坏 token 格式的风险、收益甚微。
- **provenance 三处 → 注释而非重构**：tuple 顺序是 load-bearing（有测试 pin 了"首个缺失字段"错误信息），
  整体合并会改变校验行为，故只加交叉引用注释。

**未做（可选，按一次性产物生命周期）：** `DEFAULT_NEGATIVE_PROMPT`（一次性 generate 脚本）。

---

## 1. 核心结论 (TL;DR)

**绝大多数（106/131，约 81%）ALL_CAPS 是合法边界，应原样保留。硬编码本身不等于不安全——
代表真实契约（schema 键、模型维度、文件名、协议名）的硬编码恰恰必须硬编码。真正不安全的
是那些"手写副本会随源漂移而静默腐烂"或"业务词表混进工作流代码"的常量，全仓只有一小撮。**

不要为了"清理"去搅动本就正确的常量——这正是项目 AGENTS.md "一致性优先于行数削减" 的非目标。

按动作拆分：

| 动作 | 数量 | 含义 |
|---|---|---|
| **KEEP**（保留） | 107 | schema key、模型维度、computed singleton、路径锚点、文件名、外部规范——真实边界 |
| **MOVE**（搬到 config/data 资产） | 16 | 大业务词表 / 领域分类表 / prompt 文本混进了工作流代码 |
| **PROMOTE_TO_CONFIG**（提升为配置） | 6 | 可调旋钮，含 3 个"空串可变全局"反模式 |
| **DERIVE**（从真相源派生） | 2 | 手写常量重复了已有的类型化结构，会静默腐烂 |
| **CONSOLIDATE**（跨文件合并去重） | 2 | `setup.py` 硬编码 registry；三处 provenance 字段词表重复 |

**真正的靶子只有三类**（其余都是噪音，碰了反而引入风险）：

- **(a) `danbooru.py` 的领域词表倾倒（最大单点）**——单文件塞了十几个手工 anime/safety
  tag 集合（`EXCLUDE_TAGS`、`POSE_TAGS`、`HAND_TAGS`、`SAFETY_RISK_TAGS` …），正是
  AGENTS.md 警告的"大业务词表混进工作流代码"。应整体抽到 `datasets/danbooru/config.yaml`。
- **(b) 用作配置注入的空串可变全局**——`METADATA_PATH = ""`、`HF_CACHE_DIR = ""`、
  `DOWNLOAD_DANBOORU_METADATA = True`（`danbooru.py:40-42`）。通过模块变量赋值在运行时
  注入配置，不可审计、不可重入，是最该消灭的反模式。
- **(c) 复制了类型化真相源的集合（DERIVE）**——只有 2 个，且真正"一行改完零风险"的仅
  `SUBJECT_PROMPT_TAGS`。另一个 `DEFAULT_ARTIFACT_FIELDS` 是子集、需先给源结构加标记才能干净派生。

诚实说：模型维度（Janus/NextStep）、`*_FAMILY_CAPABILITY` 工厂单例、`*_KEY` 序列化 schema、
路径锚点、checkpoint 文件名、COCO-WholeBody/MANO 关键点索引、`MEDIA_TYPES`/`FORBIDDEN_TRAJECTORY_METRICS`
等小型协议集——**全部合理，不要碰**。

---

## 2. 判据 (the rubric we applied)

沿用 AGENTS.md "架构卫生" 条款，对每个常量问两个问题：**它代表一个真实边界吗？它是否重复了
一个已有的真相源？**

**KEEP 的合法类别：**
- `schema_key` — schema 键名、序列化字段名、协议值（`runtime_role`、Hydra 的 `_self_`）
- `env_var` — 环境变量名（`VRL_DATA_ROOT`）
- `file_name` — checkpoint / 磁盘文件名（`checkpoint.pt`）、schema version 号
- `model_dim` — 模型架构维度（token 数、codebook 大小、patch 尺寸）
- `protocol_name` — 协议 / 接口 / CLI 命令名边界（family 名、`COMMAND_NAME`）
- `computed_singleton` — 工厂调用产出的模块级值（`X = make_capability(...)`），非裸硬编码
- `path_constant` — `Path(__file__).resolve().parents[N]` 锚点
- `isolated_taxonomy` — **没有**类型化真相源、被刻意隔离的分类/denylist 表（COCO-WholeBody 索引、
  MANO 骨骼链、引擎指标 denylist），外部规范即唯一真相源，无腐烂风险
- `test_fixture` — 测试 / 一次性探针夹具常量

**SMELL（需处理）的类别：**
- `duplicates_typed_structure` → **DERIVE**：手写集合/元组复制了某 dataclass/Enum/Literal 的内容，
  源增删字段时副本静默失配
- `domain_taxonomy` / `prompt_template` → **MOVE**：大业务词表、prompt 文本混进工作流代码
- 可变模块全局 / 可调旋钮 → **PROMOTE_TO_CONFIG**
- 跨文件硬编码重复 → **CONSOLIDATE**

判据核心：**类型化结构是单一真相源，能干净派生的手写副本应消解为一次派生**——但前提是
源里能**无歧义地选出**那个子集；选不出来（如"哪些字段是 artifact"无标记）就不是 DERIVE，
强行 `if name in (...)` 反而是循环废话。

---

## 3. 分类总览

| 动作 | 数量 | 涉及文件 |
|---|---|---|
| **KEEP** | 107 | `vrl/config/*`、`vrl/models/ar/*`、`vrl/models/diffusion/*/runtime.py`、`vrl/models/interfaces/runtime.py`、`vrl/models/replay_loading.py`、`vrl/trajectory/validation.py`、`vrl/trainers/checkpointing.py`、`vrl/rewards/inference.py`、`vrl/rollouts/families/registry.py`、`vrl/scripts/data/*`（路径/默认/schema 键部分）、`vrl/scripts/eval/*`、`vrl/scripts/perf/*` |
| **DERIVE** | 2 | `vrl/scripts/data/danbooru.py:90`（`SUBJECT_PROMPT_TAGS`，干净）、`vrl/trainers/data/artifacts.py:14`（`DEFAULT_ARTIFACT_FIELDS`，需先加标记） |
| **MOVE** | 16 | `vrl/models/ar/janus_pro/model.py:62/65`（2 个 R1 prompt）、`vrl/scripts/data/danbooru.py`（14 个 tag 词表，含 `HAND_FOCUS_BUCKET_TAGS`） |
| **PROMOTE_TO_CONFIG** | 6 | `vrl/scripts/data/danbooru.py`（`METADATA_PATH`、`HF_CACHE_DIR`、`DOWNLOAD_DANBOORU_METADATA`、`DEFAULT_BUCKET_WEIGHTS`、`SAFETY_TARGET_RATINGS`）、`vrl/scripts/diffusion/cosmos/anima/generate.py:29`（`DEFAULT_NEGATIVE_PROMPT`，低优先级） |
| **CONSOLIDATE** | 2 | `vrl/scripts/data/setup.py:24-28`（`_ARTIFACT_DIRS`）、provenance 字段三处重复（`artifacts.py:16` ↔ `config/validation.py:241/296`） |

---

## 4. 必须处理项 (action items)

按优先级排序：**先做 danbooru 词表抽取（最大收益）与 2 个 DERIVE（低风险），再做空串全局，最后散件。**

### P0 — danbooru.py 领域词表整体抽取（最大单点收益）

`vrl/scripts/data/danbooru.py` 把十几个手工 anime/safety tag 词表直接塞进工作流代码，正是
AGENTS.md 的"大业务词表 / 领域分类表混进工作流代码"。这些是非工程师也需审阅/更新的过滤规则，
混在代码里既难维护、又让 import 这个脚本就拖进一大坨词表。应整体抽到
**`datasets/danbooru/config.yaml`**，模块初始化时 load。

**MOVE 到 `datasets/danbooru/config.yaml`（14 个词表，每个对应一个 YAML 小节）：**

| 常量 | 行号 | YAML key | 备注 |
|---|---|---|---|
| `EXCLUDE_TAGS` | 76-86 | `anatomy.exclude_tags` | |
| `HAND_FOCUS_ALLOWED_EXCLUDE_TAGS` | 87 | `anatomy.hand_focus_allowed_exclude_tags` | |
| `SUBJECT_TAGS` | 89 | `anatomy.subject_tags` | dict |
| `POSE_TAGS` | 91-110 | `anatomy.pose_tags` | dict, 17 项 |
| `HAND_TAGS` | 111-131 | `anatomy.hand_tags` | 21 项 |
| `HAND_FOCUS_BUCKET_TAGS` | 132-143 | `anatomy.hand_focus_bucket_tags` | **是 `HAND_TAGS` 的高聚焦子集**，见下方说明 |
| `FEET_TAGS` | 144 | `anatomy.feet_tags` | |
| `ARM_TAGS` | 145-154 | `anatomy.arm_tags` | |
| `CLOTHING_TAGS` | 155-165 | `anatomy.clothing_tags` | |
| `SCENE_TAGS` | 166-175 | `anatomy.scene_tags` | |
| `ACTION_BUCKET_TAGS` | 176-194 | `anatomy.action_bucket_tags` | |
| `PROMPT_ANCHOR_TAGS` | 208-273 | `anatomy.prompt_anchor_tags` | 70+ 项 |
| `FAILURE_LABELS` | 274-287 | `anatomy.failure_labels` | |
| `SAFETY_EXCLUDED_TAGS` / `SAFETY_RISK_TAGS` | 302-313 / 314-371 | `safety.excluded_tags` / `safety.risk_tags` | risk 59 项 |

> **`HAND_FOCUS_BUCKET_TAGS` 的并行重复**：它 11 个 tag 里大量与 `HAND_TAGS` 重叠
> （`hand_focus`、`spread_fingers`、`open_hand`、`clenched_hand(s)`、`interlocked_fingers`、
> `finger_gun`、`peace_sign`、`v`、`salute` 都在 `HAND_TAGS` 里）。这是"两套手部 tag 表并行
> 维护、改一处忘另一处就静默不一致"。但它**不是纯派生**——是个语义子集（"高聚焦手部姿态"），
> 无法机械地从 `HAND_TAGS` 算出。正确做法：搬 config 时做成显式的 `hand_focus_bucket_tags`
> 小节，并在注释/schema 里写明它是 `hand_tags` 的子集关系；不要在代码里同时维护两张硬表。

**同时 PROMOTE_TO_CONFIG（可调旋钮，2 个）：**
- `DEFAULT_BUCKET_WEIGHTS`（195-207）→ `anatomy.default_bucket_weights`（实验会想调桶配额）
- `SAFETY_TARGET_RATINGS`（301）→ `safety.target_ratings`（实验会想纳入 `sensitive`）

**保留在代码内（不要误伤）：** `SAFETY_RATING_ALIASES`（290-300，输入归一化的 schema 协议，
非旋钮）、`POSITIVE_IMAGE_SOURCE`、`TEMPLATE_ID`、`SAFETY_TEMPLATE_ID`、`DOMAIN`（schema/版本
标识符），以及所有 `*_LIMIT` / `*_MIN_SCORE` / `*_SEED`（流水线不变量，仅作函数默认值）。

风险：medium——需引入 config loader 并改 14+ 处引用。建议一次性 PR + 回归 danbooru 构建脚本。

### P1 — DERIVE：从真相源派生，消除静默腐烂（低风险）

只有 2 个真正的 `duplicates_typed_structure`，其中只有第 1 个是"一行改完、零风险"。

**1. `SUBJECT_PROMPT_TAGS`** — `vrl/scripts/data/danbooru.py:90`（**干净 DERIVE，risk: low**）
```python
SUBJECT_TAGS = {"1girl": "single anime girl", "1boy": "single anime boy"}  # :89
SUBJECT_PROMPT_TAGS = ("1girl", "1boy")                                     # :90  ← 手写了上面的 keys
```
- 问题：下面这行逐字复制了上面 dict 的 keys，更新 `SUBJECT_TAGS` 而忘了同步就不一致。
- 修复：`SUBJECT_PROMPT_TAGS = tuple(SUBJECT_TAGS)`（dict 插入序稳定，结果完全等价）。
  若 `SUBJECT_TAGS` 随 P0 搬到 config，先 load 再 `tuple(...)` 派生。
- 风险：**low**（dict 顶上一行就是源，无导入顺序问题）。

**2. `DEFAULT_ARTIFACT_FIELDS`** — `vrl/trainers/data/artifacts.py:14`（**DERIVE 需先加标记，risk: medium**）
```python
DEFAULT_ARTIFACT_FIELDS = ("reference_image", "reference_video", "references")
```
- 问题：这 3 个是 `PromptExample`（`vrl/trainers/data/prompts.py:15-26`，共 8 个字段）里**承载
  外部文件路径的子集**，校验逻辑用 `getattr(example, field)` 遍历它们。`PromptExample` 新增一个
  artifact 字段时，这张表不会自动包含它。
- **不能直接 `tuple(f.name for f in fields(PromptExample))`**——那会把 `prompt`/`target_text`/
  `task_type`/`metadata` 也算进去，语义错误。`PromptExample` 里也**没有**标记区分"哪些字段是
  artifact"，所以**当前无法干净派生**。
- 真正的修复（让它成为可派生的真相源）：给 `PromptExample` 的 3 个 reference 字段加 metadata 标记，
  再从标记派生：
  ```python
  # prompts.py
  reference_image: str = field(default="", metadata={"artifact": True})
  reference_video: str = field(default="", metadata={"artifact": True})
  references: list[str] = field(default_factory=list, metadata={"artifact": True})
  # artifacts.py
  DEFAULT_ARTIFACT_FIELDS = tuple(
      f.name for f in fields(PromptExample) if f.metadata.get("artifact")
  )
  ```
  这样新增 artifact 字段只需在源处打标记，副本自动跟随。
- 备选：若不愿动 `PromptExample`，就**保留**这个常量，但加注释"curated subset of PromptExample
  artifact-bearing fields"明示这是刻意选择而非疏漏。**不要**写
  `tuple(f.name for f in fields(PromptExample) if f.name in (那三个字面量))`——那等于没消除任何重复。
- 风险：medium（动 dataclass 字段定义 + 求值顺序敏感）。

### P2 — 空串可变全局：配置注入反模式（3 个，PROMOTE_TO_CONFIG）

`vrl/scripts/data/danbooru.py:40-42`：
```python
METADATA_PATH = ""
DOWNLOAD_DANBOORU_METADATA = True
HF_CACHE_DIR = ""
```
- 问题：模块级可变全局被当作运行时配置注入点（`METADATA_PATH` 在 `382` 等处被读，`HF_CACHE_DIR`
  在 `386`，`DOWNLOAD_DANBOORU_METADATA` 在 `383`）。违反"显式函数参数"原则——配置经模块变量
  赋值传播，不可审计、不可重入、并发不安全。
- 修复：删掉这三个全局，改为构建函数的显式参数（`metadata: str | None = None`、
  `hf_cache_dir: str | None = None`、`download_metadata: bool = False`），沿调用链下传；
  CLI/调用方显式传值，`metadata=METADATA_PATH` 改为 `metadata=metadata`。
- 风险：**high**（调用链较深，需逐处改 + 回归脚本）。排在 P0/P1 之后，单独 PR。

### P3 — 散件

> **口径已修订（2026-06-13，见附录 B）**：下方原把 R1 prompts 判 MOVE→yaml、`_ARTIFACT_DIRS` 判
> CONSOLIDATE→动态化。实读源码后**全部撤销**——R1 prompts 是 byte-sensitive 协议文本（KEEP+注释），
> `_ARTIFACT_DIRS` 顶撞模块 "no framework" docstring（KEEP+可选改名）。最终口径以**附录 B**为准。

**`JANUS_R1_SELFCHECK_PROMPT` / `JANUS_R1_REGEN_PROMPT`** — `vrl/models/ar/janus_pro/model.py:62,65`
（MOVE，medium）
- 问题（`prompt_template`）：R1 生成流水线的 prompt 文本硬编码在工作流代码里（`:656/817` 使用）。
- 修复：搬到 `configs/model/ar/janus_pro/r1_prompts.yaml`（`{selfcheck_prompt, regen_prompt}`），
  在 `JanusProConfig` 初始化时加载。注意：搬的是**配置资产**（yaml），不是新建 Python lean 模块。

**`_ARTIFACT_DIRS`** — `vrl/scripts/data/setup.py:24-28`（CONSOLIDATE，medium）
```python
_ARTIFACT_DIRS = {
    "pickapic": ("pickapic",),
    "anime": ("danbooru/images", "danbooru/hand_crops"),
    "video-world": ("video_world/references", "video_world/source_videos"),
}
```
- 问题：硬编码 dict 把数据集名映射到 artifact 目录，与各脚本自己的 `COMMAND_NAME`/`DATASET_NAME`
  及路径常量重复，且命名不一致——key `"video-world"` ≠ `video_world.COMMAND_NAME="video-world-bridge"`
  ≠ `DATASET_NAME="video_world"`。`if args.dataset == "pickapic"`（`:40`）又硬编码了一次。脚本改名时
  `setup.py` 静默腐烂。
- 修复：让每个数据集模块导出 `artifact_dirs()`（或一个 `(name, dirs)` 元数据），`setup.py` 运行时
  动态调用构建 registry；条件分支改为查询模块元数据而非硬编码字符串。

**provenance 字段三处重复** — `vrl/trainers/data/artifacts.py:16` ↔ `vrl/config/validation.py:241,296`
（CONSOLIDATE，medium，**谨慎**）
- 问题：三处各自手写了一份重叠的 source-provenance 字段集：
  - `SOURCE_BACKED_VIDEO_WORLD_METADATA_FIELDS`（artifacts.py，Video2World manifest）：
    `source, source_repo, source_split, source_episode, source_video, source_frame_index, decode_method, conditioning`
  - `validation.py:241` 内联 `required_metadata`（Image2Video manifest）：
    `source_repo, source_video_url, source_frame_index, decode_method, conditioning`
  - `validation.py:296` 内联 `required_keys`（source_report payload）：
    `dataset, source_repo, source_csv, source_split, decode_method, ...`
  共享子词汇（`source_repo`、`source_frame_index`、`decode_method`、`conditioning`）在三处并行维护。
- **注意**：三套校验的是**不同 schema**（video-world manifest / i2v manifest / source-report），
  **不能整体合并**——硬合会把不属于某 schema 的字段也强加进去。正确做法是抽一个共享基集
  `_PROVENANCE_BASE_FIELDS = ("source_repo", "source_frame_index", "decode_method", "conditioning")`，
  各处 `(*_PROVENANCE_BASE_FIELDS, ...本 schema 特有字段)` 扩展，保留差异。
- 风险：medium。收益中等（消除共享子集漂移），优先级低于 P0/P1。

**`DEFAULT_NEGATIVE_PROMPT`** — `vrl/scripts/diffusion/cosmos/anima/generate.py:29`
（PROMOTE_TO_CONFIG，low，可选）
- 已有 `--negative-prompt` CLI override（`:79-82`），仅默认值硬编码。可搬到
  `configs/inference/anima_defaults.yaml`。**最低优先级**——这是个偏一次性的 generate 脚本，
  按一次性产物生命周期规则，不动也行。

---

## 5. 保留项 (KEEP) — 按类别说明为什么留

107 个保留项不逐一列举，按边界类型分桶：

**schema key / 序列化字段名（必须匹配外部契约，不可派生掉）**
- `vrl/models/replay_loading.py:20-24` 全部 `*_KEY`（`RUNTIME_ROLE_KEY` 等）1:1 映射
  `ReplayModuleLoadingProfile` 字段，是内部 dataclass 与外部 metadata dict 的序列化协议边界。
- `FULL_GENERATION_RUNTIME_ROLE` / `MINIMAL_REPLAY_RUNTIME_ROLE`（`:17-18`）绑定 `RuntimeRole` Literal。
- `MEMORY_POLICY_METADATA_KEY`、`MODEL_MEMORY_SECTIONS`、`TORCH_COMPILE_MODEL_KEY`
  （`vrl/models/interfaces/runtime.py`）协调多个独立消费者的词汇，是真实单一真相源。
- `_SELF_`（`vrl/config/loading.py:11`，Hydra 协议值）、`OPEN`（`vrl/config/unknown_keys.py:38`，
  `object()` sentinel）——协议常量，无法派生。

**模型架构维度（上游 checkpoint 不变量）**
- `JANUS_IMAGE_TOKEN_NUM/VOCAB_SIZE/PATCH_SIZE`、`NEXTSTEP_DEFAULT_TOKEN_NUM/DIM/PIXEL_SIZE`——
  token 数、codebook 大小、patch 尺寸。`JANUS_IMAGE_PIXEL_SIZE` 是透明派生（`computed_singleton`），
  保留以文档化关系。

**computed singleton（工厂调用产出，非裸硬编码）**
- 全部 `*_FAMILY_CAPABILITY`（janus/nextstep/cosmos/sd3_5/wan runtime）由 `*_family_capability(...)`
  工厂在模块加载时算出并缓存。`FAMILY_REGISTRY`（`registry.py:80`）是运行时填充的标准 registry 模式。

**小型协议集 / denylist（无类型化真相源或为刻意子集，无腐烂风险）**
- `MEDIA_TYPES`（`rewards/inference.py:14`）：注释明写它**就是**单一真相源，磁盘 artifact store
  反向 import 它——不是重复。
- `FORBIDDEN_TRAJECTORY_METRICS`（`trajectory/validation.py:18`）：是 `GenerationMetrics`（11 字段）里
  **刻意精选的 4 个引擎遥测字段** denylist（不含 `num_prompts`/`num_samples`），**不能**从
  `fields(GenerationMetrics)` 派生（会误禁合法字段）；唯一消费者就在同文件 `:98`。是小型协议
  denylist，**保留在 validation.py**（搬到新文件会违反本项目 "no new lean files" / "no single-caller
  helpers" 约定）。可加一行注释说明它是 `GenerationMetrics` 遥测字段的精选子集。
- `REQUIRED_TRAINABLE_ROLES` / `SINGLETON_TENSOR_ROLES`（`validation.py:29-30`）：后者已是
  `frozenset(REQUIRED_TRAINABLE_ROLES)` 的派生——**本就是正确写法**，是单一来源。

**路径锚点（`Path(__file__).resolve().parents[N]`）**
- `REPO_ROOT`、`CONFIGS_ROOT`、danbooru 的 `OUTPUT_DIR`/`ANATOMY_DIR`/`SAFETY_DIR` 及派生输出路径。

**file_name / checkpoint 名 / 环境变量（磁盘 & shell 契约）**
- `CHECKPOINT_SCHEMA_VERSION`、`TRAINING_CHECKPOINT_NAME`、`LORA_WEIGHTS_NAME`、`CHECKPOINT_META_NAME`、
  `DANBOORU_METADATA_FILE`；`DATA_ROOT_ENV`（`VRL_DATA_ROOT`）是 shell↔Python 环境变量契约。

**外部规范 taxonomy（外部标准即唯一真相源）**
- `BODY_KP_RANGE`、`LHAND/RHAND/HAND_KP_RANGE`、`LIMB_SYMMETRY_PAIRS`、`FINGER_CHAINS`
  （`anime_probe_common.py`）是 COCO-WholeBody / MANO 外部标准索引，仓库内无 Enum/dataclass 重复。
- `JANUS_R1_SEGMENTS`、`SOURCE_BACKED_VIDEO_WORLD_METADATA_FIELDS`（后者见 P3 的共享子词汇说明，
  但其作为 Video2World schema 的完整字段集本身保留）。

**protocol_name / CLI 命令名 + test_fixture**
- 各数据脚本 `COMMAND_NAME`（pickapic/videophy/video_world）、danbooru `ANIME_*_COMMAND`——CLI 路由边界。
- `IMAGE_SUFFIXES`、`DIFFUSION_COUNTER_PREFIX`、`METRICS`（gpm_sampler 的 CSV schema）——协议/夹具。
- eval 探针的 `PROMPTS`、`NEGATIVE_PROMPT`、`DEFAULT_MODEL`、`CONF_THRESHOLD`——一次性探针夹具（见非目标）。

---

## 6. 跨文件模式

**(1) `*_FAMILY_CAPABILITY` + family 字符串重复（diffusion runtime ↔ registry）**
family 名（`cosmos-predict2`、`cosmos-predict2.5`、`wan_2_1`、`sd3_5`、`cosmos-predict2-anima`）在
`ChunkExecutor.family` 类属性、`*_FAMILY_CAPABILITY` 单例、`vrl/rollouts/families/registry.py` 之间
重复。registry 用**字符串懒加载**（`executor_cls="vrl.models...anima.runtime:AnimaChunkExecutor"`，
`registry.py:219`）刻意**不** import 这些 runtime 模块。
- **判断：保持现状，不强抽**。只有 anima 把 family 名抽成了模块常量 `ANIMA_FAMILY`
  （`anima/runtime.py:29`），其余都内联在 capability 工厂调用里。让 runtime 反向 import registry 去
  读 family 名会**倒置层依赖**（model 层依赖 rollout 层），破坏 registry 懒加载的解耦初衷。registry
  是 family 的权威清单，model 侧的字符串视为可接受的协议常量重复。强抽得不偿失。

**(2) `*_KEY` ↔ dataclass 字段（replay_loading）**
全部 `*_KEY` 与 `ReplayModuleLoadingProfile` 字段一一对应。理论上可用 `fields()` 派生，但**保留**：
它们是序列化协议边界，显式命名比间接派生更清晰，且已进 `__all__` 属公共契约。这是"一致性优先于
行数削减"的正面案例。

**(3) `COMMAND_NAME`/`DATASET_NAME` per-file 模式 ↔ `setup.py` 硬编码 registry**
各数据脚本声明 `COMMAND_NAME`/`DATASET_NAME` 是好的（每文件协议边界），但 `setup.py:24-28` 的
`_ARTIFACT_DIRS` 与 `if args.dataset == "pickapic"` 又把这些语义硬编码了一遍且命名不一致——**这是
需要 CONSOLIDATE 的真实重复**（见 P3）；但消费侧的每文件常量本身保留。

**(4) provenance 字段三处重复** — 见 P3，已单列。

**(5) eval 探针 `PROMPTS` 跨文件镜像**
`wan_i2v_base_sample.py` 与 `wan_phys_ab_sample.py` 的 5 条 prompt 逐字相同，但注释明写
"Same 5 prompts"——是为 A/B 对比刻意镜像的夹具，不是共享源。**保留**，两个都是一次性探针。

---

## 7. 执行顺序 / 非目标

**执行顺序（按收益与风险排）：**

1. **P0 — danbooru 词表抽取**：建 `datasets/danbooru/config.yaml`，搬 14 个词表 + 2 个旋钮
   （`DEFAULT_BUCKET_WEIGHTS`、`SAFETY_TARGET_RATINGS`），加 loader，改引用。一次性 PR，回归 danbooru
   构建脚本。收益最大（清掉单文件最大的硬编码倾倒）。
2. **P1 — 2 个 DERIVE**：`SUBJECT_PROMPT_TAGS = tuple(SUBJECT_TAGS)`（零风险，可并入 P0）；
   `DEFAULT_ARTIFACT_FIELDS` 给 `PromptExample` 加 `field(metadata={"artifact": True})` 后派生。
   改完跑 `pytest tests/trainers/data/` + config-resolve 验证零回归。
3. **P2 — danbooru 空串可变全局**：删 `METADATA_PATH`/`HF_CACHE_DIR`/`DOWNLOAD_DANBOORU_METADATA`，
   改显式参数下传。风险最高，单独 PR。
4. **P3 — 散件**：Janus R1 prompts → yaml；`setup.py` `_ARTIFACT_DIRS` 动态化；provenance 共享子集
   抽 `_PROVENANCE_BASE_FIELDS`；`DEFAULT_NEGATIVE_PROMPT`（最后，可选）。

**非目标（明确不做）：**

- **不搅动合法边界**：107 个 KEEP 项一律不动——schema key、模型维度、computed singleton、路径锚点、
  checkpoint 文件名、COCO-WholeBody/MANO 索引、`MEDIA_TYPES`/`FORBIDDEN_TRAJECTORY_METRICS` 等小型
  协议集。它们不是"行数可省的样板"，是真实契约。
- **不把 `FORBIDDEN_TRAJECTORY_METRICS` 搬去新文件**：它是 4 元素 denylist、唯一消费者同文件，
  搬出会违反 "no new lean files" / "no single-caller helpers"。保留在 validation.py。
- **不强抽 diffusion family 字符串**：除已成常量的 `ANIMA_FAMILY`（也不建议为它单独发 PR）外，
  其余 family 字符串保持现状以维持跨家族一致形状 + registry 懒加载解耦；不为省几行破坏 grep 一致性。
- **不把 `replay_loading.py` 的 `*_KEY` 改成 `fields()` 派生**：显式命名的序列化边界比间接派生更清晰。
- **不碰一次性 eval 探针常量**：`wan_*_sample.py` 的 `PROMPTS`/`NEGATIVE_PROMPT`/`DEFAULT_MODEL`、
  `anime_probe_common.py` 的 `CONF_THRESHOLD`——按一次性产物生命周期规则，探针价值是它产出的答案。
- **不为清理而清理**：本 sprint 只动有明确腐烂风险或边界违规的常量，其余视为正确。

---

## 8. 校验记录（我对自动审计结果的独立复核与修正）

8 个 agent 的自动审计整体可靠，但我逐个复读源码后**修正了 3 处误判**，记录于此以便复核：

1. **`FORBIDDEN_TRAJECTORY_METRICS`**：自动审计判 `MOVE → 新建 metrics_policy.py + 从 GenerationMetrics
   派生`。复核 `vrl/generation/types.py` 发现它是 11 个字段里**精选的 4 个**（不含
   `num_prompts`/`num_samples`），无法派生；且唯一消费者在同文件 `:98`，新建模块违反项目约定。
   **改判 KEEP**。
2. **`DEFAULT_ARTIFACT_FIELDS`**：自动审计判 `DERIVE` 且给的派生式
   `tuple(f.name for f in fields(PromptExample) if f.name in (那三个字面量))` 是**循环废话**（仍硬编码
   那三个字面量）。复核 `prompts.py` 确认是 8 字段里的子集、无标记。**改为"需先给源加 field metadata
   标记才能干净派生，否则 KEEP"**。
3. **provenance 字段重复**：自动审计把 `SOURCE_BACKED_VIDEO_WORLD_METADATA_FIELDS` 判为
   `isolated_taxonomy KEEP`。我 grep 出 `config/validation.py:241,296` 另有两处重叠的手写 provenance
   集——**新增一条 CONSOLIDATE finding**（但因三套 schema 不同，只建议抽共享子集，不整体合并）。

其余确认无误的关键判断：`SUBJECT_PROMPT_TAGS` 是干净 DERIVE；`MEDIA_TYPES`/`SINGLETON_TENSOR_ROLES`
注释已明确是单一来源/已派生（KEEP）；danbooru 空串可变全局确属配置注入反模式。

---

## 附录 A：dict 型 ALL_CAPS 专项审计（必保留清单 + 删除判据）

> 用 AST 扫 `vrl/` 全部模块级 dict 型 ALL_CAPS（比正文的 `^[A-Z]` grep 更全——还抓到下划线前缀的
> 模块私有 dict）。共 **15 个**：**12 个必须保留**，**3 个**（danbooru 领域数据）已随 P0 迁出。
> 行号为审计时快照；danbooru 三项迁出后行号已变（见 A.2）。

### A.1 必须保留的 12 个（按边界类型）

| dict | 路径 | 为什么是真实边界 |
|---|---|---|
| `FAMILY_REGISTRY` | `vrl/rollouts/families/registry.py:80` | 运行时注册填充的 registry（空 `{}` 起步） |
| `_REWARD_REGISTRY` | `vrl/rewards/functions/registry.py:16` | 同上，`_register_builtins()` 填充 |
| `_FAMILY_ALIASES` | `vrl/rollouts/families/registry.py:305` | **已经是派生的**——`{alias: family for ... in FAMILY_REGISTRY}`，正是该有的写法 |
| `_CANONICAL_STRING` | `vrl/models/dtypes.py:21` | 容错拼写→规范 dtype 串归一化表；正反向单一真相源；详见 A.3 |
| `SAFETY_RATING_ALIASES` | `vrl/scripts/data/danbooru.py`（safety 区） | 解析 danbooru 外部 `g/s/q/e` 评级码；数据格式契约，详见 A.3 |
| `_MODEL_BY_TASK` | `vrl/models/diffusion/wan_2_1/runtime.py:53` | task→model 类的 dispatch 表（字符串懒加载路径） |
| `_SCORE_KEY_MAP` | `vrl/rewards/models/kling_video_reward.py:32` | 内部 key→Kling 模型维度码的外部 API 字段映射 |
| `_METADATA_PREFIX` | `vrl/models/diffusion/common/vae_decode_memory.py:76` | 注释明写 "downstream contract（bundle metadata + tests）"，schema 契约 |
| `_JANUS_LORA_DEFAULTS` | `vrl/models/ar/janus_pro/runtime.py:141` | LoRA 配方默认值，config `model.lora` 可覆盖（已有 override 路径） |
| `_NEXTSTEP_LORA_DEFAULTS` | `vrl/models/ar/nextstep_1/runtime.py:134` | 同上 |
| `_ARTIFACT_DIRS` | `vrl/scripts/data/setup.py:24` | init-dirs 命令的本地小 wiring 表（P3 改判 KEEP，见 §0/§4） |
| `_DIMENSION_DESCRIPTIONS` | `vrl/rewards/models/kling_video_reward.py:39` | 唯一边界项：是 prose 描述文本，但与 Kling 模型 prompt 格式耦合——倾向保留；若想搬只有它够格 |

### A.2 已迁出的 3 个（danbooru 领域数据，P0 完成）

| dict | 原路径 | 性质 | 去向 |
|---|---|---|---|
| `SUBJECT_TAGS` | `danbooru.py:89` | anime 领域词表 | `datasets/danbooru/config.yaml`，`dict(_ANATOMY["subject_tags"])` 加载 |
| `POSE_TAGS` | `danbooru.py:91`（18 项） | 同上 | 同上（`pose_tags`） |
| `DEFAULT_BUCKET_WEIGHTS` | `danbooru.py:195`（11 项） | 可调采样权重旋钮 | 同上（`default_bucket_weights`） |

### A.3 两个 alias 表的核验（实读 + grep 实测，不是凭注释）

两个看似"只为校验正确性、可删"的 alias 表，实测都在**解析外部输入**，不可删：

**`_CANONICAL_STRING`（dtypes.py）— 表必须留，但有 4 个死条目：**
- `resolve_torch_dtype` 拿它解析 dtype，**表里没有就 `ValueError`**。
- config 实测：`bf16`（alias）被用 **7 次**（含 `configs/base/actor.yaml`），`bfloat16`/`float32` 共 5 次 → 表 load-bearing，删不得。
- 但 `fp16`/`half`/`fp32`/`float` 这 4 个 alias **零 config 使用** = dead tolerance。技术上可删，但它们是 ML 常见简写、留着是 4 行免费容错；删了将来谁写 `fp16` 就吃莫名 `ValueError`。**建议保留**（严格输入 vs 免费容错的取舍，非正确性问题）。

**`SAFETY_RATING_ALIASES`（danbooru）— 不能删，是外部数据格式解析器：**
- Danbooru2023 原始 metadata 把分级写成单字母 `g/s/q/e`；`_record_rating` → `_normalize_rating` 用本表把它们→规范名。删单字母条目，**每条记录 rating 变 unknown，safety 数据集 build 报废**。
- identity 条目（`questionable→questionable`）也 load-bearing：`SAFETY_TARGET_RATINGS` 也要过 `_normalize_rating`，没 identity 会被判无效直接 `raise`。

### A.4 判据：硬编码 dict 何时能删

只有满足以下之一才删得：
1. **可派生** —— 信息在别处有类型化来源（如 `_FAMILY_ALIASES` 从 `FAMILY_REGISTRY` 派生）。
2. **零引用** —— grep 全仓没人用（如那 4 个 dtype alias）。

否则删不得：它在编码**某处唯一存在的信息**——外部数据格式（danbooru `g/s/q/e`、Kling API 字段）、
真实输入容错（`bf16`）、或默认值本身（LoRA 配方）。这三类不是"图方便"，是"信息只此一份"。
`defaults` 类（`_JANUS_LORA_DEFAULTS` 等）可以**搬进 config 换地方放**，但不能凭空删——默认值总得有出处。

---

## 附录 B：保留项的"写清楚"口径（语义源 + 派生表 + 准确命名）— 2026-06-13 定稿

附录 A 已证这些表 load-bearing、不可删；本附录回答下一个问题：**既然保留，怎么写才不被误读/误删。**
核心口径——**不是搬 YAML / 建 registry / 拆新文件，而是把硬编码写成「语义源 + 派生表 + 准确命名」**。
问题从来不是"它存在 / 在 .py 里"，而是 **flat dict 看不出哪些是外部输入、哪些是 canonical、哪些 identity
项是必须的**——读者据此误删。逐项（行号见 A.1，均已实读核对；这批替代原 §4 P3 的 MOVE/动态化方案）：

### B.1 `SAFETY_RATING_ALIASES`（danbooru.py:109）— DERIVE-in-place（保留 A.3 证的全部 load-bearing 条目）

A.3 已证：单字母 `g/s/q/e` 解析外部 Danbooru metadata、identity 条目供 `SAFETY_TARGET_RATINGS` 过
`_normalize_rating`——都不可删。改写成语义源 + 派生，把 identity 条目**改为自动生成（不是删除，
load-bearing 要求不变）**，消除"手写冗余"的错觉：

```python
_SAFETY_RATING_SPELLINGS = {
    "general": ("g", "safe"),
    "sensitive": ("s",),
    "questionable": ("q",),
    "explicit": ("e",),
}
SAFETY_RATING_ALIASES = {
    spelling: canonical
    for canonical, spellings in _SAFETY_RATING_SPELLINGS.items()
    for spelling in (canonical, *spellings)   # canonical 自身也被接受 → identity 自动生成
}
```

消费方仅 `:957 SAFETY_RATING_ALIASES.get(text)` 纯查表，顺序无关，派生前后值完全等价。

### B.2 `_CANONICAL_STRING`（dtypes.py:21）— 改名 + DERIVE（保留 A.3 的"免费容错"立场）

A.3 已证表 load-bearing，并主张保留 4 个零-config-使用的 alias（`fp16/half/fp32/float`）作免费容错。
本口径**不改这个立场**——派生形式同样保留全部 alias，只是：(a) 名字 `_CANONICAL_STRING` 没说出它是
"宽松输入解析器"（docstring 自述 "the lenient parser"），改名 `_DTYPE_NAME_BY_INPUT`；(b) identity
条目自动生成；(c) 源具名抽出，与 `_SAFETY_RATING_SPELLINGS` 对称：

```python
_DTYPE_SPELLINGS = {
    "bfloat16": ("bf16",),
    "float16": ("fp16", "half"),
    "float32": ("fp32", "float"),
}
_DTYPE_NAME_BY_INPUT = {
    spelling: canonical
    for canonical, spellings in _DTYPE_SPELLINGS.items()
    for spelling in (canonical, *spellings)
}
```

`resolve_torch_dtype` also owns the canonical-name-to-`torch.dtype` direction,
so both directions share one vocabulary. `dtype_to_wire_name` (named
`dtype_to_config_string` when this audit was written) preserves the historical
unknown-value pass-through behavior. Its output is a torch wire name such as
`bfloat16` or `float16`, not a public precision token.

### B.3 Janus R1 prompts（model.py:62,65）— 改判 KEEP（撤销原 §4 P3 的 MOVE）

保留在 Python，不进 YAML。加注释说明是 **byte-sensitive model protocol prompt**：
`<｜end▁of▁sentence｜>` 里的全角 `｜`（正是 `# noqa: RUF001` 在压的易混淆字符）**不能被编辑器规范化**。
搬 YAML 会有破坏 token 字节序、静默打断 R1 自校正循环的风险。

### B.4 `_ARTIFACT_DIRS`（setup.py:24）— KEEP（per A.1），可选改名 `_INIT_DIRS_BY_DATASET`

撤销原 §4 P3 的"动态化 / per-module `artifact_dirs()` 协议"——模块 docstring 明写
**"No generic artifact-manifest framework lives here"**，动态 registry 正是它刻意不要的框架。仅**可选改名**
让"它是 `init-dirs` CLI 的目录表、非通用 registry"一目了然。`args.dataset == "pickapic"` 的 HF cache
特例是 pickapic 专属，保留。

### B.5 `DEFAULT_NEGATIVE_PROMPT`（generate.py:29）— 改名降噪，非架构改造

只有一个 argparse default 用途、且已有 `--negative-prompt` 覆盖。改成 `_DEFAULT_NEGATIVE_PROMPT`
（下划线标 private 默认），保持 `add_argument` 调用可读；不进 config。

### B.6 `_JANUS_LORA_DEFAULTS` / `_NEXTSTEP_LORA_DEFAULTS`（*/runtime.py:141/134）— 改名最弱 + 一处真 smell

A.1 判 KEEP（默认值有出处）。改名 `_*_RUNTIME_LORA_DEFAULTS` **收益弱**——它们本就在 `runtime.py`，
路径已带 "runtime"，再塞 `_RUNTIME_` 与路径重复。**更值得记的相邻发现**：两个 dict 内容**逐字节相同**
（rank32 / alpha64 / q,v_proj / dropout0 / gaussian），却**解析路径不一致**——Janus 走共享 helper
`_resolve_lora_block(build, defaults)`（:162），NextStep 走内联 `dict(defaults)+update`（:155-156）。
统一两边 lora 解析路径是**单独 follow-up**，不混进本 naming PR。

### B.7 落地与非目标

全是单文件、纯改名/派生、**零行为变化**，打**一个小 PR**。硬约束：删手写表前加一次性断言确认
`SAFETY_RATING_ALIASES` / `_DTYPE_NAME_BY_INPUT` 派生前后 dict 完全相等（保护 A.3 证的全部 load-bearing
条目）。**非目标**：B 不删任何 alias（含 A.3 的 4 个 dtype 免费容错）、不搬任何表去 YAML/config（除非将来
defaults 真要外置）、不动已判 KEEP 的边界项。

---

## 关键文件引用

- `vrl/scripts/data/danbooru.py:40-42`（空串可变全局）、`:76-371`（领域词表）、`:90/132-143`（DERIVE/子集）
- `vrl/trainers/data/artifacts.py:14`（`DEFAULT_ARTIFACT_FIELDS`），源 `vrl/trainers/data/prompts.py:15-26`
- `vrl/trajectory/validation.py:18,98`（`FORBIDDEN_TRAJECTORY_METRICS`），源 `vrl/generation/types.py`（`GenerationMetrics`）
- `vrl/config/validation.py:241,296` ↔ `vrl/trainers/data/artifacts.py:16`（provenance 重复）
- `vrl/models/ar/janus_pro/model.py:62,65`（R1 prompts）
- `vrl/scripts/data/setup.py:24-28`（`_ARTIFACT_DIRS`）
- `vrl/models/diffusion/cosmos/anima/runtime.py:29` ↔ `vrl/rollouts/families/registry.py:216`（family 字符串）
- `vrl/models/replay_loading.py:20-24`、`vrl/models/interfaces/runtime.py:15-30`（KEEP 示例）
- `vrl/rewards/inference.py:14`（`MEDIA_TYPES`，KEEP 示例）
