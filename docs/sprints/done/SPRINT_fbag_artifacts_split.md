# SPRINT: 拆分 trainers/data/artifacts.py（唯一的真 grab-bag）

状态：done（2026-07-10）。父记录：`SPRINT_fbag_00_overview.md`（同在 `done/`）。

> 这是全仓 22 个审计文件里**唯一**被定为 `grab-bag-split-by-concern` 的文件（其余 15 内聚、
> 6 只有零散小问题）。判决经对抗性 verify 保留（refuted=0）。

## 0. 一句话

`vrl/trainers/data/artifacts.py`（451 行）把**两个零耦合的落盘契约**塞进一个文件：
① prompt-manifest 路径解析 + 溯源校验；② SFT clean-latents 张量分片存取。二者不共享任何
符号、常量、helper，SFT 部分甚至不在文件 docstring（"path resolution and manifest validation"）
的描述里。**动作只有两个,都是低风险**:把 SFT-latents 三件套搬到自己的模块;把重复的
`_coerce_data_root` 合并回它本来的 owner。manifest 核心原样不动。

## 1. 证据:两个契约零共享

**契约 A — manifest（留在原文件,内聚）:**
`resolve_prompt_example_artifacts`、`require_artifact_manifest[_pair]`、
`require_source_backed_video_world_manifest_pair`、`ResolvedArtifact`、`ArtifactManifestReport`,
以及 `_artifact_values`/`_source_episodes`/`_assert_readable`。常量
`DEFAULT_ARTIFACT_FIELDS`(已按 derive-from-source 规则从 `fields(PromptExample)` 派生)、
`IMAGE_SUFFIXES`、`SOURCE_BACKED_VIDEO_WORLD_METADATA_FIELDS`。这半是真内聚,和 docstring 相符。

**契约 B — SFT latents（~90 行,应搬出）:**
`save_sft_latents` / `load_sft_latents` / `SFT_LATENTS_SCHEMA_VERSION`（`vrl/trainers/data/artifacts.py:341-451`）。
这是 torch 张量持久化 + family/model 溯源契约,和契约 A **不共享一个符号、常量或 helper**。
它被 bolt 在文件末尾,是 grab-bag 的本体。

## 2. 已完成动作

### 2.1 搬 SFT-latents 到自己的模块

新建 `vrl/trainers/data/sft_latents.py`,移入 `save_sft_latents`、`load_sft_latents`、
`SFT_LATENTS_SCHEMA_VERSION`。更新调用点的 import。

**注意 [no_new_lean_files] 反例辨析**:用户的规矩是"沉降共享代码时落到已有文件,别新建单类文件"。
这里方向相反——不是沉降共享抽象,而是**拆分一个混了两个契约的文件**,把其中一个契约(~90 行、
3 个公开符号、独立的 schema 版本)独立成模块。这是 AGENTS.md "按关注点拆 grab-bag" 明确许可的,
不是"为省几行造薄文件"。搬之前先 grep 调用点确认落点(见 §3)。

### 2.2 合并重复的 `_coerce_data_root`（form-4 body-dup）

`vrl/trainers/data/artifacts.py:297-298` 与 `vrl/utils/artifacts.py:54-55` **byte-identical**,
且本文件**已经** `from vrl.utils.artifacts import (...)`（line 11）。做法:

- 在 `vrl/utils/artifacts.py` 把 `_coerce_data_root` 提升为公开 `coerce_data_root`(去下划线、
  加进 `__all__`)——它已是该模块 `resolve_artifact_path` 的内部依赖,提升后成为 artifact
  路径解析的单一 owner。
- 在 `trainers/data/artifacts.py` 删除本地 def,改从 import 取用。

这样 `utils/artifacts.py` 成为 data-root 解析的唯一 source of truth,消除双维护。

## 3. 验证

1. 搬 SFT-latents 前:`grep -rn "save_sft_latents\|load_sft_latents\|SFT_LATENTS_SCHEMA_VERSION" vrl/ tests/`
   —— 把所有 import 重指到新模块。
2. 合并 `_coerce_data_root` 前:确认两处 byte-identical(已确认),确认无第三处定义。
3. 改完跑 `tests/data/` + `tests/trainers/` + config-resolve 冒烟,确认零回归。

执行结果：旧 import 已清零；SFT shard round-trip / provenance 拒绝测试、在线加载测试、
相关配置解析与 Ruff 均通过。仓库级验证结果由本轮最终提交统一记录。

## 4. 明确不动

- **manifest 核心的 3 个 `_` helper 不合并**:`_artifact_values` 在 `require_artifact_manifest`
  内有 2 个调用点(lines 205,210),是可复用的字段规范化器,不是 1:1 单调用者拆分。
  `_source_episodes`/`_assert_readable` 同理是命名的校验步骤。
- **`SOURCE_BACKED_VIDEO_WORLD_METADATA_FIELDS` 不改成自动派生**:手写但注释说明 tuple 顺序
  语义(load 顺序)不可从 dataclass 派生——ALL_CAPS 规则里合法保留的一类。
- **`ResolvedArtifact` / `ArtifactManifestReport` 不当 derived-dead-field 处理**:它们是
  序列化/report 结构(`to_dict()` 被非测试脚本消费),不是 `Resolved*Capability` 控制结构,
  dead-field 规则不适用。

## 引用

- `vrl/trainers/data/artifacts.py`（拆分对象;SFT 部分 341-451,`_coerce_data_root` 297-298,import 11）
- `vrl/utils/artifacts.py:54-55`（合并目标 owner）
