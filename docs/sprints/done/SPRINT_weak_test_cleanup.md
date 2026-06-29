# SPRINT: 清掉镜像源码的弱测试，换成验证真实逻辑

状态：**done（2026-06-28，复核确认全部 12 处已处理；主体由 c613762 完成）**。目标：删除/重写一批「只断言一个写死的字面量/路径，或把生产代码的常量、字典推导、`get_args` 派生原样抄进断言再跟自己比」的测试。这类测试在构造上恒真——源码改了它只会要求你同步抄一遍，**永远抓不到回归**。把它们换成验证*行为/契约/分支*的测试。

> **复核结论（2026-06-28）**：12 处全部处理完毕。`c613762`（"test: replace weak mirror assertions"）做掉主体——9 处删/改名 + #1/#2/#10 **保名换体**重写成行为测试；#11/#12（perf_smoke schema 测试）随 `validation.py` 那套 magic-substring 校验整体退役而消失（§2.3 路线 2，`validation.py` 现已无 `rejection_reason`/smoke-marker 痕迹，仓库级 grep 为空，无未测路径残留）。三个"名字还在"的已逐一验证不再弱：
> - **#1 `test_family_registry_covers_current_rollout_families`**：改为断言每个 entry 的**结构契约**（diffusion 家族 `executor_cls`/`runtime_builder` 必须以 `vrl.models.diffusion.` 开头、非空 task/request_prefix、gatherer import_path 含 `:`），新增家族自动覆盖、错配 modality 会 fail。不再抄家族名列表。
> - **#2 `test_algorithm_config_dispatches_representative_kinds`**：改为加载 5 个真 experiment YAML 断言 `isinstance(build_algorithm_config(cfg), expected_type)`（分派**行为**），不再抄 `train_segments` 字面集合；另有 `test_algorithm_dispatch_is_stable_per_kind` 配套。
> - **#10 `test_seed_grid_is_identical_across_checkpoints`**：保留真契约（checkpoint 无关 `seed(0,2,1)==seed(3,2,1)` + 非退化 `!=seed(0,2,2)`），已删除重抄公式的数值锚点 `assert first == 17+2*4+1`。
>
> 保留清单（§3）两项原样在位。相关测试文件 101 passed。本轮无代码改动，仅本文件状态收尾。
>
> **顺带发现（不属本 sprint，已记录待办）**：裸 `pytest` 当前被 2 个 **collection error** 中断——`tests/models/interfaces/__init__.py:90` 的 `FAMILY_MODEL_CLASSES` 覆盖断言报 `missing families ['cosmos3','echo']`（你并行加的 cosmos3/echo 家族未同步进测试侧契约覆盖表）。排除这 2 个文件后 **1321 passed / 11 failed / 18 skipped**，11 个红全是 env-gap/既有/并行红（OCR 缺 `Levenshtein`、fp8、ray-substrate architecture、echo flow-policy 等），**无一来自本 sprint**。cosmos3/echo 契约覆盖是单独一件事。

## 0. 结论

全套 184 个 `test_*.py` 审计后确认 **12 处弱测试**（候选 12 中剔除 2 个误报，另补上用户点名的 2 个 `perf_smoke` 测试）。病根只有一个：**source-mirroring**——测试与被测代码共享同一份 source of truth，断言不可证伪。

| # | 文件::测试 | 类别 | 动作 |
|---|---|---|---|
| 1 | `test_family_registry.py::test_family_registry_covers_current_rollout_families` | list_mirror | rewrite |
| 2 | `test_load_all_experiments.py::test_algorithm_config_dispatches_representative_kinds` | exact_config | rewrite |
| 3 | `test_dance_grpo.py::test_trainer_config_defaults_to_strided` | exact_config | delete |
| 4 | `test_artifact_store.py::test_artifact_formats_set_is_derived_from_literal` ⭐ | tautological | delete |
| 5 | `test_ocr.py::test_text_normalization_core_cases` ⭐ | shadow-impl | rewrite |
| 6 | `test_ocr.py::test_normalized_edit_distance_core_cases` | shadow-impl | rewrite |
| 7 | `test_model_loading.py::test_kling_normalize_scores_emits_only_public_keys` | list_mirror | rewrite |
| 8 | `test_replay_loading.py::test_full_generation_bundle_declares_full_generation_modules` | tautological | merge+rewrite |
| 9 | `test_replay_loading.py::test_minimal_replay_bundle_does_not_declare_full_generation_modules` | tautological | merge+rewrite |
| 10 | `test_cosmos_predict25_kling_eval.py::test_seed_grid_is_identical_across_checkpoints` | tautological | rewrite（删数值锚点） |
| 11 | `test_schema.py::test_training_config_rejects_disallowed_dataset_manifest` | fixture_name | 见 §2.3 |
| 12 | `test_schema.py::test_training_data_path_rejection_reason_classifies_boundaries` | single_case | 见 §2.3 |

判定标准（用于本 sprint 及未来 review）：**「若有人改了源码的策略/列表，这个测试会抓到真实回归，还是只是需要同步抄一遍新列表？」后者即为弱测试。**

## 1. 证据（已核实）

### 1.1 恒真：`ARTIFACT_FORMATS`（#4，删）

源码 `vrl/rewards/artifacts.py:20`：

```python
ARTIFACT_FORMATS = frozenset(get_args(ArtifactFormat))
```

测试 `tests/rewards/inference/test_artifact_store.py`：

```python
assert frozenset(get_args(ArtifactFormat)) == ARTIFACT_FORMATS
```

把同一个派生重跑一遍跟自己比，无论 `Literal` 里放什么都不可能 fail。真正没被覆盖的逻辑在 `artifacts.py:38` 的 `if artifact_format not in ARTIFACT_FORMATS: raise`。

### 1.2 影子实现且行为对不上：OCR 归一化（#5，重写）

真源码 `vrl/rewards/models/ocr.py:29`：

```python
def _normalize_ocr_text(text: str) -> str:
    """flow_grpo-compatible normalization: lowercase, strip spaces."""
    return text.replace(" ", "").lower()  # 不去标点、不折叠内部空格
```

测试 `tests/rewards/functions/test_ocr.py` 内部重写了一个**语义不同**的归一化：

```python
def _normalize_text(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9\s]+", "", text.lower())  # 去标点
    return re.sub(r"\s+", " ", normalized).strip()           # 折叠空格
```

测试在验证一份连真实行为都对不上的假代码——比单纯「恒真」更糟。`_normalized_edit_distance`（#6）同理：Levenshtein 在测试里重写自比，真实 `OCRRewardModel` 打分从不被调用。

### 1.3 镜像注册表 key：family registry（#1，重写）

源码 `vrl/rollouts/families/registry.py:374`：

```python
def registered_rollout_families() -> tuple[str, ...]:
    """Return canonical rollout family keys."""
    return tuple(FAMILY_REGISTRY)
```

测试把这个 tuple 按精确顺序硬编码了一份副本。家族增删/换序只会来同步这行字面量，抓不到任何 dispatch 回归。

### 1.4 镜像 config 默认值（#2 #3）

- #2 `assert set(algo_cfg.train_segments) == {"initial_image", "selfcheck_text", "final_image"}` —— 是 `MultiSegmentTokenGRPOConfig` 默认 `train_segments` 的精确拷贝。
- #3 `assert _trainer_config().timestep_selection == "strided"` —— 单个 dataclass 字段的字面默认值，不过任何 loader/validation 分支。

两者都撞既定的 **no-exact-config tests** 约定（config 是声明，不该断言其字面值）。

### 1.5 镜像 `_SCORE_KEY_MAP` / metadata 字面量（#7 #8 #9）

- #7 复用源码同一个 `_SCORE_KEY_MAP` 重跑 dict 推导构造期望值；真逻辑（`kling_video_reward.py:682` 的 `if model_key in raw` 过滤分支）没被覆盖。
- #8/#9 源码各是一行 `return {LOADS_FULL_GENERATION_MODULES_KEY: True/False}`，测试用同一导入常量重述，恒真。

### 1.6 重抄公式做数值锚点（#10）

```python
assert first == 17 + 2 * 4 + 1  # 注释自承 "recomputed from the formula"
```

把 `_seed_for` 的公式在测试里重算。公式改了只会被一起改写，证不了任何东西。

## 2. 应该改什么

### 2.1 直接删（行为不值得单测，或恒真）

- **#4** `test_artifact_formats_set_is_derived_from_literal` → 删。若要保留该文件覆盖，新增一条「传入非法 format 触发 `artifacts.py:38` 的 raise」。
- **#3** `test_trainer_config_defaults_to_strided` → 删。可并入一条「`timestep_selection` 合法枚举值经 validation 接受、非法值被 raise」的逻辑测试。

### 2.2 重写为验证真实逻辑/契约

| # | 改成测什么 |
|---|---|
| 1 | 每个已注册 family 能成功 `build_*` / dispatch；或注册表与「实际存在 runner 的目录」一致（独立来源派生，**不抄 key**） |
| 2 | 给定 experiment YAML，loader 是否分派到 `MultiSegmentTokenGRPOConfig` 类型；**不断言 segment 内容** |
| 5 | `from vrl.rewards.models.ocr import _normalize_ocr_text`，按真实契约断言（如 `_normalize_ocr_text("Hello World") == "helloworld"`） |
| 6 | 调真实 scoring 路径；若内联在 `OCRRewardModel` 不可单测，先在源码暴露可测函数再断言，**不维护影子副本** |
| 7 | `raw` 缺某 model_key 时对应 public key 被丢弃；含额外未文档化键时不泄漏到输出。**用字面期望 dict，不复用 `_SCORE_KEY_MAP`** |
| 8+9 | 合并为一条：True/False 两路 bundle 经**消费者** `bundle_loads_full_generation_modules` 走出不同分支 |
| 10 | 保留「两 checkpoint 的 seed grid 相等」（决定论才是名字承诺的事）；**删掉重抄公式的数值锚点** |

### 2.3 `perf_smoke` 两测的根因处理（#11 #12）

这两个测试本身只是镜像 `vrl/config/validation.py` 里写死的 marker 列表（`perf_smoke`/`smoke`/`scratch`/`fixed_prompt`...）。但**真正脆的是被测策略**：`training_data_path_rejection_reason` 靠「按文件名猜是不是 smoke 数据」的魔法子串匹配——加一个新 smoke 目录就漏，正常数据撞上子串就误杀。

两条路线（实现时择一）：
1. **保策略**：把校验从「子串黑名单」改成「数据来源是否落在白名单根目录（`datasets/` / `configs/dataset/` 声明的 manifest）」，测试随之测白名单逻辑。
2. **去策略**：若 §`SPRINT_remove_smoke_datasets`（done）已从仓库物理清掉 smoke 输入，这层运行时字符串校验价值有限，可连同两测一并删除，依赖「smoke 数据不在长期数据路径」这一物理事实。

> 决策点：本 sprint 倾向路线 1（白名单派生），因为它把校验从「猜名字」升级为「查来源」，与 #1 的「从独立来源派生而非抄 key」同一原则。最终由实现者按 `validation.py` 当时形态定。

## 3. 保留清单（do NOT touch — 避免清理过头）

1. **`test_family_registry.py::test_registry_keeps_return_artifacts_as_wiring_metadata`** —— 遍历整个 `FAMILY_REGISTRY` 断言无家族意外覆写默认 wiring metadata，守护的是**跨家族一致性**（AGENTS.md 明确认可的 cross-family consistency），某 entry 被偷改会 fail。审计候选曾给「delete」，**已否决**。
2. **`test_minimal_replay_runtime_wiring.py::test_anima_runtime_spec_uses_explicit_local_paths`** —— 表面像「断言 == 输入字符串」，实为 path **carry-through 透传契约**（extractor 里插入路径改写/漏字段就会 fail），不是会随源码漂移的常量镜像。
3. **改写 #4/#7/#8/#9 时务必保住各自的「消费者」断言**：`artifacts.py:38` 的 raise 校验、kling 的 `if model_key in raw` 过滤分支、`bundle_loads_full_generation_modules` —— 那才是真逻辑，删镜像断言时别把消费者一起删了。

## 4. 非目标

- 不扩散到同文件里的其它真测试（如 `test_ocr.py` 里若有调用真实 `OCRRewardModel` 的端到端用例，与 #5/#6 的影子实现无关，按 AGENTS.md「same source + same lifecycle」不在本次范围）。
- 不顺手重构被测源码（registry / kling / replay bundle 的生产实现保持不动，只改测试；唯一例外是 §2.3 路线 1 需要动 `validation.py` 的校验逻辑）。
- 不追求「测试数量」增减，只追求「每个测试改了源码逻辑就会 fail」。

## 5. 验收

- [x] 12 处按 §2 处理完毕（删/改名 9、`perf_smoke` 2 随 §2.3 路线 2 退役、#1/#2/#10 保名重写）。
- [x] 每条重写后的测试满足不可证伪性消除（按行为/契约断言，源码策略错配即 fail）：#1 modality-import-path 契约、#2 真实 dispatch 类型、#10 checkpoint-无关 + 非退化、#5/#6 调真 `_normalize_ocr_text`。
- [x] 保留清单 3 项原样保留（`test_registry_keeps_return_artifacts_as_wiring_metadata`、`test_anima_runtime_spec_uses_explicit_local_paths`、各消费者断言）。
- [x] 弱测试范围内全绿、无新增镜像断言。**注**：裸 `pytest` 全绿因**与本 sprint 无关**的两类问题未达成——(a) cosmos3/echo 未进 `FAMILY_MODEL_CLASSES` → 2 个 collection error 中断 suite；(b) 排除后 1321 passed / 11 failed 的红全是 env-gap/既有红（Levenshtein、fp8、ray-substrate…）。两者都不在本 sprint 范围。
