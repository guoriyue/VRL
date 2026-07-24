# SPRINT: 用户可设但无效的 config 旋钮清理（planned）

状态：**RECONCILED（2026-07-24）**，对齐基线 main @ `7c748532`（= `origin/main` tip，自审计所在旧树 `88ed756e` 起累计约 63 个 cleanup/refactor 提交）。原始 5 条针对旧树采集，本次逐条独立复核后：**1 条仍需做**（STILL_VALID：`config_reads_in_code(root=...)` 死参数）+ **4 条已由 origin 落地**（`media_type` / `metadata_schema` / `target_text` / `DataConfig.source`，均在提交 `1aef2ea8 refactor(config): remove public no-op inputs` 中删除）+ **0 条情况已变**。
来源：dead-code-audit workflow（五种死代码形态 + 对抗验证 + 删除类二次字符串引用检查），审计树 `88ed756e`；本文件为对齐 `7c748532` 后的复核版。
关联：[[SPRINT_deadcode_00_overview]]；死字段/no-op 旋钮判定规则承接 [[SPRINT_trajectory_views_types_dead_fields_cleanup]]。

## 0. 一句话

本簇是**用户可设但无效**的 no-op knob 清理：用户在 YAML/CLI 设了一个期望产生效果的旋钮，它却静默无任何行为。原簇 5 条中 4 条的 `data.preprocessing.*` / `DataConfig.source` config 侧无效键已被 origin 在 `1aef2ea8` 一次性删除（schema 白名单条目 + 全部 setter YAML + 新增负向 unknown-key 测试均已落地，见 §2），**仅剩 1 条** form-1 死参数 `config_reads_in_code(root=...)` 待做——它落在 `vrl/config/lint.py`，该文件自审计以来 `git log 88ed756e..HEAD` 为空、完全未被触及，死参数原样存在。

## 1. 待删清单（仍有效）

顺序：仅剩 1 条 low-risk。

### 1.1 `config_reads_in_code(root=...)` 死参数 — dead-arg（risk=low）

- 位置：`vrl/config/lint.py:24`（定义，`root: Path | None = None`）、唯一使用点 `vrl/config/lint.py:32`（`for f in (root or REPO_ROOT / "vrl").rglob("*.py")`）、唯一内部调用 `vrl/config/lint.py:74`（`dotted = config_reads_in_code()`，不传参）。行号与原审计一致（文件未变）。
- 复核证据（2026-07-24 @ `7c748532`）：
  - `git log --oneline 88ed756e..HEAD -- vrl/config/lint.py` → **空**（文件自审计以来未变更）。
  - `grep -rn 'config_reads_in_code' vrl/ tests/ .github/ pyproject.toml` → 仅 `vrl/config/lint.py:24`（定义，`root` 形参仍在）与 `vrl/config/lint.py:74`（无参调用）。`tests/` 内无对 `config_reads_in_code(` 的直接调用。
  - 无字符串/dispatch 引用：CI（`.github/workflows/ci.yml`）跑 `python -m vrl.config.lint`，`main()` 仅经同一无参内部调用到达；`pyproject.toml` 无对应 console_script。函数体唯一使用 `root` 处是 line 32 的非默认分支，无 caller 可达。
  - 对照案例仍成立：`tests/config/test_loading_resources.py` 里所有 `root=` 用法都是 `load_config(root=tmp_path)`——因**被测试真实传参**而未被 flag；`config_reads_in_code` 无此调用，参数为 form-1 死。
- 动作：删 `config_reads_in_code` 的 `root: Path | None = None` 形参，把 line 32 的 rglob inline 成 `for f in (REPO_ROOT / "vrl").rglob("*.py")`。`Path` 因 `REPO_ROOT` 相关 import 仍需保留。无测试引用该参数，**无测试清理**。

## 2. 已由 origin 落地（本次复核确认，无需再做）

以下 4 条在 `88ed756e..7c748532` 间由提交 `1aef2ea8 refactor(config): remove public no-op inputs` 删除，本次复核逐条确认（schema 白名单条目移除 + 全部 setter YAML 清理 + 新增 `tests/config/test_unknown_keys.py` 负向断言键已成为 UNKNOWN）。**无需再做**。

- `DataConfig.preprocessing` 键 `"media_type"`（+ `prompt_image_manifest` 必填校验）— 原「必填却无 reader」的 medium 条。现 `vrl/config/schema.py` 中 `media_type` 已从 preprocessing `ConfigBlock` 元组与必填循环（现为 `("format","image_field","caption_field","conditioning")`）双删；8 个 setter YAML 全清；`test_unknown_keys.py:209-211` 断言 `data.preprocessing.media_type` 现为 UNKNOWN。活的 reward 轴 `reward.kwargs.*.media_type`（`vrl/config/validation.py:118` 等）原样保留。— `1aef2ea8`
- `DataConfig.preprocessing` 键 `"metadata_schema"`— 原「set 在活 experiment 路径却无 reader」的 low 条。已从 preprocessing `ConfigBlock` 元组移除；3 个 setter YAML 全清；仅剩 `test_unknown_keys.py:202-203` 负向断言其为 UNKNOWN。— `1aef2ea8`
- `DataConfig.preprocessing` 键 `"target_text"`— 原「10 个 setter、无 config reader」的 low 条。已从 `ConfigBlock` 元组移除；含 `ocr.yaml`、`droid_overfit_validation_sft_predict25_240p_33f.yaml` 在内的 10 个 setter YAML 全清；`test_unknown_keys.py:205-207` 断言其为 UNKNOWN。活的 manifest-row 通道 `PromptExample.target_text`（`vrl/trainers/data/prompts.py`）与 reward metadata（`vrl/rewards/models/ocr.py`）原样保留。— `1aef2ea8`
- `DataConfig.source` 字段（`source: Any = None`）— 原「YAML set、无 reader」的 low 条。字段已从 `DataConfig` 删除（现字段列表为 loader/manifest/eval_manifest/preprocessing/sampler/dataset_name/split/cache_dir/sft_latents/max_train_samples/task_type/allow_absolute_artifact_paths/artifact_data_root/source_report）；`pickapic_v1.yaml` / `pickapic_v2.yaml` 的 `source:` setter 行已删；`test_unknown_keys.py:200` 断言 `data.source` 现为 UNKNOWN。活的 `data.source_report` 字段与 payload 内 `source` 键原样保留。— `1aef2ea8`

## 3. 情况已变（需重新评估）

（无）

## 4. 验证协议

- **删除后**（仅剩 §1.1）：`ruff check vrl/config/lint.py` + `ruff format --check vrl/config/lint.py`（本任务仅触及此一 Python 文件）。
- **完成后**：`pytest tests/config/`（重点 `test_unknown_keys.py`）+ `python -m vrl.config.lint` 全绿 + `pytest -m "not e2e and not slow_test"` 子集不新增失败。§1.1 无测试文件改动——`tests/config/test_unknown_keys.py` 仅经 `unregistered_code_paths` 间接覆盖 `config_reads_in_code`，须保持通过。
- **基线**：测试与 lint 均对齐当前 checked-out main @ `7c748532`（原 `88ed756e` 基线数字已作废，§2 的 4 条 origin 删除已改变 `tests/config/` 内容——新增了 `test_unknown_keys.py` 的 4 组负向断言）。删除 §1.1 死参数后，`vrl.config.lint` 与 `ruff check vrl/config/lint.py` 须保持全绿。

## 5. Non-Goals

- 不删被「能 raise 的校验」、控制流分支、runtime/config/Ray 调用消费的字段（AGENTS 死字段规则）。`data.preprocessing` 的 `format` / `image_field` / `caption_field` / `conditioning` / `reference_image` 及 `DataConfig` 的 `source_report` / `allow_absolute_artifact_paths` / `artifact_data_root` 均有真实 reader，**保留**。
- 不为省行数扁平化 protocol / lazy-import / 跨家族一致性 thin function；`config_reads_in_code` 是独立 AST-scan lint 辅助函数，删的是它的**死参数**而非函数本身。
- 不越轴误删：`media_type` / `target_text` / `source` 三词各有一条活轴（reward-artifact `reward.kwargs.*.media_type`、manifest-row `PromptExample.target_text` / `artifact.metadata`、`data.source_report` payload 的 `source` 键）——这些活轴在 §2 的 origin 删除中已确认原样保留，本簇 §1.1 不触及。
- 不 own 其他模块的 no-op 旋钮（janus_pro `refine_mode`/`task_stages`、nextstep `token_dim`、generation `same_latent`、dpo `timestep_subset`、cosmos eval keep-model）——它们在各自 sprint 删除。
- §1.1 落在 `vrl/config/lint.py`，与在飞 sprint [[SPRINT_native_generation_engine_program]]（改 `generation/ray/`、`vrl/ray/`、`models/steps/denoise/base.py`）**无文件重叠**。

## References

- **仍需做**：`vrl/config/lint.py:24`（`config_reads_in_code` 死参数 `root`）、`:32`（唯一使用点 rglob）、`:74`（无参调用点）
- **已落地（`1aef2ea8`）**：`vrl/config/schema.py`（`media_type` / `metadata_schema` / `target_text` 白名单条目 + `media_type` 必填循环 + `DataConfig.source` 字段均已删除）；setter YAML 全清（`media_type` ×8、`target_text` ×10、`metadata_schema` ×3、`source` ×2）
- **落地后的负向测试**：`tests/config/test_unknown_keys.py:200`（`data.source` UNKNOWN）、`:202-203`（`metadata_schema`）、`:205-207`（`target_text`）、`:209-211`（`media_type`）
- **保留活轴（勿删，已确认原样）**：`vrl/config/validation.py:118`（reward `media_type` 门控）、`vrl/trainers/data/prompts.py`（`PromptExample.target_text`）、`vrl/rewards/models/ocr.py`（reward metadata `target_text`）、`data.source_report` 字段及 payload 内 `source` 键
- 关联：[[SPRINT_deadcode_00_overview]]、[[SPRINT_trajectory_views_types_dead_fields_cleanup]]、[[SPRINT_native_generation_engine_program]]（无重叠）
