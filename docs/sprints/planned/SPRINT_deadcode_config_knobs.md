# SPRINT: 用户可设但无效的 config 旋钮清理（planned）

状态：**planned（2026-07-23）**。5 条确认死代码：1 条 `medium`（`data.preprocessing.media_type` 必填却无 reader）+ 4 条 `low`（1 个 form-1 死参数 `config_reads_in_code(root=...)` + 3 个「set-but-never-read」YAML 旋钮 `metadata_schema` / `target_text` / `DataConfig.source`）。全部落在 `vrl/config/` 边界，无一触及在飞 sprint 的 `generation/ray` / `vrl/ray` / `denoise/base.py` 文件。
来源：dead-code-audit workflow（五种死代码形态 + 对抗验证 + 删除类二次字符串引用检查）.
关联：[[SPRINT_deadcode_00_overview]]；死字段/no-op 旋钮判定规则承接 [[SPRINT_trajectory_views_types_dead_fields_cleanup]]（「能 raise 的校验是合法 consumer」原则在此簇反向适用：`media_type` 的必填校验虽能 raise，但它只做**成员存在性**检查、从不读**值**，故仍是死语义）。

## 0. 一句话

这是**最高严重度**簇：用户在 YAML/CLI 里设了一个期望产生效果的旋钮，它却静默无任何行为——no-op knob。本簇主导形态是 form-1 的 data 孪生（`data.preprocessing.*` 里注册进 `ConfigBlock` 白名单、被 8~10 个 dataset preset 设值、却没有任何 Python reader 的键），最尖锐一条是 `data.preprocessing.media_type`：它不仅无 reader，还被 `schema.py:221` 列为 `loader=prompt_image_manifest` 的**必填**字段——一个强制用户填写、填了却毫无效果的旋钮。需尊重的误删风险是 **同名跨轴碰撞**：`media_type` / `target_text` / `source` 三个词都在**另一条活的轴**上真实存在（reward-artifact 轴的 `reward.kwargs.*.media_type`、manifest-row 的 `PromptExample.target_text`、`data.source_report` JSON payload 的 `source` 键），删 config 侧时**绝不能**误伤这些活 reader。

## 1. 待删清单（逐条，带证据与动作）

顺序：medium-risk 先（reviewer 需最多关注），其余 low-risk 按「先 schema 数据旋钮、后 lint 死参数」编排。

### 1.1 `DataConfig.preprocessing` 的 `"media_type"` 键（+ 必填校验）— dead-config-knob（risk=medium）

- 位置：`vrl/config/schema.py:167`（`ConfigBlock` 白名单元组）、`vrl/config/schema.py:221`（必填字段循环）
- 判死证据：
  - `grep -rn 'media_type' vrl/ tests/ --include='*.py'` → 所有 reader 都在 **reward-artifact 轴**：`vrl/config/validation.py:116`（`vr_kwargs.get("media_type")`，读的是 `reward.kwargs.kling_video_reward`）、`vrl/scripts/eval/{wan_robotics_checkpoint_eval,cosmos_predict25_kling_eval,sana_aesthetic_checkpoint_eval}.py`（构造 reward artifact 时 `media_type="video"/"image"`）。**零处**读 `data.preprocessing.media_type`：无 `preprocessing.get("media_type")`、无点路径 `data.preprocessing.media_type`、无整块 `preprocessing` 前向传递（`online.py` / `prompts.py` / `train_dpo.py` / generation bindings 均只 `.get` 具名键）。
  - `schema.py:221` 循环 `for field in ("format", "image_field", "caption_field", "media_type", "conditioning")`：其余四个键都有真实 value reader（`format`/`image_field`/`caption_field` → `vrl/trainers/data/prompts.py:126-141`、`vrl/config/validation.py:206-207`；`conditioning` → `vrl/scripts/common/online.py:880`），唯独 `media_type` 无——它是一个**必填却无效**的旋钮。
- 动作：
  1. `vrl/config/schema.py`：从 `preprocessing` 的 `ConfigBlock` 元组（line 167）删 `"media_type"`；从必填字段循环（line 221）删 `"media_type"`。
  2. 删 **全部 8 个** dataset preset 的 `data.preprocessing` 块里的 `media_type: video` 行：`vrl/config/presets/dataset/{videophy_i2v,videophy,video_world_v2w,droid_full_target_t2v,droid_full_target_v2w,droid_target_v2w,droid_overfit_validation,droid_overfit_validation_sft_predict25_240p_33f}.yaml`。
  3. 删 `tests/config/test_schema.py` 里 **data 侧** 的 4 处 fixture `"media_type": "video"`（位于 `data.preprocessing` dict 内），当前行号 **500、543、567、804**。
- 注意（medium）：
  - **同名跨轴碰撞**——**绝不可**动 `reward.kwargs.*.media_type` 这条活轴：reward preset（`vrl/config/presets/reward/*.yaml`）、experiment preset（`online_grpo_robotics_physics_4x_l4.yaml:97`、`online_grpo_unified_reward_overlap_gate.yaml:109`，均在 `reward.kwargs` 下）、`tests/config/test_schema.py` 的 reward 侧 fixture（当前行号 **776、818、853、888**，位于 `reward.kwargs.kling_video_reward` 内，须保留）、`vrl/config/validation.py:116` 的生产门控。
  - **⚠ 复核偏差**：findings 记录的 `test_schema.py` data 侧行号为 489/532/556/793、reward 侧为 765/807/842/877；本次复核实测 data 侧为 500/543/567/804、reward 侧为 776/818/853/888（各 +11）。系文件在 findings 采集后经近期提交增长 11 行所致，**归属划分与条目数量完全一致**（4 data 删 / 4 reward 留），执行时请以文件当前内容中的 `data.preprocessing` vs `reward.kwargs` 上下文归属为准，不要照抄行号。
  - 「删元组条目会破坏未知键 sweep」：`ConfigBlock` 元组是未知键白名单，`tests/config/test_unknown_keys.py::test_all_experiment_configs_have_zero_unknown_keys` 会加载每个 experiment（它们 compose 这些 dataset preset）。因此**必须同步删 8 个 YAML 的 setter 行**——留任何一行，键离开白名单后 sweep 即失败。
  - 校验方式：`pytest tests/config/test_unknown_keys.py tests/config/test_schema.py` + `python -m vrl.config.lint`。

### 1.2 `config_reads_in_code(root=...)` 死参数 — dead-arg（risk=low）

- 位置：`vrl/config/lint.py:24-32`
- 判死证据：
  - `grep -rn 'config_reads_in_code' vrl/ tests/` → 恰两处：定义 `vrl/config/lint.py:24` 与唯一内部调用 `vrl/config/lint.py:74`（`dotted = config_reads_in_code()`，**不传参**）。函数体唯一使用 `root` 处是 line 32 `for f in (root or REPO_ROOT / "vrl").rglob("*.py")`——非默认分支不可达。
  - 无字符串/dispatch 引用：CI（`.github/workflows/ci.yml:97`）跑 `python -m vrl.config.lint`，`main()` 仅经同一无参内部调用到达；`pyproject.toml` 无对应 console_script。
  - 测试（`tests/config/test_unknown_keys.py:151,158`）只 import `unregistered_code_paths` / `unknown_yaml_keys`，从不直接调 `config_reads_in_code`；`tests/config/` 内所有 `root=` 用法都是 `load_config(root=tmp_path)`（`test_loading_resources.py:71,86`）——对照案例证明 grep 正确、且 `load_config(root=...)` 因**被测试真实传参**而未被 flag。
- 动作：删 `config_reads_in_code` 的 `root: Path | None = None` 形参，把 line 32 的 rglob inline 成 `for f in (REPO_ROOT / "vrl").rglob("*.py")`。`Path` 因 `REPO_ROOT` 仍需保留 import（`lint.py:21`）。无测试引用该参数，**无测试清理**。

### 1.3 `DataConfig.preprocessing` 的 `"target_text"` 键 — dead-config-knob（risk=low）

- 位置：`vrl/config/schema.py:171`（`ConfigBlock` 白名单元组）
- 判死证据：
  - `grep -rn 'target_text' vrl/ tests/ --include='*.py'` → 所有 reader 都是 **manifest-row 通道**：`PromptExample.target_text`（`vrl/trainers/data/prompts.py:21,60-61,92`，从 JSONL manifest 行或 `.txt` 引号子串填充）、`vrl/trainers/data/artifacts.py:148`、reward metadata `vrl/rewards/models/ocr.py:65`（`artifact.metadata.get("target_text")`）及一批用该 metadata 通道的测试。**唯一**对 config 键的 Python 引用是白名单条目本身 `vrl/config/schema.py:171`；无任何 `cfg_get`/`OmegaConf.select`/点路径读 `data.preprocessing.target_text`。
  - Setters：`grep -rn 'target_text' vrl/config/presets/dataset/` → **10 个** YAML（不是 findings 初稿写的 8 个）：`video_world_v2w.yaml:12`、`drawbench_train_192.yaml:10`、`videophy.yaml:15`、`ocr.yaml:9`、`pickscore_sfw.yaml:8`、`droid_full_target_v2w.yaml:13`、`droid_target_v2w.yaml:13`、`droid_overfit_validation_sft_predict25_240p_33f.yaml:15`、`droid_full_target_t2v.yaml:14`、`droid_overfit_validation.yaml:21`。
- 动作：
  1. 从 `vrl/config/schema.py:171` 的 `ConfigBlock` 元组删 `"target_text"`。
  2. 删上列 **全部 10 个** setter YAML 的 `target_text:` 行。其中 `ocr.yaml:9` 是 `target_text: quoted_substring`，其余多为 `target_text: none`。
  3. `vrl/config/presets/dataset/ocr.yaml` 首行注释（line 1）引用了 `preprocessing.target_text` 作为 loader 契约文档：改写该 header 注释使其不再引用该键，保留一句朴素注释说明 `load_prompt_manifest` 从每行 `.txt` prompt 的引号文本提取 OCR target（契约已在 `vrl/trainers/data/prompts.py:74-78` 的 docstring 记录）。
- 注意（低，但含跨轴碰撞）：
  - **绝不可**动 manifest-row 活通道：`PromptExample.target_text`（`vrl/trainers/data/prompts.py:21,60-61,92`）、`vrl/trainers/data/artifacts.py:148`、`vrl/rewards/models/ocr.py:65` 及 `tests/rewards/service/`、`tests/trainers/online/`、`tests/rollouts/` 里通过 `metadata={"target_text": ...}` 走 metadata 通道的测试。
  - `ocr.yaml` 被多个 live experiment preset（`wan_2_1` / `janus_pro` / `nextstep_1` / `sd3_5` 等的 `online_grpo_ocr.yaml`）compose——删白名单条目却漏删 `ocr.yaml` 的 setter 会令 `test_all_experiment_configs_have_zero_unknown_keys` 与 `vrl/config/lint.py` 失败，故必须同批清理。
  - display-only 豁免**不适用**：AGENTS.md 要求豁免注解写在字段定义处（`schema.py:171` 无此注解），且把用户可见 no-op 旋钮定性为「更糟」；loader 契约已在 docstring 记录。

### 1.4 `DataConfig.preprocessing` 的 `"metadata_schema"` 键 — dead-config-knob（risk=low）

- 位置：`vrl/config/schema.py:170`（`ConfigBlock` 白名单元组）
- 判死证据：
  - `grep -rn 'metadata_schema' .`（排除 `.venv/__pycache__/docs/runs`）→ **恰 4 处**：schema 注册 `vrl/config/schema.py:170` + 3 个 dataset YAML setter（`anime_anatomy.yaml:8` = `danbooru_anatomy`、`anime_safety_stress.yaml:8` = `danbooru_safety`、`geneval.yaml:8` = `geneval`）。`vrl/` 与 `tests/` 内**零 Python reader**；每个 `data.preprocessing` 消费者都只取具名键，无一触及 `metadata_schema`，无 `**preprocessing` / `.items()` 前向传递。
  - 白名单注册项唯一作用是让未知键 walker（`vrl/config/unknown_keys.py`，仅 warn 不 raise）对一个无人消费的键闭嘴。
- 动作：从 `vrl/config/schema.py:170` 的 `ConfigBlock` 元组删 `"metadata_schema"`，并删 `vrl/config/presets/dataset/{anime_anatomy,anime_safety_stress,geneval}.yaml:8` 的 `metadata_schema:` 行。
- 注意：`dataset/anime_safety_stress` 被 live experiment `experiment/anima_preview3/online_grpo_aesthetic_nsfw_safety.yaml:9` include——属「set 在活路径、却无人读」的最坏情形。元组条目与 3 个 YAML setter 须**成对**删除以保持 `test_all_experiment_configs_have_zero_unknown_keys` 通过；无测试引用该键，无测试清理。

### 1.5 `DataConfig.source` 字段 — dead-config-knob / 死字段（risk=low）

- 位置：`vrl/config/schema.py:191`（`source: Any = None`）
- 判死证据：
  - `grep -rnF '"data.source"' vrl/ tests/ --include='*.py'` → **零命中**（exit 1）；`'data.source'` 亦零命中。对照活的兄弟字段：`allow_absolute_artifact_paths` 经 `OmegaConf.select(cfg, "data.allow_absolute_artifact_paths", ...)`（`vrl/scripts/denoise/encode_targets.py:179`）读、`source_report` 经 `require(cfg, "data.source_report")`（`vrl/config/validation.py:171`）读——证明字符串点路径就是正确 grep 形态，而 `source` 无一命中。
  - 所有 `.source` / `"source"` 命中都是**别的对象**：`data.source_report`（`validation.py:95-101` 真读）、`source_report` JSON payload 的 `source` 必填键（`validation.py:176-188`）、manifest-row provenance（`source_repo` / `source_video_url`，`vrl/scripts/data/*.py`、`vrl/trainers/data/artifacts.py`）。`vrl/scripts/data/bootstrap.py:127` 只从 `data` 切片 `("loader","manifest","eval_manifest","source_report")`——`source` 被排除在 tooling 切片外。
  - Setters：`grep -rn '^\s*source:' vrl/config/presets/` → 仅 `pickapic_v1.yaml:10`、`pickapic_v2.yaml:5`（`source: huggingface`，`data:` 块下）。
- 动作：删 `DataConfig` 的 `source: Any = None` 字段（`vrl/config/schema.py:191`），并删 `vrl/config/presets/dataset/{pickapic_v1,pickapic_v2}.yaml` 的 `source: huggingface` 行（若想留作人读 provenance，可改成 YAML 注释）。
- 注意：line-188 注释「Key registry: consumed by data/eval tooling, not validated here.」对 `allow_absolute_artifact_paths` / `artifact_data_root` / `source_report` 三个兄弟为真、对 `source` 为假。`DataConfig` 字段本身即 `find_unknown_keys` 的已知键注册表（`vrl/config/lint.py:88` 对 experiments 跑 `find_unknown_keys`），故字段删除须与两处 YAML setter **成对**进行。无测试构造 `DataConfig(source=...)` 或断言它，无测试清理。

### 关联：其他簇里的模块局部 no-op 旋钮（本簇不 own，仅交叉引用以呈现完整图景）

以下 no-op 旋钮同属「用户可设但无效」大类，但落在各自模块 sprint 里删除，不在本簇动手——列出以便 reviewer 看到 no-op 旋钮全貌：

- `janus_pro` 的 `refine_mode` / `task_stages`（[[SPRINT_deadcode_model_families]]）
- `nextstep` 的 `token_dim`（[[SPRINT_deadcode_model_families]]）
- `generation` 的 `same_latent`（[[SPRINT_deadcode_generation]]）
- `dpo` 的 `timestep_subset`（[[SPRINT_deadcode_rollouts_trainers_ray]]）
- `cosmos` eval 的 keep-model 旋钮（[[SPRINT_deadcode_scripts]]）

## 2. 验证协议

- **每条删除后**：`ruff check <touched files>` + `ruff format --check <touched files>`（仅本任务触及的 Python：`vrl/config/schema.py`、`vrl/config/lint.py`）。
- **全簇完成后**：`pytest tests/config/`（重点 `test_unknown_keys.py`、`test_schema.py`、`test_load_all_experiments.py`）+ `python -m vrl.config.lint` 全绿 + `pytest -m "not e2e and not slow_test"` 子集不新增失败。
- **基线（清理前，2026-07-23）**：fast subset **2620 passed / 7 pre-existing failures**（架构边界 + causvid/magi_1 打包摘要，与本清理无关）；`vrl.config.lint` 与 `ruff check .` 全绿。删除后这三项须保持。
- **逐条动作触及的测试文件**：
  - §1.1 `media_type`：`tests/config/test_schema.py`（删 data 侧 fixture，行号见 §1.1 复核偏差说明）、`tests/config/test_unknown_keys.py`（experiment 加载复合校验，无需改动仅须通过）。
  - §1.2 `config_reads_in_code`：无测试文件改动（`tests/config/test_unknown_keys.py` 仅经 `unregistered_code_paths` 间接覆盖，须保持通过）。
  - §1.3 `target_text`：`tests/config/test_unknown_keys.py`、`tests/config/test_schema.py`（须保持通过；manifest-row 通道测试 `tests/rewards/service/`、`tests/trainers/online/`、`tests/rollouts/` **不得改动**）。
  - §1.4 `metadata_schema`：`tests/config/test_unknown_keys.py`（无 fixture 引用该键，仅须通过）。
  - §1.5 `DataConfig.source`：`tests/config/test_schema.py`（无 `source=` 构造，仅须通过）。

## 3. Non-Goals

- 不删被「能 raise 的校验」、控制流分支、runtime/config/Ray 调用消费的字段（AGENTS 死字段规则）。特别地：`data.preprocessing` 的 `format` / `image_field` / `caption_field` / `conditioning` / `reference_image` 及 `DataConfig` 的 `source_report` / `allow_absolute_artifact_paths` / `artifact_data_root` 均有真实 reader，**保留**。
- 不动 DO-NOT-FLAG 豁免项（`_GENERATION_CUDA_RUNTIME_RESIDUAL_BYTES` / `_REWARD_CUDA_RUNTIME_RESIDUAL_BYTES` 显存残留配额、`ensure_loaded`、`process_gpu_used_bytes` NVML、sana/hunyuan `prepare_latents` 修复）。
- 不为省行数扁平化 protocol / lazy-import / 跨家族一致性 thin function；`config_reads_in_code` 是独立 AST-scan lint 辅助函数，删的是它的**死参数**而非函数本身。
- **不越轴误删**（本簇专属）：`media_type` / `target_text` / `source` 三词各有一条活轴（reward-artifact `reward.kwargs.*.media_type`、manifest-row `PromptExample.target_text` / `artifact.metadata`、`data.source_report` payload 的 `source` 键），本簇只清 `data.preprocessing.*` / `DataConfig.source` config 侧，活轴一律保留。
- 不 own 其他模块的 no-op 旋钮（janus_pro `refine_mode`/`task_stages`、nextstep `token_dim`、generation `same_latent`、dpo `timestep_subset`、cosmos eval keep-model）——它们在各自 sprint 删除，本簇仅交叉引用。
- 本簇 5 条均在 `vrl/config/`，与在飞 sprint [[SPRINT_native_generation_engine_program]]（改 `generation/ray/`、`vrl/ray/`、`models/steps/denoise/base.py`）**无文件重叠**，无需 sequence-after。

## References

- `vrl/config/schema.py:167`（`media_type` 白名单）、`:170`（`metadata_schema`）、`:171`（`target_text`）、`:191`（`DataConfig.source`）、`:221`（`media_type` 必填循环）
- `vrl/config/lint.py:24-32`（`config_reads_in_code` 死参数）、`:74`（无参调用点）、`:88`（`find_unknown_keys` over experiments）
- `vrl/config/validation.py:116`（reward `media_type` 门控，活）、`:171`（`data.source_report` 活读）、`:206-207`（`image_field`/`caption_field` 活读）
- `vrl/config/unknown_keys.py`（未知键 warn-only walker）
- `vrl/trainers/data/prompts.py:21,60-61,74-78,92,126-141`（`PromptExample.target_text` 活通道 + docstring 契约）、`vrl/trainers/data/artifacts.py:148`、`vrl/rewards/models/ocr.py:65`
- `vrl/scripts/common/online.py:880`（`conditioning` 活读）、`vrl/scripts/denoise/encode_targets.py:179`（`allow_absolute_artifact_paths` 活读）、`vrl/scripts/data/bootstrap.py:127`（tooling 切片，排除 `source`）
- dataset presets（`media_type` ×8）：`vrl/config/presets/dataset/{videophy_i2v,videophy,video_world_v2w,droid_full_target_t2v,droid_full_target_v2w,droid_target_v2w,droid_overfit_validation,droid_overfit_validation_sft_predict25_240p_33f}.yaml`
- dataset presets（`target_text` ×10）：`vrl/config/presets/dataset/{video_world_v2w:12,drawbench_train_192:10,videophy:15,ocr:9,pickscore_sfw:8,droid_full_target_v2w:13,droid_target_v2w:13,droid_overfit_validation_sft_predict25_240p_33f:15,droid_full_target_t2v:14,droid_overfit_validation:21}.yaml`
- dataset presets（`metadata_schema` ×3）：`vrl/config/presets/dataset/{anime_anatomy,anime_safety_stress,geneval}.yaml:8`
- dataset presets（`source` ×2）：`vrl/config/presets/dataset/{pickapic_v1:10,pickapic_v2:5}.yaml`
- tests：`tests/config/test_schema.py`（data 侧 `media_type` fixture 当前行 500/543/567/804）、`tests/config/test_unknown_keys.py:151,158`
- 保留活轴（勿删）：`vrl/config/presets/reward/*.yaml`、experiment `online_grpo_robotics_physics_4x_l4.yaml:97`、`online_grpo_unified_reward_overlap_gate.yaml:109`（`reward.kwargs.*.media_type`）
- 关联：[[SPRINT_deadcode_00_overview]]、[[SPRINT_trajectory_views_types_dead_fields_cleanup]]、[[SPRINT_native_generation_engine_program]]（无重叠）
