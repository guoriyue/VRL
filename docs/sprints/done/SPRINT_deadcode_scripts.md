# SPRINT: `vrl/scripts/` 死代码清理（done）

状态：**DONE（2026-07-24）**。§1 全部由 `ae3a3e96` 落地；`_video_to_cthw` finding 因现场活化撤销。

> **历史审计，禁止照 §1 对当前 HEAD 执行。** 下文保留的是对 `7c748532` 的执行前证据与动作说明。

历史基线：main @ `7c748532`。原审计 16 条复核时为 **14 条仍待做** + **1 条先行落地** + **1 条现场已变**。
原审计来源：dead-code-audit workflow（五种死代码形态 + 对抗验证 + 删除类二次字符串引用检查）。复核 grep 纪律：排除 `.venv/ third_party/ outputs/ datasets/ docs/runs/ __pycache__/ egg-info`；同名符号按 finding 的 file+context 消歧；test-only 引用在生产读者已消失时仍记为 DONE。
关联：[[SPRINT_deadcode_00_overview]]；`scripts/eval/sana_aesthetic_curve_verdict.py` 的阈值 hoist 与 [[SPRINT_sana_aesthetic_trustworthy_curve]] 的预注册协议对齐（见 §1.7）；`scripts/perf/gpu_preflight.py` 属 [[SPRINT_cosmos_video_mfu_kernels]] 的保留交付物（见 §1.14）。

> **执行状态（2026-07-24）**：§1 全部 14 条已落地 `ae3a3e96`。§3 的 `_video_to_cthw` 未做（已活化）。

## 0. 一句话

本簇是 `vrl/scripts/` 下**长期资产内部**的符号级死代码清理（数据生成器、eval 入口、perf 共享 helper 都是长期资产；`*_probe` 一次性脚本整体不在范围内，但其内部死符号在范围内）。主形态是 **dead-arg / dead-field**（没有生产者的参数、零读者的结果字典键与派生字段），另有若干 form-3 单调用者内联与 form-1/4 test-only facade。最锋利的一条是 `anime_probe_common.py` 的 `hamer_verdict` 及一整组结果字典键（`"verdict"`/`"n_hands"`/`"n_persons"`/`"annotated"` overlay）——报告入口 `probe_anime_anatomy_report.py` 的 docstring 声称展示 `verdict` 与 per-hand `finger_cv`，实际只读 `mean_finger_cv`，是文档腐烂的直接证据。误删风险主要落在 `danbooru.py`：`bucket_balance`/`candidate_pool_factor` 删除会连带触发 `bucket_weights` 局部量、`preferred_min_score` 三元式与 `build_prompt_rows` 签名的 `| None` 收窄——必须整段一起改，漏改会留下同型 form-2 死语义残留或让类型宣称接受一个运行时会崩溃的 `None`。

## 1. 待删清单（仍有效——STILL_VALID + RELOCATED）

顺序沿用原簇：medium-risk 先行（§1.1），其余 low-risk 按文件/目录聚合（`scripts/data/` → `scripts/eval/` → `scripts/generation/` → `scripts/perf/`）。RELOCATED 条目在 位置 行注明「行号已移动」，证据/动作逻辑不变。

---

### 1.1 `hamer_verdict` / `hamer_probe_image` 结果字典键 `"verdict"`/`"n_hands"`/`"annotated"` + `rtmw_metrics` 键 `"n_persons"` — dead-field（risk=**medium**，STILL_VALID）

- 位置：`vrl/scripts/eval/anime_probe_common.py`：`n_persons` 于 `:140/:162`、`hamer_probe_image` 的 `verdict`/`n_hands` 于 `:279/:276`、`no_hands_result` 于 `:366/:363`、`hamer_verdict` def 于 `:371`（复核确认行号仅在原范围 139-147/161-169/275-281/362-378 内位移约 1-2 行）
- 判死证据：
  - 模块唯一生产消费者是 `probe_anime_anatomy_report.py`（`grep -rn "anime_probe_common"` → 仅 `probe_anime_anatomy_report.py:50` import + `tests/scripts/test_anime_probe_common.py:5` 只 import `optional_finger_cv`）。
  - 结果键零读者：唯一消费者 `probe_anime_anatomy_report.py` 只读 `m["annotated"]`（`:88,:253`——RTMW overlay）、`body_coverage/hand_coverage/mean_conf/bone_ratio_err/visible_hands` 与 `mean_finger_cv`；`grep` 该消费者的 `"verdict"`/`"n_hands"`/`"n_persons"` → 零命中。
  - `hamer_verdict` 恰一个调用者（`:279`），输出落进不被读的 `"verdict"` 键 → live caller, dead semantics（form-2）。**证据订正**：`no_hands_result` 并不调用 `hamer_verdict`，而是硬编码 `"verdict": "NOT_DETECTED"`（`:366`）——审计原文对此有误。
  - 无 `json.dump`/`asdict`/`.items()`/kwargs 转发整字典；无 YAML/entry-point/字符串分发；sprint 文档仅在 `done/` 审计记录里出现。
- 动作：二选一，推荐 **(b) 删除**（(a) 反向补齐入报告的成本高、非本簇目标）：
  - (a) 把 hamer 结果 surface 进 HTML 报告——加 `"verdict"`/`"n_hands"` 列，从 `hm["hands"]` 渲染 per-hand `finger_cv`，把 hamer `"annotated"` overlay 与 RTMW 一并保存；给每个被标键一个消费者，使 docstring 成真。
  - (b) 删除：`hamer_verdict()`（单调用者、输出不被读）；`hamer_probe_image` 返回 dict 的 `"verdict"`/`"n_hands"`/`"hands"` 键；`"annotated"` 键及其绘制代码（`annotated = img_bgr.copy()`、handedness/color 计算、`cv2.circle` loop）——之后 `hand_results` 收敛为 per-hand `finger_cv` 浮点列表，`"j2d_px"`/`"handedness"` per-hand 字段随之消失；`no_hands_result` 中匹配的 `"verdict"`/`"n_hands"`/`"hands"`/`"annotated"` 键与 `cv2.putText` overlay（收敛为 `{"mean_finger_cv": None}` 或直接内联到其唯一调用点）；`rtmw_metrics` 两个返回 dict 的 `"n_persons"` 键（`:140,:162`）。**同时**把 `probe_anime_anatomy_report.py` docstring 从 `"finger_cv per hand, verdict"` 改为 `"mean finger_cv"`，与报告实际展示一致。
  - `tests/scripts/test_anime_probe_common.py` 只覆盖 `optional_finger_cv`，无需改动。
- 注意（medium）：这是唯一 medium 条，reviewer 需重点核对——(1) 删除 `"annotated"` 会连带删掉 `j2d_px`/`handedness`（它们只为 `cv2.circle` loop 服务，审计原动作漏了 `"hands"` 键，本条已补入）；(2) docstring 修正是本条自身引用的腐烂点，删除路径必须一并改，否则入口文档继续留假承诺。

---

### 1.2 `FAILURE_LABELS` — dead-field（risk=low，STILL_VALID）

- 位置：`vrl/scripts/data/danbooru.py:105, 1427`（行号不变；长期资产：danbooru 数据集生成器）
- 判死证据：`grep -rn "FAILURE_LABELS" vrl/ tests/`（排除 `__pycache__`）→ 仅 `danbooru.py:105`（定义 `FAILURE_LABELS = set(_ANATOMY["failure_labels"])`）+ `:1427`（`__all__` 条目），零消费者（另有一处 `docs/sprints/done/` 历史记录，非活引用）。`datasets/danbooru/config.yaml:219` 仍持有 `failure_labels:` 源键。`danbooru import *` 零命中；`config.yaml` 无 YAML anchor（`_ANATOMY` 各 key 逐个显式取用，无 keys 迭代/schema 校验）。
- 动作：删除 `danbooru.py:105` 的 `FAILURE_LABELS = set(_ANATOMY["failure_labels"])` 与 `__all__` 中的 `"FAILURE_LABELS"`（`:1427`）；同一生命周期内一并删除 `datasets/danbooru/config.yaml:219` 的 `failure_labels:` 键（含 `:218` 注释行）——该 YAML 键的唯一读者就是这个死常量。无测试需要更新。

---

### 1.3 `_current_http_download` — single-caller-merge（risk=low，STILL_VALID）

- 位置：`vrl/scripts/data/danbooru.py:1317-1318`（def），seam `:1314`，调用点 `:1390/:1410`（长期资产）
- 判死证据：函数体 `:1318` `return globals().get("_http_download", http_download)`——`_http_download` 在 `:1314` 恒定义，`globals().get` 的 fallback 分支不可达，函数等价于直接读全局。`grep -rn "_current_http_download\|_http_download" vrl/ tests/` → 调用者仅 `danbooru.py:1390,:1410`（两个 `_cmd` 函数），测试只 `monkeypatch.setattr(danbooru, "_http_download", ...)`（`tests/data/test_setup.py:204`，旧树为 `:149`），从不引用 getter；getter 不在 `__all__`。
- 动作：删除 `_current_http_download()`；两个调用点 `_cmd_anime_positives`（`:1390`）与 `_cmd_anime_fetch_images`（`:1410`）改为直接引用 `fetch=_http_download`（模块全局在命令执行时解析，`test_setup.py:204` 对 `danbooru._http_download` 的 monkeypatch 依旧生效）。保留 `_http_download = http_download`（`:1314`）这一测试注入 seam 本身。
- 注意：标签小瑕疵——实际有**两个**调用者（非单一），但动作对两处都处理，安全完整。真正的 test seam 是 `_http_download` 全局，须保留；getter 只是冗余间接层。

---

### 1.4 `build_anatomy_prompts(bucket_balance=, candidate_pool_factor=)` — dead-arg（risk=low，STILL_VALID）

- 位置：`vrl/scripts/data/danbooru.py:51-52, 167-168, 180-184, 674, 717-718`（行号不变；长期资产）
- 判死证据：常量 `ANATOMY_BUCKET_BALANCE=:51` / `ANATOMY_CANDIDATE_POOL_FACTOR=:52`，参数 `bucket_balance=:167` / `candidate_pool_factor=:168`，死逻辑 `bucket_weights`/`candidate_limit` 于 `:180-184`——零外部生产者（`:230,:250` 是 safety 侧同名参数，函数体无条件使用，**活的**，不在本条）。两个调用方 `build_default_manifests`（`:147`）与 `_cmd_anime_prompts`（`:1366`）只传 `metadata/download_metadata`；`register()` 的 `anime-prompts` 子命令只暴露 `--metadata`，无 `**kwargs` 转发。`"natural"` 值零生产者 → `:180` 的 `bucket_weights = DEFAULT_BUCKET_WEIGHTS if bucket_balance == "quota" else None` 恒取真值（`DEFAULT_BUCKET_WEIGHTS` 来自 `config.yaml` 11 项非空表），`candidate_pool_factor` 分支（`:184`）不可达。`tests/data/test_danbooru.py` 不 import `build_anatomy_prompts`，`bucket_weights` 仅传显式非空 dict。
- 动作：在 `vrl/scripts/data/danbooru.py`：
  1. 从 `build_anatomy_prompts` 签名删 `bucket_balance`（`:167`）与 `candidate_pool_factor`（`:168`），删常量 `ANATOMY_BUCKET_BALANCE`（`:51`）与 `ANATOMY_CANDIDATE_POOL_FACTOR`（`:52`）。
  2. 删 `:180-185` 整段（`bucket_weights` 选择、`:181-182` 校验、`:183-184` `candidate_limit` 计算），并在 `:186-196` 的 `build_prompt_rows` 调用中删掉 `candidate_limit=`（`:192`）与 `bucket_weights=`（`:195`）关键字（`DEFAULT_BUCKET_WEIGHTS` 已是其默认值），`:189` 收敛为 `preferred_min_score=preferred_min_score`。
  3. 从 `build_prompt_rows` 删 `candidate_limit` 参数（`:674`）及其早停分支（`:700-701`）。
  4. 删 `:710/:717-718` 的 `if/else`，无条件调用 `_select_quota_rows`，并把 `:677` 的注解从 `Mapping[str, float] | None = DEFAULT_BUCKET_WEIGHTS` 收窄为 `Mapping[str, float] = DEFAULT_BUCKET_WEIGHTS`。
  5. 保留 `_interleave_bucket_rows`（`_select_quota_rows` 内 `:1184/:1191` 活调用）、保留 safety 侧 `SAFETY_CANDIDATE_POOL_FACTOR`（`:61`）/`candidate_pool_factor`（`:230,:250`）与 `build_danbooru_safety_prompt_rows` 的 `candidate_limit`（生产 `:250` 与测试 `test_danbooru.py:359` 均活）。
  6. `tests/data/test_danbooru.py` 无需改动。
- 注意（load-bearing 收敛链）：这是本簇误删风险最高的一条。`:189` 的 `preferred_min_score if bucket_weights else None` 与 `:710-718` 的 `if bucket_weights` 均读同一局部 `bucket_weights`——删 `bucket_balance` 后 `bucket_weights` 恒为真值，这些三元式/分支成为同型 form-2 死语义残留，**必须在同一改动内一起收敛**（上文步骤 2、4 已含）。漏改步骤 4 的 `| None` 收窄会让签名宣称接受 `None`，而 `_bucket_quotas` 迭代 `None` 会崩溃。审计对本条给出 `verdict=revise` 提示，正是这两处；本条动作已完整覆盖。

---

### 1.5 `_iter_lerobot_v21` 局部 dict 死键 `firsts_by_episode[...]["task_index"]` — dead-field（risk=low，STILL_VALID）

- 位置：`vrl/scripts/data/video_world.py:356-358`（dict 构造），唯一读点 `:377`（原范围 356-359，实质不变；长期资产：video importer）
- 判死证据：`firsts_by_episode` 于 `:356-358` 构造为 `{episode: {"global_index": idx, "task_index": task_index}}`，唯一读点 `:377` 是 `firsts_by_episode[episode]["global_index"]`（写入 metadata `source_global_index`）；`"task_index"` 键无任何读者。同循环内 `task_index` 由 `targets[position]`（`:359/:369`）提供，与该 dict 无关。函数内局部 dict，不序列化、不出模块。
- 动作：把 `firsts_by_episode` 简化为 `{episode: idx}`（值直接存 `global_index`），删死键 `"task_index"`；`:377` 改为 `firsts_by_episode[episode]`。`task_index` 活来源是 `targets` 映射，不受影响。无测试需要更新。

---

### 1.6 `--keep-model-between-checkpoints` — dead-config-knob（risk=low，RELOCATED）

- 位置：`vrl/scripts/eval/cosmos_predict25_kling_eval.py:116-120`（argparse flag，**行号已移动**，旧树 123-127）、`:317-323`（`_keep_model_between_checkpoints`，**行号已移动**，旧树 293-299）（长期资产：eval 入口，`vrl/config/builders.py:161` 以 `python -m ...` 引用）
- 判死证据：`grep -rn "keep.model.between.checkpoints" vrl/ tests/` → 仅本文件及其测试，无 launch 脚本/doc/config 传入。函数体 `_keep_model_between_checkpoints`（`:317-323`）：flag 唯一效果是与 `--rebuild-model-between-checkpoints` 组合时 raise ValueError；单独设置是 no-op，因为 `return not bool(args.rebuild_model_between_checkpoints)` 忽略它，help 文案仍自称 `Deprecated: model reuse is now the default`。内部 eval CLI，无外部命令行兼容面。
- 动作：删除 `--keep-model-between-checkpoints` argparse flag；`_keep_model_between_checkpoints` 随之收敛为 `not args.rebuild_model_between_checkpoints`——内联进 `main()`（form-3 merge）。更新 `tests/scripts/test_cosmos_predict25_kling_eval.py` 中演练 flag 与互斥 error 的用例；`_generate_all` 测试传的是幸存 keyword 参数 `keep_model_between_checkpoints=`，不受影响。计算值 `keep_model_between_checkpoints` 在 `_generate_all` 被行为消费，保留。
- 注意：与 §3 的 `_video_to_cthw` 同处此文件——origin 已把生成路径内联进本 eval 脚本，改动前请以当前 checkout 为准逐行核对本文件的最新行号。

---

### 1.7 `evaluate(expected_updates, min_post_eval_points, endpoint_points, min_aesthetic_gain, min_gain_z, max_pickscore_relative_drop, max_pre_update_logprob_abs_diff)` — dead-arg（risk=low，STILL_VALID）

- 位置：`vrl/scripts/eval/sana_aesthetic_curve_verdict.py:88-98`（签名），回显进 `criteria` 于 `:245-251`（原范围 88-99，实质不变；长期资产：eval 入口）
- 判死证据：七个阈值 kwarg 仍在 `evaluate()` `:92-98`（默认值不变），仍原样回显进 `result["criteria"]`（`:245-251`）。`main()`（`:269` 附近）与 6 处测试调用（`test_sana_aesthetic_curve_verdict.py`）均只传 rows + `qualitative_audit=`，从不传阈值 kwarg，无非默认值生产者。七个阈值名在定义文件外唯一命中是测试里 `assert result["criteria"]["min_post_eval_points"] == 12`——读的是回显输出 dict，非 setter，hoist 后仍通过。无 `**kwargs` 转发、无 YAML/preset/entry-point。
- 动作：把七个阈值 kwarg hoist 为模块级 protocol 常量（它们是注册的 PASS/FAIL 协议值），从签名删除，只保留 `qualitative_audit` 为参数。`criteria` 输出 dict 不变。无测试需要更新。
- 注意（reconciles-with-prior-decision）：[[SPRINT_sana_aesthetic_trustworthy_curve]] 把这些值预注册为不可变协议（`预注册 300，不在中途因曲线形状延长或缩短`；`绝对增益 >= 0.10`；`至少 12 条 post-training fixed-eval 点`），模块 docstring 承诺保持 `the numerical decision mechanical`——可调 kwarg 反而与文件契约冲突。hoist 为模块级常量正对应 AGENTS.md ALL_CAPS 保留场景（协议值 / 刻意隔离的 config 表），符合而非违反架构卫生规则。

---

### 1.8 `_validate_scheduler` — dead-function（risk=low，RELOCATED）

- 位置：`vrl/scripts/eval/sana_checkpoint_compare.py:40`（import，**行号已移动**，旧树 36）、`:55`（`_validate_scheduler = require_scheduler` 别名，**行号已移动**，旧树 49）（长期资产：eval 对比脚本）
- 判死证据：`:40`（import）+ `:55`（别名），模块内零调用点（生产用 `_load_official_scheduler`/`_generate_prompt_group`）。跨仓 → 只有测试调用 `checkpoint_compare._validate_scheduler`（`test_sana_checkpoint_compare.py:204,553,561,566`——**行号已移动**，旧树 173,441,449,454；仅调用、从不 monkeypatch）。与两个 sibling 别名不同：`_validate_scheduler` 从不被 `run_comparison` 或任何生产路径调用，是只被测试续命的 re-export。删除不削弱生产校验——`load_official_scheduler`/`generate_prompt_images` 内部各自调 `require_scheduler`（`sana_inference.py:55,99`）。
- 动作：删除 `_validate_scheduler = require_scheduler` 别名（`:55`）与 `require_scheduler` import（`:40`）；把 `test_sana_checkpoint_compare.py:204,553,561,566` 改为直接调 `vrl.scripts.eval.sana_inference.require_scheduler`（`__all__` 导出、同一函数对象）——无 `test_sana_inference.py`，这四个测试是 `require_scheduler` 协议逻辑的唯一覆盖，须 repoint 而非删。**保留** sibling 别名 `_generate_prompt_group`（生产调用）与 `_load_official_scheduler`（生产调用 + 测试的 monkeypatch seam）——它们是活 monkeypatch 边界。

---

### 1.9 `_run_prefix(tf=...)` — dead-arg（risk=low，STILL_VALID）

- 位置：`vrl/scripts/eval/shared_prefix_divergence_probe.py:167`（def），唯一调用点 `:144`（原范围 144-146, 166-184，不变；一次性 probe；符号级 in-scope）
- 判死证据：函数体 `:166-184` 用 `sched`（timesteps + `sde_step_with_logprob`）、`args`（seed、noise_level）、`_embeds`、`_vel`、dtype、device——`tf` 从不出现；transformer 只经 `_vel` 闭包触达，闭包已从 `main()` 捕获 `tf`。唯一调用点 `main()` `:144`。无 `**kwargs` 转发，无测试。
- 动作：从 `_run_prefix` 签名删 `tf` 参数（`:167`），从唯一调用点删 `tf` 实参（`:145`）。无测试需更新（本 probe 无测试）。

---

### 1.10 `_new_report(args)` — dead-arg（risk=low，STILL_VALID）

- 位置：`vrl/scripts/eval/world_model_steppability_probe.py:39-53`（def），调用点 `:257`（行号不变；一次性 probe；符号级 in-scope）
- 判死证据：函数体 `:39-53` 返回的 dict 全为字面量（`probe`/`questions`/`checks`/`blockers`/`decision`/`next_step`/`env`，仅 `platform.python_version()`），零引用 `args`。`grep -rn "_new_report" vrl/ tests/` → 唯一调用点 `:257`（positional，无 `**kwargs`），另有 sibling 文件 `cosmos3_nano_generator_probe.py` 的**独立同名** helper（真读 `args.model`，故只有 steppability 这份是死的）；无测试调用者。
- 动作：从 `_new_report` 删 `args` 参数，`:257` 的 `report = _new_report(args)` → `_new_report()`。`argparse` import 经 `main()` 保持活跃。无测试需更新。
- 注意：两个独立一次性 probe 脚本间的签名对称**不是** keep-list 情形——跨家族一致性保护的是 thin function / 共享 shape，不是某个私有 helper 里没人读的参数。

---

### 1.11 `main()` 局部累加器 `logprobs` — dead-branch（risk=low，RELOCATED）

- 位置：`vrl/scripts/generation/full_sequence_denoise.py:203`（`logprobs = []` 初始化，**行号已移动**，旧树 195）、`:225`（`logprobs.append(...)`，**行号已移动**，旧树 217）（probe 脚本；符号级 in-scope）
- 判死证据：`grep -rn "logprobs" full_sequence_denoise.py` → 仅 `:203` 初始化 `logprobs = []` + `:225` `logprobs.append(sde.log_prob.detach())`，通读全文件无任何读取（仍是 write-only 局部累加器）。per-step 诊断打印直接用 `sde.log_prob`；`--check-replay` parity 用捕获的 `first_step` dict 快照——均不经该列表。唯一测试文件只演练 arg parsing / `_resolve_probe_model_build` / pre-loop `SystemExit`，从不进 denoise 循环。
- 动作：删除 `logprobs = []`（`:203`）与 `logprobs.append(sde.log_prob.detach())`（`:225`）。无测试需更新。
- 注意（sprint 命名歧义，非重叠）：[[SPRINT_native_generation_engine_program]] 引用的是**另一个** `vrl/generation/bindings/full_sequence_denoise/` 执行器包，**不是**本 `vrl/scripts/generation/full_sequence_denoise.py` 脚本，无 sprint 排序约束。

---

### 1.12 `run_e2e(iters=..., warmup=...)` — dead-arg（risk=low，RELOCATED）

- 位置：`vrl/scripts/perf/common/diffusion_runtime.py:166`（def `run_e2e(runtime, cfg, device, iters=3, warmup=2)`，**行号已移动**，旧树 137-141）（长期资产：perf 共享 helper）
- 判死证据：唯一调用者 `generation_bottleneck_profile.py:157`（**行号已移动**，旧树 154）仍 `run_e2e(runtime, cfg, device)`——从不传 `iters`/`warmup`，无 `**kwargs` 转发。无 partial/动态分发/YAML/doc/测试引用。函数体两参数均活（warmup loop、timed loop、`median of {iters}` 打印）——是没人设置的参数化，非死分支。
- 动作：从 `run_e2e` 签名（`:166`）删 `iters=3, warmup=2`，在函数体内内联为局部量（`iters = 3` / `warmup = 2`）。
- 注意（**不要**把 caller 的 `--steps/--warmup` 接进来）：审计原文的「首选动作」（wiring CLI flags）语义错误，禁止采用。`generation_bottleneck_profile.py` 的 `--steps` 文档单位是 profiled denoise steps（一 step = 一次 `step_fn`），而 `run_e2e` 的 `iters` 计整幅 encode+denoise+decode 图像。接进来会在 help 文案不变的情况下把 flag 从「6 denoise steps」悄悄改成「6 full images」（约 `num_steps` 倍工作量）。若将来确需 per-run 控制 e2e 迭代数，那是需要独立 `--e2e-iters/--e2e-warmup` flag 的新功能，非本次清理。无测试引用 `run_e2e`。**保留** `cuda_mean_ms/cuda_median_ms`（多 probe 调用点显式传 iters/warmup）与 `kernel_launches_per_step(steps=3)`（uniform API shape，跨家族一致性 keep-list）。

---

### 1.13 `build_model` — single-caller-merge（risk=low，RELOCATED）

- 位置：`vrl/scripts/perf/common/diffusion_runtime.py:42-66`（def，**行号已移动**，旧树 30-48）（长期资产：perf 共享模块）
- 判死证据：`build_model` 仍恰一个调用者：`teacache_drift_probe.py`（import `:45`、call `:204`——**行号已移动**，旧树 43/220）。无测试调用（`test_diffusion_runtime.py` 只 import `build_runtime`），无字符串/registry/YAML/pyproject。函数体仍是 `build_runtime` + dtype-vs-resolved-precision 校验（签名细化为 `(root, device, dtype, *, precision)`，概念不变），docstring 仍自称 `Compatibility facade for the recorded TeaCache drift probe`——一个 probe 的 dtype-guard 决策被 park 进共享模块（form-3）。`build_runtime` 才是真正的共享 builder。
- 动作：把 `build_model` 函数体（`build_runtime` + dtype-vs-resolved-precision 校验）移入 `vrl/scripts/perf/teacache_drift_probe.py` 作私有 helper（或内联进其 `main()`），从 `common/diffusion_runtime.py` 删除 `build_model`；`dtype_to_precision_token` 作为普通 lazy import 随函数体迁移。无测试需删。**不动** `build_runtime/make_step_fn/run_e2e` sibling——它们有多个 probe 调用者，是 `common/` 的合法共享面。

---

### 1.14 `log_gpu_preflight(report=...)` — dead-arg（risk=low，STILL_VALID）

- 位置：`vrl/scripts/perf/gpu_preflight.py:203-205`（原范围 203-206，实质不变；长期资产：`python -m vrl.scripts.perf.gpu_preflight` 的 CLI body）
- 判死证据：def 仍在 `:203`（`report: GpuPreflight | None = None`），函数体 `r = report or run_gpu_preflight()`（`:205`）。唯一调用者是模块自身 `main()`（`:221`，零参调用）；`__all__` 条目 `:226`。无测试 import（`test_gpu_preflight.py` 只 import `run_gpu_preflight`），无 partial/getattr/`**kwargs`。参数唯一效果是当传入预计算 report 时跳过 `run_gpu_preflight()`——从未落地的外部便利；`run_gpu_preflight` 本身已模块级缓存（`_REPORT_CACHE`），该便利即便原则上也冗余。
- 动作：从 `log_gpu_preflight` 删 `report: GpuPreflight | None = None` 参数，`:205` 简化为 `r = run_gpu_preflight()`。无 caller/测试更新。**保留函数本身**——它是 CLI body 及 [[SPRINT_cosmos_video_mfu_kernels]] 点名的保留交付物。`GpuPreflight` dataclass 已在定义处标注 `Display/provenance-only`，是获许豁免，不动；`run_gpu_preflight/measured_bf16_peak_tflops` 有活调用者，不动。

## 2. 已由 origin 落地（本次复核确认，无需再做）

- `RunMetrics.run_dir`（`vrl/scripts/perf/reward_overlap_benchmark.py`）— 原为 dead-field（provenance 字段零 attribute read）。复核：`RunMetrics` 类体现仅 `collect_wall/generation_wall/reward_wall/overlap/reward_queue_wait`，无 `run_dir`；构造点已是 `RunMetrics()`（原 `RunMetrics(run_dir=run_dir)`）；`grep -rnF '.run_dir'` 零 attribute 读。**已由** `7ac288d2 refactor(perf): remove dead run directory state` **落地。**

## 3. 情况已变（需重新评估）

- `_video_to_cthw`（`vrl/scripts/eval/cosmos_predict25_kling_eval.py`）— 原判为「test-only 兼容 facade，body 纯委托 `denoise_video_generation.video_to_cthw`」（form-1+4，动作是删 facade + 删喂它的 import + repoint 测试到 `denoise_video_generation.video_to_cthw`）。**该前提已不成立**：origin 把生成/解码路径内联进本 eval 脚本（refactor 提交 `a6a2cd2b`/`8fc7507f`/`ca0dfd9f`/`171e95da`），`_video_to_cthw` 现在是**完整本地实现**（ndim 4/5 检查、channel-first permute）于 `:503`，且**在生产路径被调用**——`:500` `return _video_to_cthw(decoded.detach().cpu())`（generate/decode 内）。`from ...denoise_video_generation import video_to_cthw` 这个 import 已被删除。符号现有活生产消费者，**不再是死代码**，原「删 facade」动作作废。测试仍在 `tests/scripts/test_cosmos_predict25_kling_eval.py:71` 演练它。
  - 重新评估结论：**不删**。若要清理，方向变为「本地 `_video_to_cthw` 与 `denoise_video_generation.video_to_cthw` 是否为可合并的重复实现」（form-4 重估），须先 diff 两者函数体确认语义等价——但这是一条**新的、不同的** finding，不属本簇原动作，本簇不执行。

## 4. 验证协议

- **每条删除后**：`ruff check <touched files>` + `ruff format --check <touched files>`（仅对本条改动的 Python 文件）。
- **全簇完成后**：`pytest tests/data/ tests/scripts/`（含 `tests/scripts/eval/ tests/scripts/perf/`）+ `pytest -m "not e2e and not slow_test"` 子集不新增失败。
- **基线（清理前）**：测试现基于 **main @ `7c748532`** 跑（旧审计基线 `88ed756e` 已作废）。删除前先在当前 checkout 记录 fast subset 通过数与 pre-existing 失败清单，删除后须保持不新增失败；`vrl.config.lint` 与 `ruff check .` 全绿。
- **逐条动作触及的测试文件**（须相应更新/repoint 或确认无需改动）：
  - §1.1 `anime_probe_common`：`tests/scripts/test_anime_probe_common.py`（仅覆盖 `optional_finger_cv`，无需改）。
  - §1.2 `FAILURE_LABELS`：无测试改动（连带删 `datasets/danbooru/config.yaml:219`）。
  - §1.3 `_current_http_download`：`tests/data/test_setup.py:204`（monkeypatch seam 依旧生效，无需改）。
  - §1.4 `build_anatomy_prompts`：`tests/data/test_danbooru.py`（不 import 该函数，无需改）。
  - §1.5 `firsts_by_episode`：无测试改动。
  - §1.6 `--keep-model-between-checkpoints`：`tests/scripts/test_cosmos_predict25_kling_eval.py`（更新/删除 flag 与互斥 error 用例；以当前 checkout 行号为准）。
  - §1.7 `evaluate` 阈值：`tests/scripts/eval/test_sana_aesthetic_curve_verdict.py`（读回显输出，无需改）。
  - §1.8 `_validate_scheduler`：`tests/scripts/eval/test_sana_checkpoint_compare.py:204,553,561,566`（repoint 到 `sana_inference.require_scheduler`）。
  - §1.9 `_run_prefix(tf)`：无测试（probe 无测试）。
  - §1.10 `_new_report(args)`：无测试。
  - §1.11 `logprobs`：`tests/scripts/test_full_sequence_denoise_generate.py`（不进 denoise 循环，无需改）。
  - §1.12 `run_e2e`：无测试引用，无需改。
  - §1.13 `build_model`：`tests/scripts/perf/test_diffusion_runtime.py`（只覆盖 `build_runtime`，无需改）。
  - §1.14 `log_gpu_preflight`：`tests/scripts/perf/test_gpu_preflight.py`（只 import `run_gpu_preflight`，无需改）。

## 5. Non-Goals

- 不删被「能 raise 的校验」、控制流分支、runtime/config/Ray 调用消费的字段（AGENTS 死字段规则）——例如 §1.4 保留的 safety 侧 `candidate_pool_factor`（`:250` 无条件使用）、`_interleave_bucket_rows`（`_select_quota_rows` 活调用）；§1.6 保留的 `keep_model_between_checkpoints` 计算值（`_generate_all` 行为消费）；§1.8 保留的 `_generate_prompt_group`/`_load_official_scheduler` 活 monkeypatch 边界。
- 不动 DO-NOT-FLAG 豁免项（`_GENERATION_CUDA_RUNTIME_RESIDUAL_BYTES`、`_REWARD_CUDA_RUNTIME_RESIDUAL_BYTES`、`ensure_loaded`、`process_gpu_used_bytes` NVML、sana/hunyuan `prepare_latents` 修复）；§1.14 的 `GpuPreflight` dataclass 已标注 `Display/provenance-only`，同属豁免，不动。
- 不为省行数扁平化 protocol/lazy-import/跨家族一致性 thin function——§1.12 保留 `run_e2e` 函数本身及 `kernel_launches_per_step` 的 uniform API shape，§1.13 保留 `build_runtime/make_step_fn` 共享面，§1.14 保留 `log_gpu_preflight` CLI body（sprint 交付物）。
- **cluster-specific**：`*_probe` / `*_spike` 一次性脚本**整体**不在本簇范围——只清理其内部符号级死代码（§1.9、§1.10、§1.11）；不把整支 probe 当死代码删。`scripts/eval/` 与 `scripts/data/` 的长期入口脚本本身保留，只删内部死符号。
- 不改任何 flag 的语义单位——§1.12 明确禁止把 `--steps/--warmup` 接入 `run_e2e`（denoise-step vs full-image 单位不匹配）。
- **本次复核新增**：§3 的 `_video_to_cthw` 已不是死代码（有活生产消费者），本簇不动它；§2 的 `RunMetrics.run_dir` 已由 origin 删除，本簇不重复。

## References

被触及的文件与行（以当前 main @ `7c748532` 为准；RELOCATED 条目行号已按复核更新）：
- `vrl/scripts/data/danbooru.py:51-52, 105, 167-168, 180-196, 674, 700-701, 710-718, 1314, 1317-1318, 1390, 1410, 1427` + `datasets/danbooru/config.yaml:218-219`
- `vrl/scripts/data/video_world.py:356-358, 377`
- `vrl/scripts/eval/anime_probe_common.py:140, 162, 276, 279, 363, 366, 371` + `vrl/scripts/eval/probe_anime_anatomy_report.py:50, 88, 253`
- `vrl/scripts/eval/cosmos_predict25_kling_eval.py:116-120, 317-323`（§1.6）；`:500, :503`（§3，已活化，不删）
- `vrl/scripts/eval/sana_aesthetic_curve_verdict.py:88-98, 245-251, 269`
- `vrl/scripts/eval/sana_checkpoint_compare.py:40, 55` + `vrl/scripts/eval/sana_inference.py:55, 99, 159`
- `vrl/scripts/eval/shared_prefix_divergence_probe.py:144-146, 166-184`
- `vrl/scripts/eval/world_model_steppability_probe.py:39-53, 257`
- `vrl/scripts/generation/full_sequence_denoise.py:203, 225`
- `vrl/scripts/perf/common/diffusion_runtime.py:42-66, 166` + `vrl/scripts/perf/teacache_drift_probe.py:45, 204`
- `vrl/scripts/perf/gpu_preflight.py:203-205, 221, 226`
- `vrl/scripts/perf/reward_overlap_benchmark.py`（§2，`run_dir` 字段已由 `7ac288d2` 删除）

测试文件：
- `tests/scripts/test_cosmos_predict25_kling_eval.py`（§1.6 flag 用例；§3 `:71` 仍演练已活化的 `_video_to_cthw`）
- `tests/scripts/eval/test_sana_checkpoint_compare.py:204, 553, 561, 566`
- `tests/data/test_setup.py:204`、`tests/data/test_danbooru.py`
- `tests/scripts/test_anime_probe_common.py`、`tests/scripts/eval/test_sana_aesthetic_curve_verdict.py`
- `tests/scripts/perf/test_diffusion_runtime.py`、`test_gpu_preflight.py`、`test_reward_overlap_benchmark.py`
- `tests/scripts/test_full_sequence_denoise_generate.py`

关联 sprint：
- [[SPRINT_deadcode_00_overview]]
- [[SPRINT_sana_aesthetic_trustworthy_curve]]（§1.7 阈值预注册协议）
- [[SPRINT_cosmos_video_mfu_kernels]]（§1.14 `gpu_preflight.py` 保留交付物）
- [[SPRINT_native_generation_engine_program]]（§1.11 命名歧义澄清：非本文件，无重叠）
