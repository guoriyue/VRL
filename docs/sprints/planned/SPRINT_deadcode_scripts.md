# SPRINT: `vrl/scripts/` 死代码清理（planned）

状态：**planned（2026-07-23）**。共 **16 条**确认死代码（1 条 medium：`anime_probe_common` 结果字典死键；15 条 low），来源为 dead-code-audit workflow 的对抗验证输出，覆盖 `scripts/data/`、`scripts/eval/`、`scripts/generation/`、`scripts/perf/` 四个子目录。
来源：dead-code-audit workflow（五种死代码形态 + 对抗验证 + 删除类二次字符串引用检查）。
关联：[[SPRINT_deadcode_00_overview]]；`scripts/eval/sana_aesthetic_curve_verdict.py` 的阈值 hoist 与 [[SPRINT_sana_aesthetic_trustworthy_curve]] 的预注册协议对齐（见 §1.9）；`scripts/perf/gpu_preflight.py` 属 [[SPRINT_cosmos_video_mfu_kernels]] 的保留交付物（见 §1.15）。

## 0. 一句话

本簇是 `vrl/scripts/` 下**长期资产内部**的符号级死代码清理（数据生成器、eval 入口、perf 共享 helper 都是长期资产；`*_probe` 一次性脚本整体不在范围内，但其内部死符号在范围内）。主形态是 **dead-arg / dead-field**（没有生产者的参数、零读者的结果字典键与派生字段），另有两条 form-3 单调用者内联与两条 form-1/4 test-only facade。最锋利的一条是 `anime_probe_common.py` 的 `hamer_verdict` 及一整组结果字典键（`"verdict"`/`"n_hands"`/`"n_persons"`/`"annotated"` overlay）——报告入口 `probe_anime_anatomy_report.py` 的 docstring 声称展示 `verdict` 与 per-hand `finger_cv`，实际只读 `mean_finger_cv`，是文档腐烂的直接证据。误删风险主要落在 `danbooru.py`：`bucket_balance`/`candidate_pool_factor` 删除会连带触发 `bucket_weights` 局部量、`preferred_min_score` 三元式与 `build_prompt_rows` 签名的 `| None` 收窄——必须整段一起改，漏改会留下同型 form-2 死语义残留或让类型宣称接受一个运行时会崩溃的 `None`。

## 1. 待删清单（逐条，带证据与动作）

顺序：medium-risk 先行（§1.1），其余 low-risk 按文件/目录聚合（`scripts/data/` → `scripts/eval/` → `scripts/generation/` → `scripts/perf/`）。

---

### 1.1 `hamer_verdict` / `hamer_probe_image` 结果字典键 `"verdict"`/`"n_hands"`/`"annotated"` + `rtmw_metrics` 键 `"n_persons"` — dead-field（risk=**medium**）

- 位置：`vrl/scripts/eval/anime_probe_common.py:139-147, 161-169, 275-281, 362-378`
- 判死证据：
  - 模块唯一生产消费者是 `probe_anime_anatomy_report.py`（`grep -rn "anime_probe_common"` → 仅 `probe_anime_anatomy_report.py:50` import + `tests/scripts/test_anime_probe_common.py:5` 只 import `optional_finger_cv`）。
  - 结果键零读者：`grep -rnF '"n_persons"' vrl/ tests/` → 仅定义行 `anime_probe_common.py:140,162`；`grep -rnF '"n_hands"' vrl/ tests/` → 仅定义行 `:276,363`；`build_report()` 只读 rtmw 的 `body_coverage/hand_coverage/mean_conf/bone_ratio_err/visible_hands/annotated` 与 `mean_finger_cv`（经 `optional_finger_cv(hm)`），`main()` 只对 RTMW overlay 做 `cv2.imwrite(m["annotated"])`——hamer 侧 `annotated`（`cv2.circle` loop）从不保存/渲染。
  - `hamer_verdict` 恰一个调用者 `anime_probe_common.py:279`，输出落进不被读的 `"verdict"` 键 → live caller, dead semantics（form-2）。**证据订正**：`no_hands_result` 并不调用 `hamer_verdict`，而是硬编码 `"verdict": "NOT_DETECTED"`（`:366`）——审计原文对此有误。
  - 无 `json.dump`/`asdict`/`.items()`/kwargs 转发整字典；无 YAML/entry-point/字符串分发；sprint 文档仅在 `done/` 审计记录里出现。
- 动作：二选一，推荐 **(b) 删除**（(a) 反向补齐入报告的成本高、非本簇目标）：
  - (a) 把 hamer 结果 surface 进 HTML 报告——加 `"verdict"`/`"n_hands"` 列，从 `hm["hands"]` 渲染 per-hand `finger_cv`，把 hamer `"annotated"` overlay 与 RTMW 一并保存；给每个被标键一个消费者，使 docstring 成真。
  - (b) 删除：`hamer_verdict()`（单调用者、输出不被读）；`hamer_probe_image` 返回 dict 的 `"verdict"`/`"n_hands"`/`"hands"` 键；`"annotated"` 键及其绘制代码（`annotated = img_bgr.copy()` 于 `:243`、`:255-256` 的 handedness/color 计算、`:258-260` 的 `cv2.circle` loop）——之后 `hand_results` 收敛为 per-hand `finger_cv` 浮点列表，`"j2d_px"`/`"handedness"` per-hand 字段随之消失；`no_hands_result` 中匹配的 `"verdict"`/`"n_hands"`/`"hands"`/`"annotated"` 键与 `cv2.putText` overlay（收敛为 `{"mean_finger_cv": None}` 或直接内联到其唯一调用点 `:271`）；`rtmw_metrics` 两个返回 dict 的 `"n_persons"` 键（`:140,162`）。**同时**把 `probe_anime_anatomy_report.py:30` docstring 从 `"finger_cv per hand, verdict"` 改为 `"mean finger_cv"`，与报告实际展示一致。
  - `tests/scripts/test_anime_probe_common.py` 只覆盖 `optional_finger_cv`，无需改动。
- 注意（medium）：这是唯一 medium 条，reviewer 需重点核对——(1) 删除 `"annotated"` 会连带删掉 `j2d_px`/`handedness`（它们只为 `cv2.circle` loop 服务，审计原动作漏了 `"hands"` 键，本条已补入）；(2) docstring 修正是本条自身引用的腐烂点，删除路径必须一并改，否则入口文档继续留假承诺。

---

### 1.2 `FAILURE_LABELS` — dead-field（risk=low）

- 位置：`vrl/scripts/data/danbooru.py:105, 1427`（长期资产：danbooru 数据集生成器）
- 判死证据：`grep -rn "FAILURE_LABELS" vrl/ tests/ configs/ pyproject.toml` → 仅 `danbooru.py:105`（定义 `FAILURE_LABELS = set(_ANATOMY["failure_labels"])`）+ `:1427`（`__all__` 条目），零消费者（另有一处 `docs/sprints/done/` 历史记录，非活引用）。`grep -rn "failure_labels" vrl/scripts/data/danbooru.py datasets/danbooru/config.yaml` → 仅定义行 + `config.yaml:219` 数据源。`danbooru import *` 零命中；`config.yaml` 无 YAML anchor（`_ANATOMY` 各 key 逐个显式取用，无 keys 迭代/schema 校验）。
- 动作：删除 `danbooru.py:105` 的 `FAILURE_LABELS = set(_ANATOMY["failure_labels"])` 与 `__all__` 中的 `"FAILURE_LABELS"`（`:1427`）；同一生命周期内一并删除 `datasets/danbooru/config.yaml:219` 的 `failure_labels:` 键（含 `:218` 注释行）——该 YAML 键的唯一读者就是这个死常量。无测试需要更新。

---

### 1.3 `_current_http_download` — single-caller-merge（risk=low）

- 位置：`vrl/scripts/data/danbooru.py:1313-1318`（长期资产）
- 判死证据：函数体 `return globals().get("_http_download", http_download)`——`_http_download` 在 `:1314` 恒定义，`globals().get` 的 fallback 分支不可达，函数等价于直接读全局。`grep -rn "_current_http_download\|_http_download" vrl/ tests/` → 调用者仅 `danbooru.py:1390,:1410`（两个 `_cmd` 函数），测试只 `monkeypatch.setattr(danbooru, "_http_download", ...)`（`tests/data/test_setup.py:149`），从不引用 getter；getter 不在 `__all__`。
- 动作：删除 `_current_http_download()`；两个调用点 `_cmd_anime_positives`（`:1390`）与 `_cmd_anime_fetch_images`（`:1410`）改为直接引用 `fetch=_http_download`（模块全局在命令执行时解析，`test_setup.py:149` 对 `danbooru._http_download` 的 monkeypatch 依旧生效）。保留 `_http_download = http_download`（`:1314`）这一测试注入 seam 本身。
- 注意：标签小瑕疵——实际有**两个**调用者（非单一），但动作对两处都处理，安全完整。真正的 test seam 是 `_http_download` 全局，须保留；getter 只是冗余间接层。

---

### 1.4 `build_anatomy_prompts(bucket_balance=, candidate_pool_factor=)` — dead-arg（risk=low）

- 位置：`vrl/scripts/data/danbooru.py:51-52, 167-168, 180-184, 674, 717-718`（长期资产）
- 判死证据：`grep -rn "bucket_balance\|candidate_pool_factor" vrl/ tests/` → anatomy 侧仅 `danbooru.py:167,168,180,181,182,184`，零外部生产者（`:230,:250` 是 safety 侧同名参数，函数体无条件使用，**活的**，不在本条）。两个调用方 `build_default_manifests`（`:147`）与 `_cmd_anime_prompts`（`:1366`）只传 `metadata/download_metadata`；`register()` 的 `anime-prompts` 子命令只暴露 `--metadata`，无 `**kwargs` 转发。`"natural"` 值零生产者 → `:180` 的 `bucket_weights = DEFAULT_BUCKET_WEIGHTS if bucket_balance == "quota" else None` 恒取真值（`DEFAULT_BUCKET_WEIGHTS` 来自 `config.yaml` 11 项非空表），`candidate_pool_factor` 分支（`:184`）不可达。`tests/data/test_danbooru.py` 不 import `build_anatomy_prompts`，`bucket_weights` 仅传显式非空 dict。
- 动作：在 `vrl/scripts/data/danbooru.py`：
  1. 从 `build_anatomy_prompts` 签名删 `bucket_balance`（`:167`）与 `candidate_pool_factor`（`:168`），删常量 `ANATOMY_BUCKET_BALANCE`（`:51`）与 `ANATOMY_CANDIDATE_POOL_FACTOR`（`:52`）。
  2. 删 `:180-185` 整段（`bucket_weights` 选择、`:181-182` 校验、`:183-184` `candidate_limit` 计算），并在 `:186-196` 的 `build_prompt_rows` 调用中删掉 `candidate_limit=`（`:192`）与 `bucket_weights=`（`:195`）关键字（`DEFAULT_BUCKET_WEIGHTS` 已是其默认值），`:189` 收敛为 `preferred_min_score=preferred_min_score`。
  3. 从 `build_prompt_rows` 删 `candidate_limit` 参数（`:674`）及其早停分支（`:700-701`）。
  4. 删 `:710/:717-718` 的 `if/else`，无条件调用 `_select_quota_rows`，并把 `:677` 的注解从 `Mapping[str, float] | None = DEFAULT_BUCKET_WEIGHTS` 收窄为 `Mapping[str, float] = DEFAULT_BUCKET_WEIGHTS`。
  5. 保留 `_interleave_bucket_rows`（`_select_quota_rows` 内 `:1184/:1191` 活调用）、保留 safety 侧 `SAFETY_CANDIDATE_POOL_FACTOR`（`:61`）/`candidate_pool_factor`（`:230,:250`）与 `build_danbooru_safety_prompt_rows` 的 `candidate_limit`（生产 `:250` 与测试 `test_danbooru.py:359` 均活）。
  6. `tests/data/test_danbooru.py` 无需改动。
- 注意（load-bearing 收敛链）：这是本簇误删风险最高的一条。`:189` 的 `preferred_min_score if bucket_weights else None` 与 `:710-718` 的 `if bucket_weights` 均读同一局部 `bucket_weights`——删 `bucket_balance` 后 `bucket_weights` 恒为真值，这些三元式/分支成为同型 form-2 死语义残留，**必须在同一改动内一起收敛**（上文步骤 2、4 已含）。漏改步骤 4 的 `| None` 收窄会让签名宣称接受 `None`，而 `_bucket_quotas` 迭代 `None` 会崩溃。审计对本条给出 `verdict=revise` 提示，正是这两处；本条动作已完整覆盖。

---

### 1.5 `_iter_lerobot_v21` 局部 dict 死键 `firsts_by_episode[...]["task_index"]` — dead-field（risk=low）

- 位置：`vrl/scripts/data/video_world.py:356-359`（长期资产：video importer）
- 判死证据：`grep -rn "firsts_by_episode" vrl/ tests/` → 仅构造 `:356` + 唯一读点 `:377`（`firsts_by_episode[episode]["global_index"]`，写入 metadata `source_global_index`）；`"task_index"` 键无任何读者。同循环内 `:365` 的 `task_index` 由 `targets[position]`（`:360` 构造）提供，与该 dict 无关。函数内局部 dict，不序列化、不出模块。
- 动作：把 `firsts_by_episode` 简化为 `{episode: idx}`（值直接存 `global_index`），删死键 `"task_index"`；`:377` 改为 `firsts_by_episode[episode]`。`task_index` 活来源是 `:360` 的 `targets` 映射，不受影响。无测试需要更新。

---

### 1.6 `--keep-model-between-checkpoints` — dead-config-knob（risk=low）

- 位置：`vrl/scripts/eval/cosmos_predict25_kling_eval.py:123-127, 293-299`（长期资产：eval 入口，`vrl/config/builders.py:161` 以 `python -m ...` 引用）
- 判死证据：`grep -rn "keep.model.between.checkpoints" vrl/ tests/` → 仅本文件及其测试，无 launch 脚本/doc/config 传入。函数体 `_keep_model_between_checkpoints`（`:293-299`）：flag 唯一效果是与 `--rebuild-model-between-checkpoints` 组合时 raise ValueError（`:294`）；单独设置是 no-op，因为 `return not bool(args.rebuild_model_between_checkpoints)`（`:299`）忽略它，help 文案自称 `Deprecated: model reuse is now the default`。内部 eval CLI，无外部命令行兼容面。
- 动作：删除 `--keep-model-between-checkpoints` argparse flag；`_keep_model_between_checkpoints` 随之收敛为 `not args.rebuild_model_between_checkpoints`——内联进 `main()`（form-3 merge）。更新 `tests/scripts/test_cosmos_predict25_kling_eval.py:100-115`（演练 flag 与互斥 error 的用例）；`:191` 的 `_generate_all` 测试传的是幸存 keyword 参数 `keep_model_between_checkpoints=`，不受影响。计算值 `keep_model_between_checkpoints` 在 `_generate_all`（`:320,:348`）被行为消费，保留。

---

### 1.7 `_video_to_cthw` — dead-function（risk=low）

- 位置：`vrl/scripts/eval/cosmos_predict25_kling_eval.py:41-44`（含 `:30-32` 的 feeding import）（长期资产）
- 判死证据：`grep -rn "video_to_cthw" vrl/ tests/` → 定义 `denoise_video_generation.py:81`（`__all__` `:97`）、facade `cosmos_predict25_kling_eval.py:41-44`、单一测试调用 `tests/scripts/test_cosmos_predict25_kling_eval.py:55`。函数体 `return video_to_cthw(video)`——纯委托，docstring 自称 `Compatibility facade`；生产路径从不调用 facade 或 `video_to_cthw`（真正的 layout 归一化在 `generate_one_video` 内经 `denoise_video_generation.py:78`），`:31` 的 import 仅为喂 facade。test-only caller = 死。
- 动作：删除 `_video_to_cthw` 与已无用的 `video_to_cthw` import（`:30-32`）；把 `tests/scripts/test_cosmos_predict25_kling_eval.py:50-58`（`test_video_to_cthw_accepts_btchw_layout`）改为直接 `from vrl.scripts.eval.denoise_video_generation import video_to_cthw`——测试本身有价值（验证真实 BTCHW→CTHW 语义），只删 facade。

---

### 1.8 `evaluate(expected_updates, min_post_eval_points, endpoint_points, min_aesthetic_gain, min_gain_z, max_pickscore_relative_drop, max_pre_update_logprob_abs_diff)` — dead-arg（risk=low）

- 位置：`vrl/scripts/eval/sana_aesthetic_curve_verdict.py:88-99`（长期资产：eval 入口）
- 判死证据：`grep -rn "evaluate("` → 唯一非测试调用者 `main()`（`:269`）只传 `eval_rows, train_rows, qualitative_audit=`；6 处测试调用（`test_sana_aesthetic_curve_verdict.py:50,56,66,73,84,102`）均只传 positional rows + `qualitative_audit=`，从不传阈值 kwarg。七个阈值名在定义文件外唯一命中是 `test_...:76` 的 `assert result["criteria"]["min_post_eval_points"] == 12`——读的是回显输出 dict，非 setter，hoist 后仍通过。无 `**kwargs` 转发、无 YAML/preset/entry-point。函数体把这些值当固定 gate 阈值并原样回显进 `criteria` 输出 dict（`:245-251`）——无 caller 能触达的参数化。
- 动作：把七个阈值 kwarg hoist 为模块级 protocol 常量（它们是注册的 PASS/FAIL 协议值），从签名删除，只保留 `qualitative_audit` 为参数。`criteria` 输出 dict 不变。无测试需要更新。
- 注意（reconciles-with-prior-decision）：[[SPRINT_sana_aesthetic_trustworthy_curve]] 把这些值预注册为不可变协议（`预注册 300，不在中途因曲线形状延长或缩短`；`绝对增益 >= 0.10`；`至少 12 条 post-training fixed-eval 点`），模块 docstring 承诺保持 `the numerical decision mechanical`——可调 kwarg 反而与文件契约冲突。hoist 为模块级常量正对应 AGENTS.md ALL_CAPS 保留场景（协议值 / 刻意隔离的 config 表），符合而非违反架构卫生规则。

---

### 1.9 `_validate_scheduler` — dead-function（risk=low）

- 位置：`vrl/scripts/eval/sana_checkpoint_compare.py:36, 49`（长期资产：eval 对比脚本）
- 判死证据：`grep -n "_validate_scheduler\|validate_scheduler" sana_checkpoint_compare.py` → 仅 `:36`（import）+ `:49`（`_validate_scheduler = validate_scheduler` 别名），模块内零调用点。跨仓 → 只有测试调用 `checkpoint_compare._validate_scheduler`（`test_sana_checkpoint_compare.py:173,441,449,454`，仅调用、从不 monkeypatch）。与两个 sibling 别名不同：`_validate_scheduler` 从不被 `run_comparison` 或任何生产路径调用，是只被测试续命的 re-export。删除不削弱生产校验——`load_official_scheduler`/`generate_prompt_images` 内部各自调 `validate_scheduler`（`sana_inference.py:55,99`）。
- 动作：删除 `_validate_scheduler = validate_scheduler` 别名（`:49`）与 `validate_scheduler` import（`:36`）；把 `test_sana_checkpoint_compare.py:173,441,449,454` 改为直接调 `vrl.scripts.eval.sana_inference.validate_scheduler`（`__all__` 导出、同一函数对象）——无 `test_sana_inference.py`，这四个测试是 `validate_scheduler` 协议逻辑的唯一覆盖，须 repoint 而非删。**保留** sibling 别名 `_generate_prompt_group`（生产调用 `:324`）与 `_load_official_scheduler`（生产调用 `:151/:176` + 测试 `:182` 的 monkeypatch seam）——它们是活 monkeypatch 边界。

---

### 1.10 `_run_prefix(tf=...)` — dead-arg（risk=low）

- 位置：`vrl/scripts/eval/shared_prefix_divergence_probe.py:144-146, 166-184`（一次性 probe；符号级 in-scope）
- 判死证据：`grep -rn "_run_prefix"` → 唯一调用点 `main()` `:144`，定义 `:167`。函数体（`:166-184`）用 `sched`（timesteps + `sde_step_with_logprob`）、`args`（seed、noise_level）、`_embeds`、`_vel`、dtype、device——`tf` 从不出现；transformer 只经 `_vel` 闭包触达，闭包已从 `main()` 捕获 `tf`。`sed -n '166,185p' | grep tf` 仅命中 `:167` 签名本身，函数体零使用。无 `**kwargs` 转发，无测试。
- 动作：从 `_run_prefix` 签名删 `tf` 参数（`:167`），从唯一调用点删 `tf` 实参（`:145`）。无测试需更新（本 probe 无测试）。

---

### 1.11 `_new_report(args)` — dead-arg（risk=low）

- 位置：`vrl/scripts/eval/world_model_steppability_probe.py:39-53, 257`（一次性 probe；符号级 in-scope）
- 判死证据：函数体（`:39-53`）返回的 dict 全为字面量（`probe`/`questions`/`checks`/`blockers`/`decision`/`next_step`/`env`，仅 `platform.python_version()`），零引用 `args`。`grep -rn "_new_report" vrl/ tests/` → 唯一调用点 `:257`（positional，无 `**kwargs`），另有 sibling 文件 `cosmos3_nano_generator_probe.py` 的**独立同名** helper；无测试调用者。对照：`cosmos3_nano_generator_probe.py:36-45` 的 `_new_report` 真读 `args.model`（`"model": args.model or _DEFAULT_MODEL`），故只有 steppability 这份是死的。
- 动作：从 `_new_report` 删 `args` 参数，`:257` 的 `report = _new_report(args)` → `_new_report()`。`argparse` import 经 `main()` 保持活跃。无测试需更新。
- 注意：两个独立一次性 probe 脚本间的签名对称**不是** keep-list 情形——跨家族一致性保护的是 thin function / 共享 shape，不是某个私有 helper 里没人读的参数。

---

### 1.12 `main()` 局部累加器 `logprobs` — dead-branch（risk=low）

- 位置：`vrl/scripts/generation/full_sequence_denoise.py:195, 217`（probe 脚本；符号级 in-scope）
- 判死证据：`grep -rn "logprobs" full_sequence_denoise.py` → 仅 `:195` 初始化 `logprobs = []` + `:217` `logprobs.append(sde.log_prob.detach())`，通读全文件（281 行）无任何读取。per-step 诊断打印（`:219-223`）直接用 `sde.log_prob`；`--check-replay` parity（`:268`）用 `:214` 捕获的 `first_step` dict 快照——均不经该列表。`main()` 局部量，外部机制不可达；唯一测试文件只演练 arg parsing / `_resolve_probe_model_build` / pre-loop `SystemExit`，从不进 denoise 循环。
- 动作：删除 `logprobs = []`（`:195`）与 `logprobs.append(sde.log_prob.detach())`（`:217`）。无测试需更新。
- 注意（sprint 命名歧义，非重叠）：in-flight sprint [[SPRINT_native_generation_engine_program]]（`docs/sprints/SPRINT_native_generation_engine_program.md:45,357`）引用的是**另一个** `vrl/generation/bindings/full_sequence_denoise/` 执行器包，**不是**本 `vrl/scripts/generation/full_sequence_denoise.py` 脚本。`git status --porcelain | grep full_sequence_denoise` 无命中——本文件不在未提交改动集内，无 sprint 排序约束。

---

### 1.13 `run_e2e(iters=..., warmup=...)` — dead-arg（risk=low）

- 位置：`vrl/scripts/perf/common/diffusion_runtime.py:137-141`（长期资产：perf 共享 helper）
- 判死证据：`grep -rn -w run_e2e vrl/ tests/` → 定义 `:137` + 唯一调用者 `generation_bottleneck_profile.py:154`（`run_e2e(runtime, cfg, device)`，从不传 `iters`/`warmup`），`:37` import。无 `**kwargs`/partial/动态分发/YAML/doc/测试引用。函数体两参数均活（warmup loop `:141`、timed loop `:146`、`median of {iters}` 打印 `:156`）——是没人设置的参数化，非死分支。
- 动作：从 `run_e2e` 签名（`:137`）删 `iters=3, warmup=2`，在函数体内内联为局部量（`iters = 3` / `warmup = 2`）。
- 注意（**不要**把 caller 的 `--steps/--warmup` 接进来）：审计原文的「首选动作」（wiring CLI flags）语义错误，禁止采用。`generation_bottleneck_profile.py:72` 的 `--steps` 文档单位是 profiled denoise steps（一 step = 一次 `step_fn`），而 `run_e2e` 的 `iters` 计整幅 encode+denoise+decode 图像（每次 `_e2e_once` 跑 `cfg.sampling.num_steps` 个 denoise step + encode + decode）。接进来会在 help 文案不变的情况下把 flag 从「6 denoise steps」悄悄改成「6 full images」（约 `num_steps` 倍工作量）。若将来确需 per-run 控制 e2e 迭代数，那是需要独立 `--e2e-iters/--e2e-warmup` flag 的新功能，非本次清理。无测试引用 `run_e2e`，无测试需更新。**保留** `cuda_mean_ms/cuda_median_ms`（8+ probe 调用点显式传 iters/warmup）与 `kernel_launches_per_step(steps=3)`（uniform API shape，跨家族一致性 keep-list）。

---

### 1.14 `build_model` — single-caller-merge（risk=low）

- 位置：`vrl/scripts/perf/common/diffusion_runtime.py:30-48`（长期资产：perf 共享模块）
- 判死证据：`grep -rn -w build_model vrl/ tests/` → 定义 `:30` + 唯一调用者 `teacache_drift_probe.py:43`（import）/`:220`（call）。无测试调用（`tests/scripts/perf/test_diffusion_runtime.py` 只 import `build_runtime`），无字符串/registry/YAML/pyproject。函数体 8 行：`build_runtime(cfg, device)` + `dtype_to_precision_token` 校验 dtype vs `runtime.precision.dtype` + 返回 `runtime.model`；docstring 自称 `Compatibility facade for the recorded TeaCache drift probe`——一个 probe 的 dtype-guard 决策被 park 进共享模块（form-3）。`build_runtime` 才是真正的共享 builder。docs 唯一命中 `docs/sprints/reading/cosmos-rl.md:1122` 是外部 cosmos-rl 阅读笔记的 `ModelRegistry.build_model`，不同符号。
- 动作：把 `build_model` 函数体（`build_runtime` + dtype-vs-resolved-precision 校验）移入 `vrl/scripts/perf/teacache_drift_probe.py` 作私有 helper（或内联进其 `main()`），从 `common/diffusion_runtime.py` 删除 `build_model`；`dtype_to_precision_token` 作为普通 lazy import 随函数体迁移。无测试需删。**不动** `build_runtime/make_step_fn/run_e2e` sibling——它们有多个 probe 调用者，是 `common/` 的合法共享面。

---

### 1.15 `log_gpu_preflight(report=...)` — dead-arg（risk=low）

- 位置：`vrl/scripts/perf/gpu_preflight.py:203-206`（长期资产：`python -m vrl.scripts.perf.gpu_preflight` 的 CLI body）
- 判死证据：`grep -rn "log_gpu_preflight" vrl/ tests/` → 仅定义 `:203`、模块自身 `main()` 零参调用 `:221`、`__all__` `:226`。无测试 import（`test_gpu_preflight.py` 只 import `run_gpu_preflight`），无 partial/getattr/`**kwargs`。函数体参数唯一效果是当传入预计算 report 时跳过 `run_gpu_preflight()`（`:205` `r = report or run_gpu_preflight()`）——从未落地的外部便利；`run_gpu_preflight` 本身已模块级缓存（`_REPORT_CACHE`，`:157-159`），该便利即便原则上也冗余。
- 动作：从 `log_gpu_preflight` 删 `report: GpuPreflight | None = None` 参数，`:205` 简化为 `r = run_gpu_preflight()`。无 caller/测试更新。**保留函数本身**——它是 CLI body 及 [[SPRINT_cosmos_video_mfu_kernels]]（`docs/sprints/done/SPRINT_cosmos_video_mfu_kernels.md:3,142`）点名的保留交付物。`GpuPreflight` dataclass 已在定义处（`:41-44`）标注 `Display/provenance-only`，是获许豁免，不动；`run_gpu_preflight/measured_bf16_peak_tflops` 有活调用者，不动。

---

### 1.16 `RunMetrics.run_dir` — dead-field（risk=low）

- 位置：`vrl/scripts/perf/reward_overlap_benchmark.py:100-150`（长期资产：perf benchmark）
- 判死证据：`grep -rn -w RunMetrics vrl/ tests/` → 仅本文件（定义 `:100`、ctor `:150`、返回/list hint `:130,214`），测试 import 函数而非类。`grep -rn "\.run_dir" vrl/scripts/perf/reward_overlap_benchmark.py tests/scripts/perf/test_reward_overlap_benchmark.py` → **零 attribute read**（每处 `run_dir` 都是局部变量/参数）。`read_run_metrics` 在 `:150` 构造该字段但 `summarize_arm`/`evaluate_gates`/`analyze`/`acceptance.json` 都不消费；其 raise-paths 格式化的是局部 `run_dir` 参数（`:137,146-147,162`），非字段。模块无 `dataclasses.asdict`/`vars()`/`fields()`——无序列化路径能复活它。是无消费者、无 `Display/provenance-only` 标注的 provenance 字段 → 按 dead-field 规则为死。
- 动作：删除 `RunMetrics` 的 `run_dir` 字段，并把 `:150` 的 `metrics = RunMetrics(run_dir=run_dir)` 改为 `metrics = RunMetrics()`（`run_dir` 是唯一无默认值字段，删后 `RunMetrics()` 合法）。`read_run_metrics` 所有 error 消息已用局部 `run_dir` 参数，无需改。无测试触及 `metrics.run_dir`，无测试更新。

## 2. 验证协议

- **每条删除后**：`ruff check <touched files>` + `ruff format --check <touched files>`（仅对本条改动的 Python 文件）。
- **全簇完成后**：`pytest tests/data/ tests/scripts/`（含 `tests/scripts/eval/ tests/scripts/perf/`）+ `pytest -m "not e2e and not slow_test"` 子集不新增失败。
- **基线（清理前，2026-07-23）**：fast subset 2620 passed / 7 pre-existing failures（架构边界 + causvid/magi_1 打包摘要，与本清理无关）；`vrl.config.lint` 与 `ruff check .` 全绿。删除后这三项须保持。
- **逐条动作触及的测试文件**（须相应更新/repoint 或确认无需改动）：
  - §1.1 `anime_probe_common`：`tests/scripts/test_anime_probe_common.py`（仅覆盖 `optional_finger_cv`，无需改）。
  - §1.2 `FAILURE_LABELS`：无测试改动（连带删 `datasets/danbooru/config.yaml:219`）。
  - §1.3 `_current_http_download`：`tests/data/test_setup.py`（monkeypatch seam 依旧生效，无需改）。
  - §1.4 `build_anatomy_prompts`：`tests/data/test_danbooru.py`（不 import 该函数，无需改）。
  - §1.5 `firsts_by_episode`：无测试改动。
  - §1.6 `--keep-model-between-checkpoints`：`tests/scripts/test_cosmos_predict25_kling_eval.py:100-115`（更新/删除 flag 与互斥 error 用例）。
  - §1.7 `_video_to_cthw`：`tests/scripts/test_cosmos_predict25_kling_eval.py:50-58`（repoint 到 `denoise_video_generation.video_to_cthw`）。
  - §1.8 `evaluate` 阈值：`tests/scripts/eval/test_sana_aesthetic_curve_verdict.py`（`:76` 读回显输出，无需改）。
  - §1.9 `_validate_scheduler`：`tests/scripts/eval/test_sana_checkpoint_compare.py:173,441,449,454`（repoint 到 `sana_inference.validate_scheduler`）。
  - §1.10 `_run_prefix(tf)`：无测试（probe 无测试）。
  - §1.11 `_new_report(args)`：无测试。
  - §1.12 `logprobs`：`tests/scripts/test_full_sequence_denoise_generate.py`（不进 denoise 循环，无需改）。
  - §1.13 `run_e2e`：无测试引用，无需改。
  - §1.14 `build_model`：`tests/scripts/perf/test_diffusion_runtime.py`（只覆盖 `build_runtime`，无需改）。
  - §1.15 `log_gpu_preflight`：`tests/scripts/perf/test_gpu_preflight.py`（只 import `run_gpu_preflight`，无需改）。
  - §1.16 `RunMetrics.run_dir`：`tests/scripts/perf/test_reward_overlap_benchmark.py`（不构造 `RunMetrics`，无需改）。

## 3. Non-Goals

- 不删被「能 raise 的校验」、控制流分支、runtime/config/Ray 调用消费的字段（AGENTS 死字段规则）——例如 §1.4 保留的 safety 侧 `candidate_pool_factor`（`:250` 无条件使用）、`_interleave_bucket_rows`（`_select_quota_rows` 活调用）；§1.6 保留的 `keep_model_between_checkpoints` 计算值（`_generate_all` 行为消费）；§1.9 保留的 `_generate_prompt_group`/`_load_official_scheduler` 活 monkeypatch 边界。
- 不动 DO-NOT-FLAG 豁免项（`_GENERATION_CUDA_RUNTIME_RESIDUAL_BYTES`、`_REWARD_CUDA_RUNTIME_RESIDUAL_BYTES`、`ensure_loaded`、`process_gpu_used_bytes` NVML、sana/hunyuan `prepare_latents` 修复）；§1.15 的 `GpuPreflight` dataclass 已标注 `Display/provenance-only`，同属豁免，不动。
- 不为省行数扁平化 protocol/lazy-import/跨家族一致性 thin function——§1.13 保留 `run_e2e` 函数本身及 `kernel_launches_per_step` 的 uniform API shape，§1.14 保留 `build_runtime/make_step_fn` 共享面，§1.15 保留 `log_gpu_preflight` CLI body（sprint 交付物）。
- **cluster-specific**：`*_probe` / `*_spike` 一次性脚本**整体**不在本簇范围——只清理其内部符号级死代码（§1.10、§1.11、§1.12）；不把整支 probe 当死代码删。`scripts/eval/` 与 `scripts/data/` 的长期入口脚本（数据生成器、eval 对比、report 入口）本身保留，只删内部死符号。
- 不改任何 flag 的语义单位——§1.13 明确禁止把 `--steps/--warmup` 接入 `run_e2e`（denoise-step vs full-image 单位不匹配）。

## References

被触及的文件与行：
- `vrl/scripts/data/danbooru.py:51-52, 105, 167-168, 180-196, 674, 700-701, 710-718, 1313-1318, 1390, 1410, 1427` + `datasets/danbooru/config.yaml:218-219`
- `vrl/scripts/data/video_world.py:356-359, 377`
- `vrl/scripts/eval/anime_probe_common.py:139-147, 161-169, 243, 255-260, 271, 275-281, 349, 362-378` + `vrl/scripts/eval/probe_anime_anatomy_report.py:30, 50`
- `vrl/scripts/eval/cosmos_predict25_kling_eval.py:30-32, 41-44, 123-127, 293-299, 313, 320, 348`
- `vrl/scripts/eval/sana_aesthetic_curve_verdict.py:88-99, 245-251, 269`
- `vrl/scripts/eval/sana_checkpoint_compare.py:36, 49` + `vrl/scripts/eval/sana_inference.py:55, 59, 99, 159`
- `vrl/scripts/eval/shared_prefix_divergence_probe.py:144-146, 166-184`
- `vrl/scripts/eval/world_model_steppability_probe.py:39-53, 257`
- `vrl/scripts/generation/full_sequence_denoise.py:195, 217`
- `vrl/scripts/perf/common/diffusion_runtime.py:30-48, 137-141` + `vrl/scripts/perf/teacache_drift_probe.py:43, 220`
- `vrl/scripts/perf/gpu_preflight.py:41-44, 157-159, 203-206, 221, 226`
- `vrl/scripts/perf/reward_overlap_benchmark.py:100, 130, 150, 214`

测试文件：
- `tests/scripts/test_cosmos_predict25_kling_eval.py:50-58, 100-115, 191`
- `tests/scripts/eval/test_sana_checkpoint_compare.py:173, 441, 449, 454`
- `tests/data/test_setup.py:149`、`tests/data/test_danbooru.py`
- `tests/scripts/test_anime_probe_common.py`、`tests/scripts/eval/test_sana_aesthetic_curve_verdict.py:76`
- `tests/scripts/perf/test_diffusion_runtime.py`、`test_gpu_preflight.py`、`test_reward_overlap_benchmark.py`
- `tests/scripts/test_full_sequence_denoise_generate.py`

关联 sprint：
- [[SPRINT_deadcode_00_overview]]
- [[SPRINT_sana_aesthetic_trustworthy_curve]]（§1.8 阈值预注册协议）
- [[SPRINT_cosmos_video_mfu_kernels]]（§1.15 `gpu_preflight.py` 保留交付物）
- [[SPRINT_native_generation_engine_program]]（§1.12 命名歧义澄清：非本文件，无重叠）
