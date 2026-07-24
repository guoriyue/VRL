# SPRINT: generation 栈死字段与死参数清理（planned / reconciled）

状态：**RECONCILED（2026-07-24）**，对齐 main @ `7c748532`（= origin/main tip，自审计运行的旧树 `88ed756e` 以来已落地 ~63 个 cleanup/refactor commit）。原稿共 16 条对抗验证死代码，本次逐条针对当前 checked-out 树重新复核后：**10 条仍需处理**（5 STILL_VALID + 5 RELOCATED，见 §1）、**6 条已由 origin 落地**（见 §2，无需再做）、**0 条情况已变**（§3 为空）。RELOCATED 指判死结论不变、仅行号随 origin 的重构漂移；本稿已把 §1 每条的 `位置` 行更新到当前 `file:line` 并标注「行已移」。
来源：dead-code-audit workflow（五种死代码形态 + 对抗验证 + 删除类二次字符串引用检查），原稿写于 `88ed756e`。
关联：[[SPRINT_deadcode_00_overview]]；与 [[SPRINT_native_generation_engine_program]]（in-flight，动了 `generation/ray/`）在 §1.2 / §1.7 / §1.8 三条上重叠——这三条**必须排在该 sprint 之后**；[[SPRINT_trajectory_views_types_dead_fields_cleanup]]（死字段规则的先例：能 raise 的校验/控制流分支消费者一律保留）。

> **执行状态（2026-07-24）**：§1 全部 10 条已落地 `7056ea69`；删 `same_latent` 的连带协议哈希更新见 `801a5b77`。

## 0. 一句话

这是 generation 栈「compute-once/read-everywhere」结构体和函数参数上的死字段/死参数清理：主形态是 **dead-field 规则**（生产零 reader、或 reader 只剩测试、或 form-2 活 reader 但分支无 producer），并夹带若干 dead-arg（调用方全走默认值）和一个 dead-config-knob（`same_latent` 承诺的「组内共享噪声」语义根本没实现）。最锋利的一条是 `DiffusionPreparedStageOutput.chunk_encoded`（§1.1）——它被跨 stage 边界搬运进 `run_denoise_steps(encoded=...)`，而该函数体第一句就是 `del encoded`，是教科书式的 form-2「carried-then-deleted」死字段。误删风险集中在同名字段碰撞（`.error`/`task_type` 在别的类型上都是活的）与 slots=True dataclass（删字段必须同步清掉所有 `kwarg=` 构造点，否则 `TypeError`），必须逐 receiver 消歧后再动手。注：原稿里的 token_autoregressive 一组（`sample_ids`/`request_ids`/`__len__`/`remaining_tokens`）与 `return_artifacts`/`priority` 已被 origin 删除，见 §2。

## 1. 待删清单（仍有效）

先列 4 条 medium（reviewer 需重点看），再按文件归组 6 条 low。**位置**行均为当前树 `7c748532` 的实测行号。

### 1.1 `DiffusionPreparedStageOutput.chunk_encoded`（+ `run_denoise_steps` 的 `encoded` 死参）— dead-field（risk=medium）｜RELOCATED（行已移）

- 位置：`vrl/generation/bindings/full_sequence_denoise/executor.py:72`（field，未动）、`run_denoise_steps` def 在 `496`、`del encoded` 在 `505`（原稿记 506-520）、构造 `chunk_encoded=` 在 `432`（原 442）、唯一跨 stage reader `encoded=payload.chunk_encoded` 在 `448`（原 458）；in-stage `build_chunk_encoded` 在 `409`（原 419）、`prepare_denoise_state(encoded=...)` 在 `426`（原 436，活）。
- 判死证据：
  ```
  $ grep -rn 'chunk_encoded' vrl tests --include='*.py'
  executor.py:72   chunk_encoded: dict[str, Any]                 # field def（未动）
  executor.py:409  chunk_encoded = self.build_chunk_encoded(...)  # in-stage build（活）
  executor.py:426  encoded=chunk_encoded,                         # prepare_denoise_state（活 consumer）
  executor.py:432  chunk_encoded=chunk_encoded,                   # 构造进 payload（死起点）
  executor.py:448  encoded=payload.chunk_encoded,                 # 唯一跨 stage reader
  cosmos/predict2/runtime.py + cosmos3/runtime.py: build_chunk_encoded（另一活 hook，不同符号）
  $ grep -rn 'def run_denoise_steps' vrl tests --include='*.py'
  executor.py:496  # 仅基类定义，无任何 family override 消费 encoded 参
  ```
  `DiffusionPreparedStageOutput` 除本文件与 `__init__.py` 再导出外零构造/零 reader。`run_denoise_steps` 体首句 `del encoded`（line 505），随后调 `run_denoise_loop(model=, state=, config=)`（`loop.py` 签名不含 `encoded`）。`worker.probe_chunk_size` 只读 `prepared.config`，不读该字段；无 `asdict/fields()/vars()` 迭代、无 Ray/pickle 传输。**复核确认仍是 form-2 死字段，仅 arg 块行号从 506-520 漂到 496-510。**
- 动作：删 `DiffusionPreparedStageOutput.chunk_encoded` 字段（72）；`run_prepare_stage` 里从 `DiffusionPreparedStageOutput(...)` 构造去掉 `chunk_encoded=`（432），**保留** `chunk_encoded = self.build_chunk_encoded(...)`（409）与 `prepare_denoise_state(encoded=chunk_encoded, ...)`（426，活 in-stage 消费）；`run_denoise_stage` 去掉 `encoded=payload.chunk_encoded`（448）；`run_denoise_steps` 删 `encoded` 参与 `del encoded`（496/505）。测试仅改这两处去掉 `encoded=` kwarg：`tests/generation/steps/denoise/test_preallocation.py`（lines 57, 84, 128, 142, 329, 360, 387）与 `tests/generation/bindings/full_sequence_denoise/test_executor_denoise_mode.py:25`。**不动** `test_prev_sample_mean_storage.py`/`test_ref_noise_pred_cache.py`（二者零 `run_denoise_steps`/`encoded` 引用）。`build_chunk_encoded` 及其 cosmos3/predict2 override、`__init__.py` 再导出全部保留。
- 注意：auditor 原 action 把测试文件写错了位置，验证阶段已纠正——落地前务必 grep 确认 `encoded=` 实际调用点仅在上述两文件。`vrl/models/steps/denoise/base.py` 的 `encoded`（`prepare_sampling` 协议参）是**活的 in-stage 消费路径**，勿误删。

### 1.2 `RayGenerationWorker.current_policy_version`（Ray RPC 面）— dead-function（risk=medium）｜STILL_VALID ⚠ SPRINT 重叠

- 位置：`vrl/generation/ray/worker.py:68-69`（未动）；唯一 actor 调用方 `tests/generation/ray/test_rollout_launcher.py:206`（原稿记 191，测试行已漂，target 不变）。
- 判死证据：
  ```
  $ grep -rn 'current_policy_version.remote\|current_policy_version()' vrl tests --include='*.py'
  vrl/generation/ray/worker.py:69            return self.core.current_policy_version()   # 委托体
  vrl/rollouts/orchestration/*                RolloutLifecycle.current_policy_version()   # 不同符号
  tests/generation/execution/test_worker_versioned_slots.py  core.current_policy_version()  # core 方法
  tests/generation/ray/test_rollout_launcher.py:206  ray.get(workers[0].actor.current_policy_version.remote()) == 7  # 唯一 actor 方法调用方
  ```
  actor 方法（RPC）的唯一调用方是 `test_rollout_launcher.py:206` 一处测试断言。方法体是 1 行委托 `core.current_policy_version()`，与 `worker_metadata()` 发布的 `metadata["policy_version"]` 同值。`runtime.py`/`launcher.py`/`weight_sync.py`/`protocols.py` 上的 `current_policy_version` 都是 driver 属性或协议属性，不是这个 RPC。生产侧安装 ACK 走 `update_weights` 返回值 + `require_installed_policy_version`，不经此 RPC。**复核确认仍是 TEST-ONLY 死于 RPC 面。**
- 动作：删 `RayGenerationWorker.current_policy_version`（68-69）；对应测试断言**直接删除，无需 repoint**——同一测试后续已有 `assert metadata["policy_version"] == 7`（`worker_metadata` 取的正是被删 RPC 返回的同一字段），覆盖不减。`GenerationWorkerCore.current_policy_version`（`execution/worker.py`）本条**不动**，它删除后变 test-only，归 execution-scope 审计另处理。
- 注意：⚠ **与 in-flight sprint 直接碰撞**。`tests/generation/ray/test_rollout_launcher.py` 正被 `SPRINT_native_generation_engine_program` 修改（`git status` 显示为 `M`），`worker.py` 虽不在其 worktree 改动集但同处 `generation/ray/`。**本条必须排在该 sprint 落地之后**，届时以最新 `test_rollout_launcher.py` 重新定位那条 actor RPC 断言再删（纯删除，不 repoint）。

### 1.3 `DenoiseSDEParams.same_latent`（`sampling.same_latent`）— dead-config-knob（risk=medium）｜RELOCATED（行已移）

- 位置：field `vrl/generation/steps/denoise/config.py:16`（原 18）；schema `vrl/config/schema.py:304`（原 303）；parse `vrl/generation/bindings/full_sequence_denoise/layout.py:121`（原 112）；唯一行为读 `vrl/generation/steps/denoise/loop.py:160-161`（未动）。
- 判死证据：
  ```
  $ grep -rn 'same_latent' vrl tests configs --include='*.py' --include='*.yaml'
  schema.py:304   same_latent: Any = None           # 声明（相邻字段唯一没有 # reader: 注释）
  layout.py:121   same_latent=bool(sampling.get("same_latent", False))   # parse
  config.py:16    same_latent: bool                  # field
  loop.py:160-161 elif config.sde.same_latent: raise ValueError("same_latent=True requires an explicit sampling.seed")  # 唯一行为读
  presets: denoise.yaml:17 / online_grpo_droid_overfit_validation.yaml:117  均设 false
  tests: 均传 False
  ```
  初始 latent 由各 family `prepare_sampling` 用 `request.seed` 直接播种生成，executor 对每个 chunk 传相同 `base.seed` 不做偏移——「组内共享初始 latent」完全是固定 seed + `samples_per_chunk=1` 的涌现效果，**没有任何 code path 读 `same_latent` 去让 latent 相同**。故 `same_latent=True` 除「seed 缺失时 raise」外零效果，是带着未实现承诺的 no-op knob；仓库注释已承认「the seed IS the same-latent mechanism; the flag only guards the seedless case」。全仓无任何 `True` 的 producer。**复核确认仍是 form-2 no-op knob，多处行号漂移（field 18→16、schema 303→304、parse 112→121、overfit setter 118→117）。**
- 动作：端到端删除（**不实现其语义**）——(1) `schema.py:304` 删字段；(2) `layout.py:121` 删 parse；(3) `config.py:16` 删 `DenoiseSDEParams.same_latent`；(4) `loop.py:160-161` 删 `elif ... raise` 分支（保留 seed 为 None 时 `generator=None` 路径）；(5) `presets/base/rollout/denoise.yaml:17` 删设置行，在 seed 注释补英文说明「a fixed sampling.seed with samples_per_chunk=1 makes group members share the initial latent (each chunk re-seeds the prepare_sampling generator with the same seed)」以保留 seed-is-the-mechanism 知识；(6) `online_grpo_droid_overfit_validation.yaml` 删 setter 行（现 117）及相邻注释块（结论并入 seed 注释），改写 header 过时理由为现状（seedless rollouts + 按行 eval seed）；(7) 更新注释引用 `online_grpo_droid_lora_480p_curve.yaml`、`presets/dataset/droid_overfit_validation.yaml`；(8) 更新传 `same_latent=False` 的测试/fixture（`test_preallocation.py`、`test_executor_denoise_mode.py`、`test_prev_sample_mean_storage.py`、`test_ref_noise_pred_cache.py`、`test_family_registry.py`）。
- 注意：medium 是因为它跨 schema/parse/config/loop/多个 preset/多个测试，删除面广且要重写 YAML 注释以不丢「seed 才是机制」的知识；不要图省事只删 field 留下 parse/schema 悬挂。auditor 原判断「应 implement 而非删」被证据推翻——团队已弃用 latent sharing，正确动作是删除。**落地前以最新 YAML 重新定位 setter 行。**

### 1.4 `GenerationOutput.error`（form-2：活 reader，分支无 producer）— dead-field（risk=medium）｜RELOCATED（行已移）

- 位置：field `vrl/generation/types.py:141`（原 149）；死 reader `vrl/rollouts/collector/core.py:179-183`（原 199-204）与 `tests/quality/preview.py:48-49`（原 action 漏掉的第二 reader）。
- 判死证据：
  ```
  $ grep -rn 'GenerationOutput(' vrl tests --include='*.py'
  # 5 处生产构造：full_sequence_denoise/gather.py, chunk_autoregressive_denoise/gather.py,
  #   token_autoregressive/executor.py, nextstep_1/runtime.py, janus_pro/runtime.py
  #   —— 逐个读体确认：全部省略 error=
  $ grep -rn '\.error' vrl/rollouts/collector/core.py tests/quality/preview.py
  core.py:179        if output.error:                       # 永不触发的 raise
  preview.py:48-49   if output.error is not None: raise RuntimeError(...)   # 第二个 reader
  ```
  5 个生产构造点全省略 `error=`；测试也无一构造 `error=`。worker 侧的 `error=` 都在别的类型上（`ChunkExecutionResult`、`PipelinedRequestOutOfMemory`），`RayGenerationExecutor` 把非 OOM chunk 错误统一转成 raise，OOM 走 split-retry——没有 error 字符串能到达 `GenerationOutput`。**复核确认仍是 form-2 死字段，field 行 149→141、collector reader 199→179。**
- 动作：删 `GenerationOutput.error` 字段（types.py:141）；删 `vrl/rollouts/collector/core.py:179-183` 不可达 `if output.error: raise` 分支；**同时删** `tests/quality/preview.py:48-49`（`write_preview_image` 里的 guard——`output` 是 `forward_plan` 返回的 `GenerationOutput`，slots=True 下不删会 AttributeError）；改 `tests/quality/test_preview.py`：删 `SimpleNamespace(error="oom", ...)` 的 "returned an error" parametrize 用例，并去掉 happy-path fake 里无意义的 `error=None` kwarg。
- 注意：medium 是因为 auditor 原 action 只删了 collector 一处、漏了 `tests/quality/preview.py` 的第二个 reader。落地必须先重跑 `grep -rn '\.error' vrl tests` 确认只剩这两个 `GenerationOutput` reader，并以最新行号定位。别碰同名的 `ChunkExecutionResult.error`/`PipelinedRequestOutOfMemory.error`（活）。

### 1.5 `run_sample_chunks_with_oom_retry(min_sample_count=)`（全默认值）— dead-arg（risk=low）｜STILL_VALID

- 位置：`vrl/generation/execution/chunks.py:127`（参）、`131-132`（校验）、`143`（使用）——与原稿一致。
- 判死证据：
  ```
  $ grep -rn 'min_sample_count' vrl tests --include='*.py'
  chunks.py:127/131/132/143   # 仅定义、>=1 校验、单处比较——全在 chunks.py 内
  $ grep -rn 'run_sample_chunks_with_oom_retry' vrl tests
  # 3 生产调用方 full_sequence_denoise/executor.py:267, chunk_autoregressive_denoise/executor.py:122,
  #   token_autoregressive/executor.py:96 + 测试 test_chunks.py:45 —— 全部无 kwarg
  ```
  常量 1 下 `chunk.sample_count <= min_sample_count` 就是 `<= 1`，正是 `SampleChunk.split()` 已强制的下界。无 `**kwargs` 转发、无 YAML/entry-point。**复核确认行号与判死证据全部仍成立。**
- 动作：删 `min_sample_count` 参与其 `>= 1` 校验，把 guard 硬编码为 `chunk.sample_count <= 1`（guard 本身**必须留**——它重抛原始 OOM；删掉会让 `split()` 抛 `ValueError` 掩盖 OOM）。无测试传该 kwarg，无需改测试。

### 1.6 `GenerationWorkerCore.probe_chunk_size(execute_steps=)`（全默认值）— dead-arg（risk=low）｜STILL_VALID

- 位置：`vrl/generation/execution/worker.py:303`（签名）、`394`（使用）——原稿记 302，实测 303（1 行漂）。
- 判死证据：
  ```
  $ grep -rn 'probe_chunk_size' vrl tests --include='*.py'
  ray/worker.py        self.core.probe_chunk_size(request, max_samples=max_samples)   # 唯一生产转发，只传 max_samples
  ray/worker_fleet.py  getattr(...,"probe_chunk_size"...) ... max_samples=...          # getattr 派发也只传 max_samples
  tests/.../test_chunk_memory_shadow.py  # 传 max_samples，有时 margin/knee_threshold，从不 execute_steps
  $ grep -rn 'execute_steps' vrl tests
  worker.py:303 (param) / worker.py:394 (唯一用: dataclass_replace(prepared.config, execute_steps=execute_steps))
  config.py (活 DenoiseLoopConfig 字段) / loop.py (活 reader)
  ```
  `RayGenerationWorker.probe_chunk_size` 连 RPC 边界都不暴露 `execute_steps`；`DenoiseLoopConfig.execute_steps` 本身是活的（loop 截断步数），只有 probe 这个参数从不被定制。测试断言 `set(executor.executed_steps) == {2}` 测的是默认值，改成命名常量 `= 2` 后仍绿。**复核确认仍是死参，仅签名行 302→303。**
- 动作：删 `probe_chunk_size` 的 `execute_steps` 参，改用命名局部常量（如 `_PROBE_EXECUTE_STEPS = 2`）。保留 `margin`/`knee_threshold`（测试传非默认值以确定性 pin fit 路径与 knee 规则）。`DenoiseLoopConfig.execute_steps` 保留。无需改测试。

### 1.7 `validate_colocated_replay_memory(strict=...)`（只测试传、复制 env 旋钮）— dead-arg（risk=low）｜RELOCATED（行已移）⚠ SPRINT 重叠

- 位置：`vrl/generation/ray/config.py:132`（参，原稿记 139）、env 解析 + raise `157-165`（原 164-171）；生产调用 `config.py:121`；两个 `strict=` 测试调用方 `tests/trainers/test_memory_guards.py:77-81`、`89-93`。
- 判死证据：
  ```
  $ grep -rn 'validate_colocated_replay_memory' vrl tests --include='*.py'
  config.py:121   validate_colocated_replay_memory(...)   # 生产调用，只传 bundle=/rollout_config=，无 strict=
  config.py:132   def ...( strict= )                       # 参
  config.py:290   __all__ 导出
  tests/trainers/test_memory_guards.py:80/92  strict=True  # 唯一 strict= 调用方
  $ grep -rn 'VRL_STRICT_REPLAY_MEMORY_GUARD' .   # 排除 .venv/__pycache__/egg-info
  config.py:158   # 全仓唯一命中——无 CI/YAML/脚本设置它
  ```
  生产 strictness 只由 env var 决定（`strict is None -> os.environ.get("VRL_STRICT_REPLAY_MEMORY_GUARD")`），`strict=` kwarg 仅测试用、且绕过真实生产路径——正是 TEST-ONLY 死参。**复核确认仍死，参数 139→132、env 解析 164-171→157-165（config.py 被 in-flight sprint 改，行已漂）。**
- 动作：删 `strict: bool | None = None` 参，无条件保留 env-var 解析；把两个测试调用方改成 `monkeypatch.setenv("VRL_STRICT_REPLAY_MEMORY_GUARD", "1")`——顺带让它们跑真实生产 strict 路径（当前 truthy-token 解析零覆盖）。
- 注意：⚠ **与 in-flight sprint 直接碰撞（同文件）**，本条必须**排在 `SPRINT_native_generation_engine_program` 落地之后**，届时以最新 `config.py` 重新定位行号，与 §1.8 一并处理。

### 1.8 `validate_colocated_replay_memory(log=...)`（零调用方、logging-only）— dead-arg（risk=low）｜RELOCATED（行已移）⚠ SPRINT 重叠

- 位置：`vrl/generation/ray/config.py:133`（参，原稿记 140）、`(log or logger).warning` 在 `166`（原 173）。
- 判死证据：
  ```
  $ grep -rn 'validate_colocated_replay_memory' vrl tests  # 同 §1.7 的 3 调用点
  # 生产 config.py:121 只传 bundle=/rollout_config=；测试 :77/:89 只传 strict=True——均不传 log=
  $ grep -n 'logging' vrl/generation/ray/config.py
  5:import logging          # 唯一用途是被删的注解
  18:from vrl.utils.logging import init_logger   # 不同 import，保留
  133:    log: logging.Logger | None = None,
  ```
  `log` 唯一用途是 line 166 `(log or logger).warning(message)`——纯决定哪个 logger 发警告。搬迁遗留（该函数从 `vrl/utils/memory.py` 搬入，module 级 logger 已足够）。**复核确认仍死，参数 140→133、用点 173→166。**
- 动作：(1) 删 `log: logging.Logger | None = None` 参（133）；(2) `(log or logger).warning(message)` 改 `logger.warning(message)`（166）；(3) **同时删** `import logging`（line 5，删参后成 F401 死导入——`vrl.utils.logging` 是另一 import，不受影响）。无测试传 `log=`，无需改测试。
- 注意：⚠ auditor 原 action 漏了删 `import logging`，验证阶段补上。**与 in-flight sprint 同文件碰撞**，本条与 §1.7 一并**排在 sprint 之后**，一次性处理 `config.py`。

### 1.9 `TeaCacheConfig.enabled`（无 False producer，恒 True）— dead-field（risk=low）｜STILL_VALID

- 位置：`vrl/generation/steps/denoise/teacache.py:49`（field，未动）；reader `loop.py:168`。
- 判死证据：
  ```
  $ grep -rn 'TeaCacheConfig' vrl tests --include='*.py'
  # 构造点：teacache.py from_sampling :77/:89（均硬编码 enabled=True，禁用输入 return None）
  #   + 6 处直接测试构造 test_teacache.py:36/38/42/58/72/79（均 enabled=True）
  $ grep -rn '\.enabled' vrl/generation/steps/denoise tests/generation/steps/denoise
  loop.py:168        if config.teacache is not None and config.teacache.enabled   # 第二合取恒 True
  test_teacache.py:24  assert cfg.enabled and cfg.signal == "latent"
  ```
  所有构造点都硬编码 `enabled=True`；`from_sampling` 的 `value.get("enabled", True)` **mapping key 是活的用户配置键，须留**——只有 dataclass field 死。无 YAML 设 teacache 块，perf 脚本只读 `.signal/.threshold`。**复核确认字段仍在 line 49、仍是 form-2（无 False producer）。**
- 动作：删 `TeaCacheConfig.enabled` 字段（teacache.py:49）；保留 `from_sampling` 的 `value.get("enabled", True) -> return None`（活配置键）；从两个 `from_sampling` 构造点（:77、:89）去掉 `enabled=True`；`loop.py:168` guard 简化为 `if config.teacache is not None`；`test_teacache.py`：line 24 断言去掉 `cfg.enabled`（留 `cfg.signal == "latent"`），并从 6 个直接构造（lines 36, 38, 42, 58, 72, 79）去掉 `enabled=True` kwarg（slots=True，不删会 `TypeError`）。
- 注意：auditor 原 action 只提了改 line 24 断言，漏了 6 个测试构造点与两个 `from_sampling` 构造点的 `enabled=True`——slots=True dataclass 删字段后这些 kwarg 会抛 `TypeError`，验证阶段补上。

### 1.10 `VideoGenerationRequest.task_type`（form-2：alias producer 已删，默认值从不被读）— dead-field（risk=low）｜STILL_VALID

- 位置：`vrl/generation/types.py:36`（未动）。
- 判死证据：
  ```
  $ grep -n 'task_type' vrl/generation/types.py
  ...: task_type: str | None = None       # GenerationInput.task_type —— 活，勿删
  36:  task_type: str = "text_to_video"    # VideoGenerationRequest —— 死
  $ grep -rn 'task_type=' vrl tests   # VideoGenerationRequest 的 setter
  vrl/scripts/eval/wan_i2v_logprob_parity_probe.py:115  task_type="image_to_video"
  tests/models/families/echo/test_echo_flow_policy.py:87  task_type="text_to_video"
  ```
  所有 `.task_type` 属性读都落在 `GenerationInput`（`execution/ids.py:23-24`、`magi_1/model.py:847`）、`PromptExample`、config `data.task_type`（`schema.py:187`）上——**零** `VideoGenerationRequest` 实例读。共享 executor 的 `build_video_request` 从不 set `task_type`。probe 脚本靠 `reference_image` 条件 i2v，删 kwarg 行为不变。**复核确认仍死；两个 setter 行漂移（probe 109→115、echo 75→87），`test_reward_update_flow.py` 的 `task_type=` 是 `PromptExample`（活）不在范围。**
- 动作：删 `VideoGenerationRequest.task_type`（types.py:36）；去掉两个 setter：`wan_i2v_logprob_parity_probe.py:115`、`tests/models/families/echo/test_echo_flow_policy.py:87`（slots=True，不删会 `TypeError`）。`GenerationInput.task_type` 与 config `data.task_type`（schema.py:187）是分开的活消费者，**勿动**。
- 注意：auditor 原 action 漏了 echo 测试这个第二 setter，验证阶段补上。落地前以最新行号定位两个 setter。

## 2. 已由 origin 落地（本次复核确认，无需再做）

自 `88ed756e` 以来 origin 已删除下列 6 条，本次全仓 grep 复核确认生产符号已不存在（test-only 引用即使残留也算 DONE）：

- `TokenBatch.sample_ids`（token_autoregressive 的 per-batch sample_id 便捷 property，只测试读）— 随 `token_loop.py` 重写整体删除，`landed_by` **78212af3** refactor(token): replace request-local scheduler。
- `TokenBatch.request_ids`（同上，零 reader property）— `landed_by` **78212af3**（`TokenBatch` 整类被 `TokenAutoregressiveEnvelope` 取代）。
- `TokenScheduler.__len__`（零调用方 dunder）— `landed_by` **78212af3**（`TokenScheduler` 整类删除）。
- `ActiveSequence.remaining_tokens`（纯派生 property，只测试读）— `landed_by` **78212af3**（`ActiveSequence` 类在重写中删除）。
- `GenerationRequest.return_artifacts`（生产硬编码 setter、只测试读的 no-op 旋钮）— `landed_by` **f73d2751** refactor(generation): remove dead payload fields。现仅剩 `tests/generation/execution/test_generation_contracts.py:77` 一条**新增的拒绝 guard 测试**（断言该 removed 旋钮被拒），生产字段 + setter 已删。
- `GenerationRequest.priority`（零 reader 字段）— `landed_by` **f73d2751**。现仅剩同一拒绝 guard 的 parametrize 用例 `('priority', 1)`；活的 `RayActorJob.priority`（`vrl/ray/actor_pool.py`，来自 `assignment.estimated_cost`）是不同概念，未受影响。

## 3. 情况已变（需重新评估）

（无）——本次复核未出现「判死前提被 origin 改动推翻」的 CHANGED/INDETERMINATE 项。所有原 findings 要么仍成立（§1），要么已被干净删除（§2）。

## 4. 验证协议

基线现对齐 **main @ `7c748532`**（不再是旧树 `88ed756e`）。

- **每条删除后**：`ruff check <touched files>` + `ruff format --check <touched files>`（仅本条触及的 Python 文件；YAML 改动不过 ruff）。
- **全簇完成后**：`pytest` 下列相关子目录 + `pytest -m "not e2e and not slow_test"` 子集不新增失败：
  - `tests/generation/steps/denoise/`、`tests/generation/bindings/full_sequence_denoise/`（§1.1/§1.3/§1.9）
  - `tests/generation/execution/`（§1.5/§1.6）
  - `tests/generation/ray/`（§1.2，sprint 后）
  - `tests/trainers/test_memory_guards.py`（§1.7/§1.8，sprint 后）
  - `tests/quality/test_preview.py`（§1.4）
  - `tests/models/families/echo/test_echo_flow_policy.py`（§1.10）
  - `tests/rollouts/runtime/test_family_registry.py`（§1.3）
- **config-resolve 验证**（§1.3）：`vrl.config.lint` 全绿 + 两个 cosmos_predict2 preset（`online_grpo_droid_overfit_validation.yaml`、`online_grpo_droid_lora_480p_curve.yaml`）删 `same_latent` 后仍能解析。
- **基线（清理前，对齐 `7c748532`）**：先在当前 main 上重跑 fast subset + `vrl.config.lint` + `ruff check .` 建立干净基线（原 `88ed756e` 上的 2620 passed / 7 pre-existing failures 数值已作废，须重新采集），删除后三项须保持不新增失败。
- 逐条触及测试文件：
  - §1.1：`test_preallocation.py`、`test_executor_denoise_mode.py`
  - §1.2：`test_rollout_launcher.py`（sprint 后）
  - §1.3：`test_preallocation.py`、`test_prev_sample_mean_storage.py`、`test_ref_noise_pred_cache.py`、`test_executor_denoise_mode.py`、`test_family_registry.py`
  - §1.4：`tests/quality/preview.py`（源）、`tests/quality/test_preview.py`
  - §1.5/§1.6：无需改测试
  - §1.7：`test_memory_guards.py`（sprint 后）
  - §1.8：无需改测试
  - §1.9：`test_teacache.py`
  - §1.10：`test_echo_flow_policy.py`

## 5. Non-Goals

- 不删被「能 raise 的校验」/控制流分支/runtime/config/Ray 调用消费的字段（AGENTS 死字段规则）：如 `DenoiseLoopConfig.execute_steps`（loop 截断步数，活）、`GenerationInput.task_type` 与 config `data.task_type`（活）、`RayActorJob.priority`（LPT 排序，活）、`TeaCacheConfig` 的 `enabled` mapping-key（用户配置键，活）、`build_chunk_encoded` 及其 cosmos override（活 in-stage hook）。
- 不动 DO-NOT-FLAG 豁免项：`_GENERATION_CUDA_RUNTIME_RESIDUAL_BYTES`、`_REWARD_CUDA_RUNTIME_RESIDUAL_BYTES`、`ensure_loaded`、`process_gpu_used_bytes`、sana/hunyuan 的 `prepare_latents` 修复。
- 不为省行数扁平化 protocol / lazy-import / 跨家族一致性 thin function：`run_sample_chunks_with_oom_retry` 的 OOM guard 本身保留（重抛原始 OOM）；`GenerationWorkerCore.current_policy_version` 本簇不删（归 execution-scope）；各 `__init__.py` 再导出 facade 保留。
- **不与 in-flight `SPRINT_native_generation_engine_program` 抢文件**：§1.2（`ray/worker.py` + sprint 正改的 `test_rollout_launcher.py`）、§1.7/§1.8（sprint 正改的 `ray/config.py`）三条**排在该 sprint 落地之后**，届时以最新文件重新定位行号。
- 不实现 `same_latent` 的承诺语义（团队已弃用 latent sharing，正确动作是删除而非补实现）。
- 不重复 §2 已落地项——`token_loop.py` 的 4 个 property/dunder 与 `return_artifacts`/`priority` 已由 origin（78212af3 / f73d2751）删除，勿再动。

## References

- `vrl/generation/bindings/full_sequence_denoise/executor.py:72,409,426,432,448,496,505`、`__init__.py`
- `vrl/generation/ray/worker.py:68-69`、`vrl/generation/execution/worker.py:210,303,394`、`vrl/generation/ray/worker_fleet.py`、`vrl/generation/ray/weight_sync.py`
- `vrl/generation/steps/denoise/config.py:16`、`loop.py:160-161,168`、`teacache.py:49,77,89`、`vrl/generation/bindings/full_sequence_denoise/layout.py:121`、`vrl/config/schema.py:304`
- `vrl/config/presets/base/rollout/denoise.yaml:17`、`presets/experiment/cosmos_predict2/online_grpo_droid_overfit_validation.yaml`（setter 现 117）、`online_grpo_droid_lora_480p_curve.yaml`、`presets/dataset/droid_overfit_validation.yaml`
- `vrl/generation/types.py:36,141`（`GenerationInput.task_type` 另在同文件，活；`return_artifacts`/`priority` 已删）
- `vrl/generation/execution/chunks.py:127,131-132,143`
- `vrl/generation/ray/config.py:5,121,132-133,158,166,290`（行号随 in-flight sprint 漂移）
- `vrl/rollouts/collector/core.py:179-183`、`vrl/rewards/base.py`、`vrl/ray/actor_pool.py:29,61`
- `tests/quality/preview.py:48-49`、`tests/quality/test_preview.py`
- `tests/generation/steps/denoise/test_preallocation.py`、`tests/generation/bindings/full_sequence_denoise/test_executor_denoise_mode.py`、`test_prev_sample_mean_storage.py`、`test_ref_noise_pred_cache.py`、`tests/generation/steps/denoise/test_teacache.py`、`tests/generation/execution/test_chunk_memory_shadow.py`、`tests/trainers/test_memory_guards.py`、`tests/generation/ray/test_rollout_launcher.py:206`、`tests/rollouts/runtime/test_family_registry.py`、`tests/models/families/echo/test_echo_flow_policy.py:87`、`vrl/scripts/eval/wan_i2v_logprob_parity_probe.py:115`
- 已落地（§2）：commit **78212af3**（token scheduler 重写）、**f73d2751**（remove dead payload fields）；`tests/generation/execution/test_generation_contracts.py:77-78`（return_artifacts/priority 拒绝 guard）
- 关联：[[SPRINT_deadcode_00_overview]]、[[SPRINT_native_generation_engine_program]]、[[SPRINT_trajectory_views_types_dead_fields_cleanup]]
