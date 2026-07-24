# SPRINT: generation 栈死字段与死参数清理（planned）

状态：**planned（2026-07-23）**。共 **16 条**对抗验证确认的死代码（5 medium / 11 low），覆盖 `generation/{types,bindings,execution,steps,ray,composition}`；来源为 dead-code-audit workflow 的簇输出，已在写稿前逐条 re-grep 复核。
来源：dead-code-audit workflow（五种死代码形态 + 对抗验证 + 删除类二次字符串引用检查）.
关联：[[SPRINT_deadcode_00_overview]]；与 [[SPRINT_native_generation_engine_program]]（in-flight，动了 `generation/ray/`）在 §1.2 / §1.13 / §1.14 三条上重叠——这三条**必须排在该 sprint 之后**；[[SPRINT_trajectory_views_types_dead_fields_cleanup]]（死字段规则的先例：能 raise 的校验/控制流分支消费者一律保留）。

## 0. 一句话

这是 generation 栈「compute-once/read-everywhere」结构体和函数参数上的死字段/死参数清理：主形态是 **dead-field 规则**（生产零 reader、或 reader 只剩测试、或 form-2 活 reader 但分支无 producer），并夹带若干 dead-arg（调用方全走默认值）和一个 dead-config-knob（`same_latent` 承诺的「组内共享噪声」语义根本没实现）。最锋利的一条是 `DiffusionPreparedStageOutput.chunk_encoded`（§1.1）——它被跨 stage 边界搬运进 `run_denoise_steps(encoded=...)`，而该函数体第一句就是 `del encoded`，是教科书式的 form-2「carried-then-deleted」死字段。误删风险集中在同名字段碰撞（`sample_ids`/`request_ids`/`.error`/`priority`/`task_type` 在别的类型上都是活的）与 slots=True dataclass（删字段必须同步清掉所有 `kwarg=` 构造点，否则 `TypeError`），必须逐 receiver 消歧后再动手。

## 1. 待删清单（逐条，带证据与动作）

先列 5 条 medium（reviewer 需重点看），再按文件归组 11 条 low。

### 1.1 `DiffusionPreparedStageOutput.chunk_encoded`（+ `run_denoise_steps` 的 `encoded` 死参）— dead-field（risk=medium）

- 位置：`vrl/generation/bindings/full_sequence_denoise/executor.py:72`（field）、`506-520`（`run_denoise_steps` 的 `encoded` 参 + `del encoded`）
- 判死证据：
  ```
  $ grep -rn 'chunk_encoded' vrl tests --include='*.py'
  executor.py:72   chunk_encoded: dict[str, Any]                 # field def
  executor.py:419  chunk_encoded = self.build_chunk_encoded(...)  # in-stage build（活）
  executor.py:436  encoded=chunk_encoded,                         # prepare_denoise_state（活 consumer）
  executor.py:442  chunk_encoded=chunk_encoded,                   # 构造进 payload（死起点）
  executor.py:458  encoded=payload.chunk_encoded,                 # 唯一跨 stage reader
  cosmos/predict2/runtime.py + cosmos3/runtime.py: build_chunk_encoded（另一个活 hook，不同符号）
  $ grep -rn 'def run_denoise_steps' vrl tests --include='*.py'
  executor.py:506  # 仅基类定义，无任何 family override 消费 encoded 参
  ```
  `DiffusionPreparedStageOutput` 除本文件与 `__init__.py` 再导出外零构造/零 reader。`run_denoise_steps` 体首句 `del encoded`（line 515），随后调 `run_denoise_loop(model=, state=, config=)`（`loop.py:144-149` 签名不含 `encoded`）。`worker.probe_chunk_size` 只读 `prepared.config`，不读该字段；无 `asdict/fields()/vars()` 迭代、无 Ray/pickle 传输。
- 动作：删 `DiffusionPreparedStageOutput.chunk_encoded` 字段（line 72）；`run_prepare_stage` 里从 `DiffusionPreparedStageOutput(...)` 构造去掉 `chunk_encoded=`（line 442），**保留** `chunk_encoded = self.build_chunk_encoded(...)`（line 419）与 `prepare_denoise_state(encoded=chunk_encoded, ...)`（line 436，活 in-stage 消费）；`run_denoise_stage` 去掉 `encoded=payload.chunk_encoded`（line 458）；`run_denoise_steps` 删 `encoded` 参与 `del encoded`（506-515）。测试仅改这两处去掉 `encoded=` kwarg：`tests/generation/steps/denoise/test_preallocation.py`（lines 57, 84, 128, 142, 329, 360, 387）与 `tests/generation/bindings/full_sequence_denoise/test_executor_denoise_mode.py:25`。**不动** `test_prev_sample_mean_storage.py`/`test_ref_noise_pred_cache.py`（二者无 `run_denoise_steps`/`encoded` 引用）。`build_chunk_encoded` 及其 cosmos3/predict2 override、`__init__.py` 再导出全部保留。
- 注意：auditor 原 action 把两个测试文件写错了位置（说需改 `test_prev_sample_mean_storage.py`/`test_ref_noise_pred_cache.py`），验证阶段已纠正——那两个文件零引用。改前务必 grep 确认 `encoded=` 实际调用点仅在上述两文件。`vrl/models/steps/denoise/base.py` 的 `encoded`（`prepare_sampling` 协议参）是**活的 in-stage 消费路径**，与本条删的跨 stage 搬运不是同一处，勿误删；该文件正被 in-flight sprint 修改，但本条不触及它。

### 1.2 `RayGenerationWorker.current_policy_version`（Ray RPC 面）— dead-function（risk=medium）⚠ SPRINT 重叠

- 位置：`vrl/generation/ray/worker.py:68-69`
- 判死证据：
  ```
  $ grep -rn 'current_policy_version.remote\|current_policy_version()' vrl tests --include='*.py'
  vrl/generation/ray/worker.py:69            return self.core.current_policy_version()   # 委托体
  vrl/rollouts/orchestration/*                RolloutLifecycle.current_policy_version()   # 不同符号
  tests/generation/execution/test_worker_versioned_slots.py:138/149  core.current_policy_version()  # core 方法
  tests/generation/ray/test_rollout_launcher.py:191  ray.get(workers[0].actor.current_policy_version.remote()) == 7  # 唯一 actor 方法调用方
  ```
  actor 方法（RPC）的唯一调用方是 `test_rollout_launcher.py:191` 一处测试断言。方法体是 1 行委托 `core.current_policy_version()`（返回 `self._policy_version`），与 `worker_metadata()` 发布的 `metadata["policy_version"]`（`vrl/generation/execution/worker.py:210`）是同一值。`runtime.py`/`launcher.py`/`weight_sync.py`/`protocols.py` 上的 `current_policy_version` 都是 driver 属性或协议属性，不是这个 RPC。生产侧安装 ACK 走 `update_weights` 返回值 + `require_installed_policy_version`（`vrl/generation/ray/weight_sync.py`），不经此 RPC。
- 动作：删 `RayGenerationWorker.current_policy_version`（worker.py:68-69）；`tests/generation/ray/test_rollout_launcher.py:191` 那行断言**直接删除，无需 repoint**——因为同一测试 line 194 已有 `assert metadata["policy_version"] == 7`（`worker_metadata` 取的正是被删 RPC 返回的同一字段），覆盖不减。`GenerationWorkerCore.current_policy_version`（`execution/worker.py:185`）本条**不动**，它删除后变 test-only，归 execution-scope 审计另处理。
- 注意：**与 in-flight sprint 直接碰撞**。`tests/generation/ray/test_rollout_launcher.py` 正被 `SPRINT_native_generation_engine_program` 修改（`git status` 显示为 `M`），`worker.py` 虽不在其 worktree 改动集但同处 `generation/ray/`。**本条必须排在该 sprint 落地之后**，届时以最新 `test_rollout_launcher.py` 为准重新定位 line 191/194 断言再删。auditor 原 action 的行号（189）与「repoint 成 `metadata["policy_version"]==7`」都不准——正确动作是纯删除（line 194 已存在同断言，repoint 会造成重复）。

### 1.3 `DenoiseSDEParams.same_latent`（`sampling.same_latent`）— dead-config-knob（risk=medium）

- 位置：field `vrl/generation/steps/denoise/config.py:18`；parse `vrl/generation/bindings/full_sequence_denoise/layout.py:112`；唯一行为读 `vrl/generation/steps/denoise/loop.py:160-161`；schema `vrl/config/schema.py:303`
- 判死证据：
  ```
  $ grep -rn 'same_latent' vrl tests configs --include='*.py' --include='*.yaml'
  schema.py:303   same_latent: Any = None           # 声明（相邻字段唯一没有 # reader: 注释）
  layout.py:112   same_latent=bool(sampling.get("same_latent", False))   # parse
  config.py:18    same_latent: bool                  # field
  loop.py:160-161 elif config.sde.same_latent: raise ValueError("same_latent=True requires an explicit sampling.seed")  # 唯一行为读
  presets: denoise.yaml:17 / online_grpo_droid_overfit_validation.yaml:118  均设 false
  tests: test_preallocation.py:197 / test_prev_sample_mean_storage.py:33 / test_ref_noise_pred_cache.py:54 / test_family_registry.py:168 / test_executor_denoise_mode.py:36  均传 False
  ```
  初始 latent 由各 family `prepare_sampling` 用 `request.seed` 直接播种生成（`sana/model.py:259-274`、`cosmos/predict2/model.py:326-343`），executor 对每个 chunk 传相同 `base.seed` 不做偏移——「组内共享初始 latent」完全是固定 seed + `samples_per_chunk=1` 的涌现效果，**没有任何 code path 读 `same_latent` 去让 latent 相同**。故 `same_latent=True` 除「seed 缺失时 raise」外零效果：一个带着未实现承诺（`# share noise across group`）的 no-op knob。仓库自己的注释已承认：`online_grpo_droid_overfit_validation.yaml:112-114`「the seed IS the same-latent mechanism; the flag only guards the seedless case」，run10 复盘（`docs/sprints/info/SPRINT_dino_reward_rl_trainability.md:48`）把共享噪声列为 confound。全仓无任何 `True` 的 producer。
- 动作：端到端删除（**不实现其语义**）——(1) `vrl/config/schema.py:303` 删字段；(2) `layout.py:112` 删 parse；(3) `config.py:18` 删 `DenoiseSDEParams.same_latent`；(4) `loop.py:160-161` 删 `elif ... raise` 分支（保留 seed 为 None 时 `generator=None` 路径）；(5) `presets/base/rollout/denoise.yaml:17` 删设置行，在 seed 注释补英文说明「a fixed sampling.seed with samples_per_chunk=1 makes group members share the initial latent (each chunk re-seeds the prepare_sampling generator with the same seed)」以保留 seed-is-the-mechanism 知识；(6) `online_grpo_droid_overfit_validation.yaml` 删 line 118 设置及 112-117 注释块（结论并入 seed 注释），改写 22-27 header 过时理由为现状（seedless rollouts + 按行 eval seed）；(7) 更新注释引用 `online_grpo_droid_lora_480p_curve.yaml:22`、`presets/dataset/droid_overfit_validation.yaml:11`；(8) 更新传 `same_latent=False` 的 5 处测试/fixture（见证据）。
- 注意：medium 是因为它跨 schema/parse/config/loop/多个 preset/多个测试，删除面广且要重写 YAML 注释以不丢「seed 才是机制」的知识；不要图省事只删 field 留下 parse/schema 悬挂。auditor 原判断「应 implement 而非删」被证据推翻——团队已弃用 latent sharing，正确动作是删除。

### 1.4 `GenerationOutput.error`（form-2：活 reader，分支无 producer）— dead-field（risk=medium）

- 位置：`vrl/generation/types.py:149`
- 判死证据：
  ```
  $ grep -rn 'GenerationOutput(' vrl tests --include='*.py'
  # 5 处生产构造：full_sequence_denoise/gather.py:73, chunk_autoregressive_denoise/gather.py:74,
  #   token_autoregressive/executor.py:365, nextstep_1/runtime.py:298, janus_pro/runtime.py:408
  #   —— 逐个读体确认：全部省略 error=
  $ grep -rn '\.error' vrl/rollouts/collector/core.py tests/quality/preview.py
  core.py:199        if output.error:                       # 永不触发的 raise
  core.py:203        ...: {output.error}
  preview.py:48-49   if output.error is not None: raise RuntimeError(...)   # 第二个 reader（原 action 漏了）
  ```
  5 个生产构造点全省略 `error=`；测试也无一构造 `error=`（`grep -A30 'GenerationOutput(' tests | grep 'error='` 为空）。worker 侧的 `error=` 都在别的类型上（`ChunkExecutionResult` worker.py:288-296、`PipelinedRequestOutOfMemory` worker.py:579-583），`RayGenerationExecutor` 把非 OOM chunk 错误统一转成 raise，OOM 走 split-retry——没有 error 字符串能到达 `GenerationOutput`。
- 动作：删 `GenerationOutput.error` 字段（types.py:149）；删 `vrl/rollouts/collector/core.py:199-204` 不可达 `if output.error: raise` 分支；**同时删** `tests/quality/preview.py:48-49`（`write_preview_image` 里的 `if output.error is not None: raise` guard——`output` 是 `forward_plan` 返回的 `GenerationOutput`，slots=True 下不删会 AttributeError）；改 `tests/quality/test_preview.py`：删 `SimpleNamespace(error="oom", ...)` 的 "returned an error" parametrize 用例（lines 81-84），并去掉 happy-path fake 里无意义的 `error=None` kwarg（lines 65, 87）。
- 注意：medium 是因为 auditor 原 action 只删了 collector 一处、漏了 `tests/quality/preview.py` 的第二个 reader，且行号写成 179-184（实际 199-204，因 worktree 漂移）。落地必须先重跑 `grep -rn '\.error' vrl tests` 确认只剩这两个 `GenerationOutput` reader。别碰同名的 `ChunkExecutionResult.error`/`PipelinedRequestOutOfMemory.error`（活）。

### 1.5 `GenerationRequest.return_artifacts`（生产设置、只测试读）— dead-field（risk=medium）

- 位置：`vrl/generation/types.py:57`（field）、`70`（`__init__` 参）、`91`（赋值）
- 判死证据：
  ```
  $ grep -rn 'return_artifacts' vrl tests --include='*.py'
  types.py:57/70/91                              # 定义/init/赋值
  vrl/rollouts/collector/requests.py:95          return_artifacts={"output","trajectory"}   # 唯一生产 setter（硬编码）
  tests/... ×9 构造 kwarg + tests/rollouts/runtime/test_engine_requests.py:46 与 test_janus_pro_r1_wiring.py:105 ×2 断言
  $ grep -rnF '"return_artifacts"' vrl tests ; grep -rnF "'return_artifacts'" vrl tests
  （无匹配——无字符串键派发）
  ```
  唯一生产 setter 硬编码 `{"output","trajectory"}`，生产零 reader；两个 gatherer、token executor、janus/nextstep runtime 体读确认 output/trajectory 恒无条件构建，无一分支于 `return_artifacts`。无 YAML/pyproject/CI 引用，无 `**kwargs`/`asdict`/pickle 携带。done sprint 文档提到的 `return_artifacts` 指向已删除的 `vrl/rollouts/families/registry.py`（目录已不存在）。
- 动作：删 `return_artifacts` 字段 + `__init__` 参 + 赋值（types.py:57/70/91）；删唯一生产 setter（`vrl/rollouts/collector/requests.py:95`）；去掉 9 处测试构造 kwarg（`tests/generation/execution/test_generation_contracts.py:35`、`tests/ray/test_chunk_dispatch.py:78`、`tests/models/families/janus_pro/test_r1_model.py:156/385`、`tests/e2e/test_real_checkpoint_rl.py:793`、`tests/trainers/online/test_sft_regularizer.py:80`、`tests/rollouts/replay/test_multisegment_token_logprob.py:85`、`tests/rollouts/runtime/test_janus_pro_r1_wiring.py:120`、`tests/rollouts/collector/test_runtime.py:59`）；删 2 处 test-only 断言（`test_engine_requests.py:46`、`test_janus_pro_r1_wiring.py:105`）。
- 注意：medium 是因为它坐在 engine request 契约上，未来 provider 可能想要 artifact-selection 旋钮——但无任何 sprint doc stage consumer，且 in-flight sprint DoD（`SPRINT_native_generation_engine_program.md:329`）明确要求「`GenerationRequest` 每个公开字段都有非日志行为消费者」，本字段正好违反。删除是**推进**而非违背该 sprint DoD，且不改 `generation/ray/`，可与 sprint 并行。

### 1.6 `TokenBatch.sample_ids`（只测试读）— dead-field（risk=low）

- 位置：`vrl/generation/composition/token_autoregressive/token_loop.py:101-103`
- 判死证据：
  ```
  $ grep -rn 'sample_ids' vrl tests --include='*.py'
  vrl/rewards/base.py:584   "sample_ids": [artifact.sample_id ...]   # 无关的 reward 载荷键
  token_loop.py:102         def sample_ids(...)                       # 定义
  test_token_scheduler.py:62/68/73/94/114                            # 五处测试断言（唯一 reader）
  ```
  生产零 reader；loop 靠 `batch.sequences` 直接迭代，从不读该 property。`@property` 无法被 `asdict/fields` 复活。
- 动作：删 `TokenBatch.sample_ids` property，把 `test_token_scheduler.py` 五处断言改写成 `[s.sample_id for s in batch.sequences]`（line 94 为嵌套推导变体），覆盖不减。

### 1.7 `TokenScheduler.__len__`（零调用方）— dead-function（risk=low）

- 位置：`vrl/generation/composition/token_autoregressive/token_loop.py:115-116`
- 判死证据：
  ```
  $ grep -rnE 'len\(scheduler|if scheduler|while scheduler|not scheduler|bool\(scheduler|__len__' vrl/generation/composition tests/generation/composition
  token_loop.py:115  def __len__(self) -> int:   # 仅定义，无任何 len()/真值上下文调用
  ```
  `TokenScheduler` 仅在 `TokenAutoregressiveLoop.run`（token_loop.py:284）与测试中实例化，全程只用 `pop_batch()/add_many()/push_back_unfinished()`，从不 `len()` 或布尔取值。dunder 无法被 registry/字符串派发复活。
- 动作：删 `TokenScheduler.__len__`（token_loop.py:115-116）。无测试引用，无需改测试。

### 1.8 `ActiveSequence.remaining_tokens`（只测试读）— dead-field（risk=low）

- 位置：`vrl/generation/composition/token_autoregressive/token_loop.py:68-70`
- 判死证据：
  ```
  $ grep -rn 'remaining_tokens' vrl tests --include='*.py'
  token_loop.py:69                          def remaining_tokens(self) -> int:   # 定义
  test_token_scheduler.py:40/45            assert sequence.remaining_tokens == 2 / == 0   # 唯一 reader
  ```
  纯派生 property（`max(0, max_new_tokens - position)`），生产零 reader；loop 靠 `advance()/finished/position` 推进。测试相邻行（38-39, 43-44）已断言 `position`/`finished`，删这两句零覆盖损失。
- 动作：删 `ActiveSequence.remaining_tokens` property，丢弃 `test_token_scheduler.py:40,45` 两处断言（改断言 `position`/`finished`，其实相邻行已有）。

### 1.9 `TokenBatch.request_ids`（零调用方）— dead-field（risk=low）

- 位置：`vrl/generation/composition/token_autoregressive/token_loop.py:97-99`
- 判死证据：
  ```
  $ grep -rn 'request_ids' vrl tests --include='*.py'
  vrl/rewards/base.py:583   "source_request_ids": [...]   # 不同符号
  token_loop.py:98          def request_ids(...)           # 定义，无内部/外部/测试 reader
  ```
  连 sibling `sample_ids` 有测试 reader，`request_ids` 一个都没有。`@property` on slots=True dataclass，`asdict/fields` 不携带。executor 只导入 `TokenAutoregressiveLoop`，不碰此 property。
- 动作：删 `TokenBatch.request_ids` property。无测试引用。

### 1.10 `run_sample_chunks_with_oom_retry(min_sample_count=)`（全默认值）— dead-arg（risk=low）

- 位置：`vrl/generation/execution/chunks.py:123-132`（定义/校验）、`143`（使用）
- 判死证据：
  ```
  $ grep -rn 'min_sample_count' vrl tests --include='*.py'
  chunks.py:127/131/132/143   # 仅定义、>=1 校验、单处比较——全在 chunks.py 内
  $ grep -rn 'run_sample_chunks_with_oom_retry' vrl tests
  # 3 生产调用方 full_sequence_denoise/executor.py:277, chunk_autoregressive_denoise/executor.py:124,
  #   token_autoregressive/executor.py:82 + 测试 test_chunks.py:45 —— 全部 (chunks, run_one) 无 kwarg
  ```
  常量 1 下 `chunk.sample_count <= min_sample_count` 就是 `<= 1`，正是 `SampleChunk.split()` 已强制的下界。无 `**kwargs` 转发、无 YAML/entry-point。
- 动作：删 `min_sample_count` 参与其 `>= 1` 校验，把 guard 硬编码为 `chunk.sample_count <= 1`（guard 本身**必须留**——它重抛原始 OOM；删掉会让 `split()` 抛 `ValueError` 掩盖 OOM）。无测试传该 kwarg，无需改测试。

### 1.11 `GenerationWorkerCore.probe_chunk_size(execute_steps=)`（全默认值）— dead-arg（risk=low）

- 位置：`vrl/generation/execution/worker.py:302`（签名，实测 303）、`394`（使用）
- 判死证据：
  ```
  $ grep -rn 'probe_chunk_size' vrl tests --include='*.py'
  ray/worker.py:87        self.core.probe_chunk_size(request, max_samples=max_samples)   # 唯一生产转发，只传 max_samples
  ray/worker_fleet.py:129 getattr(...,"probe_chunk_size"...) ... max_samples=...        # getattr 派发也只传 max_samples
  tests/.../test_chunk_memory_shadow.py:202/225/238/255/272/278/312  # 传 max_samples，有时 margin/knee_threshold，从不 execute_steps
  $ grep -rn 'execute_steps' vrl tests
  worker.py:303 (param) / worker.py:394 (唯一用: dataclass_replace(prepared.config, execute_steps=execute_steps))
  config.py:36 (活 DenoiseLoopConfig 字段) / loop.py:173-174 (活 reader)
  ```
  `RayGenerationWorker.probe_chunk_size` 连 RPC 边界都不暴露 `execute_steps`；`DenoiseLoopConfig.execute_steps` 本身是活的（loop 截断步数），只有 probe 这个参数从不被定制。测试断言 `set(executor.executed_steps) == {2}` 测的是默认值，改成命名常量 `= 2` 后仍绿。
- 动作：删 `probe_chunk_size` 的 `execute_steps` 参，改用命名局部常量（如 `_PROBE_EXECUTE_STEPS = 2`）。保留 `margin`/`knee_threshold`（测试传非默认值以确定性 pin fit 路径与 knee 规则）。`DenoiseLoopConfig.execute_steps` 保留。无需改测试。

### 1.12 `validate_colocated_replay_memory(strict=...)`（只测试传、复制 env 旋钮）— dead-arg（risk=low）⚠ SPRINT 重叠

- 位置：`vrl/generation/ray/config.py`：参 line 132（finding 记 139）、`strict` 解析 line 158 + `if strict` 164
- 判死证据：
  ```
  $ grep -rn 'validate_colocated_replay_memory' vrl tests --include='*.py'
  config.py:121   validate_colocated_replay_memory(...)   # 生产调用，只传 bundle=/rollout_config=，无 strict=
  config.py:128   def validate_colocated_replay_memory(   # 定义
  config.py:290   __all__ 导出
  tests/trainers/test_memory_guards.py:77/89  # 唯一 strict= 调用方
  $ grep -rn 'VRL_STRICT_REPLAY_MEMORY_GUARD' .   # 排除 .venv/__pycache__/egg-info
  config.py:158   # 全仓唯一命中——无 CI/YAML/脚本设置它
  $ grep -rn 'strict=' tests/trainers/test_memory_guards.py
  :80 strict=True / :92 strict=True
  ```
  生产 strictness 只由 env var 决定（`strict is None -> os.environ.get("VRL_STRICT_REPLAY_MEMORY_GUARD")`），`strict=` kwarg 仅测试用、且绕过真实生产路径——正是 TEST-ONLY 死参。
- 动作：删 `strict: bool | None = None` 参，无条件保留 env-var 解析；把两个测试调用方（`tests/trainers/test_memory_guards.py:77-81`、`89-93`）改成 `monkeypatch.setenv("VRL_STRICT_REPLAY_MEMORY_GUARD", "1")`——顺带让它们跑真实生产 strict 路径（当前 truthy-token 解析零覆盖）。
- 注意：⚠ **复核偏差（行号漂移）**：finding 记参数在 line 139、env 在 164-171，实测参数在 132、env 解析在 158、`if strict` 在 164；因 `vrl/generation/ray/config.py` 正被 in-flight sprint 修改（`git status` 显示 `M`），行号已漂。substantive 判死证据（三调用点、env 全仓唯一命中、strict= 仅测试）全部仍成立。**与 in-flight sprint 直接碰撞（同文件）**，本条必须**排在 `SPRINT_native_generation_engine_program` 落地之后**，届时以最新 `config.py` 重新定位行号再删。

### 1.13 `validate_colocated_replay_memory(log=...)`（零调用方、logging-only）— dead-arg（risk=low）⚠ SPRINT 重叠

- 位置：`vrl/generation/ray/config.py`：参 line 133（finding 记 140）、`(log or logger).warning` line 166（finding 记 173）
- 判死证据：
  ```
  $ grep -rn 'validate_colocated_replay_memory' vrl tests  # 同 §1.12 的 3 调用点
  # 生产 config.py:121 只传 bundle=/rollout_config=；测试 :77/:89 只传 strict=True——均不传 log=
  $ grep -n 'logging' vrl/generation/ray/config.py
  5:import logging          # 唯一用途是被删的注解
  18:from vrl.utils.logging import init_logger   # 不同 import，保留
  133:    log: logging.Logger | None = None,
  ```
  `log` 唯一用途是 line 166 `(log or logger).warning(message)`——纯决定哪个 logger 发警告。搬迁遗留（该函数从 `vrl/utils/memory.py` 搬入，module 级 logger `config.py:20` 已足够）。
- 动作：(1) 删 `log: logging.Logger | None = None` 参（line 133）；(2) `(log or logger).warning(message)` 改 `logger.warning(message)`（line 166）；(3) **同时删** `import logging`（line 5，删参后成 F401 死导入——`vrl.utils.logging` 是另一 import，不受影响）。无测试传 `log=`，无需改测试。
- 注意：⚠ **复核偏差（行号漂移）**：finding 记 `140, 173`，实测 `133, 166`（同 §1.12，config.py 被 sprint 修改）。判死证据仍成立。auditor 原 action 漏了删 `import logging`，验证阶段补上。**与 in-flight sprint 同文件碰撞**，本条与 §1.12 一并**排在 sprint 之后**，一次性处理 `config.py`。

### 1.14 `TeaCacheConfig.enabled`（无 False producer，恒 True）— dead-field（risk=low）

- 位置：`vrl/generation/steps/denoise/teacache.py:49`（field）；reader `loop.py:168`
- 判死证据：
  ```
  $ grep -rn 'TeaCacheConfig' vrl tests --include='*.py'
  # 构造点：teacache.py from_sampling :76/:88（均硬编码 enabled=True，禁用输入 return None）
  #   + 6 处直接测试构造 test_teacache.py:36/38/42/58/72/79（均 enabled=True）
  $ grep -rn '\.enabled' vrl/generation/steps/denoise tests/generation/steps/denoise
  loop.py:168        if config.teacache is not None and config.teacache.enabled   # 第二合取恒 True
  test_teacache.py:24  assert cfg.enabled and cfg.signal == "latent"
  ```
  所有构造点都硬编码 `enabled=True`；`from_sampling` 的 `value.get("enabled", True)` **mapping key 是活的用户配置键，须留**——只有 dataclass field 死。无 YAML 设 teacache 块，perf 脚本只读 `.signal/.threshold`。
- 动作：删 `TeaCacheConfig.enabled` 字段（teacache.py:49）；保留 `from_sampling` 的 `value.get("enabled", True) -> return None`（活配置键）；从两个 `from_sampling` 构造点（:76-81、:88-93）去掉 `enabled=True`；`loop.py:168` guard 简化为 `if config.teacache is not None`；`test_teacache.py`：line 24 断言去掉 `cfg.enabled`（留 `cfg.signal == "latent"`），并从 6 个直接构造（lines 36, 38, 42, 58, 72, 79）去掉 `enabled=True` kwarg（slots=True，不删会 `TypeError`）。
- 注意：auditor 原 action 只提了改 line 24 断言，漏了 6 个测试构造点与两个 `from_sampling` 构造点的 `enabled=True`——slots=True dataclass 删字段后这些 kwarg 会抛 `TypeError`，验证阶段补上。

### 1.15 `VideoGenerationRequest.task_type`（form-2：alias producer 已删，默认值从不被读）— dead-field（risk=low）

- 位置：`vrl/generation/types.py:36`
- 判死证据：
  ```
  $ grep -n 'task_type' vrl/generation/types.py
  17:    task_type: str | None = None       # GenerationInput.task_type —— 活，勿删
  36:    task_type: str = "text_to_video"   # VideoGenerationRequest —— 死
  $ grep -rn 'task_type=' vrl tests   # VideoGenerationRequest 的 setter
  vrl/scripts/eval/wan_i2v_logprob_parity_probe.py:109  task_type="image_to_video"
  tests/models/families/echo/test_echo_flow_policy.py:75  task_type="text_to_video"
  ```
  所有 `.task_type` 属性读都落在 `GenerationInput`（`execution/ids.py:23-24`、`magi_1/model.py:847`）、`PromptExample`（`trainers/data/prompts.py`）、config `data.task_type`（`schema.py:187`）上——**零** `VideoGenerationRequest` 实例读。共享 executor 的 `build_video_request`（executor.py:224-243）从不 set `task_type`。probe 脚本靠 `reference_image`（line 125）条件 i2v，删 kwarg 行为不变。
- 动作：删 `VideoGenerationRequest.task_type`（types.py:36）；去掉两个 setter：`wan_i2v_logprob_parity_probe.py:109`、`tests/models/families/echo/test_echo_flow_policy.py:75`（slots=True，不删会 `TypeError`）。`GenerationInput.task_type`（types.py:17）与 config `data.task_type`（schema.py:187）是分开的活消费者，**勿动**。
- 注意：re-grep 另见 `tests/trainers/online/test_reward_update_flow.py:117` 的 `task_type="text_to_video"`——已核实它构造的是 `PromptExample`，非 `VideoGenerationRequest`，属活消费者，**不在本条改动范围**。auditor 原 action 漏了 echo 测试这个第二 setter，验证阶段补上。

### 1.16 `GenerationRequest.priority`（零 reader）— dead-field（risk=low）

- 位置：`vrl/generation/types.py:59`（field）、`72`（`__init__` 参）、`93`（赋值）
- 判死证据：
  ```
  $ grep -n 'priority' vrl/generation/types.py
  59:    priority: int = 0
  72:        priority: int = 0,
  93:        self.priority = priority
  $ grep -rn '\.priority' vrl tests --include='*.py'
  # 仅 types.py:93 自赋值 + vrl/ray/actor_pool.py:61 RayActorJob.priority（不同类型，来自 assignment.estimated_cost）
  ```
  唯一生产构造点 `vrl/rollouts/collector/requests.py:83` 不传 priority；无字符串键/`getattr`/YAML/`asdict` 携带。活的 `RayActorJob.priority` 是完全不同的概念。
- 动作：删 `priority` 字段、`__init__` 参、`self.priority = priority` 赋值。无调用方传、无测试引用，无需改测试。
- 注意：本条已被 in-flight sprint DoD **明确点名删除**（`SPRINT_native_generation_engine_program.md:145` 与 `329-330`：「删除 `GenerationRequest.priority`」）。它不改 `generation/ray/`，可与 sprint 并行，但若 sprint 先删了应视为已完成、勿重复。

## 2. 验证协议

- **每条删除后**：`ruff check <touched files>` + `ruff format --check <touched files>`（仅本条触及的 Python 文件；YAML 改动不过 ruff）。
- **全簇完成后**：`pytest` 下列相关子目录 + `pytest -m "not e2e and not slow_test"` 子集不新增失败：
  - `tests/generation/steps/denoise/`、`tests/generation/bindings/full_sequence_denoise/`（§1.1/§1.3/§1.14）
  - `tests/generation/composition/token_autoregressive/`（§1.6-1.9）
  - `tests/generation/execution/`（§1.10/§1.11）
  - `tests/generation/ray/`（§1.2，sprint 后）
  - `tests/trainers/test_memory_guards.py`（§1.12/§1.13，sprint 后）
  - `tests/quality/test_preview.py`（§1.4）
  - `tests/rollouts/runtime/`、`tests/rollouts/collector/`、`tests/rollouts/replay/`、`tests/models/families/janus_pro/`、`tests/e2e/`、`tests/trainers/online/`、`tests/ray/`（§1.5）
  - `tests/models/families/echo/test_echo_flow_policy.py`（§1.15）
  - `tests/rollouts/runtime/test_family_registry.py`（§1.3）
- **config-resolve 验证**（§1.3）：`vrl.config.lint` 全绿 + 两个 cosmos_predict2 preset（`online_grpo_droid_overfit_validation.yaml`、`online_grpo_droid_lora_480p_curve.yaml`）删 `same_latent` 后仍能解析。
- **基线（清理前，2026-07-23）**：fast subset 2620 passed / 7 pre-existing failures（架构边界 + causvid/magi_1 打包摘要，与本清理无关）；`vrl.config.lint` 与 `ruff check .` 全绿。删除后这三项须保持。
- 逐条触及测试文件（从各 action 提取）：
  - §1.1：`test_preallocation.py`、`test_executor_denoise_mode.py`
  - §1.2：`test_rollout_launcher.py`（sprint 后）
  - §1.3：`test_preallocation.py`、`test_prev_sample_mean_storage.py`、`test_ref_noise_pred_cache.py`、`test_executor_denoise_mode.py`、`test_family_registry.py`
  - §1.4：`tests/quality/preview.py`（源）、`tests/quality/test_preview.py`
  - §1.5：见上 9 构造 + 2 断言文件
  - §1.6/§1.8：`test_token_scheduler.py`
  - §1.7/§1.9/§1.10/§1.11/§1.16：无需改测试
  - §1.12：`test_memory_guards.py`（sprint 后）
  - §1.14：`test_teacache.py`
  - §1.15：`test_echo_flow_policy.py`

## 3. Non-Goals

- 不删被「能 raise 的校验」/控制流分支/runtime/config/Ray 调用消费的字段（AGENTS 死字段规则）：如 `DenoiseLoopConfig.execute_steps`（loop 截断步数，活）、`GenerationInput.task_type` 与 config `data.task_type`（活）、`RayActorJob.priority`（LPT 排序，活）、`TeaCacheConfig` 的 `enabled` mapping-key（用户配置键，活）、`build_chunk_encoded` 及其 cosmos override（活 in-stage hook）。
- 不动 DO-NOT-FLAG 豁免项：`_GENERATION_CUDA_RUNTIME_RESIDUAL_BYTES`（`generation/execution/worker.py`）、`_REWARD_CUDA_RUNTIME_RESIDUAL_BYTES`（`rewards/base.py`）、`ensure_loaded`（`rewards/runtime.py`）、`process_gpu_used_bytes`（`utils/cuda_memory.py`）、sana/hunyuan 的 `prepare_latents` 修复。
- 不为省行数扁平化 protocol / lazy-import / 跨家族一致性 thin function：`run_sample_chunks_with_oom_retry` 的 OOM guard 本身保留（重抛原始 OOM）；`GenerationWorkerCore.current_policy_version` 本簇不删（归 execution-scope）；各 `__init__.py` 再导出 facade 保留。
- **不与 in-flight `SPRINT_native_generation_engine_program` 抢文件**：§1.2（`ray/worker.py` + sprint 正改的 `test_rollout_launcher.py`）、§1.12/§1.13（sprint 正改的 `ray/config.py`）三条**排在该 sprint 落地之后**，届时以最新文件重新定位行号；§1.5/§1.16 虽属同簇但不碰 `generation/ray/`，可并行（§1.16 且已被 sprint DoD 点名，勿重复删）。
- 不实现 `same_latent` 的承诺语义（团队已弃用 latent sharing，正确动作是删除而非补实现）。

## References

- `vrl/generation/bindings/full_sequence_denoise/executor.py:72,419,436,442,458,506-520`、`__init__.py:8,26`
- `vrl/generation/ray/worker.py:68-69`、`vrl/generation/execution/worker.py:185,210,302-303,394`、`vrl/generation/ray/worker_fleet.py:129`、`vrl/generation/ray/weight_sync.py`
- `vrl/generation/steps/denoise/config.py:18,36`、`loop.py:160-161,168,173-174`、`teacache.py:49,76,88`、`vrl/generation/bindings/full_sequence_denoise/layout.py:112`、`vrl/config/schema.py:303`
- `vrl/config/presets/base/rollout/denoise.yaml:17`、`presets/experiment/cosmos_predict2/online_grpo_droid_overfit_validation.yaml:22,112-118`、`online_grpo_droid_lora_480p_curve.yaml:22`、`presets/dataset/droid_overfit_validation.yaml:11`
- `vrl/generation/types.py:17,36,57,59,70,72,91,93,149`
- `vrl/generation/execution/chunks.py:123-132,143`
- `vrl/generation/ray/config.py:5,121,128,132-133,158,164,166,290`（行号随 in-flight sprint 漂移）
- `vrl/generation/composition/token_autoregressive/token_loop.py:68-70,97-99,101-103,115-116,284`
- `vrl/rollouts/collector/core.py:199-204`、`requests.py:83,95`、`vrl/rewards/base.py:583-584`、`vrl/ray/actor_pool.py:29,61`、`vrl/generation/ray/executor.py:111`
- `tests/quality/preview.py:48-49`、`tests/quality/test_preview.py:65,81-84,87`
- `tests/generation/steps/denoise/test_preallocation.py`、`tests/generation/bindings/full_sequence_denoise/test_executor_denoise_mode.py`、`test_prev_sample_mean_storage.py`、`test_ref_noise_pred_cache.py`、`tests/generation/composition/token_autoregressive/test_token_scheduler.py`、`tests/generation/steps/denoise/test_teacache.py`、`tests/generation/execution/test_chunk_memory_shadow.py`、`tests/trainers/test_memory_guards.py`、`tests/generation/ray/test_rollout_launcher.py`、`tests/rollouts/runtime/test_family_registry.py`、`tests/models/families/echo/test_echo_flow_policy.py`
- `tests/generation/execution/test_generation_contracts.py:35`、`tests/ray/test_chunk_dispatch.py:78`、`tests/models/families/janus_pro/test_r1_model.py:156,385`、`tests/e2e/test_real_checkpoint_rl.py:793`、`tests/trainers/online/test_sft_regularizer.py:80`、`tests/rollouts/replay/test_multisegment_token_logprob.py:85`、`tests/rollouts/runtime/test_janus_pro_r1_wiring.py:105,120`、`tests/rollouts/collector/test_runtime.py:59`、`tests/rollouts/runtime/test_engine_requests.py:46`
- 关联：[[SPRINT_deadcode_00_overview]]、[[SPRINT_native_generation_engine_program]]、[[SPRINT_trajectory_views_types_dead_fields_cleanup]]
