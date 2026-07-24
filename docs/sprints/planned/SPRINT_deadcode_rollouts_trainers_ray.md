# SPRINT: `rollouts` / `trainers` / `ray` 死代码清理（planned · RECONCILED）

状态：**RECONCILED（2026-07-24）against main @ `7c748532`**（= `origin/main` tip，自审计基线 `88ed756e` 以来落地约 63 个 cleanup/refactor commit）。原始审计（planned 2026-07-23，23 条对抗验证通过的死代码）已针对当前 checkout 逐条复核。
复核结论：**20 条仍需处理**（14 STILL_VALID + 6 RELOCATED），**3 条已由 origin 落地**（`dones` / `expected_count` / `TrainState` 累加），**2 条虽仍有效但支撑事实已随 origin 迁移**（§1.5、§1.18，见 §3——它们仍列于 §1）。无 verdict 级 CHANGED/INDETERMINATE。
来源：dead-code-audit workflow（五种死代码形态 + 对抗验证 + 删除类二次字符串引用检查）。
关联：[[SPRINT_deadcode_00_overview]]；与 [[SPRINT_trajectory_views_types_dead_fields_cleanup]]（死字段规则同源）、[[SPRINT_fbag_00_overview]]（`gather_full_state_dict` 的 cohesive-keep 判定，见 §1.4 复核）、[[SPRINT_grab_bag_file_audit]]（`require_method=False` 历史 producer，见 §1.18）、[[SPRINT_design_smell_audit]]（`mask_key` 假旋钮先例，见 §1.17）、[[SPRINT_native_generation_engine_program]]（在飞 sprint，`vrl/ray/` 文件重叠，见 §1.6/§1.8/§1.9）互有承接。

> **执行状态（2026-07-24）**：16 条已落地 `c951ceba`。§1.3 `collect_prompt_batches` **撤销**（复核发现 `remap_group_ids_` 是 per-prompt 优势归一化的承重代码，非死代码）。§1.6/§1.8/§1.9（3 条 ray/ 项）已落地 `58c4e0b3`（在飞 sprint 已合并，无 worktree 冲突）。

> 章节编号沿用原始审计（§1.1–§1.23），以保内部交叉引用不断链。已落地的三条（§1.1、§1.13、§1.19）从 §1 移入 §2，故 §1 编号有意留空档。所有 §1 条目的 KEY grep 已于 2026-07-24 对 `7c748532` 重跑，位置行号按当前树更新，逻辑判定零偏差。

## 0. 一句话

这是一个混合层死代码簇：主体是**从不被非默认调用的死形参 / 死旋钮**，其余是死字段、死分支、form-4 重复实现与一个 test-only 函数。复核后主体判定全部成立。最锋利的一条仍是 `collect_prompt_batches` 的裸字符串攒批路径（§1.3，RELOCATED——`remap_group_ids_` 定义由 `ops.py:92` 上移至 `:72`）：生产端 prompt 恒为 `PromptExample`，字符串路径连同其唯一下游 `remap_group_ids_` 在生产完全不可达，但**误删风险最高**（大量 trainer/orchestration 测试用裸 `str` 驱动真实 schedule 走该路径，删除前须逐一改写为 `PromptExample` 并注意调用粒度变化）。次高风险是 `gather_full_state_dict`（§1.4，RELOCATED——定义由 `fsdp.py:287` 上移至 `:227`）：zero 生产调用者（TEST-ONLY），但被 8 个测试当作**验证 live 生产路径的 oracle**，须**迁入测试基础设施而非删除**。

## 1. 待删清单（仍有效 · STILL_VALID + RELOCATED）

> 顺序沿用原编号。medium-risk 在前（§1.2–§1.5），其余 low-risk 按层分组（`ray/` §1.6–§1.12、`rollouts/` §1.14–§1.17、`trainers/` §1.18/§1.20–§1.22、`utils/` §1.23）。RELOCATED 条目的「位置」行已更新为当前 `7c748532` 行号，并标注移动量；「判死证据」代码块保留原始审计文本作为推理留档，行号以「位置」行与「复核」行为准。

### 1.2 `RolloutCollector.collect_unscored(seed=)` / `GenerationRequestBuilder.build(seed=)` — dead-arg（risk=medium · RELOCATED）
- 位置（当前）：`vrl/rollouts/collector/core.py:149`（`seed` 形参，def 在 :142），`:167`（转发）；`vrl/rollouts/collector/requests.py:53`（形参），`:61-62`（注入）
- 复核 2026-07-24：两符号均在。`requests.py:53/61-62` 与原引用完全一致；`core.py` 的 `collect_unscored seed` 形参由 `:169` 上移至 `:149`（转发 `:187→:167`）。生产 caller 仍不传 `seed`——`prompt_collection.py` 的 `flush_pending_prompts`(:160) 与 example-path 调用(:176) 传 `group_size/metadata/request_overrides/runtime_debug/policy_version`，wrapper(:99-105) 转发 `**kwargs` 但从不含 `seed`。**仍为死形参（非默认路径 test-only）**。
- 判死证据（原始审计）：
  ```
  $ grep -rn 'collect_unscored' vrl/ --include='*.py'
  prompt_collection.py:99/105  ← **kwargs 包装，仅本文件内被下述两处调用
  prompt_collection.py:160/176 ← 生产唯一调用链，两处均显式传参且不含 seed
  $ grep -rn 'seed' vrl/rollouts/collector/requests.py vrl/rollouts/collector/core.py
  requests.py:53/61-62  ← seed 形参 + sampling["seed"]=seed 注入
  ```
  唯一生产调用链均不传 `seed`；`build` 的唯一生产调用是 `core.py`（现 :182→随定义漂移）。仓库自身 preset 明确写明确定性种子的钦定路径是 `request_overrides.seed`——`online_grpo_droid_overfit_validation.yaml:25-27`：「the sanctioned per-prompt seed path … a top-level `sampling.seed` key does not exist」，与 `sampling["seed"]=seed` 完全同构（`requests.py:57-63` 顺序为 `sampling["seed"]=seed` 先、`sampling.update(request_overrides)` 后 → `request_overrides` 优先）。
- 动作：删除两处 `seed` 形参及 `requests.py:61-62` 的 `sampling["seed"]=seed` 注入（**保留** `sampling` dict 的 `"seed"` 键本身——它经 `request_overrides`/config `sampling` 段流入并被 executor 消费）。测试与 harness 更新：(1) `tests/rollouts/collector/_helpers.py` `collect_scored` 删 `seed` 形参与转发；(2) `tests/rollouts/collector/test_runtime.py:292` 的 `collect_scored(seed=5)` 改为 `request_overrides={"seed": 5}`（断言 L299 不变）；(3) `tests/rollouts/runtime/test_engine_requests.py:32` 的 `seed=7` 并入该调用的 `request_overrides`（断言 L44 不变）；(4) `tests/quality/preview.py` `build_preview_request` 删 `seed` 形参传递，改为 `overrides.setdefault("seed", seed)` 以**保持现行优先级**，`tests/quality/test_preview.py:48/57` 同步。
- 注意：medium。(a) `tests/quality/preview.py` 是 GPU 质量预览 harness（长期资产），功能性依赖 `seed=` 做确定性预览（`_PREVIEW_BASE_SEED+index`），**必须迁移而非删除**，且迁移须保精度序（`setdefault` 而非直接赋值）。(b) **不动** `test_runtime.py:610/975` 与 `test_janus_pro_r1_wiring.py:30/40` 的 `seed=None`——那是 `GenerationSampleRow.seed`，另一个活符号。

### 1.3 `collect_prompt_batches` 裸字符串攒批路径 + `remap_group_ids_` — dead-branch + 关联 dead-function（risk=medium · RELOCATED）
- 位置（当前）：`vrl/rollouts/orchestration/prompt_collection.py:73`（`pending_prompts`），`:157-167`（`flush_pending_prompts`），`:173-189`（str 循环，append 在 :186），`:249`（`remap` 调用）；`vrl/rollouts/batch/ops.py:72`（`remap_group_ids_` 定义，export 在 :119）
- 复核 2026-07-24：字符串路径全在。**`remap_group_ids_` 定义由 `ops.py:92` 上移至 `:72`（-20 行）**，仍导出（`ops.py:119`），仍仅被 `prompt_collection.py:249`（生产）+ 一个测试调用。生产 `collect_prompt_batches` caller（`strict_on_policy.py:89`、`continuous/producer.py:342`）喂 `PromptExample`-derived prompts，故裸字符串路径仍生产不可达。
- 判死证据（原始审计）：
  ```
  $ grep -rn 'collect_prompt_batches' vrl/ --include='*.py'
  strict_on_policy.py:89、continuous/producer.py:342  ← 生产唯二调用，prompts 均来自 trainer
  $ grep -rn 'remap_group_ids_' vrl/ tests/ --include='*.py'
  ops.py（定义）；prompt_collection.py 唯一生产调用（仅 isinstance(remap, list) 分支可达）
  tests/trainers/online/test_reward_update_flow.py:1275/1338  ← TEST-ONLY
  ```
  生产 prompts 溯源：`trainer.py rollout_schedule.next_iteration ← online.py load_prompt_examples_from_config(cfg.data)`；`vrl/trainers/data/prompts.py` 三个 loader 恒构造 `PromptExample`，`PromptExample.generation_input` → 恒命中 `hasattr` 分支、`remap` 恒为 `int`。故裸字符串攒批路径及其唯一下游 `remap_group_ids_` 在生产不可达，仅 tests 以 `["p0"]` 形式构造。
- 动作：删除 `pending_prompts`/`pending_indices`/`flush_pending_prompts`、`isinstance(remap, list)` 分支、`remap_group_ids_` 及其 `ops.py:119` export 与 `test_reward_update_flow.py:1338` 的 TEST-ONLY 用例；**保留** `split_batch_by_group`、`requests.py` `_resolve_input` 的 `str→GenerationInput` 兼容。删后收紧 `collect_prompt_batches` 的 remap 注释与 `unscored_groups` 类型（`list[int] | int → int`）。测试重写（范围远大于三文件）：
  - (a) `tests/e2e/test_real_checkpoint_rl.py:626` 的 `trainer.step([case.prompt])` 改为 `trainer.step([PromptExample(prompt=case.prompt)])`（其 fake collector 已用 `**kwargs` 接参）；
  - (b) 重写全部以裸 `str` 驱动真实 schedule 的 trainer 测试：`tests/trainers/online/test_precision_drift_guard.py`、`test_state_restore.py`、`test_step_split.py`、`test_reward_update_flow.py`、`test_trainable_state.py`、`test_trajectory_granularity.py`、`test_advantage_and_metrics.py`（用 `grep -rn 'trainer\.step(\["' tests/` 定界）；
  - (c) orchestration/continuous 测试：`test_prompt_collection.py`、`test_orchestration.py`、`continuous/test_schedule.py`、`test_owner.py`、`test_contracts.py`、`test_queue.py`、`test_scheduler.py`，及 `tests/rollouts/collector/test_runtime.py:785` 与 `tests/rollouts/collector/_collect.py`。
- 注意：medium，本簇最高误删风险。重写时**注意调用粒度变化**：裸 `str` 曾合并为一次 `collect_unscored` 调用，`PromptExample` 为每 prompt 一次调用——涉及调用次数/合批断言的测试须按新粒度改写而非机械替换。planned sprint（`SPRINT_continuous_stage_contracts_and_baseline`、`SPRINT_continuous_three_stage_pipeline_program`）仅把 `collect_prompt_batches` 当复合 task 引用，**不依赖**字符串路径，故不冲突。`train_dpo.py` 的 `OfflineDPOTrainer` 与此无关。

### 1.4 `gather_full_state_dict` — dead-function（risk=medium · RELOCATED）
- 位置（当前）：`vrl/trainers/fsdp.py:227`（定义，docstring 交叉引用 :563）
- 复核 2026-07-24：仍定义、**零生产调用者**。**定义由 `fsdp.py:287` 上移至 `:227`（-60 行）**。生产侧仅注释提及（`online.py:229`、`fsdp.py:563` docstring、`strategy.py:1001` 注释）；全部实际 caller 是测试（`test_fsdp.py` 12 处、`test_fsdp_gather_distributed.py` 3 处）。auditor 关于「若干 test site 把它当 live 生产路径 oracle」的判断仍成立。**仍为 TEST-ONLY 死函数**。
- 判死证据（原始审计）：
  ```
  $ grep -rn 'gather_full_state_dict' vrl/ tests/ --include='*.py'
  vrl/scripts/common/online.py     ← 注释提及（doc comment）
  vrl/trainers/fsdp.py:227 定义 / :563 docstring 交叉引用
  vrl/trainers/strategy.py         ← 注释「Do NOT route this through the FSDP gather_full_state_dict」
  tests/trainers/test_fsdp.py                     ← import + 多调用点
  tests/trainers/test_fsdp_gather_distributed.py  ← import + 调用点
  ```
  **零生产调用者**（三处均为注释/docstring），共 8 个测试调用点。生产 export 路径已替换：`FSDPStrategy` 经 `gather_trainable_state_dict`，`DDPStrategy` 经 unwrapped `state_dict()`，加载经 `load_trainable_state_dict`/`load_full_state_dict`。
- 动作：将 `gather_full_state_dict`（body 完整保留，含防御性 DTensor `full_tensor()` 物化）**迁入**共享测试 helper（如 `tests/trainers/conftest.py` 或 `tests/trainers/_state_dict_helpers.py`），并 repoint 全部测试调用点。**不删/不 repoint** 那几个把它当 oracle 的测试——它们覆盖 live 生产路径（`load_full_state_dict`、`FSDPStrategy.load/export_trainable_state`、legacy full-checkpoint 兼容、LoRA load 时 frozen-param 不变性）。**不要**把 `test_fsdp_gather_distributed.py` 的 frozen-param gather 移植到 `gather_trainable_state_dict`——后者按 `requires_grad` 选键，语义上无法产出 frozen params。同步把 `cpu_offload=False` rank0-only-trap 理由从被删 docstring 迁入 `gather_trainable_state_dict`/`gather_full_optimizer_state_dict` docstring（`fsdp.py:563` 交叉引用须更新），并改写 `online.py:229`、`strategy.py:1001` 两处注释使其不 dangle（直接描述 DCP `get_model_state_dict(full_state_dict=True)` 行为，不再点名已移出的函数）。
- 注意：medium，且**推翻 [[SPRINT_fbag_00_overview]] 的 cohesive-keep**：那次判的是 `fsdp.py` grab-bag 归属，非本函数 liveness。本次按 form-1 复核 caller 后确认零生产调用者——是 TEST-ONLY，故迁入测试基础设施。二者不矛盾。

### 1.5 `OfflineDPOTrainerConfig.timestep_subset` — dead-config-knob（risk=medium · STILL_VALID · 支撑事实已变 → 见 §3）
- 位置（当前）：`vrl/trainers/offline/dpo.py:60`（字段），`:204`（分支），`:218`（错误信息）
- 复核 2026-07-24：字段、非 None 分支、错误文本全在（行号与原引用一致）。`grep 'timestep_subset' vrl/` 仅命中 `dpo.py`；`vrl/config/` 零命中。**唯一生产构造点已由 `train_dpo.py` 迁至 `vrl/config/builders.py:385`**（显式 kwargs，仍不含 `timestep_subset`）；`train_dpo.py` 不再构造 config。仅 setter 是测试。**仍为 TEST-ONLY 死旋钮**。删除前须核对的是 `builders.py:385` 而非原证据里的 `train_dpo.py:85-103`。
- 判死证据（原始审计，构造点位置见上「复核」行）：
  ```
  $ grep -rn 'timestep_subset' vrl/ tests/ --include='*.py'
  vrl/trainers/offline/dpo.py:60(定义),204,205,218(读+错误信息)
  tests/trainers/test_offline_dpo_timesteps.py  ← test-only setters
  ```
  唯一生产构造显式 kwargs 建 config、无 `**` splat、**不含** `timestep_subset`；无 `asdict`/`fields()`/`vars()`/`**kwargs` 转发；`state_dict` 只序列化 `global_step` 与 optimizer。YAML 面无法达：`ActorSection`（`vrl/config/schema.py:616-635`）无此键，手写白名单 `_OFFLINE_DPO_ACTOR_FIELDS` 不含它，未知 actor 键 loud-fail。docs 唯一提及指的是另一个符号 `select_timestep_subset`（online trainer）。
- 动作：删除 `timestep_subset` 字段与 `_sample_timesteps` 中 `if self.config.timestep_subset is not None` 分支（**保留** `scheduler.timesteps`-empty fail-fast，这是唯一生产路径）；删除 `dpo.py:218` 错误信息里的悬空补救文本；更新/删除 `tests/trainers/test_offline_dpo_timesteps.py` 的三个触点。
- 注意：medium（TEST-ONLY dead knob）。删前确认 `builders.py:385` 生产构造确无 splat。

### 1.6 `RayActorGroup.launch(metadata_method=)` — dead-arg（risk=low · STILL_VALID）
- 位置（当前）：`vrl/ray/actor_group.py:42`（形参），`:80`（唯一使用）
- 复核 2026-07-24：`grep 'metadata_method'` 仅命中 `actor_group.py:42`(def) + `:80`(`getattr(actor, metadata_method).remote()`)。两个 `launch` caller 均省略它（`vrl/generation/ray/launcher.py:98`、`tests/ray/test_ray_actor_pool.py:44`，零 setter）。**仍为 form-5 死形参**。
- 判死证据（原始审计）：
  ```
  $ grep -rn 'metadata_method' vrl/ tests/ --include='*.py'
  vrl/ray/actor_group.py:42:  metadata_method: str = "worker_metadata",   ← 定义
  vrl/ray/actor_group.py:80:  metadata_refs = [getattr(actor, metadata_method).remote() ...]  ← 唯一使用
  ```
  form-5 死形参：两个 `launch` 调用者均省略它（launcher 传 `startup_method="load_policy"`+`concurrency_groups`；test 传 `startup_method="startup"`），`launch` 无 `**kwargs` 转发。默认 `"worker_metadata"` 是真协议名——`vrl/generation/ray/worker.py:74`、`_EchoWorker` 均定义它。对照 `startup_method`（有两个 producer）应**保留**。
- 动作：删除 `metadata_method` 形参，硬编码为 `actor.worker_metadata.remote()`；**保留** `startup_method`。无 caller/test 更新。
- 注意：low，但 **`actor_group.py` 属在飞 sprint [[SPRINT_native_generation_engine_program]] 的未提交 worktree**。已复核该 sprint diff **不触及** `metadata_method`，无符号级冲突；为避免与活跃 worktree 打架，**建议 sequence after the sprint** 落地或与作者协调。

### 1.7 `ClusterTopology.has_non_driver_gpus` — dead-field（risk=low · STILL_VALID）
- 位置（当前）：`vrl/ray/dependencies.py:54`（定义，docstring :44-47）
- 复核 2026-07-24：`grep 'has_non_driver_gpus'` 仅命中 `dependencies.py:54`(def) + `tests/ray/test_dependencies.py:25/33/42`（断言）。无生产消费者——`cross_node_preflight`(`vrl/ray/placement.py`) 直接读 `non_driver_gpus`/`driver_gpus`，从不读该 property。**仍为 test-only 死字段**。
- 判死证据（原始审计）：
  ```
  $ grep -rn 'has_non_driver_gpus' vrl/ tests/ --include='*.py'
  vrl/ray/dependencies.py:54:  def has_non_driver_gpus(self) -> bool:   ← 定义
  tests/ray/test_dependencies.py:25/33/42:  assert topo.has_non_driver_gpus is ...  ← 仅测试断言
  ```
  test-only property。唯一生产消费者 `cross_node_preflight` 直接读 `topology.non_driver_gpus`/`topology.driver_gpus`。docstring 声称的 `cross_node auto-detect (run_online_recipe)` 已废：`online.py` 从不调 `inspect_cluster`，`cross_node` 是 explicit config bool。docs 唯一命中是历史已删符号 `_cluster_has_non_driver_gpus`。
- 动作：删除 `has_non_driver_gpus` property；在 `tests/ray/test_dependencies.py` **直接删除**三处 property 断言（L25/33/42）——紧邻前置行已有等价的 `topo.non_driver_gpus` 直接断言。修正 `ClusterTopology` docstring 只引 `vrl.ray.placement.cross_node_preflight`，删掉 stale 的 `cross_node auto-detect (run_online_recipe)` 声称。

### 1.8 `GlobalRayPlacementOwner.placement_strategy` — dead-config-knob（risk=low · STILL_VALID）
- 位置（当前）：`vrl/ray/placement.py:226`（字段），`:406-407`（override 分支）；`_strategy()` inline 目标 `create()` 约 :250+
- 复核 2026-07-24：字段与 override 分支均在，**较原引用 :220/:401-402 下移 +3 行**，实质不变。`placement_strategy=` 作为 setter 零命中；旧 YAML 键仍被 `resources.py:1256` hard-reject；override 分支仍无 producer。**仍为 form-2 死分支**。
- 判死证据（原始审计，行号见「位置」行）：
  ```
  $ grep -rn 'placement_strategy' vrl/ tests/ --include='*.py' | grep -v chunk_placement_strategy
  vrl/ray/placement.py:  placement_strategy: str | None = None   ← 定义
  vrl/ray/placement.py:  if self.placement_strategy: return self.placement_strategy  ← 唯一读者（override 分支）
  vrl/ray/resources.py:1256:  for stale_pool_key in ("placement_strategy", ...)   ← removed-key 硬拒绝
  tests/config/test_schema.py:  @parametrize("removed_key", ["placement_strategy", ...])  ← 覆盖拒绝逻辑，保留
  ```
  form-2 死分支：唯一生产构造 `online.py` 与四处测试构造均不设它，无 `**kwargs`/`asdict` 路径。
- 动作：删除 `placement_strategy` 字段与 `_strategy()` 的 override 分支；字段删后 `_strategy()` 只剩一行 `"SPREAD" if self.resources.cross_node else "PACK"`（保留解释注释），**内联进 `create()` 并删 `_strategy()`**。无 test 改动。**并**更新 preset 注释 `vrl/config/presets/experiment/cosmos_predict2_5/online_nft_kling_video_reward_cross_node.yaml`（约 L29-30）——把 `._strategy` 指针改指 `GlobalRayPlacementOwner.create`，避免长期 config 的 doc 指针悬空。
- 注意：low，但 **`placement.py` 属在飞 sprint 未提交 worktree**。已复核 diff **不触及**该字段/分支，无符号级冲突；仍建议 **sequence after the sprint** 或与作者协调。

### 1.9 `GlobalRayPlacementOwner.ready_timeout_s` — dead-arg（risk=low · STILL_VALID）
- 位置（当前）：`vrl/ray/placement.py:227`（字段=600.0），`:264`（`ray.get` timeout），`:267`（错误信息）
- 复核 2026-07-24：字段与两处消费均在，**较原引用 :221/:258/:261 下移 +3 行**。`ready_timeout_s=` 作为 setter 零命中；唯一生产构造 `online.py` + 四处测试均用默认。**仍为「从不被行使的 override 旋钮」（死形参）**。
- 判死证据（原始审计，行号见「位置」行）：
  ```
  $ grep -rn 'ready_timeout_s' vrl/ tests/ --include='*.py'
  vrl/ray/placement.py:  ready_timeout_s: float = 600.0   ← 定义
  vrl/ray/placement.py:  ray.get(pg.ready(), timeout=float(self.ready_timeout_s))
  vrl/ray/placement.py:  f"Ray placement group not ready after {self.ready_timeout_s:.0f}s: "
  （'ready_timeout_s=' 作为 setter 零命中）
  ```
  form-5 死形参：`600.0` 默认**被消费**（`ray.get` timeout + not-ready 错误信息），但无任何构造点覆盖它；无 `**kwargs`/`asdict`/YAML 键。
- 动作：把 `ready_timeout_s` 从 dataclass 字段降级为模块级常量（如 `_PLACEMENT_READY_TIMEOUT_S = 600.0`）并在 `create()` 引用。需要快 timeout 的测试可 monkeypatch 该常量；无 test cleanup。
- 注意：low，**`placement.py` 在飞 sprint 未提交 worktree**（同 §1.8）。已复核 diff 不触及该字段；建议 **sequence after the sprint**。

### 1.10 `_to_plain` — duplicate-impl（risk=low · STILL_VALID）
- 位置（当前）：`vrl/ray/resources.py:1367`（定义）；调用点 `:1174/:1200/:1267/:1289/:1307`
- 复核 2026-07-24：body 仍与 `vrl/utils/config.py:82` `to_builtin()` 逐字节同逻辑。`grep '_to_plain'` 仅命中本文件（def + 5 调用点）。`resources.py:10` 已 import from `vrl.utils.config`。定义较原引用 :1366 下移 +1。**仍为 form-4 重复**。
- 判死证据（原始审计，行号见「位置」行）：
  ```
  $ grep -rn '_to_plain' vrl/ tests/ --include='*.py'
  vrl/ray/resources.py  ← 5 个调用点 + 定义，全在本文件
  ```
  form-4 重复：body 与 `to_builtin()` 逐字节同逻辑（try-import `DictConfig`/`ListConfig`/`OmegaConf`、except 返原值、`isinstance → OmegaConf.to_container(resolve=True)`、else 透传），仅 docstring 不同。`config.py` 是 leaf module，无循环。
- 动作：删除 `_to_plain`，五个调用点改用 `to_builtin`（扩展现有 `from vrl.utils.config import cfg_get` 一行）。无 test 改动。
- 注意：`resources.py` **不在**在飞 sprint 未提交集，无 sprint 重叠。

### 1.11 `trainer_torch_device(actual_trainer_devices=)` — dead-arg（risk=low · RELOCATED）
- 位置（当前）：`vrl/ray/resources.py:441`（形参），`:445`（body 使用）
- 复核 2026-07-24：`grep 'actual_trainer_devices'` 仅命中 `:441`(def) + `:445`(`tuple(actual_trainer_devices or resolved.trainer_devices)`)。所有 caller 单参（`train_dpo.py:141`、`online.py:749`、`resources.py:498/504`、多处测试）。**较原引用 433-443 下移 +8 行**。**仍为 form-5 死形参**。
- 判死证据（原始审计，行号见「位置」行）：
  ```
  $ grep -rn 'actual_trainer_devices' vrl/ tests/ --include='*.py'
  vrl/ray/resources.py:  actual_trainer_devices: ... | None = None,  ← 定义
  vrl/ray/resources.py:  devices = tuple(actual_trainer_devices or resolved.trainer_devices)  ← 唯一使用
  ```
  form-5：全仓无 producer，body 退化为 `tuple(None or resolved.trainer_devices)`，or-fallback 死。rank-local-device 需求由 `reward_torch_device(trainer_device=)` 服务（活 producer）。
- 动作：删除 `actual_trainer_devices` 形参与 `or` fallback，直接读 `resolved.trainer_devices`。无 caller/test 更新。

### 1.12 `format_distributed_resource_plan(actual_placement=)` — dead-arg（risk=low · RELOCATED）
- 位置（当前）：`vrl/ray/resources.py:510`（kw-only 形参），`:537-538`（append 分支）
- 复核 2026-07-24：`grep 'actual_placement'` 仅命中 `:510`(def) + `:537-538`(`if actual_placement is not None: parts.append`)。所有 caller 单参（`train_dpo.py:140`、`online.py:748`、三处测试）。**较原引用 502-534 移动**。**仍为不可达分支**。
- 判死证据（原始审计，行号见「位置」行）：
  ```
  $ grep -rn 'actual_placement' vrl/ tests/ --include='*.py'
  vrl/ray/resources.py:  actual_placement: Any | None = None,
  vrl/ray/resources.py:  if actual_placement is not None: parts.append(f"actual={actual_placement}")
  （定义外零命中）
  ```
  form-5/form-2：所有 caller 单参，该 arg 仅守末尾一个 append 分支——不可达。git 历史印证：引入于 `eaf62052`，其 caller 于 `ca3ae295` 移除；actual-vs-expected 上报现居 `vrl/ray/placement.py`。无 test 断言 `actual=` 子串。
- 动作：删除 `actual_placement` 形参及其尾部 append 分支。无 caller/test 更新。

### 1.14 `MultiSegmentTokenLogProbEvaluator._trajectory_segment_payload` 多余键 — dead-field（form-2，risk=low · STILL_VALID）
- 位置（当前）：`vrl/rollouts/evaluators/token/multi_segment_token_logprob.py:159`（def）；键 `visual` :165、`cfg` :166、`train` :167、`modality` :168、`prompt_embeds`/`attention_mask` :171-172
- 复核 2026-07-24：payload builder 仍构造 `visual/cfg/train/modality/prompt_embeds/attention_mask/prompt_attention_mask/prompt_input_ids`。payload 严格局部——`evaluate` 只把 `batch+request` 传过 `model.replay_forward`；janus 自建自读另一份。in-file 读者仍只读 `name/token_ids/token_log_probs/token_mask`。**较原引用 161-182 上移 -2 行**。**仍为死键**。
- 判死证据（原始审计，行号见「位置」行）：
  ```
  $ grep -rn '"visual"\|"cfg"\|"modality"\|"prompt_embeds"' vrl/rollouts/ --include='*.py'
  multi_segment_token_logprob.py  ← 仅本构造处
  ```
  payload 严格局部于 evaluator：janus 模型自建自读另一份 payload（`vrl/models/families/janus_pro/model.py _r1_segment_payload_from_trajectory`）。in-file 读者仅 `_segment_tensor`、`_extract_logprobs`、`_enabled_segment_names`（后者本身不可达，见 §1.16）。测试 `test_multisegment_token_logprob.py` 的 `segment['modality']` 读的是 fake model 自己的 `_segment_payload`。
- 动作：把 payload 缩减为真正被读的键 `name`/`token_ids`/`token_log_probs`/`token_mask`；删除 `visual`/`cfg`/`train`/`modality`/`prompt_embeds`/`attention_mask`/`prompt_attention_mask`/`prompt_input_ids` 的构造。无 test cleanup。
- 注意：与 §1.15、§1.16 同文件，可合并一次编辑；§1.16 另需删 payload 的 `train` 键并改测试——见 §1.16。

### 1.15 `_segment_tensor` 的 `"replay"` 子字典回退 — dead-branch（form-2，risk=low · STILL_VALID）
- 位置（当前）：`vrl/rollouts/evaluators/token/multi_segment_token_logprob.py:199`
- 复核 2026-07-24：`:199` 仍为 `replay = segment.get("replay")`；`_trajectory_segment_payload`(:159) 写的固定键集从不含 `"replay"` → 分支不可达。较原引用 199-204 一致。**仍为 form-2 死分支**。
- 判死证据（原始审计）：
  ```
  $ grep -rn '"replay"' vrl/ tests/ --include='*.py'
  multi_segment_token_logprob.py:199:  replay = segment.get("replay")   ← 回退自身
  tests/generation/execution/test_chunks_pipelined.py  ← 无关张量名
  tests/models/steps/denoise/common/test_vae_decode_memory.py  ← 无关名字过滤
  ```
  该 payload 唯一构造点 `_trajectory_segment_payload` 从不写 `"replay"` 键 → 分支不可达。删后 `None` 落到既有 `RuntimeError`，对所有可达输入语义不变。
- 动作：删除 `replay = segment.get("replay")` 回退（`if value is None` 块）。无 test cleanup。

### 1.16 `_enabled_segment_names` 的 None/Mapping 分支 + `enabled_segments` 参数类型 — dead-branch（form-2，risk=low · STILL_VALID）
- 位置（当前）：`vrl/rollouts/evaluators/token/multi_segment_token_logprob.py:32`（`enabled_segments` 参数类型），`:118-131`（方法；None 分支 :119，Mapping 分支 :125）
- 复核 2026-07-24：`enabled_segments` 仍 typed `Iterable[str] | Mapping[str, bool] | None = None`(:32)；None/Mapping 分支仍在（:119/:125）。唯一生产构造 `vrl/scripts/common/factory.py:317-320` 传 `enabled_segments=tuple(...)`（恒 tuple）。None/Mapping 分支仍无 producer。**较原引用 27-34/121-134 微移**。**仍为死分支**。
- 判死证据（原始审计，行号见「位置」行）：
  ```
  $ grep -rn 'MultiSegmentTokenLogProbEvaluator(' vrl/ tests/ --include='*.py'
  vrl/scripts/common/factory.py:  MultiSegmentTokenLogProbEvaluator(enabled_segments=enabled_segments)  ← 唯一生产构造
  tests/rollouts/replay/test_multisegment_token_logprob.py  ← 均传 tuple
  ```
  `factory.py` 即使 `train_segments` 为 None，`dict(... or {})` 后仍产出**空 tuple 而非 None**；三个测试构造点全传 tuple。None 分支读的 `"enabled"` 键在固定键集中不存在（零 producer）。`vrl/trajectory/builders.py` 的 `"enabled"`/`"train"` 处理是 rollout 侧另一份 payload，无关。
- 动作：`enabled_segments` 收紧为 `Iterable[str]`（必填，去 None 默认与 Mapping 分支及 `Mapping` import 若无他用）；`_enabled_segment_names` 收缩为一行 tuple 过滤并入 `evaluate`；**同时删除** `_trajectory_segment_payload` 中的 `"train"` 键（其唯一读者是被删的 None 分支）；删除 `tests/rollouts/replay/test_multisegment_token_logprob.py:179` 的 `metadata["train"] = True` 无效 setup。**不改动** `vrl/trajectory/builders.py`。
- 注意：与 §1.14 存在同 payload 键重叠（`train`）——若三条合并编辑，一次性删除即可，须保证不重复劳动。

### 1.17 三个 token evaluator 的 `mask_key` 构造参数 — dead-arg（form-2 假旋钮，risk=low · STILL_VALID）
- 位置（当前）：`vrl/rollouts/evaluators/token/token_logprob.py:41`、`continuous_token_logprob.py:33`、`multi_segment_token_logprob.py:33`
- 复核 2026-07-24：三 evaluator `__init__` 仍取 `mask_key="token_mask"`。`grep 'mask_key='` 仅命中三 evaluator 内部自转发 + `trajectory.py:60/104` 与 `denoise/sde_logprob.py:185`(`mask_key="mask"`)——后者是 finding 明确保留的 `TrajectorySignalBuilder` 调用。无 caller 传非默认 `mask_key` 给三 evaluator。**较原引用 +2 行**。**仍为假旋钮**。
- 判死证据（原始审计，行号见「位置」行）：
  ```
  $ grep -rn 'mask_key=' vrl/ tests/ --include='*.py'
  三 evaluator 内部 self.mask_key 转发给 builder
  vrl/rollouts/evaluators/trajectory.py  ← builder 形参
  vrl/rollouts/evaluators/denoise/sde_logprob.py:  mask_key="mask"  ← 唯一非默认传参（builder 调用，保留）
  ```
  三个 evaluator 全部构造点（`factory.py` + tests）均用默认 `"token_mask"`，无 `**kwargs`/`asdict` 转发、无 YAML、无 registry 字符串分发。`Evaluator` Protocol 仅约束 `evaluate()`。[[SPRINT_design_smell_audit]] 已删同类假旋钮 `TokenGRPOConfig.mask_key`。
- 动作：删除三个 token evaluator 的 `mask_key` 构造参数，内联 `"token_mask"`；**保留** `TrajectorySignalBuilder.single_segment`/`segment_signal` 的 `mask_key` 形参（`sde_logprob.py` 以 `mask_key="mask"` 实际使用）。无 test cleanup。

### 1.18 `enable_transformer_gradient_checkpointing(require_method=...)` — dead-arg（form-1+2，risk=low · STILL_VALID · 支撑事实已变 → 见 §3）
- 位置（当前）：`vrl/trainers/activation_checkpointing.py:128`（形参，默认 True），`:150`（`if require_method`）
- 复核 2026-07-24：`grep 'require_method'` 仅命中 `:128`(def) + `:150`(read)，零 setter。**caller 已从「单一 online.py」变为两个：`train_dpo.py:173`(`bundle, cfg`) 与 `online.py:836`(`bundle, built.root`)**，两者均不传 `require_method`，False 分支仍不可达。**finding 仍成立**（动作说明里「单一 caller」措辞已过时，见 §3）。
- 判死证据（原始审计，caller 已更新，见「复核」行）：
  ```
  $ grep -rn 'require_method' vrl/ tests/ --include='*.py'
  vrl/trainers/activation_checkpointing.py:128:  require_method: bool = True,   ← 定义
  vrl/trainers/activation_checkpointing.py:150:  if require_method:             ← 唯一内部读
  （零 setter）
  ```
  form-1 死形参 + form-2 死 False 分支。历史 `require_method=False` 的 producer（cosmos `_after_bundle_built`，见 [[SPRINT_grab_bag_file_audit]]）已删。`require_method` 恒 True 时 `continue`（静默跳过）分支不可达。
- 动作：删除 `require_method` 形参，把 `if require_method: raise / continue` 收为「trainable module 缺 `enable_gradient_checkpointing` 时无条件 raise」。无 test 更新（`test_online_lifecycle.py` monkeypatch 是 `lambda *args, **kwargs`，另有直接测试 `test_activation_checkpointing.py:35/46/59/73/96` 均不传 `require_method`）。

### 1.20 `validate_artifact_manifest(allow_absolute_paths, require_readable, reject_metadata_domain)` — dead-arg（form-1，risk=low · STILL_VALID）
- 位置（当前）：`vrl/trainers/data/artifacts.py:167-169`（三参），使用 `:180/:201/:208`
- 复核 2026-07-24：三 kw-only 参全在（:167 `allow_absolute_paths=False`、:168 `require_readable=True`、:169 `reject_metadata_domain=True`）。`require_readable`/`reject_metadata_domain` 出 `artifacts.py` 零命中；`allow_absolute_paths` 出文件仅命中 `target_dino_similarity.py`（另一函数 `resolve_artifact_path`）。caller 是 `__init__` re-export + 测试，均不传三参。**较原引用 160-229 一致**。**仍全部不可达**。
- 判死证据（原始审计）：
  ```
  $ grep -rn 'reject_metadata_domain\|require_readable\|allow_absolute_paths' vrl/ tests/ --include='*.py'
  vrl/trainers/data/artifacts.py:167/168/169/180/201/208  ← 三参定义+内部读
  vrl/rewards/models/target_dino_similarity.py:60/150     ← 无关：喂给 resolve_artifact_path 的 reward 旋钮，另一个函数
  ```
  三个 keyword-only 参非默认路径不可达。唯一 `**kwargs` 路由 `validate_artifact_manifest_pair` 及其固定签名 wrapper 只转发固定字段集；直接 caller 与 test caller 只传 `data_root`/`required_artifact_fields`。内联默认行为等价，且测试正打这些默认行为（如 `test_production_metadata_domain_is_rejected`）。
- 动作：删除三参并内联默认（`allow_absolute=False`、恒 `_assert_readable`、恒 reject `metadata.domain`）。无 test 更新。

### 1.21 `PhaseTimer.__init__(sync)` — dead-arg（risk=low · STILL_VALID）
- 位置（当前）：`vrl/trainers/online/trainer.py:151`（签名 `sync=True`），`:153`（`self.sync`）
- 复核 2026-07-24：签名仍 `def __init__(self, enabled: bool = False, sync: bool = True)`(:151)。两处构造均省略 `sync`（`trainer.py:1077` `PhaseTimer(enabled=cfg.profile)`、`test_skip_backward_agreement_distributed.py:392` `PhaseTimer(enabled=False)`）。`sync=` 仅现于 class docstring(:146)。较原引用 150-153 一致。**仍为死形参**（docstring 仍记录幽灵参数）。
- 判死证据（原始审计）：
  ```
  $ grep -rn 'PhaseTimer(' vrl/ tests/ --include='*.py'
  vrl/trainers/online/trainer.py:1077:  PhaseTimer(enabled=cfg.profile)
  tests/trainers/online/test_skip_backward_agreement_distributed.py:392:  PhaseTimer(enabled=False)
  ```
  唯二构造均不传 `sync`；无子类、无 `**kwargs` 构造、无 `.sync` 外部改写、无 YAML 键。`sync=False` 路径不可达，True 默认退化为 `torch.cuda.is_available()`。
- 动作：签名改 `def __init__(self, enabled: bool = False) -> None:`，赋值改 `self.sync = torch.cuda.is_available()`；**同时**改写 class docstring 使其不再记录 `sync=True` 参数。无 test 更新。

### 1.22 `FSDPStrategy.validate_training_state_parking` / `restore_training_state` — duplicate-impl（form-4，risk=low · RELOCATED）
- 位置（当前）：`vrl/trainers/strategy.py:866-867`（`validate_training_state_parking`），`:900-901`（`restore_training_state`）
- 复核 2026-07-24：两 override 均在且仍为 form-4 重复。`validate_training_state_parking`(:866-867) 是 `return None`，与继承的 `_TrainingStateParking.validate_training_state_parking`(:195-196) 同；`restore_training_state`(:900-901) 是纯委派 `_TrainingStateParking.restore_training_state(self, state)`（MRO 会解析到同一 mixin 方法）。`park_training_state` 是真 override，**保留**。**较原引用 762-763/798-799 下移约 +100 行**。
- 判死证据（原始审计，行号见「位置」行）：
  ```
  $ grep -n 'def validate_training_state_parking\|def restore_training_state\|^class ' vrl/trainers/strategy.py
  _TrainingStateParking   (validate = `return None`, restore = 委派目标)
  SingleProcessStrategy(_TrainingStateParking, Strategy)   ← 两方法都不定义，靠 mixin
  FSDPStrategy(_TrainingStateParking, Strategy)   (validate, park, restore)
  ```
  body 比对：`FSDPStrategy.validate_training_state_parking` 与继承版逐字节同；`FSDPStrategy.restore_training_state` body 是单行委派——正是 MRO 无 override 时的解析结果。`Strategy` 是 Protocol，mixin 赢 MRO。对照 `park_training_state`（真 override，跨 rank failure agreement），**保留**。删后 sibling `SingleProcessStrategy` 模式一致。
- 动作：删除两个 override。`park_training_state` 内 `self.restore_training_state(state)` 删后解析到相同 mixin 实现，行为不变。无 test 更新（`test_fsdp.py` 断言返回 None 经继承仍通过）。

### 1.23 `release_cuda_memory(gc_collect=)` — dead-arg（form-5，risk=low · STILL_VALID）
- 位置（当前）：`vrl/utils/cuda_memory.py:197`（形参 `gc_collect=False`），`:202`（`if gc_collect: gc.collect()`）
- 复核 2026-07-24：形参与读取均在。4 个生产调用点全传 `gc_collect=True`（`worker.py:578`、`memory_parking.py:185/429/439`），默认路径无 producer。`git show 88ed756e` 确认默认在审计基线**已是 False**，无变化。测试断言 `test_execute_request_pipelined.py:138` 仍在（finding 已含更新它）。较原引用 195-208 一致。**仍为 form-5 死形参**。
- 判死证据（原始审计）：
  ```
  $ grep -rn 'release_cuda_memory(' vrl/ tests/ --include='*.py' | grep -v for_parking
  memory_parking.py:185/429/439:  release_cuda_memory(gc_collect=True, ipc_collect=True)
  worker.py:578:                  release_cuda_memory(gc_collect=True)
  vrl/utils/cuda_memory.py:197:   def release_cuda_memory(  ← 定义（默认 gc_collect=False）
  ```
  4 个生产调用点全传 `gc_collect=True`，默认（False）路径零 producer。`gc_collect` 只守 `if gc_collect: gc.collect()`；`ipc_collect` 真正变化（worker 省略）应**保留**。`release_cuda_memory_for_parking` 是另一实现，不经此函数。
- 动作：删除 `gc_collect` 形参，body 顶部无条件 `gc.collect()`（保持在 torch import 前的现有次序）；**保留** `ipc_collect`。更新四调用点：`memory_parking.py:185/429/439 → release_cuda_memory(ipc_collect=True)`；`worker.py:578 → release_cuda_memory()`。**并**更新 `tests/generation/execution/test_execute_request_pipelined.py:138` 断言 `cleanup_calls == [{"gc_collect": True}]` 改为 `cleanup_calls == [{}]`。
- 注意：low。主体编辑在 `vrl/utils/cuda_memory.py`；触及的 `memory_parking.py`、`worker.py` **不在**在飞 sprint 的未提交集。DO-NOT-FLAG 对 `cuda_memory.py` 的豁免仅限 `process_gpu_used_bytes`（NVML），`gc_collect` 不在豁免内。

## 2. 已由 origin 落地（本次复核确认，无需再做）

自审计基线 `88ed756e` 以来，以下三条已在 `main @ 7c748532` 落地。复核确认符号已从生产消失，测试改为反向校验其缺失。**无动作**。

- **§1.1 `RolloutBatch.dones`** — RL episode-termination 残留死字段（生产写入恒 `torch.ones`，零读者）。已由 `5b2236f1 refactor(rollouts): drop post-reward batch transport` 落地；字段、`batch_builder` 三处构造、`ops.py` 两处拷贝、`estimate_batch_bytes` 候选全部移除。测试现以 `assert not hasattr(batch, "dones")`（`test_runtime.py:302`）与 `test_removed_fields_are_not_constructor_inputs`（`test_batch.py:25`）反向校验。
- **§1.13 `RewardScoringInput.expected_count`** — 与 `__post_init__` 的 `len(sample_rows)==batch_size` 校验冗余的死字段。已由 `7cfe90ef refactor(rewards): derive scoring batch facts` 落地；字段与 `batch_builder.py` 传参均移除。测试 `test_runtime.py:648` 现断言 `fields(request).isdisjoint({'prompts','expected_count','batch_size'})`。
- **§1.19 `TrainState.total_reward` / `TrainState.total_loss`** — write-only checkpointed 累加，无任何 metric/log/test 读者。已由 `a9bc6072 refactor(training): remove unused cumulative totals` 落地；`types.py` 字段、`trainer.py` 累加、`_state_dict` 写、load 恢复全部移除。（`grpo/multisegment.py` 的 `total_loss` 局部变量、`rewards` 的 `total_reward_latency_ms` 是无关同名符号。）

## 3. 情况已变（需重新评估）

本次复核**未产生 verdict 级 CHANGED/INDETERMINATE**：20 条仍需处理的 finding 判定不变。但其中 **2 条的支撑事实已随 origin 迁移**，finding 结论仍成立（故仍列于 §1），实施前须以新事实为准，勿照抄旧证据里的路径/计数：

- **§1.5 `timestep_subset`**：原证据将唯一生产构造点记为 `vrl/scripts/families/wan_2_1/train_dpo.py:85-103`。该构造已**迁至 `vrl/config/builders.py:385`**（显式 kwargs，同样不含 `timestep_subset`），`train_dpo.py` 不再构造 `OfflineDPOTrainerConfig`。删除前须核对的「无 splat」目标是 `builders.py:385`，而非旧的 `train_dpo.py`。字段仍 TEST-ONLY dead，动作不变。
- **§1.18 `require_method`**：原证据记「唯一生产 caller `online.py:909`，默认 True」。现为**两个 caller**——`train_dpo.py:173`(`bundle, cfg`) 与 `online.py:836`(`bundle, built.root`)，两者均不传 `require_method`，False 分支仍不可达，finding 不变；但动作/证据里「单一 production caller」的措辞已过时，删除后须同时确认这两个 caller 都无需改。

## 4. 验证协议
- **每条删除后**：`ruff check <touched files>` + `ruff format --check <touched files>`（仅跑本条触及的 Python 文件，先 `ruff check --fix` 再 `ruff format`，最后 `--check` 复核）。
- **全簇完成后**：`pytest tests/ray/ tests/rollouts/ tests/trainers/ tests/data/ tests/quality/ tests/generation/execution/ tests/config/ tests/scripts/` + `pytest -m "not e2e and not slow_test"` 子集不新增失败。
- **基线（清理前）**：现针对 **main @ `7c748532`** 重跑；`vrl.config.lint` 与 `ruff check .` 须全绿，fast subset 不新增失败（预存在失败与本清理无关）。删除后这三项须保持。
- **逐条触及的测试文件**（从 action 提取，无列出者表示无 test 改动）：
  - §1.2 `collect_unscored/build seed`：`tests/rollouts/collector/_helpers.py`、`tests/rollouts/collector/test_runtime.py:292/299`、`tests/rollouts/runtime/test_engine_requests.py:32/44`、`tests/quality/preview.py`、`tests/quality/test_preview.py:48/57`。
  - §1.3 `collect_prompt_batches`：`tests/e2e/test_real_checkpoint_rl.py:626`、`tests/trainers/online/{test_precision_drift_guard,test_state_restore,test_step_split,test_reward_update_flow,test_trainable_state,test_trajectory_granularity,test_advantage_and_metrics}.py`、`tests/rollouts/orchestration/{test_prompt_collection,test_orchestration}.py`、`tests/rollouts/orchestration/continuous/{test_schedule,test_owner,test_contracts,test_queue,test_scheduler}.py`、`tests/rollouts/collector/test_runtime.py:785`、`tests/rollouts/collector/_collect.py`、`tests/trainers/online/test_reward_update_flow.py:1275/1338`（`remap_group_ids_` TEST-ONLY 用例删除）。
  - §1.4 `gather_full_state_dict`：`tests/trainers/test_fsdp.py`、`tests/trainers/test_fsdp_gather_distributed.py`——**迁 helper 后 repoint**，不删 oracle 测试。
  - §1.5 `timestep_subset`：`tests/trainers/test_offline_dpo_timesteps.py:60/232/344`。
  - §1.7 `has_non_driver_gpus`：`tests/ray/test_dependencies.py:25/33/42`（删三处 property 断言）。
  - §1.16 `enabled_segments`：`tests/rollouts/replay/test_multisegment_token_logprob.py:179`（删 `metadata["train"] = True`）。
  - §1.23 `gc_collect`：`tests/generation/execution/test_execute_request_pipelined.py:138`。
  - 其余（§1.6/§1.8/§1.9/§1.10/§1.11/§1.12/§1.14/§1.15/§1.17/§1.18/§1.20/§1.21/§1.22）：**无 test 改动**，删后跑对应目录测试确认零回归即可。

## 5. Non-Goals
- 不删被「能 raise 的校验」、控制流分支、runtime/config/Ray 调用消费的字段（AGENTS 死字段规则）——如 §1.5 保留的 `scheduler.timesteps`-empty fail-fast。
- 不动 DO-NOT-FLAG 豁免项：`_GENERATION_CUDA_RUNTIME_RESIDUAL_BYTES`（`generation/execution/worker.py`）、`_REWARD_CUDA_RUNTIME_RESIDUAL_BYTES`（`rewards/base.py`）、`ensure_loaded`（`rewards/runtime.py`）、`process_gpu_used_bytes`（`utils/cuda_memory.py` NVML）、`prepare_latents` 修复（sana/hunyuan）。§1.23 的 `gc_collect` 在同文件 `cuda_memory.py` 但**不属**该豁免。
- 不为省行数扁平化 protocol/lazy-import/跨家族一致性 thin function——如 §1.6 保留 `startup_method`（两 producer）、§1.17 保留 `TrajectorySignalBuilder` 的 `mask_key`（`sde_logprob` 实用）、§1.22 保留 `park_training_state`（真 override）、§1.23 保留 `ipc_collect`（真变化）。
- **cluster-specific non-goals**：
  - **不在在飞 sprint [[SPRINT_native_generation_engine_program]] 未合并前编辑其 worktree 文件**（§1.6 `actor_group.py`、§1.8/§1.9 `placement.py`）——三条已复核无符号级冲突，但须 **sequence after the sprint** 或与作者协调后再落地。
  - §1.4 `gather_full_state_dict` **不删除、不移植到 `gather_trainable_state_dict`**——迁入测试基础设施，保留其对 live 生产路径（`load_full_state_dict`、frozen-param 不变性）的 oracle 覆盖。
  - §1.14/§1.15/§1.16 同文件三条须协调编辑，避免 payload `train`/`visual`/`cfg`/`modality` 键的重复删除。

## References
- `vrl/rollouts/batch/ops.py:72`（`remap_group_ids_`，export :119）、`vrl/rollouts/orchestration/continuous/types.py`
- `vrl/rollouts/collector/core.py:149,167`、`vrl/rollouts/collector/requests.py:53,61-62`
- `vrl/rollouts/orchestration/prompt_collection.py:73,157-167,173-189,249`、`vrl/rollouts/orchestration/strict_on_policy.py:89`、`vrl/rollouts/orchestration/continuous/producer.py:342`
- `vrl/rollouts/evaluators/token/multi_segment_token_logprob.py:32,118-131,159,165-172,199`、`vrl/rollouts/evaluators/token/token_logprob.py:41`、`vrl/rollouts/evaluators/token/continuous_token_logprob.py:33`、`vrl/rollouts/evaluators/trajectory.py:60,104`、`vrl/rollouts/evaluators/denoise/sde_logprob.py:185`
- `vrl/ray/actor_group.py:42,80`、`vrl/ray/dependencies.py:44-47,54`、`vrl/ray/placement.py:226,227,264,267,406-407`、`vrl/ray/resources.py:441,445,510,537-538,1174,1200,1256,1267,1289,1307,1367`
- `vrl/trainers/activation_checkpointing.py:128,150`、`vrl/trainers/data/artifacts.py:167-169,180,201,208`、`vrl/trainers/fsdp.py:227,563`、`vrl/trainers/offline/dpo.py:60,204,218`、`vrl/trainers/online/trainer.py:151,153,1077`、`vrl/trainers/strategy.py:866-867,900-901`
- `vrl/utils/cuda_memory.py:197,202`、`vrl/utils/config.py:82`
- 迁移后的生产构造/调用点：`vrl/config/builders.py:385`（§1.5）、`vrl/scripts/families/wan_2_1/train_dpo.py:173`+`vrl/scripts/common/online.py:836`（§1.18）、`vrl/scripts/common/online.py:749,748`+`train_dpo.py:141,140`（§1.11/§1.12）、`vrl/scripts/common/factory.py:317-320`（§1.16）
- 已落地 commit：`5b2236f1`（§1.1）、`7cfe90ef`（§1.13）、`a9bc6072`（§1.19）
- 关联 sprint：[[SPRINT_deadcode_00_overview]]、[[SPRINT_trajectory_views_types_dead_fields_cleanup]]、[[SPRINT_fbag_00_overview]]、[[SPRINT_grab_bag_file_audit]]、[[SPRINT_design_smell_audit]]、[[SPRINT_native_generation_engine_program]]、[[SPRINT_continuous_stage_contracts_and_baseline]]、[[SPRINT_continuous_three_stage_pipeline_program]]
