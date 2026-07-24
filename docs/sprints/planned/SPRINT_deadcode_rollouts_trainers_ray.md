# SPRINT: `rollouts` / `trainers` / `ray` 死代码清理（planned）

状态：**planned（2026-07-23）**。共 **23 条**对抗验证通过的死代码（5 条 medium、18 条 low），横跨 `vrl/ray/`、`vrl/rollouts/`、`vrl/trainers/`、`vrl/utils/` 四层，均来自 dead-code-audit workflow。
来源：dead-code-audit workflow（五种死代码形态 + 对抗验证 + 删除类二次字符串引用检查）。
关联：[[SPRINT_deadcode_00_overview]]；与 [[SPRINT_trajectory_views_types_dead_fields_cleanup]]（死字段规则同源）、[[SPRINT_fbag_00_overview]]（`gather_full_state_dict` 的 cohesive-keep 判定，见 §1.4 复核）、[[SPRINT_grab_bag_file_audit]]（`require_method=False` 历史 producer，见 §1.18）、[[SPRINT_design_smell_audit]]（`mask_key` 假旋钮先例，见 §1.17）、[[SPRINT_native_generation_engine_program]]（在飞 sprint，`vrl/ray/` 文件重叠，见 §1.6/§1.8/§1.9）互有承接。

## 0. 一句话

这是一个混合层死代码簇：主体是**从不被非默认调用的死形参 / 死旋钮**（dead-arg / dead-config-knob 共 13 条），其余是死字段、死分支、form-4 重复实现与一个 test-only 函数。最锋利的一条是 `collect_prompt_batches` 的裸字符串攒批路径（§1.3）——生产端 prompt 恒为 `PromptExample`，字符串路径连同其唯一下游 `remap_group_ids_` 在生产完全不可达，但**误删风险最高**：大量 trainer/orchestration 测试用裸 `str` 驱动真实 schedule 走该路径，删除前必须逐一改写为 `PromptExample` 并注意「多 prompt 合并成一次 vs 每 prompt 一次 `collect_unscored`」的调用粒度变化。次高风险是 `gather_full_state_dict`（§1.4）：它 zero 生产调用者（TEST-ONLY），但被 8 个测试当作**验证 live 生产路径的 oracle**，须**迁入测试基础设施而非删除**，此判定推翻 [[SPRINT_fbag_00_overview]] 的 cohesive-keep（那次判的是 grab-bag 归属，非 liveness）。

## 1. 待删清单（逐条，带证据与动作）

> 顺序：medium-risk 五条在前（§1.1–§1.5），其余 low-risk 按层分组（`ray/` §1.6–§1.12、`rollouts/` §1.13–§1.17、`trainers/` §1.18–§1.22、`utils/` §1.23）。所有 KEY grep 已于 2026-07-23 重跑复核，**零偏差**。

### 1.1 `RolloutBatch.dones` — dead-field（risk=medium）
- 位置：`vrl/rollouts/batch/core.py:20`
- 判死证据：
  ```
  $ grep -rn 'dones' vrl/ --include='*.py'
  vrl/rollouts/batch/core.py:20:    dones: Any           # [B] episode termination flags   ← 定义
  vrl/rollouts/collector/batch_builder.py:134/161/210:  dones=torch.ones(...)   ← 生产端唯一写入，恒为全 1
  vrl/rollouts/batch/ops.py:40:  dones=batch.dones[selector.to(...)]              ← select_batch 原样结构性拷贝
  vrl/rollouts/batch/ops.py:131: dones=batch.dones.to(device)                     ← move_*_to_device 原样拷贝
  vrl/rollouts/orchestration/continuous/types.py:60: batch.dones,                 ← estimate_batch_bytes 字节枚举
  $ grep -rnF '"dones"' vrl/ tests/ --include='*.py'   → 无字符串键读者
  $ grep -rn 'dones' vrl/algorithms/ vrl/trainers/ --include='*.py'   → (none)  ← 算法/训练器零读取
  ```
  生产端唯一写入恒为 `torch.ones(...)`（携带零信息量），全仓无任何 algorithm/trainer/evaluator 分支读它的值；两处 `ops.py` 是逐字段搬运（非语义消费），`types.py:60` 只做 `element_size*nelement` 字节估算（`[B]` bool 对 MB 级 latents 可忽略）。`RolloutBatch` 从不 `asdict`/`torch.save` 持久化；`map_tensor_tree` 按 `dataclasses.fields` 泛型重建，删字段安全。
- 动作：删除 `RolloutBatch.dones` 字段；同步删除 `batch_builder.py:134/161/210` 三处 `dones=torch.ones(...)` 构造、`ops.py:40/131` 两处拷贝、`continuous/types.py:60` 字节估算候选中的 `batch.dones`；更新 tests 中约 44 处 `RolloutBatch(...)` 构造点（全部 keyword 传参，含 `tests/rollouts/collector/test_runtime.py` 的设备断言，实际在 **L325**——finding 原文误记 L324，纯行号笔误，动作不受影响）与 `tests/rollouts/batch/test_batch.py`。
- 注意：medium。测试侧 45 处 `dones` 命中中，绝大多数是必填字段的 fixture 填充物——删字段即删填充（test-only-is-still-dead）。唯一需人工确认的是 `test_runtime.py` 的设备断言（断言 `dones` 在正确 device 上），删字段后该断言随之删除，不改其余断言。

### 1.2 `RolloutCollector.collect_unscored(seed=)` / `GenerationRequestBuilder.build(seed=)` — dead-arg（risk=medium）
- 位置：`vrl/rollouts/collector/core.py:169,187`、`vrl/rollouts/collector/requests.py:53,61-62`
- 判死证据：
  ```
  $ grep -rn 'collect_unscored' vrl/ --include='*.py'
  prompt_collection.py:99/105  ← **kwargs 包装，仅本文件内被下述两处调用
  prompt_collection.py:160/176 ← 生产唯一调用链，两处均显式传参且不含 seed
  core.py:162 ← 定义
  $ grep -rn 'seed' vrl/rollouts/collector/requests.py vrl/rollouts/collector/core.py
  requests.py:53/61-62  ← seed 形参 + sampling["seed"]=seed 注入
  core.py:169/187       ← seed 形参 + 转发
  ```
  唯一生产调用链 `prompt_collection.py:160/176` 均不传 `seed`；`build` 的唯一生产调用是 `core.py:182`。仓库自身 preset 明确写明确定性种子的钦定路径是 `request_overrides.seed`——`online_grpo_droid_overfit_validation.yaml:25-27`：「the sanctioned per-prompt seed path … a top-level `sampling.seed` key does not exist」，与 `sampling["seed"]=seed` 完全同构（且 `requests.py:57-63` 顺序为 `sampling["seed"]=seed` 先、`sampling.update(request_overrides)` 后 → `request_overrides` 优先）。
- 动作：删除两处 `seed` 形参及 `requests.py:61-62` 的 `sampling["seed"]=seed` 注入（**保留** `sampling` dict 的 `"seed"` 键本身——它经 `request_overrides`/config `sampling` 段流入并被 executor 消费）。测试与 harness 更新：(1) `tests/rollouts/collector/_helpers.py` `collect_scored` 删 `seed` 形参与转发（L19、L35）；(2) `tests/rollouts/collector/test_runtime.py:292` 的 `collect_scored(seed=5)` 改为 `request_overrides={"seed": 5}`（断言 L299 不变）；(3) `tests/rollouts/runtime/test_engine_requests.py:32` 的 `seed=7` 并入该调用的 `request_overrides`（断言 L44 不变）；(4) `tests/quality/preview.py` `build_preview_request`（L16-40）删 `seed` 形参传递，改为 `overrides.setdefault("seed", seed)` 以**保持现行优先级**（example 自带 `request_overrides["seed"]` 时覆盖预览 seed），`tests/quality/test_preview.py:48/57` 同步。
- 注意：medium。(a) `tests/quality/preview.py` 是 GPU 质量预览 harness（长期资产），功能性依赖 `seed=` 做确定性预览（`_PREVIEW_BASE_SEED+index`），**必须迁移而非删除**，且迁移须保精度序（`setdefault` 而非直接赋值）。(b) **不动** `test_runtime.py:610/975` 与 `test_janus_pro_r1_wiring.py:30/40` 的 `seed=None`——那是 `GenerationSampleRow.seed`，另一个活符号。

### 1.3 `collect_prompt_batches` 裸字符串攒批路径 + `remap_group_ids_` — dead-branch + 关联 dead-function（risk=medium）
- 位置：`vrl/rollouts/orchestration/prompt_collection.py:71,157-168,171-188,246-251`；`vrl/rollouts/batch/ops.py:92-104`
- 判死证据：
  ```
  $ grep -rn 'collect_prompt_batches' vrl/ --include='*.py'
  strict_on_policy.py:89、continuous/producer.py:342  ← 生产唯二调用，prompts 均来自 trainer
  $ grep -rn 'remap_group_ids_' vrl/ tests/ --include='*.py'
  ops.py:92 定义；prompt_collection.py:249 唯一生产调用（仅 isinstance(remap, list) 分支可达）
  tests/trainers/online/test_reward_update_flow.py:1275/1338  ← TEST-ONLY
  ```
  生产 prompts 溯源：`trainer.py:1082 rollout_schedule.next_iteration ← online.py:875 load_prompt_examples_from_config(cfg.data)`；`vrl/trainers/data/prompts.py` 三个 loader 恒构造 `PromptExample`（L92/179/245），`PromptExample.generation_input`（L40）→ 恒命中 `hasattr` 分支、`remap` 恒为 `int`。故裸字符串攒批路径（`pending_prompts`/`pending_indices`/`flush_pending_prompts` + `isinstance(remap, list)` 分支）及其唯一下游 `remap_group_ids_` 在生产不可达，仅 tests 以 `["p0"]` 形式构造。
- 动作：删除 `pending_prompts`/`pending_indices`/`flush_pending_prompts`、`isinstance(remap, list)` 分支、`remap_group_ids_` 及其 `ops.py:147` export 与 `test_reward_update_flow.py:1338` 的 TEST-ONLY 用例；**保留** `split_batch_by_group`、`requests.py` `_resolve_input` 的 `str→GenerationInput` 兼容。删后收紧 `collect_prompt_batches` 的 remap 注释（`prompt_collection.py:69-70`）与 `unscored_groups` 类型（`list[int] | int → int`）。测试重写（范围远大于三文件）：
  - (a) `tests/e2e/test_real_checkpoint_rl.py:626` 的 `trainer.step([case.prompt])` 改为 `trainer.step([PromptExample(prompt=case.prompt)])`——它经真实 `OnlineTrainer→build_rollout_schedule→collect_prompt_batches` 走字符串路径（其 fake collector `:502` 已用 `**kwargs` 接参，可原样吞下 `metadata`/`request_overrides`）；
  - (b) 重写全部以裸 `str` 驱动真实 schedule 的 trainer 测试：`tests/trainers/online/test_precision_drift_guard.py:376`、`test_state_restore.py:198`、`test_step_split.py`（126/139/164/191-192/262）、`test_reward_update_flow.py:236/927`、`test_trainable_state.py:116`、`test_trajectory_granularity.py:124`、`test_advantage_and_metrics.py:168-169`（用 `grep -rn 'trainer\.step(\["' tests/` 定界）；
  - (c) orchestration/continuous 测试：`test_prompt_collection.py`、`test_orchestration.py`、`continuous/test_schedule.py`、`test_owner.py`、`test_contracts.py`、`test_queue.py`、`test_scheduler.py`，及 `tests/rollouts/collector/test_runtime.py:785` 与 `tests/rollouts/collector/_collect.py`。
- 注意：medium，本簇最高误删风险。重写时**注意调用粒度变化**：裸 `str` 曾合并为一次 `collect_unscored` 调用，`PromptExample` 为每 prompt 一次调用——涉及调用次数/合批断言的测试须按新粒度改写而非机械替换。planned sprint（`SPRINT_continuous_stage_contracts_and_baseline`、`SPRINT_continuous_three_stage_pipeline_program`）仅把 `collect_prompt_batches` 当复合 task 引用，**不依赖**字符串路径，故不冲突。`train_dpo.py:277` 是 `OfflineDPOTrainer`，与此无关。

### 1.4 `gather_full_state_dict` — dead-function（risk=medium）
- 位置：`vrl/trainers/fsdp.py:287-326`
- 判死证据：
  ```
  $ grep -rn 'gather_full_state_dict' vrl/ tests/ --include='*.py'
  vrl/scripts/common/online.py:186     ← 注释提及（doc comment）
  vrl/trainers/fsdp.py:287 定义 / :518 docstring 交叉引用
  vrl/trainers/strategy.py:908         ← 注释「Do NOT route this through the FSDP gather_full_state_dict」
  tests/trainers/test_fsdp.py:27,277,287,298,302,435,449,456          ← import + 6 调用点
  tests/trainers/test_fsdp_gather_distributed.py:32,101,134           ← import + 2 调用点
  ```
  **零生产调用者**（三处均为注释/docstring），共 **8 个测试调用点**（finding 更正 auditor 原估的 5 个）。生产 export 路径已替换：`FSDPStrategy` 经 `gather_trainable_state_dict`（`strategy.py:659-699`），`DDPStrategy` 经 unwrapped `state_dict()`，加载经 `load_trainable_state_dict`/`load_full_state_dict`。
- 动作：将 `gather_full_state_dict`（body 完整保留，含防御性 DTensor `full_tensor()` 物化）**迁入**共享测试 helper（如 `tests/trainers/conftest.py` 或 `tests/trainers/_state_dict_helpers.py`），并 repoint 全部 8 个测试调用点。**不删/不 repoint** 那 4 个把它当 oracle 的测试——它们覆盖 live 生产路径（`load_full_state_dict` at `strategy.py:950-966`、`FSDPStrategy.load/export_trainable_state`、legacy full-checkpoint 兼容、LoRA load 时 frozen-param 不变性）。**不要**把 `test_fsdp_gather_distributed.py:101,134` 移植到 `gather_trainable_state_dict`——它们计算 `{name: value … if name not in trainable}` 即 gather **frozen** base params，而 `gather_trainable_state_dict`（`fsdp.py:329-349`）按 `requires_grad` 选键，语义上无法产出 frozen params。同步把 `cpu_offload=False` rank0-only-trap 理由从被删 docstring 迁入 `gather_trainable_state_dict`/`gather_full_optimizer_state_dict` docstring（`fsdp.py:518` 的按名交叉引用须更新），并改写 `online.py:186`、`strategy.py:908` 两处注释使其不 dangle（直接描述 DCP `get_model_state_dict(full_state_dict=True)` 行为，不再点名已移出的函数）。
- 注意：medium，且**推翻 [[SPRINT_fbag_00_overview]] 的 cohesive-keep**：那次判的是 `fsdp.py` grab-bag 归属（是否该拆分文件），非本函数 liveness。本次按 form-1 复核 caller 后确认零生产调用者——是 TEST-ONLY，故迁入测试基础设施。二者不矛盾：一个谈文件组织，一个谈函数生死。

### 1.5 `OfflineDPOTrainerConfig.timestep_subset` — dead-config-knob（risk=medium）
- 位置：`vrl/trainers/offline/dpo.py:60-62,203-221`
- 判死证据：
  ```
  $ grep -rn 'timestep_subset' vrl/ tests/ --include='*.py'
  vrl/trainers/offline/dpo.py:60(定义),204,205,218(读+错误信息)
  tests/trainers/test_offline_dpo_timesteps.py:53,62,95   ← test-only setters
  ```
  唯一生产构造 `_build_offline_dpo_trainer_config`（`vrl/scripts/families/wan_2_1/train_dpo.py:85-103`）显式 kwargs 建 config、无 `**` splat、**不含** `timestep_subset`；无 `asdict`/`fields()`/`vars()`/`**kwargs` 转发；`state_dict`（`dpo.py:398-424`）只序列化 `global_step` 与 optimizer。YAML 面无法达：`ActorSection`（`vrl/config/schema.py:616-635`）无此键，手写白名单 `_OFFLINE_DPO_ACTOR_FIELDS`（`schema.py:642-653`）不含它，未知 actor 键 loud-fail（`schema.py:941`）。docs 唯一提及 `docs/sprints/done/SPRINT_dance_grpo_validation.md:25` 指的是另一个符号 `select_timestep_subset`（online trainer）。
- 动作：删除 `timestep_subset` 字段与 `_sample_timesteps` 中 `if self.config.timestep_subset is not None` 分支（**保留** `scheduler.timesteps`-empty fail-fast，这是唯一生产路径）；删除 `dpo.py:218` 错误信息里 `OfflineDPOTrainerConfig.timestep_subset=(lo, hi)` 的悬空补救文本；更新/删除 `tests/trainers/test_offline_dpo_timesteps.py` 的三个触点（L53 默认、L62 参数、L95 的 `(5,10)`-subset 测试）。
- 注意：medium（TEST-ONLY dead knob——用户可能以为这是个可设旋钮）。删前确认 `train_dpo.py` 生产构造确无 splat。

### 1.6 `RayActorGroup.launch(metadata_method=)` — dead-arg（risk=low）
- 位置：`vrl/ray/actor_group.py:43,81`
- 判死证据：
  ```
  $ grep -rn 'metadata_method' vrl/ tests/ --include='*.py'
  vrl/ray/actor_group.py:43:  metadata_method: str = "worker_metadata",   ← 定义
  vrl/ray/actor_group.py:81:  metadata_refs = [getattr(actor, metadata_method).remote() ...]  ← 唯一使用
  ```
  form-5 死形参：两个 `launch` 调用者均省略它（`vrl/generation/ray/launcher.py:123` 传 `startup_method="load_policy"`+`concurrency_groups`；`tests/ray/test_ray_actor_pool.py:46` 传 `startup_method="startup"`），`launch` 无 `**kwargs` 转发。默认 `"worker_metadata"` 是真协议名——`vrl/generation/ray/worker.py:74`、`tests/ray/test_ray_actor_pool.py:30` `_EchoWorker` 均定义它。对照 `startup_method`（有 `"load_policy"`/`"startup"` 两个 producer）应**保留**。
- 动作：删除 `metadata_method` 形参，硬编码为 `actor.worker_metadata.remote()`（比 `getattr(actor, "worker_metadata")` 更直白，功能等价）；**保留** `startup_method`。无 caller/test 更新。
- 注意：low，但 **`actor_group.py` 属在飞 sprint [[SPRINT_native_generation_engine_program]] 的未提交 worktree**。已复核：该 sprint 的 diff hunk 位于 `@@ -9` 与 `@@ -96,7 +97` 附近，**不触及** `metadata_method`（L43/81），无符号级冲突；但为避免与活跃 worktree 编辑打架，**建议 sequence after the sprint** 落地或与 sprint 作者协调。

### 1.7 `ClusterTopology.has_non_driver_gpus` — dead-field（risk=low）
- 位置：`vrl/ray/dependencies.py:53-57`
- 判死证据：
  ```
  $ grep -rn 'has_non_driver_gpus' vrl/ tests/ --include='*.py'
  vrl/ray/dependencies.py:54:  def has_non_driver_gpus(self) -> bool:   ← 定义
  tests/ray/test_dependencies.py:25/33/42:  assert topo.has_non_driver_gpus is ...  ← 仅测试断言
  ```
  test-only property。唯一生产消费者 `cross_node_preflight`（`vrl/ray/placement.py:89-104`）直接读 `topology.non_driver_gpus`/`topology.driver_gpus`，从不读该 property。docstring 声称的 `cross_node auto-detect (run_online_recipe)` 已废：`vrl/scripts/common/online.py` 从不调 `inspect_cluster`，`cross_node` 是 explicit config bool（L118 校验、L941 调 `cross_node_preflight`）。docs 唯一命中是历史已删符号 `_cluster_has_non_driver_gpus`（另一个名字）。
- 动作：删除 `has_non_driver_gpus` property（`dependencies.py:53-57`）；在 `tests/ray/test_dependencies.py` **直接删除**三处 property 断言（L25/33/42）——紧邻的前置行（L24/32/41）已有等价的 `topo.non_driver_gpus` 直接断言，改写会造成重复。修正 `ClusterTopology` docstring（`dependencies.py:44-47`）只引 `vrl.ray.placement.cross_node_preflight`，删掉 stale 的 `cross_node auto-detect (run_online_recipe)` 声称。

### 1.8 `GlobalRayPlacementOwner.placement_strategy` — dead-config-knob（risk=low）
- 位置：`vrl/ray/placement.py:220,401-402`
- 判死证据：
  ```
  $ grep -rn 'placement_strategy' vrl/ tests/ --include='*.py' | grep -v chunk_placement_strategy
  vrl/ray/placement.py:220:  placement_strategy: str | None = None   ← 定义
  vrl/ray/placement.py:401-402:  if self.placement_strategy: return self.placement_strategy  ← 唯一读者（override 分支）
  vrl/ray/resources.py:1255:  for stale_pool_key in ("placement_strategy", ...)   ← removed-key 硬拒绝
  tests/config/test_schema.py:447:  @parametrize("removed_key", ["placement_strategy", ...])  ← 覆盖拒绝逻辑，保留
  ```
  form-2 死分支：`placement_strategy=` 作为 setter **零命中**；唯一生产构造 `online.py:923` 与四处测试构造均不设它，无 `**kwargs`/`asdict` 路径。旧 YAML 键已被 `resources.py:1255` hard-reject。override 分支 `if self.placement_strategy: …` 无 producer。
- 动作：删除 `placement_strategy` 字段（L220）与 `_strategy()` 的 override 分支（L401-402）；字段删后 `_strategy()` 只剩一行 `"SPREAD" if self.resources.cross_node else "PACK"`（保留解释注释），**内联进 `create()`（L250）并删 `_strategy()`**。无 test 改动（无 test 设字段/调 `_strategy`/断言 strategy 值；`test_schema.py:447` 覆盖的是 removed-key 拒绝，保留）。**并**更新 preset 注释 `vrl/config/presets/experiment/cosmos_predict2_5/online_nft_kling_video_reward_cross_node.yaml`（约 L29-30）——把 `._strategy` 指针改指 `GlobalRayPlacementOwner.create`，避免长期 config 的 doc 指针悬空。
- 注意：low，但 **`placement.py` 属在飞 sprint 未提交 worktree**。已复核 diff hunk（`@@ -40`、`@@ -269,7 +270`、`@@ -450,7 +452`）**不触及** L220/401/402，无符号级冲突；仍建议 **sequence after the sprint** 或与作者协调。

### 1.9 `GlobalRayPlacementOwner.ready_timeout_s` — dead-arg（risk=low）
- 位置：`vrl/ray/placement.py:221,258,261`
- 判死证据：
  ```
  $ grep -rn 'ready_timeout_s' vrl/ tests/ --include='*.py'
  vrl/ray/placement.py:221:  ready_timeout_s: float = 600.0   ← 定义
  vrl/ray/placement.py:258:  ray.get(pg.ready(), timeout=float(self.ready_timeout_s))
  vrl/ray/placement.py:261:  f"Ray placement group not ready after {self.ready_timeout_s:.0f}s: "
  （'ready_timeout_s=' 作为 setter 零命中）
  ```
  form-5 死形参：`600.0` 默认**被消费**（`ray.get` timeout + not-ready 错误信息），但无任何构造点覆盖它（生产 `online.py:923` + 四处测试均用默认）；无 `**kwargs`/`asdict`/YAML 键。属「从不被行使的 override 旋钮」（弱于 §1.8 的不可达分支，但满足死形参定义）。
- 动作：把 `ready_timeout_s` 从 dataclass 字段降级为模块级常量（如 `_PLACEMENT_READY_TIMEOUT_S = 600.0`）并在 `create()` 引用。需要快 timeout 的测试可 monkeypatch 该常量；无 test cleanup（无 test 设字段）。
- 注意：low，**`placement.py` 在飞 sprint 未提交 worktree**（同 §1.8）。已复核 diff 不触及 L221/258/261；建议 **sequence after the sprint**。

### 1.10 `_to_plain` — duplicate-impl（risk=low）
- 位置：`vrl/ray/resources.py:1366-1373`
- 判死证据：
  ```
  $ grep -rn '_to_plain' vrl/ tests/ --include='*.py'
  vrl/ray/resources.py:1173/1199/1266/1288/1306  ← 5 个调用点，全在本文件
  vrl/ray/resources.py:1366:  def _to_plain(value: Any) -> Any:   ← 定义
  ```
  form-4 重复：body 与 `vrl/utils/config.py:73-87` `to_builtin()` 逐字节同逻辑（try-import `DictConfig`/`ListConfig`/`OmegaConf`、except 返原值、`isinstance → OmegaConf.to_container(resolve=True)`、else 透传），仅 docstring 不同。`resources.py:10` 已 `from vrl.utils.config import cfg_get`（`config.py` 是 leaf module，无循环）。
- 动作：删除 `_to_plain`，五个调用点改用 `to_builtin`（扩展现有 `from vrl.utils.config import cfg_get` 一行）。无 test 改动（测试打的是 parser，非 helper）。
- 注意：`resources.py` **不在**在飞 sprint 未提交集（git status 只列 `actor_group.py`/`placement.py`/`resource_cleanup.py`），无 sprint 重叠。

### 1.11 `trainer_torch_device(actual_trainer_devices=)` — dead-arg（risk=low）
- 位置：`vrl/ray/resources.py:436,440`
- 判死证据：
  ```
  $ grep -rn 'actual_trainer_devices' vrl/ tests/ --include='*.py'
  vrl/ray/resources.py:436:  actual_trainer_devices: tuple[int,...] | list[int] | None = None,  ← 定义
  vrl/ray/resources.py:440:  devices = tuple(actual_trainer_devices or resolved.trainer_devices)  ← 唯一使用
  ```
  form-5：全仓无 producer，body 退化为 `tuple(None or resolved.trainer_devices)`，or-fallback 死。所有 caller 单参（`train_dpo.py:209`、`online.py:847`、`resources.py:493/499`、`test_resources.py` 各处、monkeypatch fake 单参）。rank-local-device 需求由 `reward_torch_device(trainer_device=)` 服务（活 producer `online.py:873`/`factory.py:96`）。
- 动作：删除 `actual_trainer_devices` 形参与 `or` fallback，直接读 `resolved.trainer_devices`。无 caller/test 更新。

### 1.12 `format_distributed_resource_plan(actual_placement=)` — dead-arg（risk=low）
- 位置：`vrl/ray/resources.py:505,532-533`
- 判死证据：
  ```
  $ grep -rn 'actual_placement' vrl/ tests/ --include='*.py'
  vrl/ray/resources.py:505:  actual_placement: Any | None = None,
  vrl/ray/resources.py:532-533:  if actual_placement is not None: parts.append(f"actual={actual_placement}")
  （定义外零命中）
  ```
  form-5/form-2：所有 caller 单参（`online.py:846`、`train_dpo.py:208`、`test_resources.py:513/594/918`、两处 monkeypatch lambda 单参），该 arg 仅守末尾一个 append 分支——不可达。git 历史印证：引入于 `eaf62052`，其 caller 于 `ca3ae295` 移除；actual-vs-expected 上报现居 `vrl/ray/placement.py:154`。无 test 断言 `actual=` 子串。
- 动作：删除 `actual_placement` 形参及其尾部 append 分支。无 caller/test 更新。

### 1.13 `RewardScoringInput.expected_count` — dead-field（form-2 冗余校验，risk=low）
- 位置：`vrl/rollouts/collector/rewards.py:29,33-38`
- 判死证据：
  ```
  $ grep -rn 'expected_count' vrl/rollouts/ tests/rollouts/ --include='*.py'
  vrl/rollouts/collector/batch_builder.py:66:  expected_count=len(self.output.sample_rows)  ← 唯一 producer
  vrl/rollouts/collector/rewards.py:29:  expected_count: int | None = None            ← 定义
  vrl/rollouts/collector/rewards.py:34/37:  校验 + 错误信息
  ```
  唯一 producer `batch_builder.py:66` 同时传 `sample_rows=tuple(self.output.sample_rows)`——`expected_count` 恒等于 `len(sample_rows)`。`__post_init__`（L46-50）已独立校验 `len(self.sample_rows) == batch_size`，故 `expected_count` 检查（L34-38）只能在 sample_rows 检查也会触发的情形触发，唯一可观测差异是错误信息文本，无 test/caller 匹配该文本。测试构造 `RewardScoringInput`（`test_runtime.py:620/645/653`）从不传它。
- 动作：删除 `expected_count` 字段及其 `__post_init__` 校验分支（L34-38），并删唯一传参点 `batch_builder.py:66`。无 test cleanup。

### 1.14 `MultiSegmentTokenLogProbEvaluator._trajectory_segment_payload` 多余键 — dead-field（form-2，risk=low）
- 位置：`vrl/rollouts/evaluators/token/multi_segment_token_logprob.py:161-182`
- 判死证据：
  ```
  $ grep -rn '"visual"\|"cfg"\|"modality"\|"prompt_embeds"' vrl/rollouts/ --include='*.py'
  multi_segment_token_logprob.py:168/169/171/174  ← 仅本构造处
  ```
  payload 严格局部于 evaluator：`evaluate()` 只把 `batch+request` 传过模型边界（`model.replay_forward(batch, request=...)`），janus 模型自建自读另一份 payload（`vrl/models/families/janus_pro/model.py:421 _r1_segment_payload_from_trajectory`，读于 L483）。in-file 读者仅 `_segment_tensor`（读 `token_ids`/`token_log_probs`/`token_mask`/`name`）、`_extract_logprobs`（`token_ids` 回退）、`_enabled_segment_names`（读 `enabled`/`train`，但该分支本身不可达，见 §1.16）。测试 `test_multisegment_token_logprob.py:157` 的 `segment['modality']` 读的是 fake model 自己的 `_segment_payload`，非 evaluator dict。
- 动作：把 payload 缩减为真正被读的键 `name`/`token_ids`/`token_log_probs`/`token_mask`；删除 `visual`/`cfg`/`train`/`modality`/`prompt_embeds`/`attention_mask`/`prompt_attention_mask`/`prompt_input_ids` 的构造。无 test cleanup（无 test 断言这些键）。
- 注意：与 §1.15、§1.16 同文件，可合并一次编辑；但 §1.16 另需删 payload 的 `train` 键并改 `test:179`——见 §1.16。

### 1.15 `_segment_tensor` 的 `"replay"` 子字典回退 — dead-branch（form-2，risk=low）
- 位置：`vrl/rollouts/evaluators/token/multi_segment_token_logprob.py:199-204`
- 判死证据：
  ```
  $ grep -rn '"replay"' vrl/ tests/ --include='*.py'
  multi_segment_token_logprob.py:202:  replay = segment.get("replay")   ← 回退自身
  tests/generation/execution/test_chunks_pipelined.py:190  ← 无关张量名
  tests/models/steps/denoise/common/test_vae_decode_memory.py:346  ← 无关名字过滤
  ```
  `_segment_tensor` 仅被本文件内 `evaluate` 以 `_segments_from_batch` 产出的 payload 调用，该 payload 唯一构造点 `_trajectory_segment_payload` 从不写 `"replay"` 键 → 分支不可达。删后 `None` 落到既有 `RuntimeError`（L205-207），对所有可达输入语义不变。
- 动作：删除 `replay = segment.get("replay")` 回退（L201-204 的 `if value is None` 块）。无 test cleanup。

### 1.16 `_enabled_segment_names` 的 None/Mapping 分支 + `enabled_segments` 参数类型 — dead-branch（form-2，risk=low）
- 位置：`vrl/rollouts/evaluators/token/multi_segment_token_logprob.py:27-34,121-134`
- 判死证据：
  ```
  $ grep -rn 'MultiSegmentTokenLogProbEvaluator(' vrl/ tests/ --include='*.py'
  vrl/scripts/common/factory.py:319:  MultiSegmentTokenLogProbEvaluator(enabled_segments=enabled_segments)  ← 唯一生产构造
  tests/rollouts/replay/test_multisegment_token_logprob.py:181/195/219  ← 均传 tuple
  $ grep -rn 'enabled_segments' vrl/scripts/common/factory.py
  factory.py:316:  enabled_segments = tuple(name for name, enabled in segment_flags.items() if bool(enabled))
  ```
  `factory.py:316` 即使 `train_segments` 为 None，`dict(... or {})` 后仍产出**空 tuple 而非 None**，Mapping 更无可能；三个测试构造点全传 tuple。None 分支读的 `segment.get("enabled", ...)` 中 `"enabled"` 键在固定键集中不存在（零 producer）。`vrl/trajectory/builders.py:932` 的 `"enabled"`/`"train"` 处理是 rollout 侧另一份 payload，与此无关。
- 动作：`enabled_segments` 收紧为 `Iterable[str]`（必填，去 None 默认与 Mapping 分支及 `Mapping` import 若无他用）；`_enabled_segment_names` 收缩为一行 tuple 过滤并入 `evaluate`；**同时删除** `_trajectory_segment_payload` 中的 `"train"` 键（其唯一读者是被删的 None 分支 L126）；删除 `tests/rollouts/replay/test_multisegment_token_logprob.py:179` 的 `metadata["train"] = True` 无效 setup。**不改动** `vrl/trajectory/builders.py` 的 `"enabled"`/`"train"` 处理。
- 注意：与 §1.14 存在同 payload 键重叠（`train`）——若三条合并编辑，`visual`/`cfg`/`modality`/`train` 等键一次性删除即可，但须保证 §1.14 的键删除与本条的 `train` 键删除不重复劳动。

### 1.17 三个 token evaluator 的 `mask_key` 构造参数 — dead-arg（form-2 假旋钮，risk=low）
- 位置：`vrl/rollouts/evaluators/token/token_logprob.py:39`、`continuous_token_logprob.py:31`、`multi_segment_token_logprob.py:32`
- 判死证据：
  ```
  $ grep -rn 'mask_key=' vrl/ tests/ --include='*.py'
  multi_segment_token_logprob.py:106、continuous_token_logprob.py:66、token_logprob.py:93  ← 三 evaluator 内部 self.mask_key 转发给 builder
  vrl/rollouts/evaluators/trajectory.py:59/103  ← builder 形参
  vrl/rollouts/evaluators/denoise/sde_logprob.py:182:  mask_key="mask"  ← 唯一非默认传参（builder 调用，保留）
  ```
  三个 evaluator 全部构造点（`factory.py:292/296/319` + tests 6 处）均用默认 `"token_mask"`，无 `**kwargs`/`asdict` 转发、无 YAML、无 registry 字符串分发。`Evaluator`（`base.py:13`）Protocol 仅约束 `evaluate()`，`__init__` 不构成协议边界。[[SPRINT_design_smell_audit]] 已删同类假旋钮 `TokenGRPOConfig.mask_key`，方向一致。
- 动作：删除三个 token evaluator 的 `mask_key` 构造参数，内联 `"token_mask"`；**保留** `TrajectorySignalBuilder.single_segment`/`segment_signal` 的 `mask_key` 形参（`sde_logprob.py:182` 以 `mask_key="mask"` 实际使用）。无 test cleanup。

### 1.18 `enable_transformer_gradient_checkpointing(require_method=...)` — dead-arg（form-1+2，risk=low）
- 位置：`vrl/trainers/activation_checkpointing.py:128-157`
- 判死证据：
  ```
  $ grep -rn 'require_method' vrl/ tests/ --include='*.py'
  vrl/trainers/activation_checkpointing.py:132:  require_method: bool = True,   ← 定义
  vrl/trainers/activation_checkpointing.py:154:  if require_method:             ← 唯一内部读
  （零 setter）
  $ grep -rn 'enable_transformer_gradient_checkpointing' vrl/ tests/ --include='*.py'
  vrl/scripts/common/online.py:909:  enable_transformer_gradient_checkpointing(bundle, cfg)  ← 唯一生产 caller，默认 True
  tests/scripts/test_online_lifecycle.py:370:  monkeypatch 整函数替换
  ```
  form-1 死形参 + form-2 死 False 分支。历史 `require_method=False` 的 producer（cosmos `_after_bundle_built`，见 [[SPRINT_grab_bag_file_audit]]:110）已删——`grep -rn 'after_bundle_built' vrl/ tests/` 零命中。`require_method` 恒 True 时 `continue`（静默跳过）分支不可达。
- 动作：删除 `require_method` 形参，把 `if require_method: raise / continue` 收为「trainable module 缺 `enable_gradient_checkpointing` 时无条件 raise」。无 test 更新（monkeypatch 是 `lambda *args, **kwargs`）。

### 1.19 `TrainState.total_reward` / `TrainState.total_loss` — dead-field（write-only，risk=low）
- 位置：`vrl/trainers/core/types.py:429-437`
- 判死证据：
  ```
  $ grep -rn 'state\.total_reward\|state\.total_loss' vrl/ tests/ --include='*.py'
  trainer.py:1382-1383 / 1821-1822  ← += 累加
  trainer.py:2063-2064              ← state_dict 写
  trainer.py:2101-2102              ← load .get 恢复
  $ grep -rnF '"total_reward"' / '"total_loss"'  → 仅上述 state_dict/load 行
  ```
  纯 write-only checkpointed state：每步累加、round-trip 进 checkpoint，但**无任何 metric emitter/log/script/test 读取**。`TrainState` 是 `slots=True`、checkpoint dict 手写键（无 `asdict`/`fields()`），删除不漏 serialization；`load_state_dict` 用 `.get(key, 0.0)`，旧 checkpoint 仍可加载。`vrl/algorithms/grpo/multisegment.py` 的 `total_loss` 是 GRPO loss 函数局部 torch 变量，无关。
- 动作：删除两字段；删 `trainer.py:1382-1383`/`1821-1822` 四行累加、`2063-2064` 两个 state_dict 条目、`2101-2102` 两个 load 恢复。无 test 更新（无 test 断言这两键或精确键集）。

### 1.20 `validate_artifact_manifest(allow_absolute_paths, require_readable, reject_metadata_domain)` — dead-arg（form-1，risk=low）
- 位置：`vrl/trainers/data/artifacts.py:160-229`
- 判死证据：
  ```
  $ grep -rn 'reject_metadata_domain\|require_readable\|allow_absolute_paths' vrl/ tests/ --include='*.py'
  vrl/trainers/data/artifacts.py:167/168/169/180/201/208  ← 三参定义+内部读
  vrl/rewards/models/target_dino_similarity.py:60/150     ← 无关：喂给 resolve_artifact_path 的 reward 旋钮，另一个函数
  ```
  三个 keyword-only 参非默认路径不可达：`require_readable`/`reject_metadata_domain` 定义外零命中；`allow_absolute_paths` 仅出现在 reward 侧另一函数。唯一 `**kwargs` 路由 `validate_artifact_manifest_pair` 及其固定签名 wrapper `validate_source_backed_video_world_manifest_pair`（`artifacts.py:263-281`，无 `**kwargs`）只转发固定字段集；直接 caller（`derive_text_video_targets.py:102`）与 test caller 只传 `data_root`/`required_artifact_fields`。内联默认行为等价（`resolve_artifact_path` 自身默认 `allow_absolute=False`；`require_readable=True` 即恒 `_assert_readable`；`reject_metadata_domain=True` 即恒 raise on `metadata.domain`），且测试正打这些默认行为（如 `test_production_metadata_domain_is_rejected`）。
- 动作：删除三参并内联默认（`allow_absolute=False`、恒 `_assert_readable`、恒 reject `metadata.domain`）。无 test 更新。

### 1.21 `PhaseTimer.__init__(sync)` — dead-arg（risk=low）
- 位置：`vrl/trainers/online/trainer.py:150-153`
- 判死证据：
  ```
  $ grep -rn 'PhaseTimer(' vrl/ tests/ --include='*.py'
  vrl/trainers/online/trainer.py:1078:  PhaseTimer(enabled=cfg.profile)
  tests/trainers/online/test_skip_backward_agreement_distributed.py:396:  PhaseTimer(enabled=False)
  ```
  唯二构造均不传 `sync`；无子类、无 `**kwargs` 构造、无 `.sync` 外部改写（仅 `__init__` 赋值 + `time()` 内两读）、无 YAML 键（唯一 YAML 命中是描述 `profile` 的注释）。`sync=False` 路径不可达，True 默认退化为 `torch.cuda.is_available()`。
- 动作：签名改 `def __init__(self, enabled: bool = False) -> None:`，赋值改 `self.sync = torch.cuda.is_available()`；**同时**改写 class docstring（L144-147）使其不再记录 `sync=True` 参数（例：「When CUDA is available, `torch.cuda.synchronize()` is called on both ends …」）。无 test 更新。

### 1.22 `FSDPStrategy.validate_training_state_parking` / `restore_training_state` — duplicate-impl（form-4，risk=low）
- 位置：`vrl/trainers/strategy.py:762-763,798-799`
- 判死证据：
  ```
  $ grep -n 'def validate_training_state_parking\|def restore_training_state\|^class ' vrl/trainers/strategy.py
  150:class _TrainingStateParking   (validate@164 = `return None`, restore@269)
  290:class SingleProcessStrategy(_TrainingStateParking, Strategy)   ← 两方法都不定义，靠 mixin
  509:class FSDPStrategy(_TrainingStateParking, Strategy)   (validate@762, park@765, restore@798)
  ```
  body 比对：`FSDPStrategy.validate_training_state_parking`（762-763）是 `return None`，与继承的 `_TrainingStateParking`（164-165）逐字节同；`FSDPStrategy.restore_training_state`（798-799）body 是单行 `_TrainingStateParking.restore_training_state(self, state)`——正是 MRO（`FSDPStrategy→_TrainingStateParking`）无 override 时的解析结果。`Strategy` 是 Protocol（`...` body，无 `@abstractmethod`），mixin 赢 MRO。对照 `park_training_state`（765-796）是真 override（跨 rank failure agreement），**保留**。删后 sibling `SingleProcessStrategy` 模式一致，跨家族一致性反倒**支持删除**。
- 动作：删除两个 override。`park_training_state` 内 `self.restore_training_state(state)`（L789）删后解析到相同 mixin 实现，行为不变。无 test 更新（`test_fsdp.py:685` 断言返回 None 经继承仍通过）。

### 1.23 `release_cuda_memory(gc_collect=)` — dead-arg（form-5，risk=low）
- 位置：`vrl/utils/cuda_memory.py:195-208`
- 判死证据：
  ```
  $ grep -rn 'release_cuda_memory(' vrl/ tests/ --include='*.py' | grep -v for_parking
  memory_parking.py:186/430/440:  release_cuda_memory(gc_collect=True, ipc_collect=True)
  worker.py:578:                  release_cuda_memory(gc_collect=True)
  vrl/utils/cuda_memory.py:195:   def release_cuda_memory(  ← 定义（默认 gc_collect=False）
  ```
  4 个生产调用点全传 `gc_collect=True`，默认（False）路径零 producer。`gc_collect` 只守 `if gc_collect: gc.collect()`；`ipc_collect` 真正变化（worker 省略）应**保留**。`release_cuda_memory_for_parking`（L246-274）是另一实现，直接调 `gc.collect()`，不经此函数。
- 动作：删除 `gc_collect` 形参，body 顶部无条件 `gc.collect()`（保持在 torch import 前的现有次序）；**保留** `ipc_collect`。更新四调用点：`memory_parking.py:186/430/440 → release_cuda_memory(ipc_collect=True)`；`worker.py:578 → release_cuda_memory()`。**并**更新 `tests/generation/execution/test_execute_request_pipelined.py:136` 断言 `cleanup_calls == [{"gc_collect": True}]` 改为 `cleanup_calls == [{}]`（`test_worker_sleep.py:1183` 用 `lambda **kwargs: None`，无需改）。
- 注意：low。本条触及 `vrl/generation/execution/memory_parking.py`、`worker.py`，但主体编辑在 `vrl/utils/cuda_memory.py`；这些文件**不在**在飞 sprint 的未提交集（git status 是 `vrl/generation/ray/` 非 `vrl/generation/execution/`）。DO-NOT-FLAG 对 `cuda_memory.py` 的豁免仅限 `process_gpu_used_bytes`（NVML），`gc_collect` 不在豁免内。

## 2. 验证协议
- **每条删除后**：`ruff check <touched files>` + `ruff format --check <touched files>`（仅跑本条触及的 Python 文件，先 `ruff check --fix` 再 `ruff format`，最后 `--check` 复核）。
- **全簇完成后**：`pytest tests/ray/ tests/rollouts/ tests/trainers/ tests/data/ tests/quality/ tests/generation/execution/ tests/config/ tests/scripts/` + `pytest -m "not e2e and not slow_test"` 子集不新增失败。
- **基线（清理前，2026-07-23）**：fast subset 2620 passed / 7 pre-existing failures（架构边界 + causvid/magi_1 打包摘要，与本清理无关）；`vrl.config.lint` 与 `ruff check .` 全绿。删除后这三项须保持。
- **逐条触及的测试文件**（从 action 提取，无列出者表示无 test 改动）：
  - §1.1 `dones`：`tests/rollouts/collector/test_runtime.py`（含 L325 设备断言）、`tests/rollouts/batch/test_batch.py`（约 44 处 `RolloutBatch(...)` 构造）。
  - §1.2 `collect_unscored/build seed`：`tests/rollouts/collector/_helpers.py`、`tests/rollouts/collector/test_runtime.py:292/299`、`tests/rollouts/runtime/test_engine_requests.py:32/44`、`tests/quality/preview.py`、`tests/quality/test_preview.py:48/57`。
  - §1.3 `collect_prompt_batches`：`tests/e2e/test_real_checkpoint_rl.py:626`、`tests/trainers/online/{test_precision_drift_guard,test_state_restore,test_step_split,test_reward_update_flow,test_trainable_state,test_trajectory_granularity,test_advantage_and_metrics}.py`、`tests/rollouts/orchestration/{test_prompt_collection,test_orchestration}.py`、`tests/rollouts/orchestration/continuous/{test_schedule,test_owner,test_contracts,test_queue,test_scheduler}.py`、`tests/rollouts/collector/test_runtime.py:785`、`tests/rollouts/collector/_collect.py`、`tests/trainers/online/test_reward_update_flow.py:1275/1338`（`remap_group_ids_` TEST-ONLY 用例删除）。
  - §1.4 `gather_full_state_dict`：`tests/trainers/test_fsdp.py`（8 调用点中 6 个 + import + 直接测试 L277）、`tests/trainers/test_fsdp_gather_distributed.py`（L32/101/134）——**迁 helper 后 repoint**，不删 oracle 测试。
  - §1.5 `timestep_subset`：`tests/trainers/test_offline_dpo_timesteps.py:53/62/95`。
  - §1.7 `has_non_driver_gpus`：`tests/ray/test_dependencies.py:25/33/42`（删三处 property 断言）。
  - §1.16 `enabled_segments`：`tests/rollouts/replay/test_multisegment_token_logprob.py:179`（删 `metadata["train"] = True`）。
  - §1.23 `gc_collect`：`tests/generation/execution/test_execute_request_pipelined.py:136`。
  - 其余（§1.6/§1.8/§1.9/§1.10/§1.11/§1.12/§1.13/§1.14/§1.15/§1.17/§1.18/§1.19/§1.20/§1.21/§1.22）：**无 test 改动**，删后跑对应目录测试确认零回归即可。

## 3. Non-Goals
- 不删被「能 raise 的校验」、控制流分支、runtime/config/Ray 调用消费的字段（AGENTS 死字段规则）——如 §1.5 保留的 `scheduler.timesteps`-empty fail-fast、§1.13 保留的 `__post_init__` `len(sample_rows)==batch_size` 校验、§1.19 保留 `TrainState` 其余被读字段。
- 不动 DO-NOT-FLAG 豁免项：`_GENERATION_CUDA_RUNTIME_RESIDUAL_BYTES`（`generation/execution/worker.py`）、`_REWARD_CUDA_RUNTIME_RESIDUAL_BYTES`（`rewards/base.py`）、`ensure_loaded`（`rewards/runtime.py`）、`process_gpu_used_bytes`（`utils/cuda_memory.py` NVML）、`prepare_latents` 修复（sana/hunyuan）。§1.23 的 `gc_collect` 在同文件 `cuda_memory.py` 但**不属**该豁免（豁免只覆盖 `process_gpu_used_bytes`）。
- 不为省行数扁平化 protocol/lazy-import/跨家族一致性 thin function——如 §1.6 保留 `startup_method`（两 producer）、§1.17 保留 `TrajectorySignalBuilder` 的 `mask_key`（`sde_logprob` 实用）、§1.22 保留 `park_training_state`（真 override）、§1.23 保留 `ipc_collect`（真变化）。
- **cluster-specific non-goals**：
  - **不在在飞 sprint [[SPRINT_native_generation_engine_program]] 未合并前编辑其 worktree 文件**（§1.6 `actor_group.py`、§1.8/§1.9 `placement.py`）——三条已复核无符号级冲突，但为避免与活跃 worktree 打架，须 **sequence after the sprint** 或与作者协调后再落地。
  - §1.4 `gather_full_state_dict` **不删除、不移植到 `gather_trainable_state_dict`**——迁入测试基础设施，保留其对 live 生产路径（`load_full_state_dict`、frozen-param 不变性）的 oracle 覆盖。此判定与 [[SPRINT_fbag_00_overview]] 的 cohesive-keep 不冲突（那次谈文件归属，本次谈 liveness）。
  - §1.14/§1.15/§1.16 同文件三条须协调编辑，避免 payload `train`/`visual`/`cfg`/`modality` 键的重复删除。

## References
- `vrl/rollouts/batch/core.py:20`、`vrl/rollouts/collector/batch_builder.py:66,134,161,210`、`vrl/rollouts/batch/ops.py:40,92-104,131,147`、`vrl/rollouts/orchestration/continuous/types.py:60`
- `vrl/rollouts/collector/core.py:162,169,187`、`vrl/rollouts/collector/requests.py:53,61-62`、`vrl/rollouts/collector/rewards.py:29,33-38`
- `vrl/rollouts/orchestration/prompt_collection.py:33,69-71,157-188,246-251,298`、`vrl/rollouts/orchestration/strict_on_policy.py:89`、`vrl/rollouts/orchestration/continuous/producer.py:342`
- `vrl/rollouts/evaluators/token/multi_segment_token_logprob.py:27-34,121-134,161-182,199-204`、`vrl/rollouts/evaluators/token/token_logprob.py:39,93`、`vrl/rollouts/evaluators/token/continuous_token_logprob.py:31,66`、`vrl/rollouts/evaluators/trajectory.py:59,103`、`vrl/rollouts/evaluators/denoise/sde_logprob.py:182`
- `vrl/ray/actor_group.py:43,81`、`vrl/ray/dependencies.py:44-47,53-57`、`vrl/ray/placement.py:220,221,250,258,261,401-402`、`vrl/ray/resources.py:436,440,505,532-533,1173,1199,1255,1266,1288,1306,1366-1373`
- `vrl/trainers/activation_checkpointing.py:128-157`、`vrl/trainers/core/types.py:429-437`、`vrl/trainers/data/artifacts.py:160-229`、`vrl/trainers/fsdp.py:287-326,518`、`vrl/trainers/offline/dpo.py:60-62,203-221`、`vrl/trainers/online/trainer.py:144-153,1078,1382-1383,1821-1822,2063-2064,2101-2102`、`vrl/trainers/strategy.py:762-763,798-799,908`
- `vrl/utils/cuda_memory.py:195-208`、`vrl/utils/config.py:73-87`
- `vrl/scripts/common/online.py:186,847,909,923`、`vrl/scripts/common/factory.py:96,292,296,316,319`、`vrl/scripts/families/wan_2_1/train_dpo.py:85-103,208-209`、`vrl/config/schema.py:616-635,642-653,941`、`vrl/config/presets/experiment/cosmos_predict2_5/online_nft_kling_video_reward_cross_node.yaml:29`
- 关联 sprint：[[SPRINT_deadcode_00_overview]]、[[SPRINT_trajectory_views_types_dead_fields_cleanup]]、[[SPRINT_fbag_00_overview]]、[[SPRINT_grab_bag_file_audit]]、[[SPRINT_design_smell_audit]]、[[SPRINT_native_generation_engine_program]]、[[SPRINT_continuous_stage_contracts_and_baseline]]、[[SPRINT_continuous_three_stage_pipeline_program]]
