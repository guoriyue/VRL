# SPRINT: Grab-bag file audit — 低内聚文件结构整治

**日期**: 2026-07-10  **状态**: EXECUTED（2026-07-10，全部 4 个 sprint 落地）
**验证**: 全量 pytest 1630 passed / 0 failed（基线 1629：-1 删除的 prefix-cache 测试，+1 diffusion replay 回归测试，+1 artifact_data_root 断言）；ruff 与 config lint 全绿；flow_matching 重构做了 HEAD-vs-新版位精确 probe（CFG 开/关、sample+replay 全等）。

**执行修正（与原计划的差异）**:
1. Sprint 1 第 1 项**撤销**：`shutdown_training_process_group` 并非零调用方——`vrl/trainers/strategy.py:346,492`（`Strategy.shutdown`）在调用，两个测试 patch 它。审计的 verify agent 看走眼。仅修正了它的过时 docstring。
2. Sprint 3 第 8 项做**窄版**：共享 `named_tensor` 落 `vrl/trajectory/views.py` 且 batch_builder 改用，但 `TrajectoryResolver.tensor` **不** delegate——delegate 会把错误类型从 `TrajectoryResolverError(ValueError)` 漂移成 `RuntimeError`，违反本计划自己的异常类型漂移红线。
3. Sprint 3 第 9 项（AR/Diffusion layout 三件套收敛）**主动跳过**：`generation/diffusion/layout.py` 是另一进程 executor-as-data reconcile 的并行改动面（记忆约定 don't redo/touch）。待 reconcile 落地后可单独补做。
4. Sprint 4 第 4 项已知行为变化（除计划记录的 cosmos/executor 尊重 `artifact_data_root` 外）：reference 路径在 load 时以 `allow_absolute=True` 解析，wan i2v 原先默认拒绝 manifest 里绝对路径的校验随之放宽（`data.allow_absolute_artifact_paths` 仍由 encode_targets 的 TARGET 侧消费，非死旋钮）。
**来源**: 53-agent 动态 workflow 审计（12 个按包扫描 agent + 每条发现独立对抗核查 + 汇总），
核查标准直接取自 AGENTS.md（薄函数保留清单 / no new lean files / no big refactors / 死代码五形式）。
38 条发现确认（30 REAL + 8 PARTIAL 修正后保留），2 条否决转为显式非目标。

## 对触发问题的直接回答

- **`vrl/trainers/fsdp.py` 的"一袋函数"风格本身是合法模式**（vLLM/torch 的 weight_utils/model_loader 先例，
  函数间无共享状态穿线）——337 行里唯一确认的问题是零调用方的 `shutdown_training_process_group`（Sprint 1 删）。
  不要为了"看起来像类"而把它类化。
- **四份 `artifacts.py` 不是四个撞名的独立概念，而是一次半途而废的迁移**：commit 98a68d4c 把纯路径工具
  （`repo_root`/`default_data_root`/`resolve_artifact_path`/`ArtifactManifestError`）抽到 `vrl/utils/artifacts.py`
  后只重指了 1 个消费方，其余 8+ 个仍从 `vrl.trainers.data.artifacts` 走 pass-through 再导出——一个符号两个
  import 家园，rewards/config/scripts 层因此背上对 trainers.data 的假依赖（Sprint 2 收尾）。
  `vrl/rewards/artifacts.py`（reward artifact 存储）与 `vrl/rollouts/collector/artifacts.py`（reward view 释放）
  是真正独立的概念，保留原地，各自只有局部小修。
- 真正的系统性问题不是"函数太散"，而是三类：**迁移收尾没做完**（双 import 家园）、
  **形式四近重复**（10 份同体 `prompt_encoder_dtype` 块、10 份同体 `_lora_dtype` override、5 份 reward `_resolve_model_root`、
  flow_matching 采样/回放双拷贝）、以及顺带揪出的**一个活 bug**（echo/cosmos3/anima 的 diffusion replay builder 断线）。

---

## Sprint 1 — 死代码清除（零行为变化，纯删除）

**目标**：按五形式清掉全部零调用方/死语义代码，净删约 400 行，不改任何运行时行为。全部 S 工作量、高置信度。

### 变更清单

1. **删 `shutdown_training_process_group`**（`vrl/trainers/fsdp.py:59-65`，零调用方）；同步改 `init_training_process_group` 的过时 docstring（fsdp.py:40，"future torchrun entrypoint" 已不成立，teardown 靠进程退出）。
2. **删 `EnginePlanner` 类**（`vrl/generation/execution/planner.py:18-50`，唯一构造点是其下方的 `build_engine_plan`）：把 chunk-size 决策（显式参数 > `sampling["samples_per_chunk"]` > `samples_per_prompt`）内联成 `build_engine_plan` 一段直线代码；`execution/__init__.py` 删 `EnginePlanner` 导入与 `__all__` 条目。5 个生产调用方只用 `build_engine_plan`，零调用方编辑。
3. **删 AR prefix-cache 投机残骸**（`vrl/nn/layers/attention/paged.py`）：`ARPrefixCacheKey`、`ARPrefixCachePolicy`、`_stable_hash`、`import json`、`from hashlib import sha256`、`__all__` 两条；`attention/__init__.py` 同步删；测试文件删 `test_prefix_cache_policy_requires_policy_version_match` 与 `_prefix_key`。保留 `Mapping, Sequence` 导入（`ARAttentionBackend.free` 还在用）。项目记忆已记录 "shared-prefix probed DEAD"。
4. **删 vLLM paged 未消费 API 面**（`vrl/nn/kernels/attention/vllm_paged.py`）：`split_kv_cache`、`write_to_paged_cache`、`paged_attn_module` property、`_REQUIRED_MODULES` 中的 `"vllm.v1.attention.ops.paged_attn"`。生产路径（`ar_decoder.py:357`）只走 `update_flash_kv_cache`。两个测试文件按发现给的方案改：import-gate 测试删 fake 分支，real-ops 测试改为走生产写路径（`make_flash_attention_impl` + scale-shim + `update_flash_kv_cache`，同样断言 120.0/376.0 求和）。
5. **删 `TrajectoryResolver._lookup_tensor`**（`vrl/trajectory/resolver.py:131-139`，零调用方且形式四——与公有 `tensor()` 同体）。保留 `_split_ref`（`replay_tensor_dict` 在用）。
6. **删 `RayActorGroup.map_method`**（`vrl/ray/actor_group.py:95-123`）及其唯一支撑导入（line 9 的 `RayActorJob, run_actor_jobs`）。**测试不整删**（PARTIAL 修正）：`test_ray_actor_group_maps_payloads_in_order` 是 launch/startup_method/shutdown 生命周期唯一的真 Ray 覆盖 → 改名 `test_ray_actor_group_launch_lifecycle`，把 `map_method` 调用换成直接 `ray.get([h.actor.echo.remote(...)])`，保留 launch 配置/metadata/shutdown 断言。
7. **删 5 个零调用方 preflight 函数**：`preflight_videoscore2_backend`、`preflight_unified_reward_video_backend`、`preflight_cosmos3_reasoner_backend`、`preflight_videocon_physics_backend`、`preflight_phymotion_backend` 及各自 `__all__` 条目。保留 kling 的（`online.py:1052-1060` 活跃调用）。**不建**通用 preflight 调度器（无消费方的机制）。
8. **删 `extract_reward_artifact` 单调用方概念裂片**（`vrl/rollouts/collector/artifacts.py:39-42`）：`batch_builder.py:246` 改直读 `self.output.output`（:244 的 `output_ref` 检查已文档化契约）；缩 import；`__all__` 删条目。`GenerationOutput` 导入保留（`release_reward_artifact_if_needed` 在用）。
9. **折叠 `vrl/math/diffusion/nft.py`**（单消费方 20 行模块）：`normalized_mse` 原样搬进 `vrl/algorithms/diffusion_nft.py` 模块级（函数体内 `import torch`，匹配该文件既有 deferred-torch 风格），docstring 去掉虚假的 "shared" 措辞；删 nft.py 文件；`math/diffusion/__init__.py` 删再导出。
10. **删 no-op 旋钮 `drop_policy`（形式二：活调用方、死语义）**：`drop_oldest` 与 `drop_oldest_stale` 行为完全相同（completed_at 单调入队 → min 即队头）。端到端删：`continuous/queue.py`（参数/校验/字段，`_pick_victim` 并入 `_enforce_caps` 成 popleft 循环）、`continuous/schedule.py:51/61/175`、`orchestration/schedule.py:164`、`trainers/core/types.py:147/166-169`、preset `continuous.yaml:27` 注释、两个测试文件的 kwargs。**不动** `docs/runs/*/resolved_config.yaml`（历史运行存档）。若将来要版本感知驱逐，实现在 `RolloutScheduler`，不在 queue。
11. **派生 Protocol 成员元组**（`vrl/models/interfaces/replay.py`）：用 `__protocol_attrs__`（py3.12+，仓库契约测试同款模式；**不用** `get_protocol_members`，那是 3.13+）派生 `_REPLAY_MODEL_METHODS` / `_RUNTIME_MODEL_METHODS`，替换 `require_replay_model`/`require_runtime_model` 里手写元组及 f-string 第三副本。

### 保持不变（及原因）
- `_over_capacity`（命名两条件谓词，保留）；`ReplayRolloutStubs`（Sprint 4 才动其周边）；kling preflight 与 `_preflight_production_video_reward`（活跃、有测试）。
- **非目标（来自 REJECTED 清单）**：`RolloutScheduler` 不改名（vLLM 先例命名，所有 *Policy 替代名与 `StalenessPolicy`/RL policy-version 词汇冲突）；`vrl/generation/ray/pipeline_runner.py` / `stage_worker.py` 及其导出、契约测试**全部保留**（parked sprint 的 foundation staging，framework adapter/protocol boundary，属 keep-list）。

### 验证清单
- `pytest tests/rollouts/orchestration/continuous/ tests/rollouts/collector/ tests/nn/ tests/ray/ tests/trajectory tests/models/interfaces/ tests/algorithms -k "not slow"`
- grep 证空：`grep -rn "EnginePlanner\|ARPrefixCacheKey\|ARPrefixCachePolicy\|map_method\|_lookup_tensor\|extract_reward_artifact\|drop_policy\|shutdown_training_process_group\|math.diffusion.nft\|preflight_videoscore2\|preflight_unified\|preflight_cosmos3_reasoner\|preflight_videocon\|preflight_phymotion" vrl tests` 只允许命中 kling 无关行/零命中。
- continuous preset config-resolve 一遍，确认无悬空 key。

---## Sprint 2 — Import 家园归一（搬家/重指，零行为变化）

**目标**：一个符号一个家。完成 98a68d4c 半途而废的 artifacts 迁移，消灭双 import 家园与循环依赖补丁。全部机械搬移。

### 变更清单

1. **artifacts 迁移收尾**（合并发现 #1/#4/#29/#36，同一工作项）：
   - `vrl/utils/artifacts.py`：`_coerce_data_root` → 公有 `coerce_data_root`（更新 :42 内部调用，进 `__all__`）。
   - `vrl/trainers/data/artifacts.py`：utils import 缩为 `ArtifactManifestError, coerce_data_root, resolve_artifact_path`（删 `DATA_ROOT_ENV`/`repo_root`/`default_data_root`——本地 `_coerce_data_root` 删除后 `default_data_root` 也无本地消费方）；删本地 `_coerce_data_root`（:297-298），:188 改调 `coerce_data_root`；`__all__` 删全部 5 个 utils 名。
   - `vrl/trainers/data/__init__.py`：删 `DATA_ROOT_ENV, ArtifactManifestError, default_data_root, resolve_artifact_path`（零 facade 消费方）。
   - 消费方全部重指到 `from vrl.utils.artifacts import ...`：`rewards/models/motion_dynamics.py:29`、`target_dino_similarity.py:32`、`config/validation.py:243`、`scripts/data/common.py:19`（**并重写 :15-18 假注释**：source of truth 是 vrl.utils.artifacts；common.py 本地再导出保留——danbooru/videophy_i2v/bootstrap 合法经它走）、`scripts/data/setup.py:35`、`scripts/diffusion/cosmos/train.py:17`、`wan_2_1/train.py:17`、`scripts/eval/future_reward_discrimination_probe.py:53`、`tests/data/test_artifact_manifest_validation.py`、`tests/data/test_video_world_manifests.py`（`validate_*` 三件套仍从 trainers.data.artifacts 导——那是它真正的家）。
2. **`replay_loading` 并入 `interfaces/runtime.py`**（发现 #18）：`LOADS_FULL_GENERATION_MODULES_KEY` + 三个 accessor 搬到 `MEMORY_POLICY_METADATA_KEY`/`MODEL_MEMORY_SECTIONS` 旁；删 `vrl/models/replay_loading.py`；重指 `ar/build.py:24`、`diffusion/build.py:28`、`utils/memory.py:11`；`RuntimeBundle` docstring 改指本模块；测试并入 `test_minimal_replay_runtime_wiring.py`，删 stale 测试文件。
3. **memory guard 搬回唯一调用方**（发现 #28，**必须与上条同批、且改 import 目标**）：`validate_colocated_replay_memory`/`_is_colocated_gpu_rollout`/`_env_flag` 从 `vrl/utils/memory.py` 搬进 `vrl/generation/ray/config.py`；config.py 顶层导入 `os` + **`from vrl.models.interfaces.runtime import bundle_loads_full_generation_modules`**（不是原发现写的 replay_loading——它已被上条删除）+ logger；`validate_driver_state` 的 lazy import 改直调（循环依赖随之消失）；memory.py 只剩 HostMemorySnapshot 半边，docstring 缩为 "Host-memory instrumentation"；重指 `tests/trainers/test_memory_guards.py`；`interfaces/runtime.py:174` docstring 改指 `vrl.generation.ray.config`（replay_loading.py:8 的那条编辑作废，模块已删）。
4. **`_cross_node_preflight` 归位**（发现 #9）：从 `launcher.py:337-362` 原样搬进 `vrl/ray/placement.py` 成公有 `cross_node_preflight`（概念同胞 `validate_actor_gpu_ids` 旁）；launcher.py 删函数并从 dependencies import 里去掉 `inspect_cluster`；`scripts/common/online.py:742-744` 的跨包私有 deferred import 改顶层 `from vrl.ray.placement import cross_node_preflight`；测试搬到 `tests/ray/` 并重指；`dependencies.py:50` docstring 指针更新。
5. **删 `import_from_path` 兼容 shim**（发现 #35，PARTIAL 修正后只做 shim 半边）：5 个调用点改 `from vrl.utils.config import import_from_path`（`stage_worker.py:14`、`launcher.py:26`（并入 :31 已有的 utils.config import 行）、`scripts/perf/common/diffusion_runtime.py:19`、`tests/models/interfaces/__init__.py:23`、`tests/e2e/test_real_checkpoint_rl.py:25`）；`dependencies.py` 删 :8-10 与 `__all__` 条目，docstring 去掉 "dynamic import"；`vrl/ray/__init__.py` 同步删。**不搬** `ClusterTopology`/`inspect_cluster` 进 placement.py（方向倒置：placement 是有状态 PG owner，本就依赖 dependencies）。
6. **`top_k_top_p_filtering` 归位共享数学层**（发现 #13）：从 `llamagen/runner.py:34-63` 原样搬进现有 `vrl/math/ar/logprob.py`（docstring 扩为 AR sampling+replay logits math；进 `__all__` 与 `math/ar/__init__.py` 再导出）；llamagen 改 import 并删 `__all__` 条目（保留 `F`，:261/270 在用）；`glm_image/runner.py:43` 重指，删 :37-38 道歉注释。
7. **rewards provenance 元组派生化**（发现 #37）：`vrl/rewards/artifacts.py::_artifact_provenance` 的 13 键手写元组改为 `("task_type", *(f for f in DEFAULT_ARTIFACT_FIELDS if f != "references"), *SOURCE_BACKED_VIDEO_WORLD_METADATA_FIELDS)`，加注释说明 task_type 是 store-local、`"references"` 因 list-valued 与标量过滤器不兼容而**有意**排除；可选把 `DEFAULT_ARTIFACT_FIELDS` 加进 trainers/data/artifacts.py 的 `__all__`。

### 保持不变
- `config/validation.py:245-249` 的独立 schema 元组（有显式 "not unified" 注释，合法分叉）；kling/dino 等 reward 内部逻辑本 sprint 不碰。

### 验证清单
- grep 证空：`grep -rn "trainers.data.artifacts import" vrl tests | grep -E "default_data_root|repo_root|DATA_ROOT_ENV|ArtifactManifestError|resolve_artifact_path"`（只允许 trainers/data/artifacts.py 自身命中）；`grep -rn "replay_loading" vrl tests`；`grep -rn "from vrl.ray.dependencies import" | grep import_from_path`；`grep -rn "_cross_node_preflight"` 全部为空。
- 循环依赖 sanity：`python -c "import vrl.generation.ray.config, vrl.utils.memory"`。
- `pytest tests/data tests/ray tests/generation/ray tests/models/interfaces tests/trainers/test_memory_guards.py tests/rewards/inference/test_artifact_store.py tests/models/ar`

---

## Sprint 3 — 近重复合并（共享实现，行为保持 + 两处有意 drift 修复）

**目标**：形式四重复收敛到单一构造点。含一个**活 bug 修复**（echo/cosmos3/anima 训练路径断线）与两个有意的 drift 修复（明确标注）。

### 变更清单

1. **【BUG】diffusion replay builder 断线 + provenance 去重**（发现 #14 + #19，同文件合并）：
   - `vrl/rollouts/families/registry.py`：echo/cosmos3/anima 三条 entry 设 `replay_runtime_builder` dotted string（复用 AR 侧既有字段，**不**在 DiffusionFamilyBuild 加新字段）；更新字段 docstring（不再 AR-only）与 :74-75 过时 wan 注释。
   - `vrl/models/diffusion/build.py::build_family_replay_runtime_bundle`：`replay_cls is None` 时若 entry 有 builder 则 `import_from_path(...)(build)`（`_check_requires_lora` 先行），否则保留响亮 ValueError。
   - 删零调用方 `build_echo_replay_runtime_bundle_from_cfg`、`build_cosmos3_replay_runtime_bundle_from_cfg` 及 `__all__` 条目；anima 的 `resolve_anima_replay_model_build` 不动（e2e 字符串引用契约）。
   - 删除 bundle 上无生产消费者的 provenance metadata；checkpoint strict-resume 的 family guard 显式接收 resolved family，不再从 bundle metadata 复制 registry identity。`assemble_replay_bundle` docstring 收窄为 "REPLAY-side ... tail"。**明确拒绝**统一 rollout/replay 的 lora+compile tail（rollout 侧量化交织在分支两臂内、顺序有文档化路径依赖——不是重复）。
   - 测试：扩展 `tests/rollouts/runtime/test_family_registry.py` 断言每个 diffusion family 解析出 replay 路径（`replay_cls` 或可导入的 `replay_runtime_builder`）——这就是本该抓住回归的测试。**落地前先 `git fetch` 复查 registry.py/build.py**（记忆：另一进程在并行 reconcile 本树）。
2. **AR runtime LoRA 默认值折叠 5 份**（发现 #12）：`vrl/models/ar/build.py` 加 `ar_model_config_base(build, lora_defaults)`（base dict + use_lora 合并 + 5 个类型化 lora_* 键）；5 个 family runtime 各替换为一行调用，**保留各家 `_*_LORA_DEFAULTS` dict**（llamagen 的 wqkv/wo 真有差异）；删 janus 的 `_resolve_lora_block`；重指 `interfaces/runtime.py:112` 与 `model_build.py:79` 两处文档引用。
3. **scripts/diffusion 六份 lazy builder + wan T2V 重复入口**（发现 #30 + #31，同文件合并）：
   - `diffusion/train.py`：`_build_bundle`/`_build_replay_bundle` → 公有 `build_bundle`/`build_replay_bundle`（唯一合法 lazy-import 副本）；删 `_after_bundle_built`，直接传 `enable_transformer_gradient_checkpointing`。
   - cosmos/train.py 删四个 builder 改 import 共享对（**保留**自家 `_after_bundle_built`——`require_method=False` 是真实家族行为）；flux、wan 同理删本地对与 wrapper。
   - 删 `train_wan_2_1_grpo`（与通用 `train_diffusion_grpo` 逐字段相同；registry alias `wan`→`wan_2_1` 全覆盖）+ `__all__` 条目 + 模块 docstring 改为 Wan I2V；`presets/recipe/online/wan_2_1_grpo.yaml` 删 `trainer:` entrypoint 回覆盖块（文件本身保留，rollout shape 被三个实验共享）；`diffusion_grpo.yaml:18-19` 注释只留 cosmos。**保留** `train_wan_2_1_i2v_grpo`（真实 hook `collector_kwargs_getter`）。
4. **Prompt-encoder precision consolidation** (finding #15):
   `diffusers_pipeline_dtypes(build, model_dtype) -> (prompt_encoder_dtype, load_kwargs)`
   is the shared load boundary used by the ten Diffusers families. VAE FP32
   remains family-owned; pipeline construction and freeze order stay unchanged.
5. **anima align/shared replay tensor 收敛**（发现 #17）：删 anima 私有 `_align_replay_tensor`/`_shared_replay_tensor`，改用 `vrl.models.diffusion.common` 的共享对；`:381` 传**在作用域内的 `batch_context`**（与 predict2/2.5 调用形状逐字节一致，不传 `{}`）；`common/tensors.py:24-27` 假豁免注释改为说明 wan 保留严格变体的真实理由；顺手修 predict2 测试 docstring 的 stale 提法。
6. **`_lora_dtype` 10 份同体 override 下沉**（发现 #38，按 PARTIAL 修正）：mixin 默认读取已在 runtime 边界归一化的 `build.parameter_dtype`，docstring 重写；删 10 个同体 override；**仅** `CosmosPredict2Model` 与 `Cosmos3Model` 加显式 `return None` override（唯二真消费 None 默认的类）；**不动** wan_2_1（自有 apply_lora 不调 `_lora_dtype`，加了就是死方法）与 predict2_5（根本不继承 mixin）。
7. **reward 模型五份 `_resolve_model_root` + `__call__` 前奏收敛**（发现 #21 + #23，同文件组合并）：
   - `hub.py` 加 `resolve_model_root(worker_config, *, default_model, family)`（含 `local_files_only` 转发、lazy snapshot_download）；videoscore2/unified/cosmos3_reasoner/videocon 四家删本地副本改调用。**有意 drift 修复①**：videocon 由此获得之前静默丢失的 `local_files_only`。**kling 不动**（pin/RuntimeError-wrap/layout 校验，真家族特有）。
   - `base.py` 加 `require_prompt_and_video_path(artifact, *, family)`，五个 `__call__` 前奏改调用。**有意 drift 修复②**：unified_reward_video 补上缺失的空 prompt raise（其 rubric 是 caption-conditioned，正确）。phymotion:74 不动（path-only）。
   - 测试：`tests/rewards/test_model_hub.py:44-66` 重指到 `hub.resolve_model_root` 并加 `local_files_only` 转发断言。
8. **collector 的 named_tensor 微收敛**（发现 #5，按 PARTIAL 修正的窄版）：`vrl/trajectory/views.py` 在 `role_tensor` 旁加 `named_tensor(segment, name)`（体 = 现 `_named_tensor`），`trajectory/__init__.py` 导出；batch_builder 删 `_named_tensor` 改调用（扩展既有 import 行）；`TrajectoryResolver.tensor` 后半段 delegate 给它（"missing tensor" 错误单一 owner）；可选把 `_split_ref` 公有化为 `split_tensor_ref` 供 `_tensor_value_from_ref` 用（保留 RewardView 专属 RuntimeError 包装）。**保留 `_optional_named_tensor`**（None 语义不同）。
9. **AR/Diffusion layout 收敛**（发现 #8，按 in-flight 改动修正）：`GenerationOutput` 的汇总遥测已删除，`max_peak_memory_mb` 随之失去消费者，不再下沉共享 helper。后续只收敛仍有行为消费者的 `require_chunk_rows` / `ordered_covering_chunks(..., row_fields, family)`；逐 chunk 峰值继续由 worker debug 边界读取，不进入 gathered output。

### 保持不变（关键非目标）
- kling `_resolve_model_root`；`batch_builder._primary_trainable_segment` / `evaluators/trajectory._primary_segment_name` / `artifacts._release_reward_view_tensors`（三者语义各异：producer 决策 vs 反序优先级 vs best-effort teardown——PARTIAL 复核结论，**不得**折进 resolver）；不在 collector 里实例化 `TrajectoryResolver`（重复校验 + 异常类型漂移）；rollout 构建 tail 不与 replay tail 统一；glm_image/llamagen/nextstep 的 runner 偏差是真实的（本 sprint 与 Sprint 4 都不碰）。

### 验证清单
- `pytest tests/models tests/rewards tests/rollouts tests/scripts tests/config/test_load_all_experiments.py tests/generation`
- 三个 wan T2V 实验 + echo/cosmos3/anima 实验 config-resolve 通过；新 registry 断言（每个 diffusion family 有 replay 路径）绿。
- grep 证空：`_resolve_lora_block`、`_build_predict2_bundle\|_build_nft_bundle`、`train_wan_2_1_grpo`、`_lora_dtype` 的 10 个删除文件、`_named_tensor\b`、`_require_rows`。
- drift 修复①②各补/改一条断言（local_files_only 转发；unified 空 prompt raise）。

---

## Sprint 4 — 训练/rollout 正确性路径（M 级合并 + 验证闸门）

**目标**：三个触碰 RL 正确性（logprob 位精度、replay 类层次、reference 条件路径）的大合并。每项独立 commit、独立闸门，任何 parity 漂移即停。

### 变更清单

1. **flow_matching 采样/回放双拷贝收敛**（发现 #25，**logprob 位一致性关键**）：`vrl/math/ar/flow_matching.py` 同文件内提取 `_flow_terminal_mean(...)`（单一 `_velocity` 闭包 + Euler 前缀环 + 4 处 CFG combine 收成 1 处 + `noise_level*sqrt(dt)` 单一构造点）与 `_isotropic_gaussian_logprob(...)`；`flow_sample_with_logprob` / `flow_logprob_at` 各保留自己的入口逻辑；修 stale Args docstring（删 "fall back to forward"，契约是 `image_head.net`）。公共 API 不变。
   **闸门**：`pytest tests/math/test_ar_flow_matching.py`（既有 sample/replay parity 测试）必须位精确通过——这是 GRPO old/fresh ratio 的防腐线。
2. **`DiffusersReplayModelBase` 落地**（发现 #16，M）：`base.py` 在 `ReplayRolloutStubs` 下方加 5 成员基类（**不放在 ReplayRolloutStubs 上**——Cosmos3ReplayModel 继承它并读 `self.pipeline` shell）；7 家整体删（sana/lumina2/cogvideox/qwen_image/hunyuan_image/hunyuan_video/cosmos-predict2）、3 家只留 `prepare_replay`（flux/mochi/pixart_sigma）、sd3_5 留 ctor+`_set_transformer` override、predict2_5 留 ctor/`set_num_steps`/nft 并**删其同体 `torch_compile_transformer`**（形式四）。wan/echo/anima/cosmos3 不动。与 Sprint 3 第 4/6 项同文件但区域不相交（from_build / `_lora_dtype` / Replay 类），**按 Sprint 3 → Sprint 4 顺序落地**避免 rebase 冲突。
   **闸门**：全家族 backbone-parity 测试 + `tests/models/diffusion/sd3_5/test_attention_processor_install.py`。
3. **Emu3/Janus paged-CFG 状态机提基**（发现 #11，M）：在**现有** `vrl/models/ar/paged_attention_helpers.py` 落 `PagedCFGARState`（kw_only；janus 的 `image_token_num` 统一改名 `total_token_num`，公共 `init_ar` kwarg 不变，3 处测试行同步）与 `PagedCFGTokenRunner`（step_ar/finalize_ar/prefill/validate/sample/`_advance_after_sample` 原样自 janus，以 `family` 类属性 + `_embed_sampled_token` hook 参数化）；两家保留文件、公有类名、`init_ar`、各自 `_sample_cfg_image_token`（含 emu3 结构 mask 与两处 RL-correctness 注释）；统一 sampling hook 签名为 `(state, hidden, position)` 并改两处 2-arg 测试直调。**不碰** glm_image/llamagen/nextstep_1（真实偏差：native mrope 无 CFG；连续 token flow loop）。
   **闸门**：`pytest tests/models/ar/emu3 tests/models/ar/janus_pro tests/generation/ar/test_janus_paged_attention_one_step.py`。
4. **reference 路径解析统一到 load 时**（发现 #2 + #32 合并，**采用 #2 的 load-time 方案，#32 的提取方案随之作废**——两者对同一段代码给了互斥修法，load-time 版一次覆盖全部 5 个散点并顺带修复 executor 无视 `data.artifact_data_root` 的缺陷）：
   - `vrl/scripts/common/online.py` 在 `load_prompt_examples_from_config`（:708）后加一个循环：就地解析 `reference_image`/`reference_video`/`references`（字段 + metadata 镜像，wan 模式），`data_root = data.artifact_data_root or default_data_root()`，`allow_absolute=True`；**行缺 reference 字段时跳过而非 raise**（t2v 配方无 reference），required-ness 校验仍留家族 getter；eval-example 加载同样处理。
   - 删 cosmos `_normalize_per_sample_reference_images` 与 wan `_i2v_collector_kwargs` 的解析循环（getter 缩为家族特有检查：cosmos 的 per_sample↔prompts_per_batch==1 守卫 + global 校验；wan 的 missing-vs-global 报错 + `metadata.setdefault("conditioning","reference_image")`）；删 `executor._reference_image_for_request` 的内联 resolve（:384-393，注释里自述的缺陷随根因消失）。Sprint 2 重指过的 `resolve_artifact_path` import 若因此无消费方，一并 grep 清理。
   - **TARGET 侧一律不动**（PARTIAL 复核的硬约束）：`target_video` 是 sft-latents shard 的 identity key，load 时解析会把 machine-specific 绝对路径写进 metadata，打爆所有既有 shard 查找——在 `PromptExample.target_video`（prompts.py:24）加注释固化此契约；`target_dino_similarity._resolve`（共享 resolver 的合法调用点，per-process data_root 旋钮）与 encode_targets 的 raw-key-then-resolve 序列**保留**；不整体接线 `resolve_prompt_example_artifacts`。
   - **有意行为变化（记录）**：cosmos/executor 从此尊重 `data.artifact_data_root`（videophy_i2v 类配方此前只有 wan 正确）。
   **闸门**：`pytest tests/rollouts/runtime/test_video_world_reference_metadata.py tests/data/test_setup.py tests/scripts/test_predict2_reference_kwargs.py tests/rewards -k dino` + manifest 校验测试；新增一条 "cosmos per_sample 在 artifact_data_root 下可解析" 断言（承接 #32 的测试意图）；跑一次 videophy_i2v smoke config-resolve。

### 保持不变
- wan I2V 收集流程的家族语义、`load_reference_image` 与 global fallback、`resolve_prompt_example_artifacts`（encode_targets 的活跃消费方）；`DiffusionSamplingStateBase` / executor-as-data 相关文件（记忆：另一进程在做薄化第十五轮 reconcile，**本 sprint 不触碰 sampling state/layout 的并行改动面，落地前逐项 git fetch + status**）。

### 验证清单（sprint 级总闸门）
- 全量 `pytest tests/`（Sprint 3 结束时基线已绿，任何新红定位到本 sprint 的三个 commit 之一）。
- 位一致性：flow_matching parity 测试 + 各家族 backbone-parity/replay-export-alignment 测试零漂移。
- grep 证空：`_normalize_per_sample_reference_images`、`_align_replay_tensor`（anima 侧）、`image_token_num`（除公共 kwarg 外）、executor 内联 `resolve_artifact_path`。

---

## 冲突自查表（全计划 sanity check）

| # | 交叉点 | 裁决 |
|---|---|---|
| 1 | 发现 #1/#4/#29/#36 四条同指 artifacts 迁移，import 保留清单互有出入 | 合并为 Sprint 2 单项；最终 import = `ArtifactManifestError, coerce_data_root, resolve_artifact_path`（`default_data_root` 随本地 `_coerce_data_root` 删除而失去消费方，一并删——调和了 #1 与 #29 的分歧） |
| 2 | #18（删 replay_loading.py）vs #28（memory guard 要从 replay_loading import） | 同批落地，#28 的 import 目标改为 `vrl.models.interfaces.runtime`；#28 中对 replay_loading.py:8 的 docstring 编辑作废 |
| 3 | #2（load-time 统一）vs #32（提取共享 normalize 函数）互斥修同一段代码 | 采用 #2（覆盖全部 5 散点 + 修 executor 旋钮缺陷）；#32 只保留其测试重指与 "cosmos 须尊重 artifact_data_root" 要求 |
| 4 | Sprint 2 重指 cosmos/wan train.py 的 artifacts import → Sprint 4 又删其部分消费点 | 顺序无碍；Sprint 4 收尾 grep 未使用 import |
| 5 | #30（builder 收敛）与 #31（wan 入口删除）同文件 | 合并为 Sprint 3 单项，先收敛后删入口，wan 文件终态只剩 i2v 用共享 builder |
| 6 | #15/#38（Sprint 3）与 #16（Sprint 4）触同一批 family model.py | 区域不相交（from_build / _lora_dtype / Replay 类）；固定 Sprint 3→4 顺序 |
| 7 | #5 与 #34 同触 resolver.py | 互补：#34 删 `_lookup_tensor`（Sprint 1），#5 让 `tensor()` delegate `named_tensor`（Sprint 3），无重叠符号 |
| 8 | 并行 worktree 风险（记忆：另一进程 reconcile wm-infra 的 layout.py/sampling-state；用户并行改分支） | Sprint 3 第 9 项与 Sprint 4 全部条目落地前强制 `git fetch` + `git status` 复查；#14 的 registry/build.py 同样先复查 |
| 9 | REJECTED 清单 | `RolloutScheduler` 命名与 `pipeline_runner.py`/`stage_worker.py` 全家在四个 sprint 中均列为非目标，防止后续误清 |

**总账**：Sprint 1 ≈ 11 项纯删（S）；Sprint 2 ≈ 7 项搬移（S）；Sprint 3 ≈ 9 项合并（S 为主 + 1 个 M + 1 个活 bug 修复）；Sprint 4 = 3 个 M 级合并 + 1 个有意行为变化，全部带 parity 闸门。
