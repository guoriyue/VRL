# Auto Per-File Architecture Audit — 索引

状态: auto-generated (per-file audit workflow, 72 agents, 262 files)

本目录下每个 `SPRINT_<path>.md` 对应一个被质疑/可改进的 `.py` 文件。审查基准 = AGENTS.md Architecture Hygiene。

## 汇总

| verdict | 数量 | 含义 |
|---|---|---|
| delete | 5 | grep 确认无引用的死代码，可删 |
| consolidate | 5 | 重复/可合并到别处 |
| question | 5 | 存在性存疑，需拍板 |
| improve | 41 | 真核心但有具体改进点（命名/拆分/derive/裁字段）|
| keep-justified | 206 | 审过无问题，未写 sprint |

## 🔴 DELETE — 死代码（5）

> **人工 grep 核实状态（2026-05-31）**：cli.py / torch.py(`TorchAttentionKernel`) / base.py(`AttentionCacheView`) / collection.py(3 helpers) — 均确认非自身引用仅剩 `__init__` re-export 或 0，**可删**。`TorchSDPAAttentionKernel`（sdpa.py，被 joint.py 使用）是 live 的，不要误删。
> **⚠️ fsdp.py 例外**：grep 确认当前 0 引用属实，但 `SPRINT_multi_gpu_training.md` 计划把它当 FSDP 地基接入。**删 vs 留是 judgment call**——若多卡训练近期要做则保留+补测，否则删。不要无脑删。

- **`vrl/config/cli.py`** (33 LOC, role=dead) — vrl/config/cli.py — 死代码，parse_and_load 零调用，所有脚本各自手搓 argparse，应删并清 __init__ re-export
  → [SPRINT_vrl_config_cli.md](SPRINT_vrl_config_cli.md)
- **`vrl/nn/kernels/attention/torch.py`** (45 LOC, role=dead) — vrl/nn/kernels/attention/torch.py — 死代码：TorchAttentionKernel 仅被 __init__ re-export，无人实例化，且与 sdpa.py 重复，应删
  → [SPRINT_vrl_nn_kernels_attention_torch.md](SPRINT_vrl_nn_kernels_attention_torch.md)
- **`vrl/nn/layers/attention/base.py`** (20 LOC, role=dead) — vrl/nn/layers/attention/base.py — AttentionCacheView 全仓无消费者，死代码，整文件可删
  → [SPRINT_vrl_nn_layers_attention_base.md](SPRINT_vrl_nn_layers_attention_base.md)
- **`vrl/trainers/fsdp.py`** (184 LOC, role=dead) — vrl/trainers/fsdp.py — 从 flow_grpo 移植的 FSDP 工具栈，全仓零引用（含 tests/docs/scripts），未接入 __init__，纯死代码应删除
  → [SPRINT_vrl_trainers_fsdp.md](SPRINT_vrl_trainers_fsdp.md)
- **`vrl/trainers/online/collection.py`** (37 LOC, role=dead) — vrl/trainers/online/collection.py — 三个 helper 全无调用方，逻辑已被 rollouts/orchestration/lifecycle.py 取代，死代码可删
  → [SPRINT_vrl_trainers_online_collection.md](SPRINT_vrl_trainers_online_collection.md)

## 🟠 CONSOLIDATE — 可合并（5）

- **`vrl/models/diffusion/cosmos/anima/model.py`** (859 LOC, role=core) — vrl/models/diffusion/cosmos/anima/model.py — 真核心但 859 行混了 RL 适配器+手写 text-adapter 网络+装载工具，建议把 adapter nn 模块拆到 adapter.py
  → [SPRINT_vrl_models_diffusion_cosmos_anima_model.md](SPRINT_vrl_models_diffusion_cosmos_anima_model.md)
- **`vrl/rewards/functions/video_reward.py`** (123 LOC, role=thin-wrapper) — vrl/rewards/functions/video_reward.py — 与 videocon_physics.py 的 __init__/_actor_runtime/_normalize_worker_config 逐字重复，应抽 RayVideoReward 基类，本文件留薄子类
  → [SPRINT_vrl_rewards_functions_video_reward.md](SPRINT_vrl_rewards_functions_video_reward.md)
- **`vrl/rewards/functions/videocon_physics.py`** (127 LOC, role=thin-wrapper) — vrl/rewards/functions/videocon_physics.py — 与 video_reward.py 100% 结构复制（仅 model factory/默认值不同），非平凡的 worker_config 归一化逻辑被抄两份，应合并到共享基类
  → [SPRINT_vrl_rewards_functions_videocon_physics.md](SPRINT_vrl_rewards_functions_videocon_physics.md)
- **`vrl/rewards/models/claude_image_qa.py`** (472 LOC, role=helper) — vrl/rewards/models/claude_image_qa.py — ~120 lines of judge-parsing/command-render helpers duplicated with codex_image_qa.py and already diverging; extract shared module, keep claude transport
  → [SPRINT_vrl_rewards_models_claude_image_qa.md](SPRINT_vrl_rewards_models_claude_image_qa.md)
- **`vrl/rewards/models/codex_image_qa.py`** (241 LOC, role=helper) — vrl/rewards/models/codex_image_qa.py — score-parsing/render helpers are near-verbatim copies of claude_image_qa.py (codex is a subset); consolidate into shared cli_image_qa_judge module
  → [SPRINT_vrl_rewards_models_codex_image_qa.md](SPRINT_vrl_rewards_models_codex_image_qa.md)

## 🟡 QUESTION — 存在性存疑（5）

- **`vrl/math/ar/flow_matching.py`** (257 LOC, role=core) — vrl/math/ar/flow_matching.py — 真核心(被 nextstep runner/model 调用)，但上游 velocity 契约未确认绑定(# TODO(nextstep-binding))且 sample/replay 两路解析不一致，存静默错配风险
  → [SPRINT_vrl_math_ar_flow_matching.md](SPRINT_vrl_math_ar_flow_matching.md)
- **`vrl/rewards/functions/nsfw_safety.py`** (47 LOC, role=thin-wrapper) — vrl/rewards/functions/nsfw_safety.py — probability_batch/_probability_from_classifier_result 是纯转发到 model 私有，仅 test 调用，泄漏 model 内部到 reward 表面
  → [SPRINT_vrl_rewards_functions_nsfw_safety.md](SPRINT_vrl_rewards_functions_nsfw_safety.md)
- **`vrl/rewards/functions/ocr.py`** (98 LOC, role=thin-wrapper) — vrl/rewards/functions/ocr.py — 顶部三个 self-described 'for tests' edit-distance helper 既不被 reward 也不被 model 用，仅 test import，测试工具误入长期模块
  → [SPRINT_vrl_rewards_functions_ocr.md](SPRINT_vrl_rewards_functions_ocr.md)
- **`vrl/scripts/data/setup.py`** (135 LOC, role=script) — vrl/scripts/data/setup.py — 与 populate.py 入口语义重叠（anime 子命令薄重复 anime-prompts），setup 命名泛，需确认两入口分工
  → [SPRINT_vrl_scripts_data_setup.md](SPRINT_vrl_scripts_data_setup.md)
- **`vrl/trainers/online/diagnostics.py`** (16 LOC, role=thin-wrapper) — vrl/trainers/online/diagnostics.py — 纯转发 compat shim，worker.py 直连 canonical 源造成两条 import 路径，疑似可删
  → [SPRINT_vrl_trainers_online_diagnostics.md](SPRINT_vrl_trainers_online_diagnostics.md)

## 🔵 IMPROVE — 可改进（41）

- **`vrl/algorithms/diffusion_nft.py`** (305 LOC, role=core) — vrl/algorithms/diffusion_nft.py — 真核心算法，但 advantage 归一化与 GRPO 逐行重复应抽共享 helper，loss 方法偏 god-method 可拆
  → [SPRINT_vrl_algorithms_diffusion_nft.md](SPRINT_vrl_algorithms_diffusion_nft.md)
- **`vrl/algorithms/dpo.py`** (182 LOC, role=thin-wrapper) — vrl/algorithms/dpo.py — 纯函数损失是核心被 offline trainer 调用，但 DiffusionDPO 适配类生产无引用(仅测试)、docstring 用途虚假，应删类
  → [SPRINT_vrl_algorithms_dpo.md](SPRINT_vrl_algorithms_dpo.md)
- **`vrl/config/validation.py`** (229 LOC, role=helper) — vrl/config/validation.py — require/path_exists 等是真边界，但 validate_data_config 与 assert_no_missing 是迁移后死代码，前者还与 schema 的 DataConfig 重复维护 sampler 白名单
  → [SPRINT_vrl_config_validation.md](SPRINT_vrl_config_validation.md)
- **`vrl/generation/ar/executor.py`** (159 LOC, role=facade) — vrl/generation/ar/executor.py — 10 个方法是对 self.layout 的纯单行转发，调用方又混用 self.layout.xxx，转发层无边界价值应收敛
  → [SPRINT_vrl_generation_ar_executor.md](SPRINT_vrl_generation_ar_executor.md)
- **`vrl/generation/capabilities.py`** (343 LOC, role=core) — vrl/generation/capabilities.py — 核心 planner 契约（重用广），但 ExecutionStageCapability.to_dict 写 profiler_label 而 from_value 读 profiler_name，round-trip 数据腐烂 bug
  → [SPRINT_vrl_generation_capabilities.md](SPRINT_vrl_generation_capabilities.md)
- **`vrl/generation/diffusion/executor.py`** (711 LOC, role=core) — vrl/generation/diffusion/executor.py — 5 个 family 共享的真核心基类，但 run_denoise_chunk 是从未被调用的死方法，应删
  → [SPRINT_vrl_generation_diffusion_executor.md](SPRINT_vrl_generation_diffusion_executor.md)
- **`vrl/generation/diffusion/layout.py`** (315 LOC, role=core) — vrl/generation/diffusion/layout.py — DiffusionRequestLayout 是真核心，但 VideoGenerationRequest 携带约 13 个全仓库无人读取的遗留死字段，应裁剪
  → [SPRINT_vrl_generation_diffusion_layout.md](SPRINT_vrl_generation_diffusion_layout.md)
- **`vrl/generation/execution/planner.py`** (480 LOC, role=core) — vrl/generation/execution/planner.py — 核心契约，但 resolve_executor_capability + _merge_runtime_caps 是无人调用的死代码，应删
  → [SPRINT_vrl_generation_execution_planner.md](SPRINT_vrl_generation_execution_planner.md)
- **`vrl/generation/execution/scheduler.py`** (118 LOC, role=core) — vrl/generation/execution/scheduler.py — plan_with_engine 是活路径，但 plan() 薄 wrapper 和 execution_unit 别名是死代码，应删
  → [SPRINT_vrl_generation_execution_scheduler.md](SPRINT_vrl_generation_execution_scheduler.md)
- **`vrl/generation/ray/launcher.py`** (409 LOC, role=core) — vrl/generation/ray/launcher.py — 核心启动逻辑，但 __all__ 死再导出 RayPlacement/create_generation_placement_group（外部零引用），且 _cfg_get/_cfg_path 是全仓 ~10 份重复 helper 之一
  → [SPRINT_vrl_generation_ray_launcher.md](SPRINT_vrl_generation_ray_launcher.md)
- **`vrl/models/ar/janus_pro/model.py`** (1357 LOC, role=core) — vrl/models/ar/janus_pro/model.py — 真核心 wrapper；死字段 _frame_constants 应删，JANUS_R1_SEGMENTS 被 runtime.py 手抄应统一来源
  → [SPRINT_vrl_models_ar_janus_pro_model.md](SPRINT_vrl_models_ar_janus_pro_model.md)
- **`vrl/models/ar/janus_pro/runtime.py`** (1246 LOC, role=core) — vrl/models/ar/janus_pro/runtime.py — god-file 塞三条管线，悬空三引号当章节分隔，_call_with_supported_kwargs 与 decode_loop 重复；拆分须跨家族一致
  → [SPRINT_vrl_models_ar_janus_pro_runtime.md](SPRINT_vrl_models_ar_janus_pro_runtime.md)
- **`vrl/models/ar/nextstep_1/model.py`** (573 LOC, role=core) — vrl/models/ar/nextstep_1/model.py — 真核心，但 docstring 仍自称 TODO scaffolding（实际只剩1处TODO）且上游 sys.path 引导复制了两份
  → [SPRINT_vrl_models_ar_nextstep_1_model.md](SPRINT_vrl_models_ar_nextstep_1_model.md)
- **`vrl/models/ar/nextstep_1/runtime.py`** (753 LOC, role=core) — vrl/models/ar/nextstep_1/runtime.py — 核心 runtime（builders+executor+gatherer，刻意对齐 janus 不该拆），但中段有一个被丢弃的伪 docstring 裸字符串
  → [SPRINT_vrl_models_ar_nextstep_1_runtime.md](SPRINT_vrl_models_ar_nextstep_1_runtime.md)
- **`vrl/models/diffusion/cosmos/predict2/model.py`** (694 LOC, role=core) — vrl/models/diffusion/cosmos/predict2/model.py — 真核心；restore_eval_state 在 base 与 ReplayModel 间几乎逐字重复（仅 scheduler 来源不同），可合并；替换张量 helper 属跨家族一致须保留
  → [SPRINT_vrl_models_diffusion_cosmos_predict2_model.md](SPRINT_vrl_models_diffusion_cosmos_predict2_model.md)
- **`vrl/models/diffusion/cosmos/predict2/runtime.py`** (337 LOC, role=registry-config) — vrl/models/diffusion/cosmos/predict2/runtime.py — runtime builders+executor 合理且跨家族一致；line 222 游离 module-level 字符串是合并残留死语句应删（predict2_5:182 同款）
  → [SPRINT_vrl_models_diffusion_cosmos_predict2_runtime.md](SPRINT_vrl_models_diffusion_cosmos_predict2_runtime.md)
- **`vrl/models/diffusion/sd3_5/model.py`** (488 LOC, role=core) — vrl/models/diffusion/sd3_5/model.py — 合格 family model；唯一问题是 _resolve_torch_dtype 与 cosmos/anima 逐字重复，应提到 common/ 共享
  → [SPRINT_vrl_models_diffusion_sd3_5_model.md](SPRINT_vrl_models_diffusion_sd3_5_model.md)
- **`vrl/models/diffusion/wan_2_1/__init__.py`** (0 LOC, role=package-init) — vrl/models/diffusion/wan_2_1/__init__.py — 唯一 0 字节的家族 init，缺兄弟家族都有的描述 docstring，补一行即可（勿加 re-export，会破坏 lazy backend import）
  → [SPRINT_vrl_models_diffusion_wan_2_1___init__.md](SPRINT_vrl_models_diffusion_wan_2_1___init__.md)
- **`vrl/models/replay_loading.py`** (361 LOC, role=helper) — vrl/models/replay_loading.py — 职责过载且文件名误导：应把通用后端加载动作从 replay 元数据中拆出，并对仅测试触达的 parser 路径表态
  → [SPRINT_vrl_models_replay_loading.md](SPRINT_vrl_models_replay_loading.md)
- **`vrl/nn/layers/attention/__init__.py`** (30 LOC, role=package-init) — vrl/nn/layers/attention/__init__.py — 包门面无人使用且 re-export 了死代码 AttentionCacheView，应摘除死项；joint/paged 门面可留
  → [SPRINT_vrl_nn_layers_attention___init__.md](SPRINT_vrl_nn_layers_attention___init__.md)
- **`vrl/rewards/functions/claude_image_qa.py`** (74 LOC, role=thin-wrapper) — vrl/rewards/functions/claude_image_qa.py — wrapper 本体 justified，但 __all__ re-export 的 DEFAULT_*/_extract/_render 是死 re-export，无外部消费者，应收窄
  → [SPRINT_vrl_rewards_functions_claude_image_qa.md](SPRINT_vrl_rewards_functions_claude_image_qa.md)
- **`vrl/rewards/functions/codex_image_qa.py`** (59 LOC, role=thin-wrapper) — vrl/rewards/functions/codex_image_qa.py — import 了 _extract/_render 却 body 未用，仅死 re-export 进 __all__，会触发 F401，应收窄
  → [SPRINT_vrl_rewards_functions_codex_image_qa.md](SPRINT_vrl_rewards_functions_codex_image_qa.md)
- **`vrl/rewards/functions/registry.py`** (137 LOC, role=registry-config) — vrl/rewards/functions/registry.py — MultiReward+内置表是真核心/真 taxonomy，但 register_reward() 导出却零调用是死 API，应删或让 _register_builtins 真正用它
  → [SPRINT_vrl_rewards_functions_registry.md](SPRINT_vrl_rewards_functions_registry.md)
- **`vrl/rewards/inference.py`** (321 LOC, role=interface-boundary) — vrl/rewards/inference.py — reward 推理契约边界，整体合理；media_type 合法值集合在 inference/artifacts 两处手抄，应抽单一来源
  → [SPRINT_vrl_rewards_inference.md](SPRINT_vrl_rewards_inference.md)
- **`vrl/rewards/models/kling_video_reward.py`** (776 LOC, role=core) — vrl/rewards/models/kling_video_reward.py — cohesive self-owned VideoAlign-derived model adapter; ALL_CAPS are real boundaries, but _torch_dtype duplicates base.resolve_dtype and prompt-template table could be externalized
  → [SPRINT_vrl_rewards_models_kling_video_reward.md](SPRINT_vrl_rewards_models_kling_video_reward.md)
- **`vrl/rewards/types.py`** (40 LOC, role=core) — vrl/rewards/types.py — RewardTrajectoryStep 从未实例化、steps/seed 全调用点恒为[]/0 的死字段，应删收敛
  → [SPRINT_vrl_rewards_types.md](SPRINT_vrl_rewards_types.md)
- **`vrl/rollouts/collector/batch_builder.py`** (420 LOC, role=core) — vrl/rollouts/collector/batch_builder.py — 真核心转换器，但 RolloutBatchBuildContext.extra 是死字段且类名带引擎味(Context)与周边 Runtime 概念碰撞
  → [SPRINT_vrl_rollouts_collector_batch_builder.md](SPRINT_vrl_rollouts_collector_batch_builder.md)
- **`vrl/rollouts/collector/requests.py`** (124 LOC, role=core) — vrl/rollouts/collector/requests.py — 核心请求适配器没问题，但 RolloutEngineRequestBuilder 别名是改名遗留薄别名，仅测试在用，应删
  → [SPRINT_vrl_rollouts_collector_requests.md](SPRINT_vrl_rollouts_collector_requests.md)
- **`vrl/rollouts/orchestration/types.py`** (105 LOC, role=interface-boundary) — vrl/rollouts/orchestration/types.py — 整体是合理共享类型边界，但 RolloutScheduleState 有两个死字段(current_policy_version/pending_rollout)应删
  → [SPRINT_vrl_rollouts_orchestration_types.md](SPRINT_vrl_rollouts_orchestration_types.md)
- **`vrl/scripts/ar/janus_pro/train.py`** (108 LOC, role=script) — vrl/scripts/ar/janus_pro/train.py — 模块是合法 family entrypoint（跨家族与 nextstep_1 一致，保留），但 train_janus_pro_grpo 无任何 config/test 引用且与 ocr 版字节重复，应删除该死 export
  → [SPRINT_vrl_scripts_ar_janus_pro_train.md](SPRINT_vrl_scripts_ar_janus_pro_train.md)
- **`vrl/scripts/common/__init__.py`** (34 LOC, role=package-init) — vrl/scripts/common/__init__.py — 包级 facade 全仓库零引用，所有消费者直接 import 子模块；应收缩为 marker 或只暴露 run_online_recipe+OnlineRecipeDefinition
  → [SPRINT_vrl_scripts_common___init__.md](SPRINT_vrl_scripts_common___init__.md)
- **`vrl/scripts/common/factory.py`** (332 LOC, role=core) — vrl/scripts/common/factory.py — 装配中枢是真核心，但本地 build_rollout_config_from_cfg 与上游同名遮蔽(薄 wrapper)、且 __all__ 公开面过宽(7个符号仅内部用)，需收窄+改名
  → [SPRINT_vrl_scripts_common_factory.md](SPRINT_vrl_scripts_common_factory.md)
- **`vrl/scripts/data/bootstrap.py`** (115 LOC, role=script) — vrl/scripts/data/bootstrap.py — _MANIFEST_POPULATE_HINTS 是手抄的路径→命令映射，会与 populate/danbooru 真实子命令脱钩；--run 仅半实现只认 pickapic
  → [SPRINT_vrl_scripts_data_bootstrap.md](SPRINT_vrl_scripts_data_bootstrap.md)
- **`vrl/scripts/data/danbooru.py`** (1858 LOC, role=script) — vrl/scripts/data/danbooru.py — 1858 行 god-file 混 4 条管线；确证死代码(_load_reward_rollouts/_metric_report/_pairwise_auc/_mean)与无人用的 back-compat alias 应删；ALL_CAPS taxonomy 是 justified 不动
  → [SPRINT_vrl_scripts_data_danbooru.md](SPRINT_vrl_scripts_data_danbooru.md)
- **`vrl/scripts/diffusion/wan_2_1/train_dpo.py`** (325 LOC, role=script) — vrl/scripts/diffusion/wan_2_1/train_dpo.py — 真实在用的离线 DPO 驱动，但 CSV header 手抄 DPOStepMetrics 字段(会漂移)、export 块重复两遍、单函数职责过载
  → [SPRINT_vrl_scripts_diffusion_wan_2_1_train_dpo.md](SPRINT_vrl_scripts_diffusion_wan_2_1_train_dpo.md)
- **`vrl/trainers/data/artifacts.py`** (373 LOC, role=core) — vrl/trainers/data/artifacts.py — 核心 manifest 校验，但 repo_root/default_data_root 与 scripts/data/common.py 逐字重复，应收敛单一来源
  → [SPRINT_vrl_trainers_data_artifacts.md](SPRINT_vrl_trainers_data_artifacts.md)
- **`vrl/trainers/online/__init__.py`** (17 LOC, role=package-init) — vrl/trainers/online/__init__.py — re-exports dead collection helpers + 私有 _validate_ema 进 __all__，外部只用 OnlineTrainer，应收窄
  → [SPRINT_vrl_trainers_online___init__.md](SPRINT_vrl_trainers_online___init__.md)
- **`vrl/trainers/online/trainer.py`** (899 LOC, role=core) — vrl/trainers/online/trainer.py — 核心训练循环，但 _step_impl 内嵌一次性 grad-split/first-step 调试脚手架(写/tmp、三处重复样板)，应抽离隔离
  → [SPRINT_vrl_trainers_online_trainer.md](SPRINT_vrl_trainers_online_trainer.md)
- **`vrl/trainers/weight_sync.py`** (184 LOC, role=interface-boundary) — vrl/trainers/weight_sync.py — 整体是活跃的 trainer↔rollout 同步边界，但 InMemoryWeightSyncer 全仓从未实例化，是死的占位类应删除
  → [SPRINT_vrl_trainers_weight_sync.md](SPRINT_vrl_trainers_weight_sync.md)
- **`vrl/trajectory/storage.py`** (179 LOC, role=core) — vrl/trajectory/storage.py — 核心 storage policy，但 _VALID_DEVICES/_VALID_DTYPES frozenset 手抄了 Literal 成员，应 get_args derive
  → [SPRINT_vrl_trajectory_storage.md](SPRINT_vrl_trajectory_storage.md)
- **`vrl/trajectory/validation.py`** (417 LOC, role=interface-boundary) — vrl/trajectory/validation.py — 核心契约校验器(真边界,保留);必需角色三元组 (action/old_log_prob/mask) 在同文件写两遍(L26 frozenset + L233 字面 tuple)应统一,且 _looks_runtime_only 末尾 ray/actor 判断是恒为真的死分支
  → [SPRINT_vrl_trajectory_validation.md](SPRINT_vrl_trajectory_validation.md)
