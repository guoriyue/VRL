# Auto Per-File Architecture Audit — 索引

状态: auto-generated (per-file audit). 已解决的条目随实现一并删除。

剩余待处理: 32 个文件。审查基准 = AGENTS.md Architecture Hygiene。

## 汇总

| verdict | 数量 |
|---|---|
| delete | 1 |
| consolidate | 1 |
| question | 3 |
| improve | 27 |

## 🔴 DELETE（1）

- **`vrl/trainers/fsdp.py`** (184L, dead) — vrl/trainers/fsdp.py — 从 flow_grpo 移植的 FSDP 工具栈，全仓零引用（含 tests/docs/scripts），未接入 __init__，纯死代码应删除  
  → [SPRINT_vrl_trainers_fsdp.md](SPRINT_vrl_trainers_fsdp.md)

## 🟠 CONSOLIDATE（1）

- **`vrl/models/diffusion/cosmos/anima/model.py`** (859L, core) — vrl/models/diffusion/cosmos/anima/model.py — 真核心但 859 行混了 RL 适配器+手写 text-adapter 网络+装载工具，建议把 adapter nn 模块拆到 adapter.py  
  → [SPRINT_vrl_models_diffusion_cosmos_anima_model.md](SPRINT_vrl_models_diffusion_cosmos_anima_model.md)

## 🟡 QUESTION（3）

- **`vrl/rewards/functions/nsfw_safety.py`** (47L, thin-wrapper) — vrl/rewards/functions/nsfw_safety.py — probability_batch/_probability_from_classifier_result 是纯转发到 model 私有，仅 test 调用，泄漏 model 内部到 reward 表面  
  → [SPRINT_vrl_rewards_functions_nsfw_safety.md](SPRINT_vrl_rewards_functions_nsfw_safety.md)
- **`vrl/rewards/functions/ocr.py`** (98L, thin-wrapper) — vrl/rewards/functions/ocr.py — 顶部三个 self-described 'for tests' edit-distance helper 既不被 reward 也不被 model 用，仅 test import，测试工具误入长期模块  
  → [SPRINT_vrl_rewards_functions_ocr.md](SPRINT_vrl_rewards_functions_ocr.md)
- **`vrl/trainers/online/diagnostics.py`** (16L, thin-wrapper) — vrl/trainers/online/diagnostics.py — 纯转发 compat shim，worker.py 直连 canonical 源造成两条 import 路径，疑似可删  
  → [SPRINT_vrl_trainers_online_diagnostics.md](SPRINT_vrl_trainers_online_diagnostics.md)

## 🔵 IMPROVE（27）

- **`vrl/algorithms/diffusion_nft.py`** (305L, core) — vrl/algorithms/diffusion_nft.py — 真核心算法，但 advantage 归一化与 GRPO 逐行重复应抽共享 helper，loss 方法偏 god-method 可拆  
  → [SPRINT_vrl_algorithms_diffusion_nft.md](SPRINT_vrl_algorithms_diffusion_nft.md)
- **`vrl/algorithms/dpo.py`** (182L, thin-wrapper) — vrl/algorithms/dpo.py — 纯函数损失是核心被 offline trainer 调用，但 DiffusionDPO 适配类生产无引用(仅测试)、docstring 用途虚假，应删类  
  → [SPRINT_vrl_algorithms_dpo.md](SPRINT_vrl_algorithms_dpo.md)
- **`vrl/config/validation.py`** (229L, helper) — vrl/config/validation.py — require/path_exists 等是真边界，但 validate_data_config 与 assert_no_missing 是迁移后死代码，前者还与 schema 的 DataConfig 重复维护 sampler 白名单  
  → [SPRINT_vrl_config_validation.md](SPRINT_vrl_config_validation.md)
- **`vrl/generation/ar/executor.py`** (159L, facade) — vrl/generation/ar/executor.py — 10 个方法是对 self.layout 的纯单行转发，调用方又混用 self.layout.xxx，转发层无边界价值应收敛  
  → [SPRINT_vrl_generation_ar_executor.md](SPRINT_vrl_generation_ar_executor.md)
- **`vrl/generation/execution/planner.py`** (480L, core) — vrl/generation/execution/planner.py — 核心契约，但 resolve_executor_capability + _merge_runtime_caps 是无人调用的死代码，应删  
  → [SPRINT_vrl_generation_execution_planner.md](SPRINT_vrl_generation_execution_planner.md)
- **`vrl/generation/execution/scheduler.py`** (118L, core) — vrl/generation/execution/scheduler.py — plan_with_engine 是活路径，但 plan() 薄 wrapper 和 execution_unit 别名是死代码，应删  
  → [SPRINT_vrl_generation_execution_scheduler.md](SPRINT_vrl_generation_execution_scheduler.md)
- **`vrl/models/ar/janus_pro/model.py`** (1357L, core) — vrl/models/ar/janus_pro/model.py — 真核心 wrapper；死字段 _frame_constants 应删，JANUS_R1_SEGMENTS 被 runtime.py 手抄应统一来源  
  → [SPRINT_vrl_models_ar_janus_pro_model.md](SPRINT_vrl_models_ar_janus_pro_model.md)
- **`vrl/models/ar/janus_pro/runtime.py`** (1246L, core) — vrl/models/ar/janus_pro/runtime.py — god-file 塞三条管线，悬空三引号当章节分隔，_call_with_supported_kwargs 与 decode_loop 重复；拆分须跨家族一致  
  → [SPRINT_vrl_models_ar_janus_pro_runtime.md](SPRINT_vrl_models_ar_janus_pro_runtime.md)
- **`vrl/models/ar/nextstep_1/model.py`** (573L, core) — vrl/models/ar/nextstep_1/model.py — 真核心，但 docstring 仍自称 TODO scaffolding（实际只剩1处TODO）且上游 sys.path 引导复制了两份  
  → [SPRINT_vrl_models_ar_nextstep_1_model.md](SPRINT_vrl_models_ar_nextstep_1_model.md)
- **`vrl/models/ar/nextstep_1/runtime.py`** (753L, core) — vrl/models/ar/nextstep_1/runtime.py — 核心 runtime（builders+executor+gatherer，刻意对齐 janus 不该拆），但中段有一个被丢弃的伪 docstring 裸字符串  
  → [SPRINT_vrl_models_ar_nextstep_1_runtime.md](SPRINT_vrl_models_ar_nextstep_1_runtime.md)
- **`vrl/models/diffusion/cosmos/predict2/runtime.py`** (337L, registry-config) — vrl/models/diffusion/cosmos/predict2/runtime.py — runtime builders+executor 合理且跨家族一致；line 222 游离 module-level 字符串是合并残留死语句应删（predict2_5:182 同款）  
  → [SPRINT_vrl_models_diffusion_cosmos_predict2_runtime.md](SPRINT_vrl_models_diffusion_cosmos_predict2_runtime.md)
- **`vrl/models/diffusion/sd3_5/model.py`** (488L, core) — vrl/models/diffusion/sd3_5/model.py — 合格 family model；唯一问题是 _resolve_torch_dtype 与 cosmos/anima 逐字重复，应提到 common/ 共享  
  → [SPRINT_vrl_models_diffusion_sd3_5_model.md](SPRINT_vrl_models_diffusion_sd3_5_model.md)
- **`vrl/models/diffusion/wan_2_1/__init__.py`** (0L, package-init) — vrl/models/diffusion/wan_2_1/__init__.py — 唯一 0 字节的家族 init，缺兄弟家族都有的描述 docstring，补一行即可（勿加 re-export，会破坏 lazy backend import）  
  → [SPRINT_vrl_models_diffusion_wan_2_1___init__.md](SPRINT_vrl_models_diffusion_wan_2_1___init__.md)
- **`vrl/models/replay_loading.py`** (361L, helper) — vrl/models/replay_loading.py — 职责过载且文件名误导：应把通用后端加载动作从 replay 元数据中拆出，并对仅测试触达的 parser 路径表态  
  → [SPRINT_vrl_models_replay_loading.md](SPRINT_vrl_models_replay_loading.md)
- **`vrl/rewards/inference.py`** (321L, interface-boundary) — vrl/rewards/inference.py — reward 推理契约边界，整体合理；media_type 合法值集合在 inference/artifacts 两处手抄，应抽单一来源  
  → [SPRINT_vrl_rewards_inference.md](SPRINT_vrl_rewards_inference.md)
- **`vrl/rewards/models/kling_video_reward.py`** (776L, core) — vrl/rewards/models/kling_video_reward.py — cohesive self-owned VideoAlign-derived model adapter; ALL_CAPS are real boundaries, but _torch_dtype duplicates base.resolve_dtype and prompt-template table could be externalized  
  → [SPRINT_vrl_rewards_models_kling_video_reward.md](SPRINT_vrl_rewards_models_kling_video_reward.md)
- **`vrl/rewards/types.py`** (40L, core) — vrl/rewards/types.py — RewardTrajectoryStep 从未实例化、steps/seed 全调用点恒为[]/0 的死字段，应删收敛  
  → [SPRINT_vrl_rewards_types.md](SPRINT_vrl_rewards_types.md)
- **`vrl/rollouts/collector/batch_builder.py`** (420L, core) — vrl/rollouts/collector/batch_builder.py — 真核心转换器，但 RolloutBatchBuildContext.extra 是死字段且类名带引擎味(Context)与周边 Runtime 概念碰撞  
  → [SPRINT_vrl_rollouts_collector_batch_builder.md](SPRINT_vrl_rollouts_collector_batch_builder.md)
- **`vrl/scripts/ar/janus_pro/train.py`** (108L, script) — vrl/scripts/ar/janus_pro/train.py — 模块是合法 family entrypoint（跨家族与 nextstep_1 一致，保留），但 train_janus_pro_grpo 无任何 config/test 引用且与 ocr 版字节重复，应删除该死 export  
  → [SPRINT_vrl_scripts_ar_janus_pro_train.md](SPRINT_vrl_scripts_ar_janus_pro_train.md)
- **`vrl/scripts/common/factory.py`** (332L, core) — vrl/scripts/common/factory.py — 装配中枢是真核心，但本地 build_rollout_config_from_cfg 与上游同名遮蔽(薄 wrapper)、且 __all__ 公开面过宽(7个符号仅内部用)，需收窄+改名  
  → [SPRINT_vrl_scripts_common_factory.md](SPRINT_vrl_scripts_common_factory.md)
- **`vrl/scripts/data/bootstrap.py`** (115L, script) — vrl/scripts/data/bootstrap.py — _MANIFEST_POPULATE_HINTS 是手抄的路径→命令映射，会与 populate/danbooru 真实子命令脱钩；--run 仅半实现只认 pickapic  
  → [SPRINT_vrl_scripts_data_bootstrap.md](SPRINT_vrl_scripts_data_bootstrap.md)
- **`vrl/scripts/data/danbooru.py`** (1858L, script) — vrl/scripts/data/danbooru.py — 1858 行 god-file 混 4 条管线；确证死代码(_load_reward_rollouts/_metric_report/_pairwise_auc/_mean)与无人用的 back-compat alias 应删；ALL_CAPS taxonomy 是 justified 不动  
  → [SPRINT_vrl_scripts_data_danbooru.md](SPRINT_vrl_scripts_data_danbooru.md)
- **`vrl/scripts/diffusion/wan_2_1/train_dpo.py`** (325L, script) — vrl/scripts/diffusion/wan_2_1/train_dpo.py — 真实在用的离线 DPO 驱动，但 CSV header 手抄 DPOStepMetrics 字段(会漂移)、export 块重复两遍、单函数职责过载  
  → [SPRINT_vrl_scripts_diffusion_wan_2_1_train_dpo.md](SPRINT_vrl_scripts_diffusion_wan_2_1_train_dpo.md)
- **`vrl/trainers/data/artifacts.py`** (373L, core) — vrl/trainers/data/artifacts.py — 核心 manifest 校验，但 repo_root/default_data_root 与 scripts/data/common.py 逐字重复，应收敛单一来源  
  → [SPRINT_vrl_trainers_data_artifacts.md](SPRINT_vrl_trainers_data_artifacts.md)
- **`vrl/trainers/online/trainer.py`** (899L, core) — vrl/trainers/online/trainer.py — 核心训练循环，但 _step_impl 内嵌一次性 grad-split/first-step 调试脚手架(写/tmp、三处重复样板)，应抽离隔离  
  → [SPRINT_vrl_trainers_online_trainer.md](SPRINT_vrl_trainers_online_trainer.md)
- **`vrl/trajectory/storage.py`** (179L, core) — vrl/trajectory/storage.py — 核心 storage policy，但 _VALID_DEVICES/_VALID_DTYPES frozenset 手抄了 Literal 成员，应 get_args derive  
  → [SPRINT_vrl_trajectory_storage.md](SPRINT_vrl_trajectory_storage.md)
- **`vrl/trajectory/validation.py`** (417L, interface-boundary) — vrl/trajectory/validation.py — 核心契约校验器(真边界,保留);必需角色三元组 (action/old_log_prob/mask) 在同文件写两遍(L26 frozenset + L233 字面 tuple)应统一,且 _looks_runtime_only 末尾 ray/actor 判断是恒为真的死分支  
  → [SPRINT_vrl_trajectory_validation.md](SPRINT_vrl_trajectory_validation.md)
