# Auto Per-File Architecture Audit — 索引

状态: auto-generated (per-file audit). 已解决的条目随实现一并删除。

剩余待处理: 12 个文件。审查基准 = AGENTS.md Architecture Hygiene。

## 汇总

| verdict | 数量 |
|---|---|
| delete | 0 |
| consolidate | 1 |
| question | 2 |
| improve | 9 |

## 🟠 CONSOLIDATE（1）

- **`vrl/models/diffusion/cosmos/anima/model.py`** (847L, core) — 真核心但混了 RL 适配器+手写 text-adapter 网络+装载工具，建议把 adapter nn 模块（`AnimaLLMAdapter`/`TransformerBlock`/`Attention`/`RotaryEmbedding`/`rotate_half`/`apply_rotary_pos_emb`）拆到 `adapter.py`
  → [SPRINT_vrl_models_diffusion_cosmos_anima_model.md](SPRINT_vrl_models_diffusion_cosmos_anima_model.md)

## 🟡 QUESTION（2）

- **`vrl/rewards/functions/nsfw_safety.py`** — `probability_batch`/`_probability_from_classifier_result` 是纯转发到 model 私有，仅 test 调用，泄漏 model 内部到 reward 表面
  → [SPRINT_vrl_rewards_functions_nsfw_safety.md](SPRINT_vrl_rewards_functions_nsfw_safety.md)
- **`vrl/rewards/functions/ocr.py`** — 顶部三个 'for tests' edit-distance helper 既不被 reward 也不被 model 用，仅 test import，测试工具误入长期模块
  → [SPRINT_vrl_rewards_functions_ocr.md](SPRINT_vrl_rewards_functions_ocr.md)

## 🔵 IMPROVE（9）

- **`vrl/algorithms/diffusion_nft.py`** (core) — advantage 归一化与 GRPO 逐行重复应抽共享 helper，`compute_batch_timestep_loss` 偏 god-method 可拆
  → [SPRINT_vrl_algorithms_diffusion_nft.md](SPRINT_vrl_algorithms_diffusion_nft.md)
- **`vrl/algorithms/dpo.py`** — ⚠️ 复核结论：`DiffusionDPO` 类是**活的离线适配器**（schema/builders/offline trainer 在用、有真测试），原审计"应删类"已被否决；仅 docstring 措辞可校正
  → [SPRINT_vrl_algorithms_dpo.md](SPRINT_vrl_algorithms_dpo.md)
- **`vrl/generation/ar/executor.py`** (facade) — 10 个方法是对 `self.layout` 的纯单行转发，调用方又混用 `self.layout.xxx`，转发层无边界价值应收敛
  → [SPRINT_vrl_generation_ar_executor.md](SPRINT_vrl_generation_ar_executor.md)
- **`vrl/models/replay_loading.py`** (helper) — 职责过载且文件名误导：应把通用后端加载动作从 replay 元数据中拆出，并对仅测试触达的 parser 路径表态
  → [SPRINT_vrl_models_replay_loading.md](SPRINT_vrl_models_replay_loading.md)
- **`vrl/rewards/models/kling_video_reward.py`** (core) — dtype 重复已统一到 `vrl/models/dtypes.py`；剩**可选**的 prompt-template 表外置
  → [SPRINT_vrl_rewards_models_kling_video_reward.md](SPRINT_vrl_rewards_models_kling_video_reward.md)
- **`vrl/rollouts/collector/batch_builder.py`** (core) — 死字段 `extra` 已删、packers 已合并；剩**可选**的 `RolloutBatchBuildContext`→`...Spec` 改名（口味）
  → [SPRINT_vrl_rollouts_collector_batch_builder.md](SPRINT_vrl_rollouts_collector_batch_builder.md)
- **`vrl/scripts/common/factory.py`** (core) — 本地 `build_rollout_config_from_cfg` 与上游同名遮蔽（薄 wrapper）；`__all__` 过宽（注：`build_reward_from_cfg`/`build_algorithm_and_evaluator_from_cfg` 实被测试引用，须保持可 import）
  → [SPRINT_vrl_scripts_common_factory.md](SPRINT_vrl_scripts_common_factory.md)
- **`vrl/scripts/data/bootstrap.py`** (script) — `_MANIFEST_POPULATE_HINTS` 是手抄的路径→命令映射，会与真实子命令脱钩；`--run` 仅半实现只认 pickapic（注：`populate`→`setup` 改名已落地）
  → [SPRINT_vrl_scripts_data_bootstrap.md](SPRINT_vrl_scripts_data_bootstrap.md)
- **`vrl/scripts/diffusion/wan_2_1/train_dpo.py`** (script) — CSV header drift 已修（`fields(DPOStepMetrics)`）；剩 export 块重复两遍、单函数职责过载
  → [SPRINT_vrl_scripts_diffusion_wan_2_1_train_dpo.md](SPRINT_vrl_scripts_diffusion_wan_2_1_train_dpo.md)
