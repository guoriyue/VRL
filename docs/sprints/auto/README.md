# Auto Per-File Architecture Audit — 索引

状态: auto-generated (per-file audit). 已解决的条目随实现一并删除。

剩余待处理: 6 个文件。审查基准 = AGENTS.md Architecture Hygiene。

## 汇总

| verdict | 数量 |
|---|---|
| delete | 0 |
| consolidate | 1 |
| question | 0 |
| improve | 5 |

## 🟠 CONSOLIDATE（1）

- **`vrl/models/diffusion/cosmos/anima/model.py`** (847L, core) — 真核心但混了 RL 适配器+手写 text-adapter 网络+装载工具，建议把 adapter nn 模块（`AnimaLLMAdapter`/`TransformerBlock`/`Attention`/`RotaryEmbedding`/`rotate_half`/`apply_rotary_pos_emb`）拆到 `adapter.py`
  → [SPRINT_vrl_models_diffusion_cosmos_anima_model.md](SPRINT_vrl_models_diffusion_cosmos_anima_model.md)

## 🔵 IMPROVE（5）

- **`vrl/algorithms/diffusion_nft.py`** (core) — advantage 归一化与 GRPO 逐行重复应抽共享 helper，`compute_batch_timestep_loss` 偏 god-method 可拆
  → [SPRINT_vrl_algorithms_diffusion_nft.md](SPRINT_vrl_algorithms_diffusion_nft.md)
- **`vrl/generation/ar/executor.py`** (facade) — 10 个方法是对 `self.layout` 的纯单行转发，调用方又混用 `self.layout.xxx`，转发层无边界价值应收敛
  → [SPRINT_vrl_generation_ar_executor.md](SPRINT_vrl_generation_ar_executor.md)
- **`vrl/models/replay_loading.py`** (helper) — 职责过载且文件名误导：应把通用后端加载动作从 replay 元数据中拆出，并对仅测试触达的 parser 路径表态
  → [SPRINT_vrl_models_replay_loading.md](SPRINT_vrl_models_replay_loading.md)
- **`vrl/scripts/common/factory.py`** (core) — 本地 `build_rollout_config_from_cfg` 与上游同名遮蔽（薄 wrapper），改私有名收敛；`__all__` 收窄已撤回（`build_reward_from_cfg`/`build_algorithm_and_evaluator_from_cfg` 被 4 个测试直接 import，须保持公共可 import）
  → [SPRINT_vrl_scripts_common_factory.md](SPRINT_vrl_scripts_common_factory.md)
- **`vrl/scripts/data/bootstrap.py`** (script) — `_MANIFEST_POPULATE_HINTS` 是手抄的路径→命令映射，会与真实子命令脱钩；`--run` 仅半实现只认 pickapic（注：`populate`→`setup` 改名已落地）
  → [SPRINT_vrl_scripts_data_bootstrap.md](SPRINT_vrl_scripts_data_bootstrap.md)
