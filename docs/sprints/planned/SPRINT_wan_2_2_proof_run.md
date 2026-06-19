# SPRINT: Wan 2.2 A14B GRPO proof run（落地验证）

状态：planned（2026-06-18）。**实现已完成**并归档（`done/SPRINT_wan_2_2_dual_expert.md`：dual-stage
加载 / boundary 路由 / replay `transformer_2` / 默认 low-noise 训练 + 23 项 CPU 单测全过）。本 sprint =
那个实现 sprint 唯一的剩余项：**把代码在真机上跑通一次 Wan 2.2 A14B GRPO，验证 replay 契约与 reward 信号**。

> 拆分理由：实现是"代码 + CPU 单测"（已 done）；proof run 是"真机 + 多卡 + 端到端契约验证"，性质不同、
> 卡在不同资源（GPU），故独立成 sprint。决策依据见 `reading/SPRINT_moe_support_decision.md`。

---

## 0. 当前缺口（为什么还没跑）

| 缺口 | 现状 | 谁能补 |
|---|---|---|
| **experiment 入口** | `configs/experiment/diffusion/wan_2_2/` **不存在**——没有 GRPO recipe，连 launch 路径都没有 | 纯 config，**现在就能做**（任务 1） |
| **2×14B 显存策略** | 实现 sprint §2#6 标"中"：inactive 专家 CPU-offload / generation memory policy 对 dual-stage 的接入未在真机验证 | 接 `done/SPRINT_generation_memory_policy.md` 的系统，需在目标拓扑上验（任务 2） |
| **真机 proof run** | 无 run 证据；2×14B 需多卡 | **阻塞在多卡硬件**（任务 3） |

## 1. 任务 1 — 建 experiment recipe（code-only，可立即做）

按实现 sprint §3.2「proof run 应优先走 I2V」，先建 I2V-A14B 入口：

- `configs/experiment/diffusion/wan_2_2/online_grpo_physics_i2v.yaml`，镜像现有
  `configs/experiment/diffusion/wan_2_1/online_grpo_physics_i2v.yaml`，但 `defaults` 指
  `/model/diffusion/wan_2_2/i2v_a14b`（带 `boundary_ratio` + 默认 low-noise `trainable_transformers`）。
- 显式声明 `model.trainable_transformers`（默认 = low-noise `transformer_2`；要两个专家时设 `both`）。
- rollout/trainer 其余 knob 对齐 wan_2_1 物理 run，先小 `total_epochs` 做 smoke。

## 2. 任务 2 — 2×14B 显存策略（真机前置）

- inactive 专家 CPU-offload：eval 侧已有 `--offload model`（`wan_i2v_base_sample.py`）；训练/rollout 侧需把
  非活跃 transformer 下放，接 `done/SPRINT_generation_memory_policy.md` 的 generation memory policy。
- 在目标多卡拓扑上确认峰值显存（2×14B 不同驻、按 boundary 切换活跃专家）。

## 3. 任务 3 — proof run + 契约验证（阻塞在多卡）

跑一个短 GRPO run，验证实现 sprint §5 的 finishing criteria：

- **硬指标**：rollout 记录的 `old_log_prob` 与训练 replay 重算**逐步一致**，**跨 boundary 不漂移**
  （high/low 专家切换点是最容易 off-by-one 的地方——路由公式 `t >= boundary_ratio *
  num_train_timesteps` 必须 rollout/replay 两侧一致）。
- 短 GRPO run 端到端不崩、reward 可动。
- 拆 guard 后 2.1 单塔路径**回归不变**（不破坏现有 1.3B / I2V-14B 训练）——已有 CPU 单测兜底，真机再确认一次。

## 4. 任务 4 — 下游解锁（proof 通过后）

- **wan_2_2 compile**：proof 通过后即可把 `configs/model/diffusion/wan_2_2/{a14b,i2v_a14b}.yaml` 的
  `torch_compile.enable` 翻 `true`（compile 路径 = 共享 wan DiT，已验证 parity-safe；此前 defer 的唯一
  理由就是"base run 未验"——见 compile 默认态/parity 审计）。

## 5. 验证标准（finishing criteria）

- I2V-A14B GRPO smoke 端到端跑通、reward 曲线可动；
- `old_log_prob` rollout vs replay 跨 boundary 逐步一致（契约硬指标）；
- 2.1 单塔（1.3B / I2V-14B）训练回归不变；
- 目标拓扑峰值显存在预算内（2×14B + offload）。

## 6. 阻塞 / 非目标

- **阻塞**：任务 3 需多卡（2×14B）。任务 1（recipe）/任务 2（显存策略接线）不阻塞，可先行。
- **非目标**：Wan 2.2-5B `expand_timesteps` 变体（机制不同，`_validate_wan_pipeline` 已 fail-loud）；
  任何 expert-parallelism / Megatron MoE kernel（双专家是时间步路由，用不上）。

## 7. 引用

- 实现 sprint（已 done）：`done/SPRINT_wan_2_2_dual_expert.md`
- 决策：`reading/SPRINT_moe_support_decision.md`
- 模型/路由：`vrl/models/diffusion/wan_2_1/model.py`（`_uses_low_noise_transformer` / `boundary_ratio`
  路由 :282-292 / `_normalize_trainable_transformers` / replay context）、`runtime.py`（replay bundle 载 `transformer_2`）
- config：`configs/model/diffusion/wan_2_2/{a14b,i2v_a14b}.yaml`；镜像入口
  `configs/experiment/diffusion/wan_2_1/online_grpo_physics_i2v.yaml`
- 显存：`done/SPRINT_generation_memory_policy.md`、`vrl/scripts/eval/wan_i2v_base_sample.py`（`--offload model`）
- 测试兜底：`tests/models/diffusion/wan_2_1/{test_backbone_parity,test_model_loading}.py`、
  `tests/models/interfaces/test_minimal_replay_runtime_wiring.py`（23 passed）
