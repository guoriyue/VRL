# SPRINT: Wan 2.2 双专家（A14B）RL 支持

状态：**done（实现已落地 main：43dbb6e/d622406/7e3e072，23 项 CPU 单测全过；2026-06-18 归档至 done/）**。A14B dual-stage 模型加载、timestep 路由（`_uses_low_noise_transformer`/`_validate_wan_pipeline`）、replay runtime（`transformer_2`）、默认 low-noise 专家训练（`_normalize_trainable_transformers`）、A14B config（`configs/model/diffusion/wan_2_2/`）与 23 项 CPU 单测均已落地并经核实（`tests/models/diffusion/wan_2_1/{test_backbone_parity,test_model_loading}.py` + `tests/models/interfaces/test_minimal_replay_runtime_wiring.py` = 23 passed）。**剩余的真实 Wan 2.2 A14B GRPO proof run 已拆出独立 sprint：`planned/SPRINT_wan_2_2_proof_run.md`**（缺 `configs/experiment/diffusion/wan_2_2/` 入口 + 需多卡）。Wan 2.2-5B `expand_timesteps` 仍为非目标。

> 排序澄清：MoE 决策文档（`reading/SPRINT_moe_support_decision.md`）= "要不要/为什么"；本文 =
> "怎么做"。二者一对，不存在"先 Wan 2.2 再做别的 MoE"。

---

## 0. Core Decision（含 go/no-go）

把 Wan 2.2 A14B（high-noise `transformer` + low-noise `transformer_2`，按 `boundary_ratio` 的
SNR 阈值二选一）接入 RL 训练/rollout。当前状态：

- **base 推理已能跑**（`wan_i2v_base_sample.py:14,102`，带 offload）。
- **A14B 训练/rollout 基础支持已接入**：`vrl/models/diffusion/wan_2_1/model.py` 读取
  `boundary_ratio` / `transformer_2`，按 diffusers 公式 `t >= boundary_ratio *
  scheduler.config.num_train_timesteps` 选择 high-noise `transformer`，否则选择 low-noise
  `transformer_2`；`runtime.py` 的 minimal replay bundle 同步加载 `transformer_2`。
- **默认只训练 low-noise 专家**：dual-stage 且未显式设置 `model.trainable_transformers` 时，
  `trainable_modules == {"transformer_2": ...}`。需要两个专家时可显式设为 `both` / `all`。

**它便宜在哪**：时间步路由双专家，任一时刻只 1 个 ~14B transformer 活跃——**无** gate、**无**
token 路由、**无** expert-parallelism、**无** Megatron kernel。拦路虎是 replay 契约，不是 MoE 系统。

**go/no-go**：代码层 go 已执行，策略选择采用最小显存起点：默认只训 low-noise 专家。下一步不是再改
family/module 结构，而是跑真实 Wan 2.2 A14B GRPO proof run。

## 1. 当前单塔契约（要复制/扩展的对象）

Wan 2.1 的 replay 契约（rollout 记录 → 训练重算 logprob 做 GRPO）：

```
forward_step(state, step_idx)            # 单个 transformer 在某 timestep 前向
export_replay_tensors(state)             # → {prompt_embeds, neg_embeds, timesteps, latents, ...}
build_replay_state(replay_tensors, i)    # 训练时重建 state，逐步重算 old_log_prob
WanT2VReplayModel(ReplayRolloutStubs)    # replay-only（无 text encoder/VAE/pipeline）
model.trainable_modules                  # weight-sync 下发的可训状态（LoRA 在单个 transformer）
```
证据：`vrl/models/diffusion/wan_2_1/model.py:296-319,367-378`、`runtime.py:134,187`。

## 2. 范围（dual-expert 触及的 7 处）

| # | 区域 | 改动 | 难度 |
|---|---|---|---|
| 1 | 模型加载 | 已支持 2 个 transformer + `boundary_ratio`；缺 `transformer_2` 会 fail loud | done |
| 2 | forward 派发 | 已按 diffusers boundary 公式选 high/low 专家 | done |
| 3 | **replay/logprob 契约（核心）** | 已把 `boundary_ratio` / `num_train_timesteps` 放进 batch context，replay 按同公式派发；CPU 单测覆盖跨 boundary 路由 | done |
| 4 | replay 模型 | `WanT2VReplayModel` / `WanI2VReplayModel` 已能持有并路由 `transformer_2` | done |
| 5 | **可训范围 / weight-sync（关键决策）** | 默认 `transformer_2`；可用 `model.trainable_transformers` 显式选 `transformer` / `transformer_2` / `both` | done |
| 6 | 显存策略 | 2×14B 不同驻；inactive 专家 CPU-offload（eval 已有 `--offload model`）；接 generation memory policy 系统 | 中 |
| 7 | config/registry + 拆 guard | 新 Wan 2.2 model config（如 `/model/diffusion/wan_2_1/i2v_a14b`）；变体选择分支到 dual-stage 类；移除 `_ensure_single_transformer_wan_i2v` 的 `NotImplementedError`（`model.py:699-702`） | 低-中 |

> 变体当前靠 `_resolve_model_cls(task_variant)` 选 t2v/i2v（`runtime.py:59`）。2.2 的"双阶段"与
> t2v/i2v **正交**——需新增一个维度（新 model 类或 dual-stage flag），不是塞进 task_variant。

## 3. 待决项（落地前必须拍板）

1. **训一个专家还是两个？** 决定显存、weight-sync 体量、LoRA 配置、replay 模型持有谁。
   - 候选：只训 low-noise（细节专家，多数去噪步在它）/ 只训 high-noise / 两个都训。
   - 当前默认：先只训 **low-noise**（覆盖大部分步、显存/同步减半），验证通路后再评估两个。
2. **先 T2V-A14B 还是 I2V-A14B？** 你的物理 run 用 I2V
   （`online_grpo_physics_i2v.yaml`），所以 proof run 应优先走 I2V。
3. **expert-per-step 是否真能从 timestep+boundary 推出**（§2#3 的好消息）→ 写码前用一个 step 验证
   路由公式与 diffusers pipeline 一致，避免 off-by-one。
4. **复用还是新建 replay 类**：已复用现有 replay 类，增加 `transformer_2` 持有与路由。

## 4. 非目标

- **Wan 2.2-5B `expand_timesteps` 变体**：不同机制（`runner.py:84-86` 单列），本 sprint 不做。
- **任何 expert-parallelism / Megatron MoE kernel**：双专家是时间步路由，用不上（见决策文档 §3）。
- **不预先承诺训两个专家**：先 §3.1 拍板，避免一上来背 2× 显存/同步。

## 5. 验证标准（finishing criteria）

- 路由公式逐步对齐 diffusers WanPipeline（§3.3）；
- rollout 记录的 `old_log_prob` 与 replay 重算**逐步一致**（跨 boundary 不漂移）——这是契约正确性的硬指标；
- 一个短 GRPO run 端到端不崩、reward 可动；
- 拆掉 guard 后 2.1 单塔路径回归不变（不破坏现有 1.3B/I2V-14B 训练）。

当前验证：

- `tests/models/diffusion/wan_2_1/test_backbone_parity.py` 覆盖 A14B high/low boundary 路由；
- `tests/models/diffusion/wan_2_1/test_model_loading.py` 覆盖 dual-stage 接受与 `expand_timesteps` 拒绝；
- `tests/models/interfaces/test_minimal_replay_runtime_wiring.py` 覆盖 replay bundle 加载 `transformer_2` 且默认只同步 low-noise 专家；
- 尚未完成真实 Wan 2.2 GRPO proof run。

## 6. 来源 / 证据

- `vrl/models/diffusion/wan_2_1/model.py`（dual-stage 加载、boundary 路由、replay context、
  默认 low-noise `trainable_modules`、`expand_timesteps` fail-loud）
- `vrl/models/diffusion/wan_2_1/runner.py`（A14B 仍复用 I2V channel-concat runner；5B
  `expand_timesteps` 保持非目标）
- `vrl/models/diffusion/wan_2_1/runtime.py`（minimal replay bundle 加载 `transformer_2`，
  复用模型自己的 LoRA/full-finetune/compile 逻辑）
- `configs/model/diffusion/wan_2_2/a14b.yaml`、`configs/model/diffusion/wan_2_2/i2v_a14b.yaml`
  （官方 Wan 2.2 A14B boundary ratio + 默认 low-noise 训练配置；7e3e072 迁此）
- `vrl/scripts/eval/wan_i2v_base_sample.py:14,102`（双专家事实 + `boundary_ratio`/`guidance_scale_2`）
- `configs/experiment/diffusion/wan_2_1/online_grpo_physics_i2v.yaml`（现有 I2V GRPO 入口）
- 决策依据：`reading/SPRINT_moe_support_decision.md`（Wan 2.2 = 唯一值得做的 MoE；机制与外部来源）
