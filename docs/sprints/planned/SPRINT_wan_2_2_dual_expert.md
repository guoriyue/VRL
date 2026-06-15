# SPRINT: Wan 2.2 双专家（A14B）RL 支持

状态：planned / 设计（**未承诺、未开工**）。这是 [[SPRINT_moe_support_decision]] 结论指向的**唯一
具体 MoE 落地项**——不是 MoE 的"前置"，它**就是**本 infra 该做的那个 MoE。先把 dual-stage
replay 契约设计想清再写码（改错会静默污染 GRPO）。

> 排序澄清：MoE 决策文档（`reading/SPRINT_moe_support_decision.md`）= "要不要/为什么"；本文 =
> "怎么做"。二者一对，不存在"先 Wan 2.2 再做别的 MoE"。

---

## 0. Core Decision（含 go/no-go）

把 Wan 2.2 A14B（high-noise `transformer` + low-noise `transformer_2`，按 `boundary_ratio` 的
SNR 阈值二选一）接入 RL 训练/rollout。当前状态：

- **base 推理已能跑**（`wan_i2v_base_sample.py:14,102`，带 offload）。
- **训练/rollout 被显式拒绝**：`vrl/models/diffusion/wan_2_1/model.py:699-702`
  （`boundary_ratio is not None` → `NotImplementedError("Wan 2.2 dual-stage I2V needs a separate
  replay contract")`）；`runner.py:82-86`（"future 2.2 upgrade path"）。

**它便宜在哪**：时间步路由双专家，任一时刻只 1 个 ~14B transformer 活跃——**无** gate、**无**
token 路由、**无** expert-parallelism、**无** Megatron kernel。拦路虎是 replay 契约，不是 MoE 系统。

**go/no-go**：只有想要 Wan 2.2 质量时才做。决策前先答 §3 的两个待决项（训一个还是两个专家、
先 T2V 还是 I2V）——它们决定成本。**本 sprint 是设计，落地前需显式拍板。**

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
| 1 | 模型加载 | 载入 2 个 transformer + 读 `boundary_ratio`；当前 `_load` 只载单 `pipeline.transformer`、freeze 其余（`model.py:114-121`） | 低 |
| 2 | forward 派发 | `forward_step` 按 timestep vs boundary 选专家；当前单塔（`model.py:260-276`） | 低 |
| 3 | **replay/logprob 契约（核心）** | `export_replay_tensors`/`build_replay_state` 要让重算路由到**当步用过的同一专家**。好消息：路由由 `timestep + boundary_ratio` **确定性**决定 → 不必存"哪个专家"，只需把 `boundary_ratio` 进 replay tensors，replay 端按同公式派发 | **高** |
| 4 | replay 模型 | `WanT2VReplayModel` 需持有**两个**专家（或当前所需那个）并正确路由；判断是否需新 replay 类还是复用 | 中 |
| 5 | **可训范围 / weight-sync（关键决策）** | LoRA 打**一个还是两个**专家？`model.trainable_modules` 要枚举对的专家；weight-sync 与 rollout worker 加载相应可训状态 | **中-高** |
| 6 | 显存策略 | 2×14B 不同驻；inactive 专家 CPU-offload（eval 已有 `--offload model`）；接 generation memory policy 系统 | 中 |
| 7 | config/registry + 拆 guard | 新 Wan 2.2 model config（如 `/model/diffusion/wan_2_1/i2v_a14b`）；变体选择分支到 dual-stage 类；移除 `_ensure_single_transformer_wan_i2v` 的 `NotImplementedError`（`model.py:699-702`） | 低-中 |

> 变体当前靠 `_resolve_model_cls(task_variant)` 选 t2v/i2v（`runtime.py:59`）。2.2 的"双阶段"与
> t2v/i2v **正交**——需新增一个维度（新 model 类或 dual-stage flag），不是塞进 task_variant。

## 3. 待决项（落地前必须拍板）

1. **训一个专家还是两个？** 决定显存、weight-sync 体量、LoRA 配置、replay 模型持有谁。
   - 候选：只训 low-noise（细节专家，多数去噪步在它）/ 只训 high-noise / 两个都训。
   - 推荐起点：先只训 **low-noise**（覆盖大部分步、显存/同步减半），验证通路后再评估两个。
2. **先 T2V-A14B 还是 I2V-A14B？** 你的物理 run 用 I2V（`online_grpo_physics_i2v.yaml`），但拒绝 guard
   也在 I2V 路径——先做哪个取决于目标 reward 任务。
3. **expert-per-step 是否真能从 timestep+boundary 推出**（§2#3 的好消息）→ 写码前用一个 step 验证
   路由公式与 diffusers pipeline 一致，避免 off-by-one。
4. **复用还是新建 replay 类**（§2#4）。

## 4. 非目标

- **Wan 2.2-5B `expand_timesteps` 变体**：不同机制（`runner.py:84-86` 单列），本 sprint 不做。
- **任何 expert-parallelism / Megatron MoE kernel**：双专家是时间步路由，用不上（见决策文档 §3）。
- **不预先承诺训两个专家**：先 §3.1 拍板，避免一上来背 2× 显存/同步。

## 5. 验证标准（finishing criteria）

- 路由公式逐步对齐 diffusers WanPipeline（§3.3）；
- rollout 记录的 `old_log_prob` 与 replay 重算**逐步一致**（跨 boundary 不漂移）——这是契约正确性的硬指标；
- 一个短 GRPO run 端到端不崩、reward 可动；
- 拆掉 guard 后 2.1 单塔路径回归不变（不破坏现有 1.3B/I2V-14B 训练）。

## 6. 来源 / 证据

- `vrl/models/diffusion/wan_2_1/model.py:114-121`（加载/freeze）、`:260-276`（forward_step）、
  `:296-319`（replay 投影/重建）、`:367-378`（`WanT2VReplayModel`）、`:699-702`（2.2 拒绝 guard）
- `vrl/models/diffusion/wan_2_1/runner.py:82-86`（2.2 deferral）
- `vrl/models/diffusion/wan_2_1/runtime.py:59`（变体选择）、`:95-104,134,175-187`（LoRA/trainable/weight-sync）
- `vrl/scripts/eval/wan_i2v_base_sample.py:14,102`（双专家事实 + `boundary_ratio`/`guidance_scale_2`）
- `configs/experiment/diffusion/wan_2_1/online_grpo_physics_i2v.yaml`（现有 I2V GRPO 入口）
- 决策依据：`reading/SPRINT_moe_support_decision.md`（Wan 2.2 = 唯一值得做的 MoE；机制与外部来源）
