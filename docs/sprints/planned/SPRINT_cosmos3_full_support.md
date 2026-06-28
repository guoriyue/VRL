# SPRINT: Cosmos3 全量支持 — reasoner(judge)+ generator(policy)双塔，按 MR 拆分

状态：**planned（2026-06-27）**。性质：**模型支持执行计划 + 架构边界决策**。承接 `SPRINT_physical_ai_model_support.md`（Cosmos3 Tier-1 probe），并用 NVIDIA 自家的 `~/Desktop/cosmos-rl` 作权威蓝本，把"能不能训"从推断升级为**有参考实现的可执行拆分**。实现落在**另一个 repo**，本 sprint 是设计 + MR 边界 + 契约。

## 0. 一句话

Cosmos3（`model_type="cosmos3_omni"`）是统一 omni 模型 = **AR reasoner（视觉→文本理解）+ diffusion generator（→图/视频/音/动作）+ audio + action 塔**。两塔在 RL 回路里角色不同、解锁状态不同，所以**两塔都接，但按 MR 分期**：

```text
diffusion generator  ->  被 RL 训练的 policy   ->  diffusion seam（复用 flow_grpo / Predict2）
AR reasoner          ->  reward/judge + 理解服务 ->  reward-model seam（仿 Kling，vLLM 服务）
audio / action 塔     ->  本 sprint 非目标，track only（action 走 VLA/Env seam）
```

**核心修正（相对上一版）**：生成器**不是**被 diffusers 永久卡死。diffusers 没有 Cosmos3 pipeline 是真的，但 **NVIDIA 原生 `cosmos-rl` 已经有完整的"生成器 denoise + 每步可复算 logprob + GRPO"实现**——所以 generator RL 的路径是"移植已知契约"，不是"探索是否可能"。

## 1. 权威事实（已读源，非推断）

### 1.1 生成器 = diffusion DiT（cosmos-rl 确认）

- 网络：`MinimalV1LVGDiT`（DiT + AdaLN + LoRA），`cosmos-rl/cosmos_rl/policy/model/wfm/networks/minimal_v1_lvg_dit.py`。
- denoise：EDM / Rectified-Flow 预条件（`c_skip/c_out/c_in/c_noise`），`x0_pred = c_skip*x_t + c_out*net(...)`，`t2v_model.py:1298-1343`。**不是自回归。**
- forward 签名 `denoise(x_t, sigma, condition: T2VCondition)`——**standalone DiT**。

### 1.2 generator 与 reasoner/conditioner 解耦（关键，解掉 MoT 风险）

- conditioner（含 `qwen_omni` 文本编码器）**不可训练**：`assert sum(p.requires_grad for p in conditioner.parameters()) == 0`（`t2v_model.py:248`）。
- generator 只吃 `condition` object（embeddings/tokens），**不在架构层 fuse**。→ **可以单独 forward 生成器**，不需要跑整个 omni。`qwen_omni` 只是可选的 online 文本编码器（`compute_online=True` 才加载）。

### 1.3 RL logprob 契约（cosmos-rl 的核心，可直接复用）

每个 denoise step 是一个高斯转移 `x_{t-1} = μ(x_t,σ_t) + std·ε`，policy-gradient 的 ratio 来自**当前策略均值 μ_θ vs 参考均值 μ_ref 的高斯似然比**（`wfm_trainer.py:464-490`）：

```text
μ_θ   = get_mu_from_model(x0_fn_θ,   x_t, σ_t, σ_next, cond)   # 当前策略走一步 solver
μ_ref = rollout 时存下的 mu_old                                  # 参考策略（frozen net_ref）
log_prob_ratio = -( ||x_sample-μ_θ||² - ||x_sample-μ_ref||² ) / (2·σ_next²·(η²+2η))
ratio = exp(mean(log_prob_ratio))         # p_θ / p_ref
loss  = max(-A·ratio, -clip(ratio,1±ε)·A) + kl_beta·KL + data_beta·SFT
```

- η 来自 `s_churn/(sample_steps+1)`（EDM 随机性），`std = σ_next·√(η²+2η)·s_noise`。
- **这就是 VRL 已有的 `vrl/algorithms/flow_matching.sde_step_with_logprob`(DanceGRPO/Flow-GRPO)同构**——P3 不需要新算法，只需把 Cosmos 的 scaling/solver 喂进去。
- 参考模型 `net_ref` 每 16 iter 刷新；`train_on=[0..7]` 选哪些 step 训练；轨迹存 `{noise_x, sigma, sigma_next, mu_old, sample, x0_pred}`。
- ⚠️ **移植坑**：cosmos-rl 当前把方差归一化 `/(2·std²)` 注释掉了（`wfm_trainer.py:486` TODO），用的是未归一化的 `-(diff_current-diff_old)`。移植时**用归一化的正确式**，别照搬 bug。

### 1.4 reasoner = Qwen3-VL（vLLM 已支持）

- `vllm/model_executor/models/cosmos3.py` = `Cosmos3ForConditionalGeneration(Qwen3VLForConditionalGeneration)`，WeightsMapper **drop 生成塔/audio/action，只留 "Reasoner-only part"**。所以 reasoner 服务可直接用 vLLM。
- config `cosmos3_omni`（`vllm/transformers_utils/configs/cosmos3.py`）。

### 1.5 模型与规模

- HF 真实存在：`nvidia/Cosmos3-Nano`(16B)、`Cosmos3-Super`(64B)、`Cosmos3-Super-Text2Image/Image2Video`、`Cosmos3-Nano-Policy-DROID`。
- 16B → 单卡 32GB **full-param 必 OOM**；generator RL 只能 LoRA（AdaLN-LoRA 是原生支持）或多卡 FSDP（`SPRINT_multi_gpu_training`）。

## 2. MR 拆分 + 依赖 DAG

```text
MR0 (contract probe) ──┬──> MR1 (reasoner-judge)            [独立，已解锁]
                       └──> MR2 (generator adapter) ──> MR3 (diffusion family) ──> MR4 (generator RL run)
                                                                                   MR5 (action/audio) = 非目标/track
```

reasoner 线(MR1)与 generator 线(MR2→4)**并行独立**，只共享 MR0 的盘点。

### MR0 — 契约盘点（probe，不训练）

- 刷新 `cosmos3_nano_generator_probe.py` / `cosmos3_nano_policy_droid_probe.py` 的 decision note。
- 对 `nvidia/Cosmos3-Nano` 实测：组件图（reasoner/generator/vae/conditioner/media-tokenizer）、generator 是否能 `denoise(x_t,σ,cond)` 单独 forward（按 §1.2 预期 yes）、scheduler 暴露的 σ 序列。
- 把 `support_matrix.py` 的 `cosmos3_nano.logprob` 从 `unknown` 落定为 `gaussian_step_ratio`（按 §1.3）。
- **验收**：两塔 component graph + tensor/logprob 契约写清；负结果也算交付。**KILL**：若 Cosmos3-Nano 的 generator 与 reasoner 真的架构级 fuse、无法单独 forward（与 cosmos-rl WFM 不同）→ 记录并把 generator 线降级为"只用 cosmos-rl 原生训练，不进本仓库 seam"。

### MR1 — Reasoner-as-judge（已解锁，先拿价值）

把 Cosmos3 reasoner 接成 video reward，仿 `kling_video_reward`：

- 新增 `vrl/rewards/models/cosmos3_reasoner_reward.py`（`RewardModel` 子类）：vLLM 起 `Cosmos3ForConditionalGeneration` 服务（参考 `vllm_omni_diffusion_profile.py` 的 vLLM 用法）或 HF Qwen3-VL；prompt 模板 + 视频 artifact → 结构化分数（task_success / contact_realism / temporal_consistency / physical_plausibility）。
- 新增 `configs/reward/cosmos3_reasoner.yaml`（仿 kling：`reward_name` + `worker_config`，`execution: pool`）。
- **复用** `RewardInferenceArtifact` 落盘 mp4 + pool execution，不改 transport 边界。
- **验收（discrimination probe，data-factory §2）**：reward 能区分 real / good-gen / bad-gen / perturbed-negative，AUC>0.7 才当主 reward；否则当 guard。

### MR2 — Generator 访问 adapter（KILL-RISK gate）

把原生生成器的 denoise + 每步 logprob 暴露成可训练句柄（**移植 cosmos-rl 契约**，§1.3）：

- 句柄 1：`denoise_step(x_t, σ_t, σ_next, cond) -> (mu, std)`（= cosmos-rl `get_mu_from_model` + `rl_update_step_fn`）。
- 句柄 2：scheduler 的 `σ` 序列 + solver（2ab / RK），使 ratio 可复算。
- 句柄 3：可挂 LoRA / 可 backward 的 `net`（AdaLN-LoRA 原生支持）。
- 验证：first-step log-prob diff ≈ 0（训推一致，同 Predict2 判据）。
- **KILL 条件**（命中即停）：generator 只暴露 server-level API、σ 序列不可得、或 step 非高斯可复算 → generator RL 不接本仓库，只在 cosmos-rl 原生跑。

### MR3 — Generator diffusion family（gated on MR2）

接入 VRL diffusion seam，**复用 flow_grpo**：

- `vrl/models/diffusion/cosmos3/{model,runner,runtime}.py`：`encode_conditioning`(走 reasoner/text-encoder 出 condition)/`prepare_sampling`/`forward_step`/`sde_logprob`（直接调 `vrl/algorithms/flow_matching`，喂 Cosmos 的 c_skip/c_out/c_in/c_noise + solver）。
- `register_rollout_family(...)`（`vrl/rollouts/families/registry.py`），family=`cosmos3`，diffusion 分支。
- `configs/model/diffusion/cosmos3/nano.yaml`、`configs/experiment/diffusion/cosmos3/online_grpo_*.yaml`。
- 配置对齐 cosmos-rl 默认：`sample_steps≈10`、`num_rollout=8`、`clip_ratio=1e-4`、`kl_beta=0.01`、resolution 480、latent `state_t=24/state_ch=16`、solver `2ab`。

### MR4 — Generator RL run（gated on MR3）

- 16B → **LoRA 优先**（单卡可行性 smoke）；full-param 等多卡 FSDP（cosmos-rl 用 FSDP+CP+TP）。
- 数据：机器人 per-sample reference（复用 `video_world_v2w`，见 data-factory sprint）。
- reward：MR1 的 Cosmos3-reasoner judge（或 Kling 过渡）。
- **验收**：clip_fraction>0、first-step logprob diff≈0、生成 artifact **肉眼**连贯（参考 480p 教训：neighbor-diff 统计骗人，必须看）、eval reward >2σ 才算"学到"。

### MR5 — action / audio 塔（非目标）

track only。action 走 `SPRINT_physical_ai_model_support.md` 的 VLA/Env seam，不塞进 diffusion。

## 3. 工程落点（按 MR）

| MR | 路径 | 复用/参考 |
|---|---|---|
| MR0 | 扩 `vrl/scripts/eval/cosmos3_nano_generator_probe.py`；落定 `vrl/models/support_matrix.py` logprob | — |
| MR1 | `vrl/rewards/models/cosmos3_reasoner_reward.py`、`configs/reward/cosmos3_reasoner.yaml` | `kling_video_reward.py`、vLLM `cosmos3.py` |
| MR2 | generator adapter（另 repo 的 native 封装层） | cosmos-rl `wfm/models/t2v_model.py`、`trainer/wfm_trainer.py:405-634` |
| MR3 | `vrl/models/diffusion/cosmos3/{model,runner,runtime}.py`、registry entry、configs | `vrl/models/diffusion/cosmos/predict2/*`、`vrl/algorithms/flow_matching` |
| MR4 | `configs/experiment/diffusion/cosmos3/*` | data-factory `video_world_v2w` |

**保持不变**：Predict2/2.5/Wan/Echo diffusion seam 不动；reward transport 不动；reasoner 不塞进 AR 图像生成家族（它是 VLM）；不用 vLLM cosmos3.py 当 generator 参考（它 drop 了生成塔）。

## 4. 全局验收

- **MR1**：reasoner-judge 对机器人视频出可区分结构化分数（AUC>0.7）。**这一条独立可交付，不依赖 generator。**
- **MR2**：拿到可训练 denoiser + 可复算高斯-step logprob（first-step diff≈0），或明确 KILL 并记录。
- **MR3/4**（若 MR2 过）：clip_fraction>0 + 生成连贯 + eval reward >2σ。
- 16B 资源边界写清：单卡只能 LoRA/judge-only，full-param 需多卡。

## 5. 非目标

- 不把 reasoner 接成 AR 图像生成家族（它是视觉→文本 VLM）。
- 不用 vLLM cosmos3.py 当 generator 参考。
- MR2 契约未过前不接 generator 训练（不伪造 logprob，verl 训推一致铁律）。
- 不照搬 cosmos-rl 的未归一化 logprob bug（§1.3）。
- 不做 audio/action 训练；不上 Super 64B 训练；不为 16B full-param 单卡硬上。

## 6. 风险

| 风险 | 处理 |
|---|---|
| Cosmos3 generator 与 cosmos-rl WFM 结构不同、真 MoT fuse | MR0 KILL gate 探明；不可单独 forward 则只用 cosmos-rl 原生 + 本仓库只接 reasoner |
| 16B 单卡 OOM | LoRA + judge-only 先行；full-param gated 多卡 FSDP |
| reasoner judge 被 hack（看似对、物理错） | discrimination probe + 多 lens；judge 只当 reward 之一 |
| logprob 方差归一化错（照搬 cosmos-rl TODO） | 用 §1.3 归一化式 + first-step diff≈0 验 |
| generator 训练需 CP/TP 才能放下激活 | 对齐 cosmos-rl 的 FSDP+CP；先小 shape/LoRA smoke |

## 7. 参考

- **权威蓝本（generator RL 契约）**：`~/Desktop/cosmos-rl/cosmos_rl/policy/model/wfm/`（`models/t2v_model.py`、`networks/minimal_v1_lvg_dit.py`、`networks/vlm_qwen/qwen_omni.py`）、`cosmos_rl/policy/trainer/wfm_trainer.py:405-634`(logprob+GRPO)、`cosmos_rl/policy/config`(RLConfig)
- **reasoner 服务参考**：`~/Desktop/vllm/vllm/model_executor/models/cosmos3.py`、`transformers_utils/configs/cosmos3.py`
- **本仓库复用**：`vrl/algorithms/flow_matching`(sde_step_with_logprob)、`vrl/models/diffusion/cosmos/predict2/*`、`vrl/rewards/models/kling_video_reward.py`、`vrl/scripts/eval/cosmos3_nano_generator_probe.py`、`vrl/models/support_matrix.py`
- **承接/下游**：`SPRINT_physical_ai_model_support.md`、`SPRINT_cosmos_robotic_data_factory_domain_rl.md`(reward 缺口)、`SPRINT_multi_gpu_training.md`(16B 多卡)
- 模型：`nvidia/Cosmos3-Nano`(16B) / `Cosmos3-Super`(64B)，HF collection https://huggingface.co/collections/nvidia/cosmos3
