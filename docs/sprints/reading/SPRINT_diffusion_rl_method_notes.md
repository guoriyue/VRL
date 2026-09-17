# SPRINT：diffusion/flow RL 方法笔记——DanceGRPO / Flow-GRPO / MixGRPO / SRPO / Flow-Factory（reading）

状态：**reading / 存档（2026-09-16）**。KIND：**reading**。每节末尾有"VRL 现状"，
对照的是 2026-09-16 的 HEAD（`vrl/algorithms/grpo/continuous.py`、
`vrl/generation/steps/denoise/config.py`、`vrl/rewards/functions/registry.py`）。
缺口汇总在 [[SPRINT_industry_post_training_gap_backlog]]。相关旧档：
[[SPRINT_dance_grpo_validation]]（机制验证，done）、
[[SPRINT_anime_rl_post_training_literature]]（reward hacking 证据）。

PDF 已在 `docs/papers/diffusion-flow-rl/`（dancegrpo、flow-grpo、mixgrpo 等）。

---

## 1. DanceGRPO（arXiv 2505.07818，ByteDance Seed）

### 1.1 方法
- SDE 化采样（diffusion 与 rectified flow 各一条反向 SDE，ε_t 控制随机性）；策略
  = 转移 `p(z_{t-1}|z_t, c)`，由 SDE 离散化的高斯给出。
- 目标：`J = E[1/G Σ_i 1/T Σ_t min(ρ A_i, clip(ρ, 1±ε) A_i)]`，
  `A_i = (r_i − mean)/std` **按 prompt 组**归一。
- **组内共享初始噪声**：同 prompt 的 G 个样本用同一初始噪声——视频上不共享会发散
  并加剧 reward hacking。
- **timestep 子采样**：训练时随机丢 40%（τ=0.6）；消融：前 30% 的步最关键，但全程
  训练仍需要以做精修；省 ~40% 计算无损。
- **多 reward 合成在 advantage 级**：`A_i = Σ_k (r_i^k − μ^k)/σ^k`，避免不同 RM 的
  尺度问题。用到 HPS-v2.1、CLIP、VideoAlign（VQ/MQ/TA）、二值阈值 reward；
  VisionReward-Video 29 维不稳定。
- **Best-of-N**：从 16/64/256 的池里只训 top-k 与 bottom-k 样本，加速收敛。
- 超参（Table 6）：lr 1e-5 AdamW；32 prompt/batch；12 样本/prompt；每轮 4 次梯度更新；
  clip ε=1e-4；ε_t=0.3；τ=0.6；采样步 50（SD）/ 25（FLUX、HunyuanVideo）。
- 结果：SD HPS 0.239→0.365；HunyuanVideo 运动质量 +181%、视觉质量 +56%。
- 教训：只用 HPS 出"油腻"伪影，CLIP 正则保住自然感；**CFG 让训练不稳**，FLUX /
  HunyuanVideo 建议关，CFG 依赖的模型要同时优化 cond+uncond（显存翻倍）；DDPO 式目标
  放进 flow SDE 会发散，GRPO clip 才稳；首个稳定跑到 10k+ prompt 的方法。

### 1.2 VRL 现状
| 项 | 状态 |
|---|---|
| GRPO loss + 组归一 advantage | 有（`GRPO`, `GroupAdvantageEstimator`） |
| 随机 timestep 子集 | 有（`actor.timestep_selection=random`，done sprint 验证过机制） |
| 组内共享初始噪声 | **无**（只有 `v_grpo.py` 的 `_group_shared_noise` 在 loss 侧；生成侧无开关） |
| 多 reward advantage 级合成 | **无**（`MultiReward` 出加权总分，分量保留在 `RewardOutput.components`，但 advantage 只看总分） |
| Best-of-N top/bottom-k | **无**（只有非零 advantage 掩码） |
| CFG 下的训练分支 | 有 `DenoiseCFGMode`，但没有审计"训练时 uncond 分支是否也走 replay" |

---

## 2. Flow-GRPO（arXiv 2505.05470）

### 2.1 方法
- ODE→SDE：`dx_t = [v + σ_t²/(2t)(x_t + (1−t)v)]dt + σ_t dw`，`σ_t = a√(t/(1−t))`，
  由 Fokker-Planck 匹配保证边缘分布不变。
- **denoising reduction**：训练 T=10 步，推理 T=40 步，4× 提速无损——RL 只依赖相对
  reward。
- 目标同 GRPO + `β KL`。
- Reward：GenEval（规则）、OCR（编辑距离）、PickScore。
- 超参：G=24（更小会崩）；a=0.7（<0.3 收敛差，>0.7 边际递减）；β=0.04（组合）/
  0.01（偏好）；LoRA α=64 r=32。
- reward hacking：偏好任务无 KL 会塌成单一风格；"KL 不等价于 early stopping"。
- 泛化：训 2–4 个物体计数，泛化到 12。

### 2.2 VRL 现状
| 项 | 状态 |
|---|---|
| SDE + KL 闭式 | 有 |
| 训练步数 < 推理步数 | `sampling.num_steps` 只有一份；`config/schema.py` 里 grep 不到 eval 级 sampling 覆盖 |
| G≥24、a=0.7 默认 | recipe 里各自设置，无统一默认核对 |

---

## 3. MixGRPO（arXiv 2507.21802，Tencent Hunyuan）

### 3.1 方法
- 滑动窗口 `S=[t_l, t_r)` 内 SDE，窗外 ODE；窗口从低 SNR（高随机）向高 SNR 移动：
  w=4、每 τ=25 步移一次、步幅 s=1；progressive-constant 调度最好。
- **MixGRPO-Flash**：窗口**之后**的 ODE 段可用二阶 DPM-Solver++（不参与策略比值）；
  窗口**之前**加速会放大数值误差、损害 reward 可靠性。
- 数字：vs DanceGRPO，ImageReward 1.629 vs 1.436；每轮 291.3s→150.3s（~50%）；
  π_θ 的 NFE 14→4；Flash 再降到 83.3s（71%）。
- Reward：HPS-v2.1、PickScore、ImageReward、UnifiedReward 等权多 reward 更稳。
- 教训：早期步定全局结构、后期步精修，课程式从早到晚最稳；4 个选定步 > 随机 14 步；
  CPS（coefficients-preserving sampling）比标准 SDE 离散少高频伪影；RM 不够强时
  reward hacking 仍在。

### 3.2 VRL 现状
| 项 | 状态 |
|---|---|
| 窗口 SDE + 窗外 ODE | 有（`rollout.sde.window_size/window_range`, `timestep_selection=sde_window`） |
| 窗口滑动调度 | **无**：`loop.py:250` 是固定 `sde_window=[lo, hi)`，没有随训练步移动的 progressive 调度 |
| 窗后高阶 ODE 求解器 | **无**（窗外仍是同一阶 Euler） |
| CPS | 有（`SdeType = flow_grpo | cps | ddim`） |

---

## 4. SRPO（arXiv 2509.06942，Tencent Hunyuan）

### 4.1 方法
- **Direct-Align**：向图像注入已知高斯噪声到 t，用闭式 `x₀ = (x_t − σ_t ε_gt)/α_t`
  单步恢复干净图；早期 timestep（5% 噪声）也能高质量恢复；多条噪声序列的 reward 用
  衰减因子 λ(t) 聚合。
- **语义相对偏好**：同一样本上正负 prompt（"Realistic photo" vs "CG Render"）的
  reward 之差 `r_SRP = f_img(x)ᵀ(C₁ − C₂)`；负梯度做正则。
- 训练：FLUX.1-dev，HPSv2.1 / PickScore / ImageReward，HPDv2；32 H20，**10 分钟收敛**；
  训练 25 步，推理 50 步。
- reward hacking 观察：只在晚期 timestep 直接优化会过拟合——HPSv2 偏红、PickScore 偏紫、
  ImageReward 过曝、细节被抹平。
- 人评：真实感 3.7×、美学 3.1×。

### 4.2 VRL 现状
| 项 | 状态 |
|---|---|
| 可微 reward 直接反传 | **无**（grep `refl / reward_backprop / direct_align` 为空） |
| 正负 prompt reward 差 | **无**（`rewards/` 无 negative_prompt 概念） |

---

## 5. Flow-Factory（arXiv 2602.12529）

### 5.1 框架
- 注册表式三层解耦：BaseAdapter（模型）/ BaseTrainer（算法）/
  Pointwise & Groupwise RewardModel（reward）+ SDESchedulerMixin；O(M×N)→O(M+N)。
- 算法：Flow-GRPO、MixGRPO、GRPO-Guard、DiffusionNFT、AWM（advantage-weighted
  matching）；SDE 动力学 Flow-SDE / Dance-SDE / CPS。
- Reward 接口：pointwise `score(x)→ℝ`、groupwise `rank({x_i})→ℝ^k`；同一模型自动去重
  只加载一次；聚合可选加权和或 **GDPO 式逐 reward 归一**。
- 训练/推理分离：NFT 与 AWM 对求解器无关，任何 ODE solver 都能出轨迹。
- 效率：预计算并缓存条件 embedding 与 VAE latent，冻结件卸出 GPU；FLUX.1-dev
  8×H200：峰值 61.08→53.14 GB（−13%），每步 144.02→82.68s（1.74×）。
- 复现了 Flow-GRPO / NFT / AWM 在 FLUX.1-dev + PickScore 上的结果。

### 5.2 VRL 现状
| 项 | 状态 |
|---|---|
| 模型/算法/reward 解耦 | 有（family registry、`Algorithm` protocol、reward registry） |
| GRPO-Guard、NFT、FlashGRPO、FlowDPPO、V-GRPO | 有 |
| AWM | **无** |
| groupwise reward 接口（排序型） | **无**（`RewardFunction` 是 pointwise） |
| 逐 reward 归一聚合 | **无**（同 §1） |
| 条件 embedding / latent 预缓存 | 部分：prompt encoder 冻结可卸载；无离线缓存到磁盘 |
| 同一 RM 多处引用自动去重 | 需核对 `MultiReward.from_dict` 是否按模型路径去重 |
