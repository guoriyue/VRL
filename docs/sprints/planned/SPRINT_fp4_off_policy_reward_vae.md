# SPRINT: MXFP4 离-policy-path —— reward model + VAE 上 Blackwell FP4（越 bf16 上限）

状态：**planned / high-risk proof-gated（2026-06-27）**。性质：**唯一能越过 bf16 硬件上限的单卡 compute 杠杆,但有损**——只能用在 **policy/log-prob path 之外**的组件(reward model、VAE),那里有损不污染 old_log_prob。Blackwell 第 5 代 tensor core 的 MXFP4/NVFP4 经 `tcgen05.mma` 硬件块缩放,custom Triton/CUTLASS 可调,~4x bf16。

> 来由([[SPRINT_lossless_diffusion_rl_research]] §验证 7-9 + 研究):bf16 GEMM 实测已到消费卡硬件上限(NCU tensor SOL 45% ≈ 最优方阵 47%)。唯一越界的是 sub-bf16(MXFP4 ~2x FP8 ~4x bf16,sm_120 确认有真 FP4 加速)。但 FP4 改输出 → **上 policy denoise 会改 Gaussian transition mean → 污染 old_log_prob**(禁止);只能离 policy path。
> **2026-06-27 外部佐证(deep-research):** 这条"FP4 离 policy / bf16 上 policy"路线已是发表 SOTA——`FP4 Explore, BF16 Train`(arXiv:2604.06916, 2026-04)用近乎相同的契约(FP4 只跑 rollout、训练 bf16、log-prob 用 bf16 重算)报 ~2.5-3x rollout / 1.5-2x 端到端。配方可信,但其数在 H100;5090(sm_120,无 TMEM)的 FP4 tensor 路径不同 → **幅度必须本机重测,不能直接引用**。SVDQuant/Nunchaku NVFP4 在 5090 上对 FLUX 报 3x over bf16(仅 image,video 未实现);SageAttention3 FP4 attention 确认能在 sm_120 跑(5x over FA-2,有损)。

## 0. 一句话

policy denoise forward **保持 bf16**(动它就破 old_log_prob)。但 **reward model(Kling 视频 reward)和 VAE 不在 log-prob path 上** → 它们上 MXFP4(~4x)只会轻微改 reward/pixel,**只要保住组内 reward 排序,group-relative advantage 仍有效**。这是干净的越界手段。

## 1. 正确性：为什么 reward/VAE 的 FP4 是 RL-safe（条件)

GRPO 梯度 = `Σ A_i ∇log π`,`A_i` = 组内 reward 中心化。old_log_prob 来自 **denoise SDE 步**(bf16,不动)。reward/VAE 只进 `A_i`:
- **reward model FP4**:改 reward 绝对值,但 advantage 只看**组内相对排序** → **保住排序 = advantage 不变 = RL-safe**。
- **VAE FP4**:VAE decode 产 pixels 喂 reward → FP4 VAE 轻微改 pixels → 改 reward → 同样**只要排序保住**就安全(且全样本同一 FP4 VAE,系统性偏移被中心化吃掉)。
- **关键铁律**:**denoise policy network 绝不上 FP4**(改 old_log_prob,§[[SPRINT_lossless_diffusion_rl_research]] §2.2 verl 铁律)。

## 2. 为什么是 Triton（不是现成库）

- bf16 GEMM 上 **Triton 打不过 cuBLAS**(研究证伪,只追平)→ 给 reward/VAE 的 bf16 写 Triton 无意义。
- 但 **MXFP4 需要 `tcgen05.mma` 块缩放**,cuBLAS 不一定为 reward/VAE 的形状选 FP4 → custom Triton/CUTLASS(NVIDIA 官方 Triton Blocked-Scaled Matmul tutorial / CUTLASS SM100 block-scaled GEMM)是落点。
- caveat:FP4 近无损 PTQ(≤1%)只在**大 LLM 语言任务**证过,**未在视频 reward model / VAE 上验** → 必须自验排序 parity(§4)。

## 3. Phase plan

- **P0 — 量 reward+VAE 占 rollout compute 多少**:若 reward/VAE 只占 rollout wall-clock 一小片(denoise 主导),FP4 它们端到端收益小 → 先用 [[SPRINT_video_rollout_stage_overlap.md]] P0 的 stage 归因定值不值得。**reward+VAE compute 占比大才做。**
- **P1 — reward model MXFP4**:Kling 视频 reward 的大 GEMM 上 MXFP4(Triton tcgen05);**排序 parity 门**(§4)。
- **P2 — VAE MXFP4**:VAE decode 的 conv/GEMM 上 FP4;同排序 parity 门。
- **P3 — 端到端**:rollout wall-clock 下降 + eval reward 曲线不退化(排序保住则不退化)。

## 4. 验收（排序 parity，不是逐位）

- **reward 排序 parity**:同一批样本,FP4 reward/VAE vs bf16,组内 **Spearman 排序相关 ≈ 1.0**(advantage 只依赖排序)。排序变了就退回 bf16。
- **advantage parity**:`group_relative_advantages` 的 sign + 相对大小与 bf16 基线一致(`adv_zero_rate` 不升)。
- rollout wall-clock 下降 ≥ 目标(P0 占比定上界)。
- eval reward 曲线不退化(排序保住 → 学习信号不变)。
- **denoise log-prob 逐位不变**(证明没碰 policy path)——这是 RL-safe 的硬证。

## 5. 非目标

- **denoise policy network 绝不上 FP4**(污染 old_log_prob;要在 policy 上省 compute 走 [[SPRINT_approximate_single_gpu_perf.md]] 的稀疏注意力-当-policy,不是 FP4)。
- 不追绝对 TFLOPS 数(B200 数不转移到 5090,只有结构性 4x)。
- 不在 reward/VAE compute 占比小时硬做(P0 门)。
- 不和稀疏注意力 / async reward 同实验叠加(一次一变量)。

## 6. 关键文件

- reward:`vrl/rewards/runtime.py:InProcessRewardRuntime`、`vrl/rewards/functions/kling_video_reward.py`、`vrl/rewards/models/kling_video_reward.py`
- VAE:各家 `vrl/models/diffusion/<family>/` 的 decode 路径
- FP4 kernel 先例:`vrl/nn/quantization/fp8.py`（fp8 swap 机制,FP4 同构扩展)
- 排序 parity:`vrl/algorithms/advantages.py:group_relative_advantages`
- 证据:[[SPRINT_lossless_diffusion_rl_research.md]] §验证 7-9、记忆 `project_lossless_diffusion_rl_research`
