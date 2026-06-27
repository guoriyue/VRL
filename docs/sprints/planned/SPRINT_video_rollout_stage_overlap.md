# SPRINT: Video rollout staged pipeline —— async reward + stream 重叠（填非-DiT 空转）

状态：**planned（2026-06-27 实测重定向）**。性质：EXACT(无损)吞吐杠杆。**实测后主线变了:真正的 prize 是 reward stage(实测 14%,见下),不是生成侧那 9% 边界;而 reward 是 ≥2-GPU 杠杆(单卡显存装不下 rollout+reward 两模型,被 `release_rollout_before_reward` offload barrier 逼成串行)。**

> **2026-06-27 实测结论(kernel-union + NVTX,真 cosmos run,推翻本 sprint 原假设):**
> - **生成侧不是大头**:denoise 循环 GPU-bound(96-98%);单卡 rollout 的 36% idle 几乎全在 **per-sample chunk 边界**(sbs=1=7 个边界),已用 **`sample_batch_size=4` 单卡回收(1.50x,64%→89% busy,已落配置)**。剩 ~9% 是非-tensor 边界,被 NCU 证明 tensor core 已到 5090 bf16 天花板(43%≈47%)→ 单卡 P2 重叠只能藏非-tensor 部分,ROI 低。
> - **真正的大 bubble = reward stage**:`collector.reward_score` 实测 **12.6s/group wall、rollout GPU 0% busy、= reward+denoise 的 14%**。根因:`reward execution: pool`(独立 actor)+ `release_rollout_before_reward` 把 rollout 2B 模型 offload 腾显存才能装 reward 模型 → **单卡被显存逼成串行(offload→reward→restore),不是 tensor 争用**。
> - **所以本 sprint 的 async-reward 从"deferred 配菜"升为主线,且明确是 ≥2-GPU 杠杆**:reward pool 放第二张卡 → reward(group N) ∥ rollout(group N+1) + 干掉 offload barrier → 藏掉 14%。单卡无法重叠(显存)。详见 [[SPRINT_diffusion_rollout_stage_pipeline]] §4(多卡 gate 已由此开)。

> 实测来由([[SPRINT_approximate_single_gpu_perf.md]] §2):flux image rollout sbs=1 GPU 只 49% 利用率、31% 空转 → 调大 `sample_batch_size` 填到 77%(2.7x,干净)。**video 不一样**:240p_33f sbs 难调大(OOM),空转来自更大的串行非-DiT stage(33帧 VAE + Kling 视频 reward model 前向 + 33帧×35步 CPU SDE/logprob)。repo 已有 profile 说 cosmos **denoise 是 SM 100% compute-bound** → 空转在 denoise 之外。

## 0. 一句话

video rollout 的 GPU 在 denoise 时满载,但在 **reward 打分 / CPU 数学 / VAE / orchestration** 时空着。这些都不在 policy path 上(不碰 old_log_prob)→ **可与下一组的 denoise 错峰重叠,无损**。现在它们串行(`release_rollout_before_reward` 是个硬 barrier:rollout 释放 → 才打分 → 才训练)。

## 1. 正确性：为什么这是 EXACT（无损）

- reward 打分**在 policy path 之外**:old_log_prob 来自 denoise 的 SDE 步,reward 只进 advantage。reward 算得早算得晚不改 log-prob → async 安全。唯一约束:reward 必须在该 batch 的 advantage/训练步**之前**完成。
- CPU SDE/logprob 数学:是 denoise 步**之间**的 host 计算,与下一步 GPU forward 无数据依赖时可 overlap(双缓冲)。
- VAE decode:产 pixels 喂 reward,也在 log-prob 之外。
- **全程不改采样分布、不改 loss、不改 old_log_prob** → 无 IS、无 variance 代价,是真无损(与 §[[SPRINT_approximate_single_gpu_perf.md]] 的近似杠杆 ① ② 不同)。

## 2. 当前串行点（代码证据）

- **reward barrier**:`vrl/rollouts/collector/core.py:222` `release_rollout_before_reward`——现在 rollout 模型先 release(offload),再打分。reward 期间 GPU 空(denoise 不在跑)。
- **reward scorer**:`vrl/rollouts/collector/rewards.py:RewardScorer`、`vrl/rewards/runtime.py:LocalRewardRuntime`(Kling 本地视频 reward model)。
- **stage 计时**:`vrl/generation/diffusion/executor.py:154,478` `stage_durations` + `record_function`——P0 归因用。
- **CPU 数学**:`vrl/math/diffusion/flow_matching.py:sde_step_with_logprob`(math fp32,denoise 循环内每步调)。

## 3. Phase plan

- **P0 — 干净 stage 归因 ✅ 已做（nsys + NVTX wiring, 真 cosmos run 2026-06-27, 在 ~/Desktop/vrl2/VRL）**:
  - **方法(可复用)**:① `record_function` 加 NVTX wiring(`vrl/utils/profiling.py`,gated on `VRL_PROFILE_COLLECT=1`,push/pop `torch.cuda.nvtx`)→ nsys 看得到 stage;② `VRL_PROFILE_COLLECT=1 nsys profile --trace=cuda,nvtx --trace-fork-before-exec=true python -m vrl.scripts.train ...`（**`--trace-fork-before-exec=true` 是抓 Ray worker 的关键**,没它 nsys 录 0 kernel）;③ `nsys stats --report nvtx_gpu_proj_sum`。
  - **实测(cosmos predict2 2B, 240p_33f, 8样本×35步, GPU-time vs wall-clock):**
    ```
    stage                       GPU-time  wall    GPU忙%   解读
    generation.denoise_forward   78.2s    65.7s  ~100%    GPU 饱和（DiT 计算,动不了）
    generation.decode_latents     7.28s    7.28s ~100%    VAE,GPU 忙
    generation.prepare_sampling   4.86s    4.87s ~100%    GPU 忙
    generation.prompt_encode      1.05s    1.06s ~99%     GPU 忙
    generation.scheduler_step     0.14s   12.70s ~1%      ← 不是 idle(projection 假象)
    ```
  - **⚠️ 量 GPU-idle 必须用 kernel-interval UNION,别用 nsys projection。** projection 的 "Proj Time" 算 range 内 *launch* 的 kernel 的 GPU 时间,异步会溢出 range → GPU-time > wall(denoise_forward 78.2s GPU > 65.7s wall),误导。**正确:`nsys export --type sqlite` 后对 `CUPTI_ACTIVITY_KIND_KERNEL` 的 [start,end] 求并集 / 窗口 wall = 真 GPU-busy。**
  - **实测真值(kernel-union):**
    ```
    denoise 循环内:     96-98% GPU-busy（compile on/off 都是）= GPU-BOUND
    rollout 整体:       64% busy / 36% idle
    134s 窗口拆解:      denoise 78.4s(58%)+VAE 6.4s(5%)+prepare 4.2s(3%)+encode 0.8s(1%)
                       + 未归因 orchestration gap 44.3s(33%)  ← 真 idle 在这
    ```
  - **决定性:denoise 循环 GPU-bound(动不了;compile 1.25x 是 fusion 减工作量不是填 idle);真 idle = ~33% per-sample orchestration gap**(8 样本 × ~5.5s/样本 Ray actor handoff/Python/latent transfer/非-denoise setup,GPU 基本空,全是 <0.08s 微 gap 无大 stall)。step_index 已传、generator 已 CUDA、scheduler_step 非 idle —— 全排除。
  - **gap 源已 pin(CUDA-API + gap 结构实测):** 44.9s idle = **7 个 sample 边界**(sbs=1,8 样本串行),每个 ~7.9s、**81% GPU-idle**;step 之间 0.0s gap(背靠背)。每边界 = cudaMemcpyAsync 8.0s(latent/observation GPU↔CPU 搬)+ cudaLaunchKernel 7.8s(VAE/encode 碎 kernel launch 开销)+ 10.7s 真 VAE/encode GPU 活 + ~30s 纯 host Python(建下个样本/写 trajectory buffer/decode setup)。**非 sync-bound**(边界里 cudaStreamSynchronize 仅 0.1s;那 12.6s sync 在 denoise 循环内被覆盖)。**非 Ray relaunch**(一个 request 内)。
- **结论(验证后):staged-pipeline 对 video 有真杠杆,在 per-sample 边界(orchestration 层),不在 denoise/SDE。** 潜在 ~1.3-1.5x rollout,与 compile 正交。
- **P1 ✅ 已验证(2026-06-27)— `sample_batch_size`↑ 是答案,1 行配置**:实测 cosmos 240p_33f / n_samples=8 / 1×32GB:

  | sbs | chunks | 边界 | rollout wall | GPU-busy | speedup | 显存 |
  |----|----|----|----|----|----|----|
  | 1（当前）| 8 | 7 | 134.1s | 64% | 1.00x | fit |
  | 2 | 4 | 3 | 103.0s | 77% | 1.30x | fit |
  | 4 | 2 | 1 | 89.2s | 89% | **1.50x** | **FIT ✓** |

  **设 `rollout.sample_batch_size=4` → 1.50x rollout,89% GPU-busy,放得下**(online_grpo_fullparam_8bit_240p.yaml:52 当前是 1)。每翻倍 sbs 砍半边界 → 填 boundary idle。**精确无损**(sbs 只控制 n 样本里几个共享一次 forward batch 维,每样本 noise/denoise/log-prob 不变;planner 本就支持 fixed-size sample chunks)。**推翻旧假设"video can't batch (OOM)"——240p_33f 至少 batch 到 4。** sbs=8(1 chunk,0 边界)未测,OOM 风险 + 边际小(~89%→~94%)。
- **P2 — pipeline 边界与下一 denoise(结构修法,sbs 受显存逼成 1 时走这条)**:把 sample N 的 VAE-decode+transfer+Python-setup 与 sample N+1 的 denoise forward 双缓冲重叠 → 隐藏 45s idle。验收:逐位不变(纯重叠)。
- **P3 — 砍 transfer + 常驻 actor**:8s memcpy(observation/latent GPU→CPU)留 GPU/批量搬;colocated 每轮 relaunch 重载模型 → 常驻 actor 去 orchestration 空转。

## 4. 验收（无损铁律）

- **逐位不变**:reward 值、old_log_prob、advantage、loss 与串行基线逐位相等(只是 wall-clock 变)——这是"无损"的硬证。
- rollout+reward wall-clock 下降 ≥ 目标(由 P0 的 reward/CPU-math 占比定上界)。
- eval reward 曲线、drift guard、TIS-RS 与基线一致(无损 → 必须完全一致)。
- 不引入 off-policy / staleness(这是纯重叠,不是异步 RL)。

## 5. 非目标

- 不做多卡 rollout/train 分离(那是另一轴,需 ≥2 GPU)。
- 不改采样/loss/log-prob(任何改这些的就不是本 sprint 的无损杠杆)。
- 不在 image 上做(image 调大 `sample_batch_size` 已够,见 §[[SPRINT_approximate_single_gpu_perf.md]] §2)。
- 不把 reward 做 off-policy 复用(staleness 是 multi-GPU async 的事)。

## 6. 关键文件

- `vrl/rollouts/collector/core.py:222`（reward barrier `release_rollout_before_reward`）
- `vrl/rollouts/collector/rewards.py:RewardScorer`、`vrl/rewards/runtime.py:LocalRewardRuntime`
- `vrl/generation/diffusion/executor.py:154,478`（`stage_durations` / P0 归因）
- `vrl/math/diffusion/flow_matching.py:sde_step_with_logprob`（CPU 数学）
- 实测依据:[[SPRINT_approximate_single_gpu_perf.md]] §2、记忆 `project_rollout_bound_class_probe`
