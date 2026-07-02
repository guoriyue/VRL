# SPRINT: Video rollout staged pipeline —— async reward + stream 重叠（填非-DiT 空转）

> **2026-07-01 架构更正（reward 执行层收敛到 inline + sleep/wake）：** d90ce04 的
> resident-reward Ray parking 被 revert 后，本轮把整个 **Ray reward actor pool 删掉了**
> （`vrl/rewards/ray/`、`vrl/ray/runtime.py` 的 `RayActorMethodRuntime`+`release_after_call`、
> resources/factory 的 pool 计数与 kwargs 注入）。替代：所有 reward **进程内打分**
> （`LocalRewardRuntime`），重型 reward 用 `reward.kwargs.<name>.sleep_offload=true` 借
> rollout lease 的 sleep/wake 语义在打分间隙 park 到 CPU（kling 已开，实测 ~1.6s/score
> steady vs 旧 pool 的 ~8s/步 actor 重载）。`release_rollout_before_reward`（rollout 先让卡）
> 保留不变——inline 大 reward 依然依赖它。本 sprint 内提到的 "reward pool 放第二张卡"
> 的多卡形态失去了传输层：cross_node 配方已在 header 标注 STALE；若将来要跨节点 reward，
> 需要新的 remote 传输而不是复活 actor pool。

状态：**planned → 机制基本落地,等待多卡吞吐验证（2026-06-27 二次复核,见 §7）**。性质：EXACT(无损)吞吐杠杆。**实测后主线变了:真正的 prize 是 reward stage(实测 14%,见下),不是生成侧那 9% 边界;而 reward 是 ≥2-GPU 杠杆(单卡显存装不下 rollout+reward 两模型,被 `release_rollout_before_reward` offload barrier 逼成串行)。** 二次复核（对照通过的测试 + 既有配置）发现：async-reward + per-stage-placement 这条线**绝大部分已经建好且已测**——placement/release 契约、`reward.gpu_pool` 语法、continuous producer 的 reward(N)∥generate(N+1) 重叠、late-group 版本丢弃**都已存在**。本轮又补上 cosmos continuous + `reward.gpu_pool=dedicated` 配方和 late-reward draining / non-draining 正确性测试。真正剩的是 ≥2/3-GPU 吞吐验证（唯一仍需第二/第三张卡）和单卡 worker/pool I/O overlap。详见 §7。

> **2026-06-27 实测结论(kernel-union + NVTX,真 cosmos run,推翻本 sprint 原假设):**
> - **生成侧不是大头**:denoise 循环 GPU-bound(96-98%);单卡 rollout 的 36% idle 几乎全在 **per-sample chunk 边界**(sbs=1=7 个边界),已用 **`sample_batch_size=4` 单卡回收(1.50x,64%→89% busy,已落配置)**。剩 ~9% 是非-tensor 边界,被 NCU 证明 tensor core 已到 5090 bf16 天花板(43%≈47%)→ 单卡 P2 重叠只能藏非-tensor 部分,ROI 低。
> - **真正的大 bubble = reward stage**:`collector.reward_score` 实测 **12.6s/group wall、rollout GPU 0% busy、= reward+denoise 的 14%**。根因:`reward execution: pool`(独立 actor)+ `release_rollout_before_reward` 把 rollout 2B 模型 offload 腾显存才能装 reward 模型 → **单卡被显存逼成串行(offload→reward→restore),不是 tensor 争用**。
> - **所以本 sprint 的 async-reward 从"deferred 配菜"升为主线**:reward pool 放第二张卡 → reward(group N) ∥ rollout(group N+1) + 干掉 offload barrier → 藏掉 14% 的 reward **tensor compute**(这部分单卡藏不了,要 ≥2-GPU)。详见 [[SPRINT_diffusion_rollout_stage_pipeline]] §4。
> - **⚠️ 更正(2026-06-27 晚,micro-benchmark 实测推翻上面两条悲观结论):** ① 上面说"单卡无法重叠(显存)/ROI 低"是**错的**——(a) 显存装得下(reward 常驻实测峰值 20.8GB/32GB),(b) 单卡 `compute∥(copy+CPU)` 实测能重叠(快 20%,copy engine+CPU ≠ tensor core)。② 边界是 **33%(44s)不是 9%**,其中 ~38s 是 copy(8s)+Python(30s)= **单卡可藏**(藏进下一个 denoise)。③ 单卡藏不了的只有 reward 的**纯 tensor compute**(那才需 ≥2-GPU)。**净:单卡 staged pipeline 是真杠杆(~33% 边界),见下方 P2/P3。**

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
- **P2/P3 — 单卡 staged pipeline(把边界的 copy+CPU 藏进下一个 denoise)。实测背书,2026-06-27:**
  - **能藏什么 / 不能藏什么(micro-benchmark 实测,`vrl/scripts/perf/single_gpu_overlap_{compute,copy}_probe.py`):**
    ```
    compute ∥ compute        → NEUTRAL（2×2B forward,双流 1.1% = 噪声;tensor core 串行)
    compute ∥ (copy+CPU)     → 藏得了（denoise 1680ms ∥ 400ms copy+CPU → 并发 1687ms ≈ compute 单独,快 20%)
    ```
    copy engine(DMA)和 CPU 是和 tensor core **分开的硬件** → 数据搬运 + Python 编排能藏进 GPU compute。**44.3s 边界里 ~38s 是 copy(8s)+Python(30s),全可藏；不能藏的只有 reward 的纯 tensor compute。**
  - **结构 = staged pipeline(sample N+1 denoise ∥ sample N 的 VAE/transfer/reward-prep/Python-setup)。两个独立 enabler:**
    - **① async copy + 独立 stream**:把同步/SM-kernel 的拷贝改 `cudaMemcpyAsync`(走 copy engine,让出 SM)→ 重叠在硬件上真并发。
    - **③ 常驻 actor**:去掉 colocated per-cycle relaunch 的模型重载 + Python 开销(边界 ~30s host 工作的大头)。
    - **双缓冲 latent/observation = 此 pipeline 的【内在存储】,不是独立杠杆**(N+1 写新 latent 时 N 的还在被 reward/VAE 读,必须 ≥2 份;缓冲深度 = pipeline 重叠级数,是 pipeline 参数)。
  - **验收**:reward/old_log_prob/advantage 逐位不变(纯重叠,无 staleness)；rollout wall 下降 ≤ 边界占比(33%)。
- **⚠️ 不要走的弯路(实测撞过)**:让 reward **模型常驻**和 rollout 同卡并发(`reward.resident_overlap` enabler,resources.py +33 行已建、20.8GB 实测装得下)——但 ① reward 的 Qwen2-VL **compute** 和 denoise 抢 tensor core(藏不了),② Ray reward pool actor + resident rollout 在单卡 **死锁 hang**。正确路是上面的"藏 copy+CPU",**不需要 reward 模型常驻**,绕开死锁。
- **更正旧结论**:本 sprint 顶部曾写"单卡无法重叠(显存)"——**错**。显存装得下(20.8GB),且 copy+CPU 能重叠;单卡 staged pipeline 是真杠杆(~33% 边界),不是只能 ≥2-GPU。≥2-GPU 才需要的是藏 reward 的 **tensor compute**(单卡那部分藏不了)。

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

## 7. 2026-06-27 二次复核 — 机制基本落地（对照通过的测试与既有配置）

本节是对 §0-§6 计划的**实证复核**：本 sprint 提的 async-reward + per-stage-placement，
**大部分机制已经存在并已被测试覆盖**，不是待从零开工的 build。本轮又补上 recipe 与
late-reward correctness tests。下面每条都给代码/测试证据。完整版见
[[SPRINT_diffusion_rollout_stage_pipeline]] §4.4。

**(a) placement + release barrier 的消除契约：已建，已测。** §1/§2 的核心是「reward 放第二
卡 → 干掉 `release_rollout_before_reward` offload barrier」。这个推导**已经实现**：reward 与
rollout GPU disjoint 时，resolver 自动把 `release_rollout_before_reward` derive 成 `False`，
barrier 自动消失。证据 `tests/ray/test_resources.py`：

```text
:579-665  test_dedicated_reward_gpu_derives_resident_lifecycle_when_unset
          test_lifecycle_plan_resident_when_roles_disjoint
            disjoint reward GPU -> rollout resident, release_rollout_before_reward=False, 无 handoff
:867-917  test_colocated_reward_on_dedicated_gpu_owns_its_own_bundle
            reward 独占 GPU1 -> 自己的 bundle，与 rollout bundle disjoint
本 session 跑：15 passed。
```

**(b) per-stage placement 配置面（reward stage）：已建，已测。**
`distributed.resources.reward.gpu_pool: auto|rollout|dedicated` 已有完整语法 + 5 条测试
（`test_resources.py:1019-1113`，含 auto 抢空闲卡、与 legacy 等价、互斥校验、未知值拒绝）。

**(c) reward(N) ∥ generate(N+1) 重叠：已是 continuous 模式既有行为。**

```text
vrl/rollouts/orchestration/continuous/producer.py
  bounded asyncio inflight 集合，每组一个 task（含 generation+reward），
  max_inflight_groups>=2 时第 N 组与第 N+1 组同时在飞 -> §1 的「async 安全重叠」已就绪
configs/.../sd3_5/online_grpo_ocr_single_gpu_async_debug.yaml
  已有一份可跑的单卡 continuous async 配方（max_inflight_groups: 2）
```

**(d) late group 正确性：version-stamp + max_stale + drop_stale 已兜住**
（`staleness.py` 的 `StalenessPolicy.admit/too_stale`、`scheduler.py:8-10`）。reward 与 policy
无关，迟到组按版本丢弃，不污染 on-policy 训练——这正是 §4「不引入 staleness」的硬保证。

**本轮已补：**

```text
(1) configs/experiment/diffusion/cosmos_predict2/online_grpo_async_reward.yaml
    组合 cosmos continuous + rollout.gpu_pool=dedicated + reward.gpu_pool=dedicated。
(2) tests/ray/test_resources.py::test_cosmos_async_reward_recipe_resolves_resident_reward_overlap
    锁住 recipe 解析出的 disjoint reward layout、resident reward、no rollout-before-reward release、
    continuous.max_inflight_groups>=2。
(3) tests/rollouts/orchestration/continuous/test_contracts.py::
      test_late_reward_finishes_before_version_bump_under_draining
      test_late_reward_group_dropped_under_non_draining_max_stale_0
    锁住 late reward 在 draining / non-draining 两条 barrier 语义下不会污染训练。
```

**真正剩下的缺口：**

```text
(1) 吞吐验证（需要 ≥2/3 GPU）：实测那 ~14% reward stage 被藏住、barrier 消失、
    验证 §[[SPRINT_diffusion_rollout_stage_pipeline]] 的 1/max 模型。
(2) 单卡 worker/pool I/O overlap：`forward_chunks_pipelined` 必须落在 `worker._to_cpu`
    + Python orchestration 这个生产瓶颈上;executor 内部 stage 再拆不是答案。
(3) data-parallel rollout 摸底：denoise 吞吐扩展优先用 N 个完整 rollout actor。
    任何 denoise stage-split 前必须先证明它打得过 data-parallel baseline。
```

**教训：先核对已有测试/配置，再 scope build。** 早先 do-next 把已存在的 placement/release
契约、gpu_pool 语法、async 重叠当成「要新建」来立项——根因是其 reader 映射了**源码**却没映射
**既有测试套件**（`tests/ray/test_resources.py` 的 placement/lifecycle 测试 + async debug
配方本就证明这些机制是活的）。scope build 前必须 grep `tests/` 与 `configs/` 确认是否已被测被配。
