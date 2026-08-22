# SPRINT: 单卡近似加速杠杆 —— bf16 饱和之后还能做什么（已验证研究 + 本机实测）

状态：**INFO measurement archive（2026-07-18）**。本文只保存已验证研究与本机 profiling，
不是可执行 Sprint，也不拥有后续 action；所有实现责任都归到下文列出的现有 proof-gated Sprint。

> 来由:[[SPRINT_lossless_diffusion_rl_research]] 判"单卡 bf16 饱和 → 无无损杠杆"后,追问"Triton kernel / staged pipeline 是不是漏了"。研究纠正两处:① 单卡有真(近似)杠杆;② 我把"47% 空转"证据用错了(那是多卡训练阶段闲置,非单卡 stream 重叠)。

## 0. 一句话

两个独立结论:① **per-step compute**:DiT forward bf16 已饱和,要省只能近似——video 稀疏注意力(当 policy)或 MXFP4(离 policy);Triton 在 policy bf16 GEMM 上打不过 cuBLAS(证伪)。② **rollout pipeline 吞吐(实测新发现,§2)**:sbs=1 时整条 rollout 为 **49% GPU 利用率、31% 完全空转**；sbs=12 后完全空转降到 **14%**。DiT forward 饱和 ≠ rollout 饱和，但 31% 是 batching 前基线，不能全部算成 orchestration 收益；14% 才是 batching 后的残余上界。

## 1. 三个单卡近似杠杆（按 RL-safety × ROI 排）

### 杠杆 ①：稀疏视频注意力当 policy（最高 RL-safe 视频 ROI）⭐
- **数据**:**SLA 在 RTX 5090 实测 13.7x kernel / 2.2x 端到端(Wan2.1-1.3B)**——目标卡 + 目标模型族;VSA 2.53x e2e / 8x attn-FLOP（无 diffusion-loss 退化,after FT）;SVG2 2.30x;STA 2.98x(over FA2,可移植数)。video attention 极可压(13% 计算 = 95% recall）。
- **为什么 video 才值**:本机实测 attention 占比随帧数 11%(1帧)→47%(16帧),image 只 4-13%。所以稀疏注意力是 **video-only** 杠杆。
- **RL 正确性(关键)**:全部 approximate → **不能 drop 到冻结 policy(改 old_log_prob)**。RL-safe 的唯一方式:**稀疏注意力的网络 = 你训练的 policy**(in-loop 训,稀疏 forward 自洽就是 old_log_prob)。VSA/SLA 是 trainable 的(~2000-3000 FT 步,<0.1% 预训成本)。
- **后续责任边界**：VSA/SLA 是否进入 cosmos/wan policy 由
  `docs/sprints/parked/SPRINT_efficient_rollout_program.md` 的 attention/efficiency 轴负责；
  若 processor seam 不足、确实需要 repo-owned layer semantics，再触发
  `docs/sprints/parked/SPRINT_diffusion_native_transformer_executor.md` 的算法门。本文不拥有实现阶段。
- 来源:SLA 2509.24006、VSA 2505.13389、SVG2 2505.18875、STA 2502.04507、Compact 2508.12969。
- **2026-06-27 外部佐证(deep-research):VSA/SLA 确认是 trainable（稀疏模式烘焙进权重，非 inference-only），但跨 VSA/SLA/SLA2/PSA/LinVideo 全部检查——无一篇把稀疏注意力放进 RL 循环。** 所以"稀疏注意力当 policy 在 GRPO loop 里训"是干净的**原创方向**(杠杆里唯一兼具大数量级 2.2-2.5x 和无 prior-art 的)。唯一未被任何工作回答的正确性风险:若稀疏 top-k block routing 依赖 KV 内容,replay 时 batch 组成/数值微小差异可能改变选中 block → 破坏 old_log_prob 自洽。若归属 Sprint 解冻，第一道 proof gate 必须验证 rollout/replay routing 确定性一致。

### 杠杆 ②：MXFP4/NVFP4 离 policy path（Triton tcgen05）
- **数据**:Blackwell 第 5 代 tensor core `tcgen05.mma` 硬件块缩放 → MXFP4/NVFP4 **~2x FP8 / ~4x bf16**;**sm_120(5090)确认有真 FP4 加速**。custom Triton/CUTLASS 可调(NVIDIA Triton Blocked-Scaled Matmul tutorial / CUTLASS SM100)。
- **RL 正确性**:有损 → **只能离 policy/log-prob path**:**reward model、VAE、text-encoder 用 FP4 完全 OK**(不碰 old_log_prob);上 policy denoise 会改 Gaussian transition mean → 污染 old_log_prob(除非 IS 修正)。
- **caveat**:近无损 PTQ(≤1%)只在大 LLM 语言任务证过,**未在 video-DiT log-prob 上验**;且绝对数是 B200,5090 的吞吐没人测过(只有结构性 4x)。
- **Triton 在 bf16 上证伪**:研究实测 Triton 在 Blackwell 只**追平** cuBLAS,不超 → 给 policy 的 bf16 GEMM 写 Triton = 白写。Triton 的真家在 FP4 + 离 policy 组件。

### 杠杆 ③：单卡 staged pipeline（本机实测结论,见 §2）
- 我之前把"47% 空转"(2606.19004)当依据是**错的**:那是多卡系统训练阶段 spot-GPU 闲置(扩散同时完成无 straggler),**和单卡 stream 重叠 VAE/reward/text-encode 完全两轴**。单卡真相未被任何引用量化。
- 单卡逻辑:denoise/VAE/reward 都 GPU-bound → 抢 SM,真重叠不了;能回收的是 **orchestration 空转**(Ray actor 每轮 relaunch 重载模型、CPU 调度、stage 间数据搬运、denoise 步间 python)。靠**常驻 actor / prefetch / async reward**修,不是 kernel。
- **§2 本机 profiling 的唯一事实口径**：sbs=1 时完全 idle 为 **31%**；增大到 sbs=12
  后残余完全 idle 为 **14%**。前者不能全部归因给 orchestration，后者才是 batching 后的残余上界。

## 2. 本机 profiling：单卡 rollout 的 GPU-idle（staged pipeline 可回收多少）

方法:`nvidia-smi -lms 100` 采全卡 GPU 利用率,跑真 flux RL rollout(Flux 12B LoRA, 256², sbs=1, text-encode → denoise×10 → VAE → aesthetic reward, 真 Ray orchestration),量 rollout 期间 GPU 利用率分布。

**实测(2026-06-27, 96s rollout 窗口, 959 个 100ms 采样):**
```
GPU util:  mean 49%   median 69%
  完全 idle(0%):    31% 的时间
  <30%(近 idle):    39%
  <70%:             51%
  >=90%(饱和):      仅 17%
```

**结论:整条 rollout pipeline 远未饱和——这纠正了"rollout compute-bound"的说法。** DiT forward 本身饱和(之前 NCU 测的),但 rollout **只有 17% 时间在跑那个饱和 forward**,31% 完全空转。staged pipeline / batching 有真头空间。

**idle 的两个来源(需分离)**:
1. **小 batch 欠载**:sbs=1 + 256² 小图,单样本喂不满 GPU(我们之前测过 256² 是 under-utilized 区间)。**修法=调大 `sample_batch_size`**(已知杠杆,不是 stream 重叠)。
2. **orchestration / stage 间空转**:Ray per-cycle actor relaunch、CPU 侧 SDE/logprob 数学、stage 间数据搬运、denoise 步间 python。**修法=常驻 actor + async reward + stream prefetch**。

**分离实测(2026-06-27, flux sbs=1 vs sbs=12 A/B):**
```
sbs=1 : rollout 96s | util mean 49% | idle(0%) 31% | 饱和(>=90%) 17%
sbs=12: rollout 35s | util mean 77% | idle(0%) 14% | 饱和(>=90%) 75%
```
**分离清楚:31% idle 里 ~17 个点(→14%)是小 batch 欠载,~14% 是 orchestration 残余。**
- **小 batch 欠载 → `sample_batch_size=12`:rollout 直接 2.7x(96s→35s),util 49%→77%,无损一行配置。这是最大最便宜的 win。** flux 默认 sbs=1 白丢了 2.7x。
- **orchestration 残余 14%**:即便填满 batch 仍有 14% 全空转(Ray relaunch / CPU SDE-logprob / stage 间)→ stream 重叠/async/常驻 actor 的真目标,但收益小(~14%)、工程量大。

> **方法**:`nvidia-smi --query-gpu=utilization.gpu -lms 100`(全卡、进程无关,适合 Ray 多进程;粗但够分离 batch vs orchestration)。要精确归因到 stage(denoise/VAE/reward)需 nsys 时间线。

**cosmos video（predict2 2B, 240p_33f, sbs=1, dummy ref profiling-only）实测：比 image 更空**
```
                   util mean  median  idle(0%)  <30%   >=90%   rollout
flux  image sbs=1    49%       69%     31%       39%    17%     96s
flux  image sbs=12   77%       98%     14%       20%    75%     35s
cosmos video sbs=1   42%        4%(!)  26%       58%    41%     182s/8样本
```
**Cosmos video 是双峰**:41% 时间满载(denoise 33帧×35步)、但 **58% 时间 <30%(median 仅 4%)**。video 比 image 更空,因为非-DiT stage 大得多:**33帧 VAE decode + Kling 视频 reward model 前向 + 33帧×35步的 SDE/logprob CPU 数学**——全是 GPU 在等。
- video 的 sbs 难调大(240p_33f at sbs>1 可能 OOM)→ 欠载部分不好用 batch 填;
- 更多 idle 是**真·串行的非-DiT stage**(VAE/reward/CPU 数学)→ 单卡上它们都 GPU-bound 抢不了,但 **reward model 可与下一批 denoise 错峰(async reward)、CPU SDE/logprob 数学可与 GPU denoise 重叠(stream)** → 这俩是 video 的真 staged-pipeline 杠杆,比 image 更值。
- **要精确归因(VAE vs reward vs CPU 数学各占多少 idle)需 nsys 时间线**；是否继续由
  `docs/sprints/reading/SPRINT_diffusion_rollout_stage_pipeline.md` 的 profiling gate 裁决。


## 3. 诚实天花板（确认）

```
单卡 EXACT 杠杆:   无（DiT forward bf16 饱和 + elementwise compile 融）；唯一 exact = 多卡
单卡近似杠杆:
  ① 稀疏注意力（video, 2x e2e, 5090 实测）→ 必须当 policy 训（RL-safe，video-only）
  ② MXFP4 GEMM（4x）→ 只能离 policy path：reward/VAE/text-enc（RL-safe）
单卡 EXACT 吞吐杠杆（实测新发现,§2）:
  ③ staged pipeline / 调大 batch → rollout 实测只 49% 利用率、31% 空转！
     - 小 batch 欠载 → 调大 sample_batch_size（exact，简单）
     - orchestration 空转 → 常驻 actor + async reward + stream 重叠（exact）
策略修正:DiT forward 饱和 ≠ rollout 饱和。基线完全 idle=31%，batching 后残余=14%（③）;
         per-step compute 杠杆是 video 的 ①②（近似）。
```

## 4. 正确性铁律（贯穿）

任何动 policy denoise forward 的近似(稀疏 attn / FP4 / cache)都改 old_log_prob → **只在三种情况 RL-safe**:(a) 近似网络**就是**训练的 policy(稀疏 attn 当 policy);(b) 用在**离 policy path** 的组件(reward/VAE/text-enc);(c) 显式 IS 修正(TIS/RS,有 bias/variance 代价)。见 [[SPRINT_lossless_diffusion_rl_research]] §2.2 verl 铁律。

## 5. 结案与后续责任归属

- **P0 已完成**：§2 的测量已经回答问题；基线完全 idle=31%，batching 后残余=14%。
- **稀疏视频 attention**：移交
  `docs/sprints/parked/SPRINT_efficient_rollout_program.md`；需要自有层语义时再由
  `docs/sprints/parked/SPRINT_diffusion_native_transformer_executor.md` 接实施边界。
- **reward/VAE MXFP4**：移交
  `docs/sprints/parked/SPRINT_fp4_off_policy_reward_vae.md` 的 profiling-triggered gate。
- **request finalize / stage overlap**：分别由
  `docs/sprints/planned/SPRINT_rollout_finalize_overlap_ga.md` 和
  `docs/sprints/reading/SPRINT_diffusion_rollout_stage_pipeline.md` 持有；本文不再保留 P3。

## 6. 非目标

- 不把任何近似(稀疏/FP4/cache)drop 到**冻结** policy 上(污染 old_log_prob)。
- 不给 policy 的 bf16 GEMM 写 Triton(研究证伪,只追平 cuBLAS)。
- 不把 image 当稀疏注意力目标(attn 才 4-13%,不值)。
- 不重启已证伪的 paged-store/stepwise-batching/shared-prefix。

## 7. 源（已验证）

- 稀疏注意力:SLA 2509.24006（5090 实测）、VSA 2505.13389、SVG2 2505.18875、STA 2502.04507、Compact 2508.12969、SVG 2502.01776
- FP4/MXFP:NVIDIA Triton-on-Blackwell blog、NVFP4 blog、2512.02189（B200 微基准）、2509.23202（5090 FP4 加速）
- staged pipeline 误用纠正:2606.19004（"47% 空转"= 多卡训练阶段,非单卡 stream）
- 全文 + RL 正确性:[[SPRINT_lossless_diffusion_rl_research]]、记忆 `project_lossless_diffusion_rl_research`
