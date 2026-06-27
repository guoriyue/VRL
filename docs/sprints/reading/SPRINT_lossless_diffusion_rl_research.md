# SPRINT: 无损（exact）加速 diffusion RL —— 外部研究综述 + 对标我们的实测

状态：**reading / 已验证（2026-06-26）**。性质：**deep-research 工作流产出(23 源 → 101 claims → 25 对抗验证 → 22 确认 / 3 证伪)，逐条对标我们本轮三个 probe 的实测，产出"无损杠杆清单 + 天花板判断 + 门控"，不是功能移植。**

> 问题：对 diffusion/flow-matching RL 后训练(GRPO/Flow-GRPO，SD3.5/Flux/Wan/Cosmos)，有哪些**无损(不改输出分布)**的系统/kernel 手段能省显存(装更大 batch)或省计算?
> 实测前提(本仓库已测，RTX 5090 / SD3.5-medium 1024²)：rollout DiT **compute-bound**(ms/sample batch 1→16 全平),compile 后 **94% MFU**(69%→94%，1.37x)。证据：记忆 `project_rollout_bound_class_probe`、`project_lossless_diffusion_rl_research`、`vrl/scripts/perf/{attention_fraction,rollout_bottleneck,dit_mfu,backward_mfu}_probe.py`。
> 相关：[[SPRINT_signal_paged_rollout]]（shared-prefix = 唯一待证的无损 group 级杠杆）、[[SPRINT_training_mfu_compile]]、[[SPRINT_rl_safe_feature_cache_probe]]（近似路径的正确性门）、本轮证伪的 [[SPRINT_paged_trajectory_store]] / `SPRINT_diffusion_stepwise_batching_probe`。

## 0. 一句话（诚实的天花板 —— 二次修正：image 和 video 都在 bf16 硬件天花板上）

**最终结论：image 和 video 都已接近 bf16 硬件峰值。早前"video 有 45% 头空间"是拿错峰值(419)算的,已被 GEMM 微基准证伪。**

- **真实峰值**:RTX 5090 bf16 dense(fp32 累加)实测 **~232 TFLOPS**(`gemm_peak_probe`),不是 419——419 是 fp8/稀疏的 "AI TOPS" 数。消费卡 bf16 tensor 用 fp32 累加是半速。
- **IMAGE(SD3.5)**:compile 后 ~94%(对真实峰值)。已饱和。
- **VIDEO(cosmos)**:按真实 232 峰值重算,40.7 TFLOP / 187ms = 218 TFLOPS = **~94% 饱和**(不是 51%)。GEMM 实测 ~232-245 TFLOPS = **就在 bf16 天花板上**,`cutlass_80` kernel 不是"旧/慢",是到顶了。
- **结论**:bf16 上没有 GEMM 无损杠杆(已到硬件极限)。想更快只能 fp8/fp4(**有损 → 只能离 policy path**,repo 已有 fp8 rollout)。注意力也已被复核收敛：Blackwell sm_120 没有 FA-3，flash≈cuDNN，batch=4 flash attention tensor SOL 已到方阵 GEMM 的 91%。其余无损大杠杆只剩 RL pipeline(分离 rollout/train,需多卡)。

> 用词更正:这些技术是**算法级 exact(fp32 累加、重排 IO),不是 bit-for-bit**。3 个被证伪的 claim 全是 over-claim bit-exactness。对 RL 这恰是对的标准:它们**不引入系统性 policy-vs-old-policy 偏差**(只有浮点重排噪声,naive 参考也有);cache/量化引入的才是系统性偏差。

## 0.5 已 profile 的模型汇总（RTX 5090 sm_120, torch 2.11+cu128）

> caveat:SD3.5 是**真权重**(HF checkpoint);cosmos/wan 是**合成权重**(真架构+真维度,随机权重)——profile timing/MFU/kernel 有效(只依赖 shape/kernel),不可用于质量/数值正确性。MFU 数字(94%/51%/56%)是早期用错峰值(419)算的、**不可信**;饱和判断以 NCU tensor-pipe SOL 为准(验证 9)。

| 模型 | 类型 | 真/合成 | 跑了什么 probe | 关键结果(已用 NCU 校正) |
|---|---|---|---|---|
| **SD3.5-medium** (2.24B) | image DiT 1024² | 真权重 | rollout_bottleneck / dit_mfu / backward_mfu / attention_fraction / shared_prefix_divergence / CUDA-graph | rollout compute-bound(ms/sample 平);compile 1.37x;train compile ~1.2x+省~20%显存;attn 4-13%(2048² 才 35%);**shared-prefix 多样性塌=死**;CUDA graph +0.7% |
| **cosmos-predict2.5** (1.96B) | video DiT | 合成 | video_dit_mfu / video_op_breakdown / **ncu** | 帧数↑→attn 11→47%;per-op MATMUL 60%/ATTN 36%/NORM 3.5%;**NCU: GEMM tensor SOL 45% ≈ 方阵 47% = 饱和** |
| **wan_2_1** (14.29B) | video DiT | 合成 | video_dit_mfu | attn 6-17%(d_model 5120 更 linear-bound) |

**非模型微基准(5090 实测):** 纯 bf16 方阵 GEMM 真实峰值 **~232 TFLOPS**(非 vendor 419);SDPA flash≈cuDNN(~185 TFLOPS,噪声内);NCU 方阵 GEMM tensor SOL **47%** = bf16+fp32 累加消费卡上限。
**未 profile(缓存但没跑):** cosmos-predict2 (2B)、cosmos-predict2.5-14B、FLUX 系列、Anima。

## 0.6 本机实测验证（2026-06-26 初测；2026-06-27 NCU 复核, RTX 5090）

跑了三个 probe 把报告里对我们最相关的两条轴实测掉(数存 `project_lossless_diffusion_rl_research`；脚本保留用于换 GPU / PyTorch / 模型形态后重跑)。

### 验证 1：注意力占比随分辨率（→ FA-3 对我们有没有用）
`vrl/scripts/perf/attention_fraction_probe.py`：

```
side   img_tok   seq     lin TFLOP  attn TFLOP  attn%   ms/fwd
512    1024      1357    12.2       0.5         4%      54.5
768    2304      2637    23.7       2.1         8%      86.6
1024   4096      4429    39.7       5.8        13%     160.3
1536   9216      9549    85.7      26.9        24%     449.3
2048   16384    16717   150.0      82.4        35%    1020.6
```

**结论：attention 是 O(seq²),图像 ≤1024² 只占 4-13% FLOP → FA-3/注意力 kernel 升级天花板很低(linear/MLP-bound,compile 已啃)。1536²+ 升到 24-35%,video token 数更高才会 attn-dominated → 那时 FA-3 才值。** 所以无损 compute 杠杆随 seq 增长从"MLP 融合(compile,已有)"漂移到"attention(FA-3,video 才需要)"。

### 验证 2：shared-prefix 多样性保留（→ 报告点名的唯一待证无损 group 杠杆）—— 负结果
`vrl/scripts/eval/shared_prefix_divergence_probe.py`，G=6 T=28 768²，用 repo 真实 `sde_step_with_logprob`：

```
            noise_level=1.0          noise_level=1.4(高噪声上界)
k    fwd_saved  retention            retention
0    0%         89%(sanity)          87%
4    11.9%      44%                  61%
8    23.8%      38%                  48%
14   41.7%      31%                  36%
20   59.5%      22%                  29%
24   71.4%      13%                  23%
28   83.3%      0%                   0%
```

**结论:只共享前 4/28 步(省 12%)多样性就塌到 44%;[[SPRINT_signal_paged_rollout]] P0 的验收门(1.5x forward≈k14 时保留 ≥70%)实测只有 31-36%,直接没过。** 提高 SDE 噪声(1.0→1.4,已超 Flow-GRPO 默认)有缓解(k4 44→61%)但救不回来——扩散早期步锁全局结构是本质。**→ shared-prefix 作为无损 group 级杠杆,对 SD3.5 GRPO 实测为死。**

> caveat:这是 latent 多样性(reward variance 的代理),非 reward 本身;单 prompt / SD3.5 / 768²。但 k=4 就塌到 44% 的悬崖太陡,reward-model 版 P0 翻盘概率很低。signal_paged 据此应 park(见该 sprint §9 自己写的关闭条件)。

### 验证 4：CUDA graphs（→ SGLang 重投的 launch-bound 利器对我们有没有用）
`vrl/scripts/perf/dit_mfu_probe.py --compile-mode reduce-overhead`（CUDA graphs）vs `default`（纯 inductor 融合），1024² batch4：

```
eager                         628.6 ms   69%
compile default(无 graph)       463.5 ms   94%
compile reduce-overhead(graph)  460.4 ms   94%   ← +0.7%,在噪声里
```

**CUDA graphs 在我们 compute-bound 的 DiT 上 ≈ 零收益(460 vs 463)。** SGLang/vLLM 重投 CUDA graphs,是因为 LLM 自回归 decode 是 launch-bound(batch-1、几千小 kernel、GPU 在 kernel 间饿);compile 之后我们的 launch 泡泡已经没了,graph 没东西可回收。**这条实测把"SGLang custom 东西 = 解 launch-bound serving"这个 regime 差异钉死了。**

### 验证 3：融合 AdaLN kernel 头空间（→ image/video 都不作为当前 kernel sprint）
`vrl/scripts/perf/dit_mfu_probe.py` 已测 SD3.5 eager 69% → compile 94%,compile 靠的就是融合那些 norm/AdaLN/elementwise 的 bandwidth-bound 算子。**所以手写融合 AdaLN 在 SD3.5 image 上只剩 ~6% 残余头空间(94→100%),不值得。** video 侧后来由 per-op + NCU 复核收敛：compile 后 NORM_ELEM 约 3.5%，主 compute kernel 已接近同机 bf16 上限，也不值得开单卡 AdaLN kernel sprint。

### 验证 5：旧解析结果：VIDEO DiT 看似没饱和（已被验证 7/9 推翻）
`vrl/scripts/perf/video_dit_mfu_probe.py`（合成真实 dims,扫 latent 帧数,compile A/B @ 8 帧）：

```
cosmos-predict2.5 (1.96B, d_model=2048)        wan_2_1 (14.29B, d_model=5120)
frames vid_tok attn%  ms/fwd                    frames vid_tok attn%  ms/fwd
1      880     11%    38                         1      390     6%     91
4      3520    21%    116                        4      1560    7%     283
8      7040    32%    232                        8      3120    11%    499
16     14080   47%    596                        16     6240    17%    1070

eager(8帧) MFU 42% → compile 51% (1.23x)        eager 48% → compile 56% (1.17x)
对比 SD3.5 image: compile 94% (1.37x)
```

**这段是旧解析，不再作为结论。** 当时的 51% / 56% 来自 `video_dit_mfu_probe` 的解析 FLOP 估计，并且使用了错误的 vendor peak=419 分母；后续 `gemm_peak_probe` 和 NCU tensor SOL 证明主 bf16 compute kernel 已到同机有效上限。保留这张表只用于说明当时为什么启动 per-op/NCU 复核，不能再解释成"近一半 tensor-core 算力空着"。

旧家族差异只保留为 shape 事实:
- **cosmos**：attention FLOP 占比随帧数升到 32-47%(16 帧近 attn-dominated)，但 Blackwell 上没有可用 FA-3，flash≈cuDNN，batch=4 NCU flash attention 已接近方阵 GEMM tensor-SOL 上限。
- **wan**：14B 大模型 + d_model 5120，attention 11-17%，更 linear-bound；但不能从这个表直接推出手写 AdaLN/GEMM kernel 有杠杆。

> 纠偏:合成权重对 kernel shape 有效，但解析 FLOP 估计和 peak 分母不足以判饱和。51% vs 94% 的差不是"估计误差小所以可信"，而是分母口径错。最终以验证 7/9 为准。

### 验证 6：video 头空间到底在哪（per-op 分解，修正"融合 AdaLN"猜测）
`vrl/scripts/perf/video_op_breakdown_probe.py`（torch.profiler,cosmos 8 帧,compile 前后按 kernel 分桶）：

```
            EAGER          COMPILED
MATMUL      49.1%          60.2%   cutlass_80_tensorop (Ampere SM_80 kernel 跑在 Blackwell SM_120)
ATTENTION   28.5%          36.3%   pytorch_flash::flash_fwd (FlashAttention-2)
NORM_ELEM   20.5%           3.5%   ← compile 已融掉(triton_red_fused_layer_norm)
```

**修正 1(compile 已融 AdaLN):** torch.compile 把 AdaLN/RoPE/modulation 从 20.5% 融到 3.5% → 手写融合 AdaLN 端到端只剩 ~2-3%,不是杠杆,砍掉。

**修正 2(GEMM 已到 bf16 天花板,不是"用了旧 kernel"——见验证 7/9):** `cutlass_80` GEMM 实测 ~245 TFLOPS,而 GEMM 微基准测出 5090 bf16 dense 真实峰值就 ~232 TFLOPS；NCU 又确认 cosmos 主 GEMM 45.29-45.33% tensor SOL，和 8192³ 方阵 GEMM 47.48% 同区间 → **GEMM 到顶了,不是慢**。"60% 时间在 Ampere kernel = 头空间"的判断**错了**(基于错峰值 419)。注意力也被验证 8/9 关闭。落地 [[SPRINT_cosmos_video_mfu_kernels]]。

### 验证 7：5090 真实 bf16 GEMM 峰值（→ 推翻"video 有 45% 头空间"，那是错峰值算的）
`vrl/scripts/perf/gemm_peak_probe.py`（cuBLAS bf16 dense, fp32 累加）：

```
4096³  162 | 8192³  214 | 12288³  232 | 16384³  231  TFLOPS  → 真实峰值 ~232
```

**之前所有 MFU 拿 peak=419 算 → 错。419 是 5090 的 fp8/稀疏 "AI TOPS",bf16 dense(fp32 累加,消费卡半速)实测 ~232。** 按 232 重算:SD3.5 image ~94%、cosmos video ~94%(40.7 TFLOP/187ms=218)、cosmos GEMM ~245=到顶。**image 和 video 都已饱和在 bf16 硬件极限,没有 bf16 GEMM 无损杠杆。** torch 已为 sm_120 编译(`arch_list` 含 sm_120),`cutlass_80` 的 s16816 MMA 跑到了 bf16 上限——不是配置错。**教训:MFU 的分母必须每台机器实测,别用 vendor headline。**

### 验证 8：注意力杠杆也死了 + 落地 gpu_preflight（最终收敛）
- **FA-3 是 Hopper(sm_90)专属**:flash-attn 最新 2.8.3 = FA-2,Blackwell sm_120 没有 FA-3 build;torch 已内置 FA-2(`pytorch_flash::flash_fwd`),装 flash-attn ≈ 零增益。
- **SDPA 后端实测打平**:cosmos attention shape 上 flash 187 / cuDNN 183 TFLOPS,胜负随热态/warmup 在 ~5% 噪声带翻转;attention 已 ~80-89% of bf16 GEMM 峰值 = 实际上限附近。**无稳健注意力杠杆。**
- **唯一落地交付**:`vrl/scripts/perf/gpu_preflight.py`(perf-only + 测试)——MFU probe 显式 log 本机真实 bf16 峰值(MFU 正确分母)、arch 匹配、最快 SDPA 后端;所有 MFU probe 默认峰值改成实测(`measured_bf16_peak_tflops`)。**根因(用 419 当峰值)根治。**
- **最终结论**:image 和 video 都已饱和在 bf16 硬件极限,**单卡无损 kernel 杠杆全部证伪**。剩下的只有 fp8(有损,离 policy path,repo 已有)和多卡 pipeline 分离(需 ≥2 GPU)。

### 验证 9：NCU 硬件计数器复核（→ 纠正"解析 MFU 不可信"，用 tensor-pipe SOL 重新验证饱和）
解析 MFU(FLOP 估计 ÷ 方阵峰值)**只能筛查，不能单独判饱和**。可信方法 = Nsight Compute 的 **tensor-pipe SOL**(`sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`),并**和同机方阵 GEMM 比**(不是和 vendor TFLOPS 比)。2026-06-27 复跑环境：RTX 5090 sm_120，driver 580.159.03，torch 2.11.0+cu128，`arch_list` 含 `sm_120`，NCU 2025.3.1。

```
kernel                          tensor SOL   DRAM SOL   解读
纯 8192³ 方阵 bf16 GEMM           47.48%       11.59%     同机 bf16+fp32 累加对照上限
cosmos 主 GEMM (cutlass_80)       45.29-45.33% 7.66-7.69% = 和方阵同区间 → 饱和
cosmos attention (flash,batch=4)  43.42%       2.35%      接近方阵上限；不是内存带宽瓶颈
gpu_preflight GEMM peak           233 TFLOPS   -           正确 MFU 分母，不是 419 headline
```

**关键:一个方阵 GEMM 也只到 47.48% tensor SOL → 45-47% 不是"一半算力闲着",而是 bf16+fp32 累加(消费卡半速)的有效上限区间。** cosmos 主 GEMM 跑到同一利用率 → **真饱和,经硬件计数器证实**(不是形状/旧 kernel 问题)。attention batch=4 也到方阵上限的 91%，只剩小幅 kernel 差异，不是大杠杆。超过它只能 fp8/fp4(无半速惩罚,有损,离 policy path)。**教训叠加:MFU 要每台机器实测峰值(验证 7),判"饱没饱和"要用 NCU tensor SOL 对比同机方阵 GEMM。**

## 1. 无损杠杆清单(已验证)

### 1.1 省显存(让更大 batch 装下,exact)

| 技术 | 收益(带 source) | 对我们 |
|---|---|---|
| **FlashAttention(exact)** | "same output/gradient up to numerical tolerance";显存线性于 seq-len,2K→**10x**、4K→**20x**(attention 层中间显存,非整模) | SDPA 已用 flash backend(`dit_mfu_probe` 实测 flash=True)→ 基本已吃到 |
| **融合 AdaLN kernel(AdaptiveLoad, video DiT)** | AdaLN **自身**激活 **-61.9%**,forward kernel **3.21-3.39x**,但 backward 仅 1.28-1.51x,系统级 **+27.2%**(16 卡) | 外部结果成立，但本仓库 cosmos 侧 compile 后 NORM_ELEM 约 3.5%，端到端不构成当前杠杆 |
| **activation checkpointing** | 标准,recompute 换显存 | 已有(grad-ckpt),实测 26.8→9.8GB;是 MFU 税(60%→41%) |
| **ZeRO-Offload / FSDP 分片** | 单卡装 **>13B(10x)** | 多卡/大模型容量手段,非吞吐 |

来源：FlashAttention(arxiv 2205.14135 / dao-ailab repo "EXACT Attention")、AdaptiveLoad(arxiv 2605.17923)、ZeRO-Offload(arxiv 2101.06840 USENIX ATC'21)。

### 1.2 省计算(不改输出分布)

| 技术 | 收益(带 source) | 对我们 |
|---|---|---|
| **FlashAttention-3** | Hopper H100 **1.5-2.0x** exact(FP16,740 TFLOPs/s,75% util),靠 TMA/warp-specialization 调度 | H100/sm_90 专属；Blackwell sm_120 当前无 FA-3 build，torch flash≈cuDNN，已证伪为本机杠杆 |
| **torch.compile / inductor fusion** | **1.4-1.85x**(Flux.1-Dev H100,6.431s→3.483s,无量化) | **这条 band 正好框住我们实测的 1.37x**——已在吃 |

来源：FA-3(arxiv 2407.08608 / pytorch.org blog)、diffusers-torchao(sayakpaul repo)、pytorch torch-compile-diffusers blog。

## 2. RL 专项:剩下最大的无损杠杆 = 分离 pipeline + 正确性铁律

### 2.1 分离 rollout/train(disaggregation)

开源异步 RL 已收敛到:**把推理 GPU 和训练 GPU 物理分开,用有界 rollout buffer(producer/consumer)连接**,因为生成主导 wall-clock。
- verl Fully Async：**2.35-2.67x**(streaming~1.6x → +staleness+partial_rollout→2.35x)。
- AReaL(NeurIPS 2025, arxiv 2505.24298)：**up to 2.77x**,且 matched/improved final accuracy。

**对我们的 transfer caveat(关键)**：verl/AReaL/APRIL 全是**自回归 LLM**,收益来自"用长尾生成藏延迟"。**diffusion 是固定 T 步、无长尾、无 KV cache**——**架构能搬(分离 GPU 池 + 有界 buffer + IS/staleness 修正 + old_log_prob-from-rollout),但 2.3-2.8x 的量级不一定搬得过来。** → 见 §5 开放问题。这正是我们上一轮 overlap 讨论的结论:轴对、量级存疑(记忆 `project_two_level_async`)。

来源：HF async-RL-landscape survey(16 库)、verl fully_async docs、AReaL(2505.24298)。

### 2.2 正确性铁律(外部权威背书我们的 feature-cache 不变量)

verl 逐字：**"old_log_prob 必须用 rollout 参数和 tokens 算,不能用 trainer 算"**(`algorithm.rollout_correction.bypass_mode` 默认 True)。

直接推论:**任何让 rollout 去噪网络 ≠ trainable 网络的近似(DeepCache/TeaCache/蒸馏/fp8/int4)都静默污染 old_log_prob,破坏 GRPO/Flow-GRPO。** 这就是 [[SPRINT_rl_safe_feature_cache_probe]] 的 `cached ≠ trainable` 硬不变量的外部背书。报告点名 **int4 偏差最大、对 RL 直接不安全**(ViDiT-Q "unacceptable degradation";SVDQuant 也只 "approaches" BF16,从不 bit-exact)。

### 2.3 off-policy 复用:只有一种 exactly lossless

- **version rejection(丢弃过期 rollout)= 唯一 exactly lossless**("Simple and correct, but wastes compute")。
- **IS / Truncated-IS reweighting(我们的 TIS/RS)= 近似但有修正**,拿 bias/variance 换("cost in gradient variance")——**不是无损**。
- **partial-rollout resumption** 混策略(APRIL 实测 ~40% off-policy tokens),也非无损。

来源：HF survey、AReaL、APRIL(arxiv 2509.18521)、verl fully_async。

## 3. "diffusion 版 PagedAttention" 到底有没有(medium confidence)

**没有找到任何无损的"diffusion 连续批处理/分页"能带来 per-step 计算收益的文献。** 因为没有 KV cache,diffusion 正确的"分页"类比就是**有界经验/轨迹 buffer + staleness 队列**(memory/pipeline 杠杆,非计算杠杆)。最接近"diffusion 分页/缓存"的 **TreeGRPO(shared-prefix)** 和 **DeepCache/TeaCache** 都**改输出**,不无损。这是综合性否定(无单一 source 直接断言),标 medium。

## 4. 边界:被证伪的 3 个 claim(= "无损"的真实边界)

| 被证伪 claim | 投票 | 含义 |
|---|---|---|
| 融合 AdaLN kernel "bit-exact / loss 曲线全等" | 0-3 | 是数值等价,**非 bit-identical** |
| ZeRO-Offload "preserves computation exactly / lossless" | 0-3 | 是容量技术,CPU fp32 optimizer step 数学等价但**非 bit-exact** |
| activation checkpointing 在 FSDP2 混精下"静默破坏 bit-exactness" | 1-2 | 配置正确时不破坏;过度断言 |

**结论:全场"lossless"= 算法级 exact，不是 bit-for-bit。** 对 RL 这是对的标准(无系统性偏差)。

## 5. 落到我们该做什么 + 待证 probe

```
已撞天花板,别再投:  paged store / stepwise batching(compute-bound 已证伪,本轮 probe)
继续吃的无损项:      torch.compile 训练侧逐 recipe 开([[SPRINT_training_mfu_compile]])
已证伪的单卡 kernel 项(video 51% MFU 是旧错判 → 见 [[SPRINT_cosmos_video_mfu_kernels]]):
  ① 融合 AdaLN-Zero kernel：compile 后 NORM_ELEM 仅 ~3.5%，端到端不构成杠杆
  ② Blackwell GEMM/FA-3 注意力：NCU 确认主 GEMM/attention 接近同机 bf16 上限，且 FA-3 不支持 sm_120
仍值得看的无损系统项:
  ③ 分离 rollout/train pipeline —— 唯一的大无损 pipeline 杠杆,但 diffusion 固定步,量级要 probe(需 ≥2 GPU)
近似项只能离 policy path:量化/cache/蒸馏 只能放 frozen reward/critic,绝不上 rollout/trainable 网络
```

### 待证开放问题(各自值得一个 probe)
1. **shared-prefix 有没有无损版**(对我们最直接)：同 prompt 的 GRPO group 共享早期去噪前缀(SDE 分叉前确定性 latent),能不能既省 forward 又保住 per-sample `old_log_prob`?= [[SPRINT_signal_paged_rollout]] 核心。报告认为这是**唯一真正待证的无损 group 级杠杆**。
2. **固定步 diffusion 的 rollout:train GPU 配比**：没有长尾可藏,async overlap 还剩多少(对照 LLM 的 2.3-2.8x)?
3. **Flow-GRPO 的 stale/partial 能否用 IS 在 `sde_step_with_logprob` 上安全修正**,variance/bias 代价 vs LLM token 级 IS?
4. **多卡 rollout/train pipeline 的真实 diffusion-RL 收益**——单卡 kernel 已到 bf16 上限，剩下的大无损空间在系统 pipeline，而不是 AdaLN/FA3 kernel。

## 6. 非目标

- 不把算法级 exact 当 bit-for-bit 承诺(§4 边界)。
- 不把 LLM 异步 RL 的 2.3-2.8x 量级直接外推到固定步 diffusion(§2.1 caveat)。
- 不把量化/cache/蒸馏放上 policy path(§2.2 铁律)。
- 不重启 paged store / stepwise batching(本轮 compute-bound 已证伪)。

## 7. 源(已验证,按 angle)

- **exact kernels / compute**：FA-3(arxiv 2407.08608 / pytorch.org/blog/flashattention-3)、FlashFormer(2505.22758,低 batch 专用=regime 澄清)、AdaptiveLoad(2605.17923)、diffusers-torchao(github sayakpaul)、torch-compile-diffusers(pytorch blog)。
- **exact memory / training**：FlashAttention(2205.14135 / github dao-ailab)、ZeRO-Offload(2101.06840)。
- **RL rollout infra**：AReaL(2505.24298)、HF async-RL-landscape、APRIL(2509.18521)、verl fully_async docs。
- **approximate / 正确性风险**：ViDiT-Q(2408.06995)、SVDQuant(2411.05007)、Flow-GRPO(2505.05470)。
- **diffusion paged/batching**：TreeGRPO(2512.08153)、vLLM-Omni diffusion continuous batching docs。

完整逐条 claim + evidence + 投票留档于 deep-research 输出(task wwo36sy04)。
