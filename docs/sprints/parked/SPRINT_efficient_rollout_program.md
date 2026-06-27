# SPRINT: Efficient RL Rollout — lever portfolio（经 2026 文献核对后改版）

状态：**PARKED — 触发条件：≥2 GPU 可用（2026-06-27 移入 parked/）**。原 planned / proof-gated（2026-06-24 文献版），2026-06-27 本机实测复核把单卡 compute 主轴判到头：两个 OPEN 中心（同时完成调度、跨帧分页分配器）都需要多卡或未引入的因果视频 backbone → 等"≥2 GPU"这个 EVENT 才解冻。当前单卡真 blocker 已转向 RL 学习（[[SPRINT_cosmos_predict2_2b_trustworthy_curve]]），不是 rollout 效率。这是 efficient-rollout 的**总纲**：不是一个 cache、一个 paged KV、一个蒸馏。它把"减少 rollout compute"拆成一组**正交杠杆**。重心已移到两个系统层 OPEN 中心（同时完成调度 + 跨帧分页分配器）；其余杠杆 compose+cite。**但 2026-06-27 在 RTX 5090 上的 ncu/GEMM 硬件实测把"单卡省 compute"整条主轴判到头了——见 §-0.5。**

---

## -0.5. 2026-06-27 本机实测复核（NCU + GEMM 峰值 + 探针）—— 单卡"省 compute"主轴到头

本总纲是文献核对版。2026-06-27 在 RTX 5090（单卡 32GB, sm_120）跑了 ncu / GEMM 峰值 / 多个探针，把"减 rollout compute"**按硬件真值重判**（全文 [[SPRINT_lossless_diffusion_rl_research]]）：

- **单卡 bf16 compute 已饱和（硬件天花板）**：NCU tensor-pipe SOL 实测 cosmos GEMM 45% ≈ 最优方阵 GEMM 47%（消费卡 bf16+fp32 累加的物理上限）；rollout compute-bound、compile 后到顶。→ **Lever F（forward-only 量化）不是"干净 compose"，是唯一能越 bf16 上限的杠杆——但有损（fp8/fp4），只能离 policy path**；bf16 上无无损 compute 杠杆。
- **本会话实测证伪/否定（已各自 park）**：
  - shared-prefix tree（Lever C / `signal_paged`）：P0 实测只共享前 4/28 步,多样性就塌到 44% retention → **死**（不是 prior-art 问题,是 variance 塌,§2 原理实锤）。
  - paged trajectory store / stepwise batching：trajectory buffer 非峰值、ms/sample 随 batch 平 → **证伪 park**。
  - 单卡 kernel（融合 AdaLN / Blackwell GEMM / FA-3 / SDPA 切换）：NCU 全部到顶/无 build/噪声内 → `cosmos_video_mfu_kernels` 收敛为 `gpu_preflight`。
- **两个 OPEN 中心的现实**：① Lever G（同时完成调度 / 砍 47% 空转）：**单卡 overlap 物理不可能**（colocated 串行,已证）→ **blocked on ≥2 GPU**,且固定步 diffusion 的 overlap 量级存疑。② §6（跨帧分页）：仍需先引因果视频 backbone,repo 未引 → 停在设计。
- **战略结论**：单卡"省 compute"到头（compute 饱和 + 无损 kernel 杠杆证伪）。本总纲降级为 **multi-GPU 解锁后才执行**；当前真 blocker 已转向 **RL 学习**（[[SPRINT_cosmos_predict2_2b_trustworthy_curve]]），不是 rollout 效率。唯一还活的单卡 perf = 训练侧 compile（[[SPRINT_training_mfu_compile]]，1.2x）+ fp8 rollout（有损,离 policy path,已有）。

> 判饱和的可信方法（本会话教训）：MFU 分母每台机器实测（`gpu_preflight`）；饱和判断用 NCU `sm__pipe_tensor_cycles_active` 对比最优方阵 GEMM——别用 vendor headline 或解析 FLOP MFU（419 误诊把饱和的 DiT 当 51%）。

---

## 0. 一句话

三条原理把整张图钉死：

1. **正确性**：rollout 的 FLOPs 只买 **within-group reward variance**，它是 group-relative RL 梯度信号的全部来源。每个效率杠杆只有两种：(a) 删掉**产生零 variance** 的 FLOPs（免费），或 (b) 拿 variance 换 FLOPs（必须实测证明不压平学习）。

2. **方法层护城河已经很薄**：trainer / GRPO 算法族正在**本季度**被 Flow-Factory（2602.12529，9 算法×全模型开箱）和 verl-omni（v0.1.0rc1，2026-06-23）免费送掉。本 repo 的 `families/registry.py` + flux 四算法 recipe 正在和它们收敛。**不要在这层做差异化。**

3. **系统层护城河仍厚，且扩散专属**：LLM-RL 的异步 overlap 套路**结构性地无法迁移到扩散**——扩散 rollout 几乎**同时完成**，没有 straggler 可流水化，GPU 空转约 47%（2606.19004，2026-06）。这就是 Bet 2 的真正落点：**不是抄 vLLM 的通用 rollout 效率，而是做 LLM 框架继承不了的扩散/视频专属系统能力。**

---

## 1. 为什么"想得太简单了"是对的

- **它不是一个 trick，是一个 portfolio**，不同杠杆作用在不同轴上（prompt / step / group 结构 / 精度 / 调度 / 算法），每个有独立的正确性契约，混在一起会互相污染。
- **"复用旧轨迹 / paged KV / LCM"这三个反复想到的方向，推理界都做透了**；价值不在再造 cache，而在让它们和 policy-gradient RL 兼容——但**连这一步，2026 也已经做掉一大半**（§1.5）。
- **正确性比吞吐难。** 只报 speedup 不报 reward curve 的 sprint 等于没做。

---

## 1.5. 2026 先验检查（残酷但必须诚实）

一轮联网检索（认知截止 2026-01，今 2026-06）后，原草稿杠杆的真实状态：

| 原杠杆 | 我以为 | 2026 真实 prior-art | 状态 |
|---|---|---|---|
| **C** shared-prefix tree | novel | BranchGRPO(2509.06040)、TreeGRPO(2512.08153)、Multi-GRPO(2512.00743)、Expand-Prune(2512.15347)、**TMPO(2605.10983) 已做动态分叉** | **DONE / cite，不 claim** |
| **A** 自适应 group | "orchestration 无 state，是空白" | **AEGPO(2602.06825) 扩散原生自适应分配 5×**、SuperFlow(2512.17951) variance-aware + value-tracker 杀零优势 | **DONE / cite，可借机理** |
| **B** 高熵步 SDE+cache | 雏形已在，需统一 | Flash-GRPO(2605.15980, 单步 6×)、V-GRPO(2604.23380, ELBO surrogate, 2×MixGRPO)、E-GRPO(2601.00423)、EGSPO-SA(2603.12554, **可证无偏的逐步分解**) | **机理 DONE；feature-cache-in-RL 仍空白** |
| **E** few-step 蒸馏 RL | "最深的开放问题" | 解了两遍：显式高斯 GRPO（RTDMD 2605.26108 / AdvDMD 2604.28126 / RAVEN 2605.15190 / SDPO 2411.11727）+ 绕过 log-prob（DiffusionNFT 2509.16117 **25×**，2602.04663 反超 2×） | **方法 DONE；视频上的经验槽仍空** |
| **G** 异步 overlap | "现成，砍 idle" | **2606.19004 证明 LLM 异步套路对扩散结构性失效，空转 47%，verl-omni 仅省 14%** | **OPEN，且最值得做** ⭐ |
| §6 视频 paged-KV | "底座不存在，park" | 因果底座已成熟（Wan2.1 Causal-FT、MAGI-1 2505.13211、Self-Forcing++ 2510.02283、LongLive 2509.22622）；Astrolabe/AR-CoPO 已在其上做 RL；**但全是固定滑窗，跨帧分页无人做（2603.27469 明列 future work）** | **底座 DONE；分页分配器层 OPEN** ⭐ |

**结论**：portfolio 重心从"GRPO 方法杠杆"整体移到**两个系统层 OPEN 槽**——§5-G（同时完成调度）和 §6（跨帧分页分配器）。方法杠杆（A/B/C/E）保留为**正确性契约 + 与 prior-art 组合**的工程基线，不作为本 sprint 的 novelty claim。

---

## 2. 唯一的正确性原理：variance = signal

GRPO 梯度 `g = Σ_i Σ_t A_i · ∇log π(a_t^i|s_t^i)`，`Σ_i A_i = 0`。组内 reward 无 variance → advantage 全零 → 梯度全零 → 这条 rollout 100% 浪费（历史 dead-epoch / flat-reward 根因）。排序标尺：

```text
Lever 省的是不是"产生 variance 的 FLOPs"？
  否 -> 免费，先做
  是 -> 必须实测 variance retention，proof-gate
```

**2026 强化**：TAGRPO(2601.05729)、SAGE-GRPO(2603.21872) 独立确认——朴素把 DanceGRPO/Flow-GRPO 搬到现代 I2V（Wan2.2、HunyuanVideo-1.5）**不涨 reward**，因为忽略组内关系信号。这与本 repo memory 里 flux/cosmos 平坦曲线发现一致：低信号区放大的正是这个 variance 塌缩，修法是组内关系/轨迹损失，不是加 epoch。

---

## 3. 护城河：RL rollout efficiency ≠ inference serving efficiency

### 3.1 算法层四约束（推理服务不处理）

| # | RL rollout 约束 | 推理服务处理 | 现有代码负责 |
|---|---|---|---|
| C1 | 每个**可训步**有 well-defined 随机 log-prob | 否 | `sde_step_with_logprob` → `SegmentSignal.log_prob` |
| C2 | within-group reward variance 必须活着 | 否 | `group_relative_advantages`，`adv_zero_rate` |
| C3 | 权重每步在动，behavior ≠ target | 否 | `behavior_policy_version` + 弱 on-policy |
| C4 | version gap 可 IS 纠偏 | 否 | TIS/RS（`apply_truncated_importance_weight` / `apply_rejection_sample_mask`） |

### 3.2 系统层护城河（2026 新证据，本 sprint 真正的中心）

**2606.19004（2026-06，系统 Spotlight）**：LLM-RL 靠"长短序列错峰 → straggler 流水化"来 overlap rollout 与 train；**扩散 rollout 步数固定、几乎同时完成，没有 straggler**，所以这套**结构性失效**，spot GPU 空转 ~47%；verl-omni 的 async-reward 只买 ~14%。**这是扩散/视频 RL 系统层最未解、最专属、且本 repo 已有底座的问题。** 它把原来当脚注的 Lever G 提成中心。

---

## 4. 一个 schema 升级解锁大半 levers（mask 改为 tri-state）

落点仍是两个 ownership：generation 写每步性质契约，algorithm 按需声明字段。现有雏形：`TrajectorySegment.trainable` + per-step `mask`；算法 `needs_kl_intermediates` + `SignalRequest`；off-policy TIS/RS。

**2026 修正（关键）**：原草稿把 `step_kind` 当三选一枚举，把 `trainable` 当独立布尔。文献证明 **`cached` 与 `trainable` 不是正交两轴，是同一个 per-step 变量的两态**：

- log-prob 只存在于 SDE/可训步；ODE 确定性步**无 log-prob 项**（Flash-GRPO 2605.15980 明示"只在 ODE→SDE 转换步算梯度"）。
- "把可训步推向确定性"是已知偏差源（Sampler Stochasticity 2510.10767："SDE 训练随机性越高，ODE 推理质量越好"）。
- 逐步梯度**可证无偏地分解**（EGSPO-SA 2603.12554），这是"只训子集步"的正式许可。

所以 schema 升级为**单个 per-step 三态**，让"可训+cached"在类型上不可表达：

```text
step_kind[t] ∈ { trainable_sde, ode_fresh, ode_cached }
  trainable_sde : 有随机 log-prob，进 PG，必须 fresh forward
  ode_fresh     : 确定性，无 log-prob，不进 PG，算一次
  ode_cached    : 确定性 + feature cache 复用，无 log-prob，不进 PG
segment 级：behavior_policy_version    # C3/C4 组 IS
algorithm 侧：required_step_kinds       # NFT 只要 final sample，不触发逐步 logprob
```

**正确性不变式（贯穿所有 cache/skip/quant 杠杆）**：

```text
cached  ⟹ ¬trainable
trainable ⟹ stochastic-SDE-with-fresh-forward
违反 = log-prob 静默算错 = 梯度错且不报错（最危险）
CI 断言：held-out 轨迹上 cached-vs-fresh 的"无梯度泄漏"等价性检查
```

先做 §4 + Lever D，再上 B。

---

## 5. Lever portfolio（按 2026 后的"该做程度"重排）

| Lever | 轴 | 2026 状态 | 本 sprint 取舍 |
|---|---|---|---|
| **D** 冻结无损 cache | 卫生 | 工程 | 先做，零风险 baseline |
| **B** 三态 step mask | step | 机理 DONE；feature-cache-in-RL 空白 | 统一现有零件到三态 + 不变式；cache 当**研究项**门控 |
| **G** 同时完成调度 | 调度 | **OPEN ⭐** | **中心一**：砍 47% 空转 |
| **§6** 跨帧分页分配器 | 视频系统 | **OPEN ⭐** | **中心二**：因果底座上自造分页层 |
| **F** forward-only 量化 | 精度 | 工程（FP4 已有 Sol-RL 2604.06916） | compose，干净普适 |
| **A** 自适应 group | prompt | DONE（AEGPO/SuperFlow） | 借机理，cite，不 claim |
| **C** shared-prefix tree | group 结构 | DONE（Tree/Branch/TMPO） | compose，cite；已有 sprint 需降级 |
| **E** few-step 视频 RL | 算法 | 方法 DONE；视频经验槽空 | 经验性补 Echo 槽，cite RAVEN/RTDMD |

### Lever D — 冻结量无损 cache（先做，卫生）

text-embed 算一次缓存；KL 的 ref log-prob frozen，collect 时算一次写进 trajectory，replay 永不重算。证据：`prompt_embeds` 已存；ref 现经 `disable_adapter()` 在 replay **重算**（`sde_logprob.py`），可前移缓存，把 ref forward 从热路径删掉。variance 代价零。

### Lever B — 三态 step mask（机理已被 2026 做掉，本 sprint 做"正确收口 + cache 研究门控"）

机理不再 novel：Flash-GRPO（单可训步 6×）、V-GRPO（ELBO surrogate 2×MixGRPO）、E-GRPO（熵选步）、EGSPO-SA（无偏分解）。本 repo 三个零件**已存在但各自为政**：`select_sde_window` / `_train_timestep_indices` / `TeaCacheState`。本 sprint 的工作=把它们收口到 §4 三态 + 不变式，并把**唯一仍空白的点**——feature-cache 进生成式 RL loop 的 log-prob 安全性（检索确认无人发表）——当**高风险研究项**，只 cache `ode_*` 步，配 CI 等价断言。

> 设计默认值参考 Flash-GRPO/Flow-GRPO-Fast：可训子集可以小到 **1–2 步**，其余 ODE/cached；用熵（E-GRPO）选哪几步。先验证"1 trainable step + 全 cache 其余"再谈滑窗。

### Lever G ⭐ — 同时完成调度（中心一，OPEN）

问题（2606.19004）：扩散 rollout 同时完成 → 无 straggler → LLM 异步 overlap 失效 → 空转 47%。本 repo 已有底座：`continuous/`（producer+queue+consumer+staleness）、weight-sync barrier、Ray actor relaunch、TIS/RS。

可做的扩散专属调度（这是 novelty 所在）：
```text
- 异构分辨率/步数 co-batch（vLLM-Omni 只支持同形 homogeneous，明确实验性）
- reward 计算（需近全程去噪 + VAE decode，是延迟大头）与下一轮 rollout 错峰填空转
- 把"同时完成"变成"故意错峰发射"：不同 prompt 给不同 step 预算/分辨率，制造 straggler 让流水化重新可用
- behavior_policy_version + behavior_logprob 进 §4 契约，让 off-policy 填空转的 IS 组得对（NFT tolerates_off_policy=False，对它关闭）
```
验收：iteration GPU 空转占比从 ~47% 基线下降 ≥ 目标，且 eval reward 不退化、IS mismatch 在阈内。

### §6 / Lever 视频分页 ⭐ — 跨帧 paged-KV 分配器（中心二，OPEN，见 §6）

### Lever F — forward-only 量化（compose）

rollout 是 `no_grad`（`run_denoise_steps` 在 `torch.no_grad()` 内），可激进量化不碰训练精度。FP4 已有 prior-art：Sol-RL（2604.06916，NVIDIA，FP4 探索池 + BF16 训练，4.64×）。本 repo compose 即可，契约：量化只作用于生成 noise_pred，replay 重算 log-prob 仍训练精度。

### Lever A / C / E（降级为 cite + compose）

- **A**：AEGPO(2602.06825) 扩散原生自适应分配已 5×；SuperFlow(2512.17951) 用 value-tracker 杀零优势。借其机理给 orchestration 补 per-prompt variance state（本 repo `RolloutScheduleState` 现仅 `rollout_id`），但作为工程对齐，不作 novelty。
- **C**：见 `SPRINT_signal_paged_rollout.md`，但该 sprint 需加 prior-art 段（Tree/Branch/TMPO 已覆盖 ~80%）。唯一仍未占的角度：**antithetic / K-correlated noise（2506.06185, −1/(K−1)）压 group size 到 2**，且和 tree 结合做 image/video GRPO——检索称这是"最可利用的空白"。若要在 C 上保留 novelty，押这里。
- **E**：方法已解（RTDMD/AdvDMD/RAVEN/DiffusionNFT）。仍空的是**经验槽**：把成熟 image 配方扩到蒸馏 few-step **视频** at scale。本 repo `echo`（LTX-2.3 派生、DMD ~8 步）正是这个槽的活标本——拿 RAVEN 的 CM-GRPO（把蒸馏步显式参数化成高斯 z=α·x̂+σ·ε，只在选定步注 σ）在 Echo 上跑，是有价值的经验工作，但 cite RAVEN，不 claim 方法。

---

## 6. 跨帧 paged-KV 分配器（中心二，重写：底座已成熟，分配器层仍空）

原草稿判"视频 paged-KV 走不通、park"——**这个结论被 2026 推翻一半，必须更新**。

**底座已成熟**：因果 / 块因果 + 可缓存过去帧 KV 的视频模型在 2026 已是一条成熟谱系——MAGI-1（2505.13211，块因果 + KV-range 窗）、Self-Forcing++（2510.02283）、Causal Forcing（2602.02214）、LongLive（2509.22622，有界 KV + KV-recache）、**Wan2.1 Causal-FT**（贴合本 repo 现有 Wan stack，是整条谱系蒸馏起点）。而且 RL 已经在其上开跑：**Astrolabe（2603.17051，滚动 KV + 前向过程 RL，只对局部 clip 窗口更新）**、**AR-CoPO（2603.17461，块级 forking，只重 rollout 一个块而非整段）**。

**但分配器层仍空**：所有这些模型用的是**固定大小滑动窗口** KV（VideoMLA 2605.30351 称这是收敛后的范式），**没有人做 vLLM 式的跨帧分页分配器（页表 + 页分配 + ref-count + COW over the temporal axis）**——KV 量化 33 方法实证（2603.27469）**明确把"PagedAttention 思想迁到视频帧"列为 future work**；OmniMem（2605.30519）的块检索是最近的现成原语但仍非页表。

**所以你最初的 paged-attention 直觉，scope 对了就是真空白**：

```text
不做：对当前 full-bidirectional 视频模型（cosmos predict2 / wan T2V / LTX-2 双向 / SD3.5 / Flux）做 paged KV  —— category error（每步全帧重算，无可缓存 KV）
做：  在一个因果底座（Wan2.1 Causal-FT / MAGI-1）之上，造跨帧 TrajectoryBlock 分页分配器
       页 = 帧块的 KV；页表 per-rollout；ref-count 让同 prompt group 的 rollout 共享历史帧 KV；COW 在分叉点
       与 Lever G 天然耦合：分页让长 horizon 视频 rollout 不重算、不碎片、可错峰调度
```

落地前置：repo 需引入一个因果视频 backbone（Wan2.1 Causal-FT 最省事，贴现有 stack）。在那之前，当前 full-attention 视频族的省算只能走 Lever B（时空 attention O(N²) 大头 → 滑窗/稀疏 attention kernel）+ Lever F。

**这是本 sprint 两个 OPEN 中心里更难、更高天花板的一个**：它同时是系统基建（分页分配器）+ 视频前沿 + 贴合你 Triton kernel 背景，而 2603.27469 已经替你确认了它没被做。

---

## 7. 排序与 phase plan（围绕两个 OPEN 中心重排）

```text
P0  §4 三态 schema + 不变式 + CI 等价断言；Lever D 无损卫生   # 解锁后续、零风险
P1  Lever G 同时完成调度（中心一）                           # 砍 47% 空转，repo 有底座，OPEN
P1' Lever B 收口（三态收编 sde_window/timestep/teacache）     # 与 P1 并行，机理已知
P2  §6 跨帧分页分配器（中心二）：先引 Wan2.1 Causal-FT，再造 block 分页   # 最高天花板，OPEN
P3  Lever F（rollout FP4/FP8，compose Sol-RL）
P4  Lever E 经验槽：Echo + CM-GRPO（cite RAVEN）
A/C: 不立项 novelty；按需 compose AEGPO/TreeGRPO 机理对齐工程
```

每个 P 统一 proof-gate：
```text
省算/省 idle：forward 数 / wall-clock / GPU 空转占比 下降 ≥ 目标
不退化：eval reward curve 不显著退化，adv_zero_rate 不显著上升
正确性：rollout-vs-replay log-prob mismatch 在阈内（cache/skip/quant/stale 必查）
任一不过 -> 记负结果并关该 lever，不硬推
```

---

## 8. 风险

- **R1 variance 塌（B/C/E/§6 高危）**：晚分叉 / few-step / 共享历史帧都可能压平组内 variance。`adv_zero_rate` 当 fail-fast。
- **R2 静默梯度错（B/F/G/§6）**：cache/量化/stale/共享 KV 步若没正确 mask 出 PG，log-prob 错且不报错。§4 不变式 + mismatch 检查是唯一防线。
- **R3 杠杆互相污染**：一次只隔离一个变量（沿用 R4 教训：TeaCache × shared-prefix 不同实验）。
- **R4 被 scoop（新增，最现实）**：方法层（A/B/C/E）2026 已密集出活，立项即过时。**只在两个 OPEN 系统中心投 novelty**，方法层一律 compose+cite。
- **R5 §6 前置依赖**：跨帧分页要先引因果 backbone（Wan2.1 Causal-FT）；若 repo 不引，§6 只能停在设计。

---

## 9. 非目标

- **不在 trainer / 算法族层做差异化**——Flow-Factory（2602.12529）+ verl-omni（v0.1.0rc1, 2026-06-23）本季度免费送掉；本 repo `families/registry.py` 维持可用即可，不扩。
- 不做跨 iteration 旧 trajectory replay buffer（off-policy、stale advantage、IS 爆炸；注意 OP-GRPO 2604.04142 已用受控回放达成 34% 步，要做也是 compose 它，不是裸 replay）。
- 不对当前 full-attention 视频模型移植 paged KV（§6，category error）。
- 不把任何 cache/shared-prefix/共享 KV 作为普通 serving 的透明默认。
- 不同时叠多个 lever 跑同一实验。
- 不在 proof-gate 前把 `TrajectoryBatch` 改成 block-native。

---

## 10. 与已有 sprint 的关系

- `SPRINT_signal_paged_rollout.md` = 本总纲 **Lever C**，但**需补 prior-art 段**：BranchGRPO/TreeGRPO/Multi-GRPO/TMPO 已覆盖 ~80%，其唯一可保留的 novelty 是 antithetic-noise × tree 压 group size（§5-C）。建议把它从"独立 killer capability"降级为"compose + 一个窄 novelty 押注"。
- `SPRINT_diffusion_rollout_system.md`（reading）= 旧"AR paged KV 不适用"边界；§6 用 2026 证据细化为"对 full-attention 视频不适用，但因果底座 + 分页分配器是新空白"。
- `SPRINT_rollout_vllm_migration.md`（done）= TeaCache 结论 + vLLM-Omni 边界；§3.2 用 2606.19004 补"LLM 异步对扩散失效"。
- 本总纲 = 上层战略 Bet 2（"RL-specific 扩散/视频系统能力"）的工程展开，且经 2026 核对后**收窄到系统层两中心**。

## 11. 参考代码

- `vrl/generation/diffusion/executor.py`：`run_denoise_steps`（`no_grad`、TeaCache gate、dense replay）。
- `vrl/generation/diffusion/layout.py`：`select_sde_window` / `DiffusionSDEParams`。
- `vrl/trainers/online/trainer.py`：`_train_timestep_indices`（denoising reduction / DanceGRPO）。
- `vrl/math/diffusion/flow_matching.py`：`sde_step_with_logprob`（C1 随机 log-prob 定义点）。
- `vrl/algorithms/advantages.py`：`group_relative_advantages`（C2）。
- `vrl/algorithms/logprob_mismatch.py`：TIS/RS（C4）。
- `vrl/trajectory/types.py`：`TrajectorySegment.trainable` / per-step `mask`（§4 三态落点）。
- `vrl/rollouts/evaluators/types.py`：`SignalRequest` / `needs_kl_intermediates`（algorithm-declares-fields 雏形）。
- `vrl/rollouts/orchestration/continuous/`：producer/queue/consumer/staleness + weight-sync barrier（Lever G 底座）。
- `vrl/rollouts/orchestration/.../types.py`：`RolloutScheduleState`（仅 `rollout_id`，A 缺的 per-prompt state）。
- `vrl/models/diffusion/echo/`：DMD ~8 步 flow-matching（Lever E 视频经验槽标本）。
- `vrl/models/diffusion/{cosmos,wan_2_1}/`、`vrl/models/ar/paged_attention_helpers.py`：§6 当前视频 full-attention / 无 KV 的证据（对照因果底座）。

## 12. 论文 / 外部参考（2026-H1 核对版，⭐=已确认 Jan-2026 之后）

**系统层（本 sprint 中心依据）**
- 扩散 RL 系统开放问题（同时完成、空转 47%）— https://arxiv.org/html/2606.19004v1 ⭐
- verl-omni（trainer 层商品化证据）— https://github.com/verl-project/verl-omni ; https://vllm.ai/blog/2026-05-14-verl-omni ⭐
- Flow-Factory（9 算法×全模型）— https://arxiv.org/abs/2602.12529 ⭐
- TRL v1 删除 DDPO — https://huggingface.co/blog/trl-v1 ⭐

**step 级 / 正确性（Lever B、§4）**
- Flash-GRPO（单步 6×，视频）— https://arxiv.org/abs/2605.15980 ⭐
- V-GRPO（ELBO surrogate，2×MixGRPO / 3×DiffusionNFT）— https://arxiv.org/abs/2604.23380 ⭐
- E-GRPO（高熵步）— https://arxiv.org/abs/2601.00423 ⭐
- EGSPO-SA（可证无偏逐步分解）— https://arxiv.org/abs/2603.12554 ⭐
- Sampler Stochasticity（cache 反向警告）— https://arxiv.org/abs/2510.10767
- MixGRPO — https://arxiv.org/abs/2507.21802 ; Flow-GRPO — https://arxiv.org/abs/2505.05470
- TeaCache 2411.19108 / DeepCache 2312.00858 / ToCa 2410.05317（within-sample feature cache，未进 RL loop）

**group / tree / 自适应（Lever A/C，已 DONE）**
- AEGPO（扩散原生自适应分配 5×）— https://arxiv.org/pdf/2602.06825 ⭐
- SuperFlow（variance-aware + value-tracker 杀零优势）— https://arxiv.org/pdf/2512.17951
- BranchGRPO 2509.06040 / TreeGRPO 2512.08153 / Multi-GRPO 2512.00743 / Expand-Prune 2512.15347 / **TMPO（动态树）2605.10983** ⭐
- AR3PO 2509.25808 / 2-GRPO=2510.00977（AR-LLM）/ AERO 2602.14338 ⭐
- antithetic K-noise（−1/(K−1)，最可利用空白）— https://arxiv.org/abs/2506.06185 ; Coupled-GRPO 2506.20639

**few-step 蒸馏 RL（Lever E，方法 DONE）**
- DiffusionNFT（25×，ICLR'26 Oral）— https://arxiv.org/abs/2509.16117
- 反超 DiffusionNFT 2×（final-sample ELBO）— https://arxiv.org/abs/2602.04663 ⭐
- RTDMD 2605.26108 ⭐ / AdvDMD 2604.28126 ⭐ / Flash-DMD 2511.20549 / TDM-R1 2603.07700 ⭐ / SDPO 2411.11727 / AWM 2509.25050
- **RAVEN（CM-GRPO，蒸馏 few-step 视频）— https://arxiv.org/html/2605.15190** ⭐ ; DMD 2311.18828

**视频系统 / 因果底座 / 分页（§6 中心）**
- KV 量化 33 方法实证（明列 paged-over-frames 为 future work）— https://arxiv.org/abs/2603.27469 ⭐
- MAGI-1 2505.13211 / Self-Forcing++ 2510.02283 ⭐ / Causal Forcing 2602.02214 ⭐ / LongLive 2509.22622 / Wan2.1 Causal-FT
- VideoMLA 2605.30351 ⭐ / OmniMem 2605.30519 ⭐ / TempCache 2602.01801 ⭐
- **Astrolabe（滚动 KV + 前向过程 RL）2603.17051** ⭐ / **AR-CoPO（块级 forking RL）2603.17461** ⭐ / Reward-Forcing 2601.16933 ⭐

**视频 RL / reward（方法论警示 + 战略）**
- TAGRPO（朴素 GRPO 在现代 I2V 不涨）2601.05729 ⭐ / SAGE-GRPO 2603.21872 ⭐ / Wan-R1 2603.27866 ⭐ / DanceGRPO 2505.07818
- OP-GRPO（受控 off-policy，34% 步）2604.04142 ⭐ / Sol-RL（FP4 rollout，4.64×）2604.06916 ⭐
- VideoRLVR 2605.15458 ⭐ / GenEval2 2512.16853 / VideoReward 2501.13918 / GRPO-for-Generation Survey 2603.06623 ⭐

> 验证说明：arXiv `26MM.xxxxx` = 2026 年 MM 月。部分 2026 号与 headline 数字在认知截止之后由联网检索取得、个别 PDF 仅读到摘要；推进任一 lever 前按 real-run gating 纪律对原文复核一遍。优先精读：**2606.19004（中心一依据）、2603.27469 + Astrolabe + Wan2.1 Causal-FT（中心二依据）、Flash-GRPO + EGSPO-SA（§4 三态依据）**。
