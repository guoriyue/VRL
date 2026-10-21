# SPRINT：rollout / replay 数值一致——从"绕开"到"定位根因"

状态：**planned（2026-10-20）**。来源是 PyTorch Conference NA 2026 Day 1 的以下内容：
010 TitanRL keynote、051 TitanRL demo、091 Agentic RL、114 OpGuard、海报 P13、P23、P56、P87、P102。
逐场笔记不入库，结论和证据都写在本文件里。

## 0. 结论先行

VRL 已经能做到**测量**：有 parity gate，有 `LogprobMismatchStats`，有 TIS/RS。缺的是两样能力：

1. **定位**：偏差是从哪个 op、哪个 block 开始的。
2. **根因**：去掉"gen batch == replay batch、compile off"这类绕行约束。

本 sprint 只做诊断能力和一次性实验。生产默认值不变。batch-invariant 模式只作为诊断模式：会上测得它吞吐慢 1.7–5 倍；在 off-policy 设置下，reward 没有提升（091）。

## 1. 当前代码现实（HEAD a914644c6 核对）

- **gate 配置**：`ReplayParityConfig`（`vrl/trainers/core/types.py:74-97`）。默认 `max_abs_logprob_diff=0.01`，`every_update=False`，即只在第一次 update 前检查。
- **gate 逻辑**：在 `vrl/trainers/online/trainer.py:1832-1910`。开了 TIS/RS/recompute 时，超阈值只记 warning，不拦截；只有非有限值，或差值超过 `CORRECTED_REPLAY_MAX_ABS_LOG_RATIO`，才 fail。gate 失败或首次通过时，会记录 `trainable_state_digest`。
- **漂移统计**：`LogprobMismatchStats`（`vrl/algorithms/logprob_mismatch.py:40-56`）提供 `logprob_abs_diff_*`、`ratio_abs_dev_*`、`mismatch_kl`、`mismatch_k3_kl`。没有把"训推失配"和"staleness"分开的统计。continuous 模式只有一个 `continuous.stale_policy_versions` 表盘（`vrl/trainers/metrics_io.py:72`）。
- **deterministic 开关**（`vrl/run.py:68-82`）：设置 `use_deterministic_algorithms`、cudnn deterministic、cuBLAS workspace。它**不保证 batch invariance**：同一个样本在 batch 4 和 batch 8 里，算出的结果仍可能不同。TF32 开关只在 `vrl/models/precision.py:55-56`。全仓没有 `batch_invariant_ops`，没有 `sdpa_kernel` 固定后端，也没有 DebugMode 或 op 级哈希。
- **训练侧 attention 后端**：ring CP 强制用 `_native_cudnn`（`vrl/trainers/context_parallel.py:54-57`）。rollout 侧 Ulysses 用的是 `F.scaled_dot_product_attention`，后端由 PyTorch 启发式挑选（`vrl/models/sequence_parallel.py:202`）。两侧 kernel 不同。
- **FSDP root 的输入 cast**：root `fully_shard` 用默认 `MixedPrecisionPolicy`，即 `cast_forward_inputs=True`；只有 block 显式关掉了（`vrl/trainers/fsdp.py:200-217`）。核对结果分两类：
  - SD3.5、FLUX、Qwen-Image(-2.1)、Krea2、MOSS、Wan、Cosmos2.5 都会在 family forward 里把 timestep 显式转成 transformer dtype（例如 `sd3_5/model.py:87`、`wan_2_1/model.py:361`），所以 root 的 cast 对它们是 no-op。
  - **SANA 例外**：它有意保留 fp32 timestep（`sana/model.py:89-95`）。在 FSDP 下，root 会把它降成 fp16，而非 FSDP 的 rollout 保持 fp32。目前 SANA 没有 FSDP preset，所以这还是潜在风险，不是现存 bug。
- **工具**：parity 探针是 `vrl/scripts/generation/full_sequence_probe.py`，参数 `--check-replay`。`trainable_state_digest`（`vrl/trainers/diagnostics.py:13`）按 rank 计算本地分片的摘要，只在 gate 处调用，没有跨 rank 比对，也没有在 rollout 侧回算。
- **量化**：FP8/NVFP4 rollout 在每次权重安装后都会 `_requantize_weight`（`fp8.py:83`、`fp4.py:203`、`base.py:127,158`）。训练侧量化在配置解析时就被拒绝（`docs/PRECISION.md`）。

## 2. 外部证据（只列和决策相关的）

- **TitanRL 的 BITWISE_PARITY 清单（010/051）**：
  - mm/addmm/log_softmax 换成 thinking-machines 的 `batch_invariant_ops`；
  - FA3 设 `num_splits=1`；
  - NCCL 用 tree 算法、单 channel；
  - 关闭 TF32 和 reduced-precision reduction；
  - RoPE 保持 fp32，所以 FSDP 设 `cast_forward_inputs=False`。
  - 效果：Qwen3-8B 上 kl 恰好为 0；纯计算慢 2.4–2.9 倍，端到端几乎不变。
- **OpGuard（114 / P13，OSDI'26，Apache-2.0，`pip install opguard`）**：
  - 方法：在 op 边界记录 XOR 指纹；先固定所有良性随机性（随机数、kernel 选择、通信拓扑、确定性算法），再找两次运行之间最长的逐位相同前缀，第一个不一致的 op 就是排查起点。
  - 支持 FSDP、torch.compile、HF；对 RL rollout 的 kernel 覆盖率 95.3%。
- **FP8-RL（P56）**：只有 rollout 用 FP8、训练用 BF16 时，mismatch KL 在后期持续上升，约第 700 步飙升并导致崩溃，TIS 压不住。两侧用同一套 FP8 recipe 后，漂移接近 BF16。
- **SkyRL（099）**：把重要性比率分解为两项：π_old/μ_old 是训推失配，π/π_old 是 staleness。
- **reduce 顺序（会场与 Luminal 工程师的交流，不在下载的议程材料里）**：浮点加法不满足结合律，同一组数按不同顺序归约，结果会不同。编译器只要改变 tiling、split-K 或融合方式，就会改变归约顺序。BITWISE_PARITY 清单里的大多数条目，本质上都是在固定归约顺序。

### 2.1 VRL 里改变归约顺序的位置（HEAD a914644c6 核对）

| # | 来源 | rollout 与 replay 为什么不同 | 对应的对照组或步骤 |
|---|---|---|---|
| R1 | GEMM 的 split-K 和 tile 选择 | cuBLAS 和 Triton 按 M（batch × token）挑 kernel，batch 一变，归约顺序就变。`allow_bf16_reduced_precision_reduction` 在 torch 2.11 下默认是 True，VRL 没有改它（只在 `vrl/trainers/trace.py:155-156` 记录） | P0(a)、P1 |
| R2 | attention 的 KV 归约 | flash、cudnn、efficient 三种后端的 split-KV 和 online-softmax 合并顺序都不同。ring CP 还要跨 rank 按 ring 顺序合并部分 softmax 结果 | P0(d)、P3.2 |
| R3 | Inductor 的融合归约 | 开 compile 后，norm、softmax 这类行归约会被融合或拆分（2.11 的 `torch._inductor.config.split_reductions=True`，`deterministic=False`）。这就是已知 LoRA parity bug 里"compile on"那一半 | P0(b) |
| R4 | **VRL 自己的 logprob 归约** | `sde_step_with_logprob` 先算逐元素 Gaussian 对数密度，再对 C×H×W（视频是 C×T×H×W）做 `.mean`（`vrl/math/denoise/flow_matching.py:262-268`）。数学 dtype 默认是 fp32（`precision.denoise_math`）。行归约 kernel 的分块方式和整个张量的形状有关，所以同一个样本在 batch 4 和 batch 8 里，归约顺序可能不同 | 新增 P0.4 |
| R5 | 小 σ 步的放大 | 不是归约顺序本身，而是放大器：logprob 里有 `(prev_sample − mean)² / (2·noise_scale²)`。noise_scale 在最后几步很小，所以 R1–R3 在 model output 上留下的微小差异，在这些步会被成倍放大成 logprob 差 | P0.4 的按步拆分 |
| — | LoRA 合并 | 已核对，**不是来源**：VRL 里没有 `merge_adapter`、`fuse_lora` 或 `merge_and_unload`，rollout 和 replay 都走未合并的 peft 前向，计算顺序相同 | — |
| — | 梯度的 reduce-scatter 顺序 | NCCL 的 ring 和 tree 算法、rank 数，会影响训练能否逐位复现，但**不影响** rollout 与 replay 的 logprob parity。Ulysses 的 all-to-all 只搬数据，不做归约 | 不在本 sprint |

## 3. 阶段

### P0. 一次性定位工具（只放在 `vrl/scripts/`，不进库结构）

1. 写一个探针：同一份 batch，分别跑 rollout 路径（`vrl/generation/steps/denoise/loop.py`）和 replay 路径（`replay_forward_with_latents`），对每个 transformer block 的输出做哈希或计算 max-abs，报告**第一个**超阈值的 block 或 op。
   - 先试 torch 2.11 自带的 DebugMode tensor hashing。
   - 不够用再接 OpGuard。
2. 用它跑四组对照，每组输出 block 级的偏差曲线：
   - (a) gen batch ≠ replay batch；
   - (b) compile 开 vs 关；
   - (c) FP8 rollout vs bf16；
   - (d) ring CP（cudnn）vs 单卡 SDPA。
3. **验收**：每组都能给出"第一个分歧的位置"。复现一次已知的 LoRA parity bug 配置（关掉现在的 batch 约束），工具必须能指到正确的位置。
4. **归约顺序归因（§2.1）**：
   - 探针报告第一个分歧的 op 时，同时标出它属于 R1、R2、R3 中的哪一类。
   - 单独测 R4：对完全相同的 `model_output`、`sample`、`prev_sample`，分别在 batch=1 和 batch=N 下调用 `sde_step_with_logprob`，逐位比较每个样本的 `log_prob`。再用 fp64 算一遍作为参照，得到 R4 的误差底。这个测试不需要模型，只需几十行，可以先做。
   - 每组对照的 |Δlogprob| 都按去噪步拆开报告，验证 R5：如果差异集中在小 σ 的最后几步，说明主要问题是放大，而不是某个 op 本身误差大。

### P1. batch-invariant 诊断模式（实验）

0. **最便宜的一步先做：关掉 fp16/bf16 GEMM 的 reduced-precision reduction。** 2026-10-20 在 5090（sm120）+ torch 2.11 上实测，脚本是一次性的，不入库：
   - 默认设置（True）下，K=5600 这类长 K 的 down-proj 有 batch 依赖：同一个样本在 B=1 和 B≥2 时结果不同，bf16 下 max|Δ| 约 3e-2，fp16 下约 4e-3，误差也更大。
   - 设为 False 后，在 tok 256/1024/4096、K 2240/5600 的所有组合上，B=1–32 都逐位一致，误差还更小。速度只有 tok=256、K=5600 这一档慢约 16%，其余持平。
   - 同条件下，SDPA（flash/efficient/cudnn 三种后端）各自都与 batch 无关；layer_norm 和 rms_norm 也与 batch 无关。但**不同后端之间**，bf16 差约 1e-3，就是 R2。
   - FP8 rowwise（`fp8.py` 的做法）同条件下与 batch 无关，`use_fast_accum` 开关结果完全相同。总误差 3.7e-2 几乎全部来自量化，累加误差只有 1.7e-3。所以 FP8 rollout 要改善的是量化本身（T3 four-over-six、两侧同一套 recipe），而不是归约。
     - 一个小改进：bias 现在是在 GEMM 输出 bf16 之后才加（`fp8.py:114-115`），改用 `_scaled_mm(bias=...)` 在 kernel 内融合，相对误差从 2.34e-3 降到 1.66e-3。`use_fast_accum` 保持默认的 False。
     - 测不到的部分：Hopper 的 FP8 tensor core 累加只保留约 14 位尾数（DeepSeek-V3 每 128 个元素提升一次到 fp32）。如果将来在 H100 上跑 FP8 rollout，需要单独测。
   - R4 已经顺带测了：logprob 式的 fp32 `.mean` 相对 fp64 的误差约 1e-7，随 batch 的变化约 1.2e-7，不是逐位一致，但远低于 1e-6。P0.4 只需要在真实的 `sde_step_with_logprob` 上复核一次；P1 里"换成固定顺序"那一条，只有追求逐位一致时才做。
   - 做法：在 `apply_float32_precision`（`vrl/models/precision.py:35`）旁边，同一个 owner 里设置 `allow_bf16/fp16_reduced_precision_reduction=False`，`trace.py:155` 已经在记录这个状态。然后在 gen batch ≠ replay batch 下跑一次 `full_sequence_probe --check-replay`。上线前要在 L4、L40S 上复测，cuBLAS 的启发式随架构变化。
1. 再在 P0 的探针里手动开启：`batch_invariant_ops`、`allow_tf32=False`、`allow_bf16_reduced_precision_reduction=False`，同时固定 SDPA 后端。compile 打开的那一组，再加上 `torch._inductor.config.deterministic=True`（2.11 已有），必要时再加 `split_reductions=False`。
   - 逐项打开，每打开一项记一次 |Δlogprob|，得到"每个归约来源各贡献多少"的表，不只给全开时的最终结论。
   - 如果 P0.4 测出 R4 本身就超过 1e-6，logprob 的 `.mean` 要换成与 batch 无关的固定顺序：例如先 reshape 成固定的 chunk，再分两级归约，或者在 fp64 下累加。否则下面的 1e-6 目标永远达不到。
2. **判定**：在 rollout batch ≠ replay batch 的条件下，`full_sequence_probe --check-replay` 的 |Δlogprob| 能否降到 ≤1e-6。
   - **能**：才把它做成 `trainer.deterministic` 旁边的一个诊断开关（同一个 seam，不新建模块），并在 `docs/PRECISION.md` 写明代价。
   - **不能**：记录残差来自哪个 op，然后停在这里。
3. **注意**：rollout 用 bf16/fp8，replay 用 fp32 时，精度本身就不同。BI 只能消除"同精度下的 batch 依赖"。

### P2. 长跑漂移可见性

1. 用现有 `mismatch_kl` 跑一次长程 FP8 或 NVFP4 rollout，至少 500 次 update，开 `every_update=True`，观察斜率。测到上升后才做下一步：只加一个"斜率超阈值就 warn"的监控，**不加**新的配置组合。
2. continuous 模式下，把重要性比率拆成两项分别记录：
   - "训推失配"：同版本下 replay 与 rollout 的差，这部分已有；
   - "staleness"：版本差带来的部分。
   - 只加指标，不改 loss。

### P3. 小项（各自独立，各一个 commit）

1. **FSDP root cast**：两种做法二选一，要先读 `MixedPrecisionPolicy` 的语义和各 family 的输入：
   - 让 root 也设 `cast_forward_inputs=False`，由 family 自己负责 cast（大多数 family 已经这样做了）；
   - 或者在 SANA 首次上 FSDP preset 时，用 P0 的工具验证一次。
   - **不要**为此给每个 family 都加防御性 cast。
2. **ring CP 与 SP 后端不一致**：在 `docs/PRECISION.md` 写明这类 recipe 的 parity 阈值不能是 bit-exact。或者等 `SPRINT_video_context_parallel.md` 推进时，固定同一个后端。
3. **SDC 检测（可选）**：周期性比对各 DP rank 上同一个 replicated 参数的摘要。只在出现可疑的静默错误时才做。

## 4. 不做

- 不把 batch-invariant 设成生产默认。
- 不自写 OpGuard 式的 kernel 指纹框架：用现成的 DebugMode 或 OpGuard。
- 训练侧 FP8 前向是 P56 指出的根本解法，但它依赖 Blackwell 硬件和训练量化 seam。等 P2 测出 FP8 rollout 的后期漂移确实存在，再单独立项。

## 5. 实施记录（2026-10-20，RTX 5090，torch 2.11）

### 已落地
| 项 | commit 主题 | 结论 |
|---|---|---|
| P0 探针 | `scripts(generation): add a rollout/replay parity divergence probe` | `vrl/scripts/generation/parity_divergence_probe.py`：逐模块定位第一个分歧点，按步给出 max/mean \|Δlogprob\|；对照开关 `--replay-batch`、`--replay-compile`/`--rollout-compile`、`--quantize ... --replay-dequantize`、`--rollout-sdpa/--replay-sdpa`、`--reduced-precision-reduction`、`--batch-invariant` |
| P1 第 0 步 | `models(precision): reduce fp16/bf16 GEMMs in full precision` | 生产默认关掉 cuBLAS reduced-precision reduction |
| 奖励进程同样处理 | `rewards(runtime): reduce half-precision GEMMs in full precision in reward processes` | Qwen3-VL judge 的分数不再随同 batch 的其他图变化 2e-2 |
| P3.1 FSDP cast | `trainers(fsdp): compute in the stored dtypes and leave inputs to the model` | 根因比计划里设想的大：除了 root 的输入 cast，actor 策略还把 diffusers 固定为 fp32 的模块（Wan 的 norm 和 `scale_shift_table`、SANA）降成了 bf16/fp16 |

### 实测（gen batch 4 → replay batch 1，除注明外 8 步）
| 家族 | 对照 | max \|Δlogprob\| | 第一个分歧点 |
|---|---|---|---|
| SD3.5 Medium bf16 | reduced-precision reduction 开 | 1.1e-1 | `context_embedder`（K=4096 的长 K GEMM） |
| SD3.5 | 关（现在的默认） | 3.7e-9 | 无 |
| SD3.5 | 关，batch 8→1、4→3 | 3.0e-8 | 无 |
| SANA 1600M fp16 | 开 / 关 | 1.5e-3 / 7.5e-4 | LoRA 的 `to_q.lora_A`（N=32 的瘦 GEMM） |
| SANA | 关 + batch_invariant_ops | 1.1e-3 | `ff.conv_point`（1x1 卷积，batch_invariant_ops 不覆盖） |
| Wan2.1 1.3B（不 compile，6 步） | 开 / 关 | 5.0e-3 / 3.7e-9 | — |
| SD3.5 | SDPA flash vs cudnn / flash vs efficient | 8.8e-2 / 8.8e-2 | attention 之后第一个 Linear |
| SD3.5 | 只 replay compile（regional） | 1.4e-1 | — |
| SD3.5 | 两侧 compile，replay 不开 grad，batch 相同 | 0（逐位一致） | — |
| SD3.5 | 两侧 compile，replay 不开 grad，batch 4→1 | 4.3e-2 | — |
| SD3.5 | 两侧 compile，replay 开 grad（真实训练），batch 相同 | 6.9e-2 | inference 图和 autograd 图不是同一张图 |
| Wan2.1 1.3B preset（默认开 compile） | 同 batch、replay 开 grad | 1.3e-2 | 超过 0.01 的 gate |
| SD3.5 | FP8 rowwise rollout vs bf16 master replay | 9.2e-1（最后一步） | 第 0 块 `attn.to_q` |
| SANA | FP8 rollout vs fp16 master replay | 6.1e2 | `caption_projection.linear_1`；排除它后仍有 5.1e2，SANA 的 fp16 激活接近上限，不适合 FP8 |
| SD3.5 | NVFP4 max scaling / four-over-six | 平均 0.82 / 0.73 | 见吞吐 sprint T3 |

所有差异都集中在最后一两步（R5 的放大效应）：例如 SD3.5 在 reduction 开着时，第 0–5 步只有 8e-5 到 1e-3，第 7 步才到 1.1e-1。

### P1 判定
- SD3.5 和 Wan 只靠"关掉 reduced-precision reduction"就达到了 ≤1e-6 的目标，不需要 batch_invariant_ops，所以**不做** batch-invariant 诊断开关。
- SANA 剩余约 1e-3 来自 cuBLAS 瘦 GEMM 和 cuDNN 1x1 卷积的选核，在 0.01 的 gate 之下。要逐位一致，需要卷积也 batch-invariant，代价不值得。
- batch_invariant_ops 原版的 Triton 配置在 sm120 上超出 shared memory，探针里用的是把 `num_stages` 改成 2 的本地副本。

### 对现有约束的影响
- "gen batch == replay batch"这条约束：SD3.5 和 Wan 在默认设置下已经可以放开（3.7e-9），SANA 也在 gate 内（7.5e-4）。preset 里的 `samples_per_generation_batch: 1` 大多出于显存考虑，没有改。
- "compile off"这条约束仍然成立：compile 后的 replay（开 grad）和任何 rollout 都不是同一张图，SD3.5 差 7e-2，Wan 差 1.3e-2。所以开 compile 的 preset（SD3.5 PickScore、Wan 1.3B）首步 parity gate 会失败，或者需要 TIS/RS 兜底。

### R4（logprob 归约）
fp32 `.mean` 在真实流程里的误差底是 3e-9 到 3e-8（上面"关"的几行），不需要改成固定顺序。

### P2 长跑漂移
- 500 次 update 的 FP8 长跑**没有做**：单卡和另一会话共享，GPU 时间优先给了多目标实验。FP8 在 SD3.5 上第 0 次 update 前就已经差 0.9（最后一步），已经足够说明 FP8 rollout 必须开 TIS/RS。

### P3
- P3.2：attention 后端之间差 8.8e-2（SD3.5），已写进本表；ring CP 和 SP 只能在多卡上测。
- P3.3（SDC）：按计划不做。
