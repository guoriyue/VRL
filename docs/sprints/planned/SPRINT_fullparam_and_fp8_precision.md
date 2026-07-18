# SPRINT: full-param 替换 LoRA + FP8/FP4 精度轴（吞吐杠杆）(planned)

状态：**部分落地**。P3 precision/drift correction 与 rollout FP8 kernel/
zero-swap guard 已落地。P1.5 的 native-FP16 FP32-master capacity/numerics gate
已于 2026-07-13 通过；长期 held-out learning 仍未由五步 pilot 证明。旧的
direct FP16/BF16 parameter update 证据无效。

> **2026-07-13 correctness correction:** AdamW8bit 只压缩 optimizer moments，
> 不提供 FP32 master parameter。SANA 原生 FP16 的 1.604B 参数在 `lr=5e-6`
> 首步只有 4.345% 元素改变，完整 attention tensors 被 FP16 ULP 冻结。
> `FP32MasterWeightOptimizer` 现从低精度 trainables 自动派生，保存 sub-ULP
> residual，并在成功 step 后发布 rounded visible weights；FP16 backward 另由
> GradScaler 保护。新的五步 GPU pilot 已在 RTX 5090 32GB 上完成：5/5 update、
> 5/5 exact pre-update parity、finite non-zero gradients、严格断点续训、完整
> master residual checkpoint 均通过。该结果解锁 SANA 1.6B 的 full-parameter
> systems path，但不等于 held-out quality improvement。五步后的单 prompt
> official-DPM probe 仍是正常图像，但 aesthetic/PickScore 均小幅下降；不得据此
> 启动或宣传长期 quality curve。

> **2026-06-27 复核（[[SPRINT_lossless_diffusion_rl_research]] + 本轮工作）：**
> - **full-param 轴（P1.5）：旧容量结论只覆盖 optimizer moments。**
>   `actor.optim.optim_8bit` 不改变 forward/old_log_prob，因此对 policy forward
>   精度透明；它本身不保证低精度参数更新正确。长期 full-param 还需要 FP32
>   master、checkpoint/resume 与真实 activation peak gate。
> - **fp8/fp4 轴：NCU 确认 bf16 单卡 compute 已饱和**（GEMM tensor SOL 45% ≈ 最优方阵 47% = 硬件上限），所以 **fp8/fp4 是唯一能越过 bf16 上限的 compute 杠杆——但有损，只能离 policy path**。"吞吐杠杆"在 bf16 上已无空间；fp8 是越界手段不是无损优化。原 P1（QKV 融合）低 ROI 结论被 NCU 加强（compile 已把非-matmul 融到 3.5%，手写融合端到端 ~2-3%）。
> - **2026-06-27 外部佐证（deep-research）：本 sprint 的精度-split 契约 = 发表 SOTA。** `FP4 Explore, BF16 Train`（arXiv:2604.06916, 2026-04）用近乎相同的设计（FP4 只跑 rollout、训练全程 bf16、**log-prob 在 bf16 重算所以 old_log_prob 永不被 FP4 污染**）报 ~2.5-3x rollout / 1.5-2x 端到端，独立印证了这里的 TIS/RS + 受控 rollout 轴方向。caveat：其数在 H100，5090（sm_120）FP4 路径不同 → 幅度本机重测。我们已落地的 fp8 rollout（1.40x、漂移 6.6%）是同一契约的 fp8 实例；FP4 是下一档但仍 gated（dtype 齐、kernel 路由待接）。

## 0. 来历

[[SPRINT_gemm_utilization]]（`done/`）测完后，剩下的两条都不是"实现一个已知安全改动"，而是受显存/多卡/kernel 支持门控的较大杠杆，故单列、低优先。

## 1. P1.5 — full-param 替换 LoRA（仍是 LoRA 的家族）

- **surface**：`vrl/config/presets/model/diffusion/sd3_5/medium.yaml`、
  `vrl/config/presets/model/diffusion/wan_2_1/` 与 `wan_2_2/` 的模型 preset
  仍以 LoRA 为主。
- **why**：全参训练是 LoRA 之外**最大的非-FP8 吞吐/质量杠杆**（GEMM 不再被 adapter 旁路稀释）。cosmos predict2.5 已是全参基线。
- **gate**：受显存/多卡门控——SANA 1.6B 已通过单 RTX 5090 的五步
  FP32-master pilot；该结果不能外推到 14B。14B 全参仍需多卡、offload 或更大
  显存，并按家族重新跑真实 backward/resume gate。

## 2. P3 — FP8/FP4 精度轴（FP8 已落地；FP4 仍 gated）

- **why**：FP8 是 Hopper/Blackwell 上的下一档吞吐杠杆。
- **硬件门控已作废**：本机是 RTX 5090（Blackwell, sm_120），FP8(e4m3/e5m2) + FP4(e2m1) dtype 齐全，`torch._scaled_mm` 在 sm_120 实测可跑（fp8 GEMM vs bf16 平均相对误差 ~3.8%）。原 sprint 假设的"无 FP8 硬件"前提不成立——真正的门变成 **rollout 引擎的 diffusion 前向是否走 `_scaled_mm`**（kernel 路由），而非硅片。
- **已落地（精度-修正地基，全部单测 + 5090 实测过）**：
  - `vrl/config/precision.py`：基础 dtype 与 quantization 已拆开；受控 FP8 rollout 通过 `rollout: {dtype: bf16, quantization: {format: fp8}}` 表达，FP4 为保留词，policy resolver 在 runtime 构造前拒绝。
  - `vrl/models/dtypes.py`：fp8/fp4/e4m3/e5m2 → torch dtype（缺失 dtype 的 torch build 显式报错，不静默退化）。
  - **TIS/RS 自动派生**：用户通过 rollout quantization block 表达意图；`build_trainer_config` 在 training/rollout stage policy 不同且没有显式 expert block 时自动注入 `PrecisionCorrectionConfig(tis_mode="truncate", rs_mode="seq_mean_k1")`。TIS 在 PPO clip 前截断 rollout→replay 重要性权重，RS 丢弃整段 out-of-band 轨迹；两个 GRPO 家族共用 `vrl/algorithms/logprob_mismatch.py` 的原语，`tis_clip_fraction` / `rs_seq_masked_fraction` 作为健康指标上报，不是普通用户调参面。
  - drift guard 也随 split 自动派生为 catastrophic gate：非 split 仍保留逐位 parity 语义；FP8 rollout split 下默认只对非有限或极端 log-ratio fail-fast，让 TIS/RS 处理正常低精度漂移。FP4 当前无法构造 resolved policy。显式 `trainer.precision_drift_guard` / `trainer.precision_correction` 仍作为 expert override 保留。
  - 验证资产：`vrl/scripts/perf/fp8_rollout_drift_probe.py`——真 `_scaled_mm` fp8 GEMM 测漂移 → 喂 `compute_logprob_mismatch_stats` → guard 触发 → TIS 在轨迹累积漂移的尾部 engage。实测：单步漂移温和（ratio_dev max ~0.14），沿 T=35 去噪步累积后 max ~0.87，cap=1.5 时 TIS 截断 ~1.2% 轨迹。
- **kernel 已接（2026-06-19，见 [[SPRINT_fp8_rollout_gemm_kernel]]）**：`Fp8Linear`（`vrl/nn/quantization/fp8.py`，真 `_scaled_mm`）+ `base.quantize_transformer_fp8` + 三家 rollout builder（sd3_5 / wan / cosmos predict2）全链路 wiring 已落地。5090 实测：fp8 比 bf16 **快 1.40x**（geomean）、24-block DiT 端到端漂移 **6.6%**。`precision.rollout.quantization.format=fp8` live run 可起。predict2_5/anima wiring + 假旋钮 backstop guard 已落地（2026-06-19，[[SPRINT_rollout_optimization_layer]] P0）。
- **历史**：用户 2026-06-14 曾 park，分析保留在 [[SPRINT_gemm_utilization]]。

## 3. FP8 kernel 状态

旧的“下一步让 fp8 alive”实施清单已由
[[SPRINT_fp8_rollout_gemm_kernel]] 完成，不再是活计划。当前剩余工作只包括
新增 family 的真实 swap/漂移验收，以及本 sprint 的 full-param master-weight
capacity gate；不得重复实现已经存在的 `_scaled_mm` 路径。

## 4. 非目标

- 不重开 gemm 已落的 profiler / QKV / compile 结论。
- 不让 `precision.rollout.quantization.format=fp8` 变成假旋钮，也不让保留的
  `format=fp4` 深入 runtime 才崩溃：FP8 token + GEMM kernel 已可 live，FP4 由 config
  policy resolver 在 runtime 构造前明确拒绝。FP8 的 family-level swapped-count guard
  **已落地**（见文件头状态：build.py 单一 swap 点 0 命中报错 + worker 载入 backstop
  数 `QuantizedLinear`），builder 漏接 swap 不再可能静默回 bf16。

## 相关
- [[SPRINT_gemm_utilization]]（`done/`，父 sprint）
- `vrl/scripts/perf/gemm_projection_breakdown.py`、`vrl/scripts/perf/compile_benchmark.py`（已落地的测量工具）
