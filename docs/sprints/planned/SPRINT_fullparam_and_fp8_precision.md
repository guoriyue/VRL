# SPRINT: full-param 替换 LoRA + FP8/FP4 精度轴（吞吐杠杆）(planned)

状态：**部分落地（2026-06-18 拆出）；2026-06-27 两条轴都被本机实测推进**（见下方复核）。GEMM 审计的 profiler（P0）+ QKV 融合 A/B（P1，低 ROI 不落 runtime）+ torch.compile（P2，默认开）已全部落地、那个 sprint 已归档 `done/`。本 sprint：**P3 的精度-修正地基（config 轴 + dtype + TIS + drift guard + 5090 实测）已落地**（详见 §2），rollout fp8 kernel 路由已由 [[SPRINT_fp8_rollout_gemm_kernel]] 落地并 live 验证；~~剩余是假旋钮 family guard / recipe wiring~~ **假旋钮 guard 已落地（核对 2026-07-08）**：双层——`apply_rollout_quantization` 在唯一通用 swap 点（`vrl/models/diffusion/build.py:108`）0 命中即 `RuntimeError`，外加 family/scheme 无关的 backstop `assert_rollout_quantization_applied`（`vrl/models/loader.py:149`，worker 载权重时数 `QuantizedLinear`，`vrl/generation/execution/worker.py:687`），CPU 测试齐（`tests/nn/quantization/test_fp8.py` 2026-07-08 复跑绿）。剩余仅 **P1.5（full-param）**，原受多卡显存门控——**门已解，见复核**。

> **2026-06-27 复核（[[SPRINT_lossless_diffusion_rl_research]] + 本轮工作）：**
> - **full-param 轴（P1.5）：单卡显存门已解。** 本轮建了 `actor.optim.optim_8bit`（bitsandbytes AdamW8bit，Blackwell sm_120 实测通），int8 Adam 状态让 full-param 2B 在单卡 32GB fit（fp32 Adam 状态 ~16GB 原本 OOM）。**对 RL 安全**（量化优化器状态不是 forward，不碰 old_log_prob）。落地见 [[SPRINT_cosmos_predict2_2b_trustworthy_curve]] / 记忆 `project_fullparam_8bit_adam`。caveat：8-bit 只解优化器状态，**video 激活值仍要小 shape**（240p_33f）。
> - **fp8/fp4 轴：NCU 确认 bf16 单卡 compute 已饱和**（GEMM tensor SOL 45% ≈ 最优方阵 47% = 硬件上限），所以 **fp8/fp4 是唯一能越过 bf16 上限的 compute 杠杆——但有损，只能离 policy path**。"吞吐杠杆"在 bf16 上已无空间；fp8 是越界手段不是无损优化。原 P1（QKV 融合）低 ROI 结论被 NCU 加强（compile 已把非-matmul 融到 3.5%，手写融合端到端 ~2-3%）。
> - **2026-06-27 外部佐证（deep-research）：本 sprint 的精度-split 契约 = 发表 SOTA。** `FP4 Explore, BF16 Train`（arXiv:2604.06916, 2026-04）用近乎相同的设计（FP4 只跑 rollout、训练全程 bf16、**log-prob 在 bf16 重算所以 old_log_prob 永不被 FP4 污染**）报 ~2.5-3x rollout / 1.5-2x 端到端，独立印证了这里的 TIS/RS + 受控 rollout 轴方向。caveat：其数在 H100，5090（sm_120）FP4 路径不同 → 幅度本机重测。我们已落地的 fp8 rollout（1.40x、漂移 6.6%）是同一契约的 fp8 实例；FP4 是下一档但仍 gated（dtype 齐、kernel 路由待接）。

## 0. 来历

[[SPRINT_gemm_utilization]]（`done/`）测完后，剩下的两条都不是"实现一个已知安全改动"，而是受显存/多卡/kernel 支持门控的较大杠杆，故单列、低优先。

## 1. P1.5 — full-param 替换 LoRA（仍是 LoRA 的家族）

- **surface**：`configs/model/diffusion/sd3_5/medium.yaml:14`、`wan_2_1/{14b.yaml:11, i2v_14b.yaml:14, 1_3b.yaml:12}`、`wan_2_2/{a14b.yaml:14, i2v_a14b.yaml:18}` 仍 `use_lora: true`。
- **why**：全参训练是 LoRA 之外**最大的非-FP8 吞吐/质量杠杆**（GEMM 不再被 adapter 旁路稀释）。cosmos predict2.5 已是全参基线。
- **gate**：受显存/多卡门控——全参 14B 在单卡放不下，需多卡或更大卡。先确认目标家族 + 硬件再动。

## 2. P3 — FP8/FP4 精度轴（精度-修正机器已落地 2026-06-18；rollout kernel 待接）

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

## 3. 下一步：让 fp8 rollout 真正 alive（kernel sprint）

根因（实测）：普通 `nn.Linear` 在 float8 上 `"addmm_cuda" not implemented for 'Float8_e4m3fn'`——fp8 不是 drop-in dtype，GEMM 必须显式"量化激活+量化权重+scale → `torch._scaled_mm` → 高精度累加"，非矩阵乘部分留 bf16 master。所以"alive"要做的是把生成引擎前向的策略 GEMM 换成 fp8 路径：

1. **选量化路径**：优先 **torchao `Float8Linear`**（module swap，和 PyTorch 原生 `_scaled_mm` 一脉、最省事）；备选 TransformerEngine fp8 linear、或 cosmos-rl 式 runtime patch（`apply_fp8_linear_patch` 动态量化）。
2. **改前向**：在 `vrl/generation/` 的 diffusion DiT 前向里，把**策略 GEMM**（QKV / MLP 投影）的 `nn.Linear` swap 成 fp8 linear；**跳过** embedding / 最后投影 / norm / SDE-logprob 数学（这些留 bf16 master，绝不 fp8）。这一步是 per-架构的。
3. **scaling recipe**：先 per-tensor dynamic amax（最简）；不够再上 per-row 或 128×128 per-block（slime 式）。静态 vs 动态量化也在这步定。
4. **Guard resolution**: FP8 is live. The rollout's ordinary
   parameter/autocast precision remains `policy.rollout.dtype`; quantization
   replaces selected GEMMs instead of storing the whole model as float8. FP4
   fails as reserved/unavailable in `QuantizationPolicy`, before runtime
   construction. The family zero-swap guard and worker-side `QuantizedLinear`
   backstop already prove that an FP8 builder applied the swap.
5. **数值验证**：把 `vrl/scripts/perf/fp8_rollout_drift_probe.py` 换成真实 DiT 跑一遍，量 `ratio_dev` / `rs_seq_masked_fraction` 分布。默认 auto correction 应可直接跑；只有当实测长期大面积截断/拒绝时，才进入 expert override 校准。
6. **吞吐验证**：用 `gemm_projection_breakdown.py` / `compile_benchmark.py` 量 rollout 段实际加速，再按 colocated 串行的 Amdahl 折算端到端收益（rollout 占比是关键，先测）。

地基（本 sprint 落地的 precision policy / drift guard / TIS / probe）就是给这一步用来**测量+修正+验证**漂移的；kernel 接上后它们零改动直接生效。

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
