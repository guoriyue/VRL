# SPRINT: full-param 替换 LoRA + FP8/FP4 精度轴（吞吐杠杆）(planned)

状态：**部分落地（2026-06-18 从 [[SPRINT_gemm_utilization]] 拆出）**。GEMM 审计的 profiler（P0）+ QKV 融合 A/B（P1，低 ROI 不落 runtime）+ torch.compile（P2，默认开）已全部落地、那个 sprint 已归档 `done/`。本 sprint：**P3 的精度-修正地基（config 轴 + dtype + TIS + drift guard + 5090 实测）已落地**（详见 §2），剩 rollout kernel 路由；**P1.5（full-param）仍 planned**，受多卡显存门控。

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
  - `vrl/config/precision.py`：`_CANONICAL` 加 `fp8`/`fp4`；重开**受控的 `rollout` 轴**（scalar 形式仍禁止 split，只有显式 `{forward: bf16, rollout: fp8}` 可达；fp8/fp4 仅 rollout 轴合法，forward/math 拒绝）。
  - `vrl/models/dtypes.py`：fp8/fp4/e4m3/e5m2 → torch dtype（缺失 dtype 的 torch build 显式报错，不静默退化）。
  - **TIS**（truncated importance sampling）`tis_mode=off|truncate|clip|mask`，在 PPO clip 前截断 rollout→replay 重要性权重，防 fp8 漂移在负优势样本上撑爆未截断梯度。**配置和原语都在算法层漂移模块 `vrl/algorithms/logprob_mismatch.py`**（`PrecisionCorrectionConfig` + `apply_truncated_importance_weight`，与 `compute_logprob_mismatch_stats` 并排 = 测量+修正一处）；**旋钮不在 GRPOConfig**，而在 trainer 层 `trainer.precision_correction`（与 `precision_drift_guard` 并排），trainer `__init__` 注入算法的 `precision_correction` 槽。两个 GRPO 家族共用同一原语，`TrainStepMetrics.tis_clip_fraction` 上报。
  - 既有 drift guard（`vrl/trainers/online/precision_guard.py`，`auto` 在 rollout!=compute 时 →fail）对 fp8 直接复用，无需改动——guard 测量/拦，TIS 在 loss 端 bound，是它的对侧。
  - 验证资产：`vrl/scripts/perf/fp8_rollout_drift_probe.py`——真 `_scaled_mm` fp8 GEMM 测漂移 → 喂 `compute_logprob_mismatch_stats` → guard 触发 → TIS 在轨迹累积漂移的尾部 engage。实测：单步漂移温和（ratio_dev max ~0.14），沿 T=35 去噪步累积后 max ~0.87，cap=1.5 时 TIS 截断 ~1.2% 轨迹。
- **kernel 已接（2026-06-19，见 [[SPRINT_fp8_rollout_gemm_kernel]]）**：`Fp8Linear`（`vrl/nn/quantization/fp8.py`，真 `_scaled_mm`）+ `base.quantize_transformer_fp8` + 三家 rollout builder 全链路 wiring 已落地。5090 实测：fp8 比 bf16 **快 1.40x**（geomean）、24-block DiT 端到端漂移 **6.6%**。`precision.rollout=fp8` live run 可起。剩真实 checkpoint 的端到端验证。
- **历史**：用户 2026-06-14 曾 park，分析保留在 [[SPRINT_gemm_utilization]]。

## 3. 下一步：让 fp8 rollout 真正 alive（kernel sprint）

根因（实测）：普通 `nn.Linear` 在 float8 上 `"addmm_cuda" not implemented for 'Float8_e4m3fn'`——fp8 不是 drop-in dtype，GEMM 必须显式"量化激活+量化权重+scale → `torch._scaled_mm` → 高精度累加"，非矩阵乘部分留 bf16 master。所以"alive"要做的是把生成引擎前向的策略 GEMM 换成 fp8 路径：

1. **选量化路径**：优先 **torchao `Float8Linear`**（module swap，和 PyTorch 原生 `_scaled_mm` 一脉、最省事）；备选 TransformerEngine fp8 linear、或 cosmos-rl 式 runtime patch（`apply_fp8_linear_patch` 动态量化）。
2. **改前向**：在 `vrl/generation/` 的 diffusion DiT 前向里，把**策略 GEMM**（QKV / MLP 投影）的 `nn.Linear` swap 成 fp8 linear；**跳过** embedding / 最后投影 / norm / SDE-logprob 数学（这些留 bf16 master，绝不 fp8）。这一步是 per-架构的。
3. **scaling recipe**：先 per-tensor dynamic amax（最简）；不够再上 per-row 或 128×128 per-block（slime 式）。静态 vs 动态量化也在这步定。
4. **拆 guard**：`vrl/scripts/common/online.py` 现在 fp8/fp4 直接 `NotImplementedError`；接上后改成"engine 报告支持 fp8 才放行"的能力检查，并让 `rollout_weight_dtype` 在 fp8 时 storage 仍 bf16（master），量化在 GEMM 内部做，而不是把模型存成 float8。
5. **数值验证**：把 `vrl/scripts/perf/fp8_rollout_drift_probe.py` 换成真实 DiT 跑一遍，量 `ratio_dev` 分布，据此校准 `trainer.precision_correction` 的 cap；drift guard 设 `warn` 先跑一轮观察。
6. **吞吐验证**：用 `gemm_projection_breakdown.py` / `compile_benchmark.py` 量 rollout 段实际加速，再按 colocated 串行的 Amdahl 折算端到端收益（rollout 占比是关键，先测）。

地基（本 sprint 落地的 precision policy / drift guard / TIS / probe）就是给这一步用来**测量+修正+验证**漂移的；kernel 接上后它们零改动直接生效。

## 4. 非目标

- 不重开 gemm 已落的 profiler / QKV / compile 结论。
- 不让 `precision.rollout=fp8/fp4` 变成假旋钮或深处崩溃：token 已加（硬件 kernel 实测可用），但 live rollout 在引擎 GEMM 未接前由 `vrl/scripts/common/online.py` 显式 `NotImplementedError` 拦住，fp8/fp4 当前是"测量/修正/验证-only"。引擎 scaled_mm 接上后再放开。

## 相关
- [[SPRINT_gemm_utilization]]（`done/`，父 sprint）
- `vrl/scripts/perf/gemm_projection_breakdown.py`、`vrl/scripts/perf/compile_benchmark.py`（已落地的测量工具）
