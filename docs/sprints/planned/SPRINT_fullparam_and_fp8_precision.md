# SPRINT: full-param 替换 LoRA + FP8/FP4 精度轴（吞吐杠杆）(planned)

状态：**planned（2026-06-18 从 [[SPRINT_gemm_utilization]] 拆出）**。GEMM 审计的 profiler（P0）+ QKV 融合 A/B（P1，低 ROI 不落 runtime）+ torch.compile（P2，默认开）已全部落地、那个 sprint 已归档 `done/`；这里收剩下两个**都受硬件门控**的吞吐杠杆。

## 0. 来历

[[SPRINT_gemm_utilization]]（`done/`）测完后，剩下的两条都不是"实现一个已知安全改动"，而是受显存/多卡/kernel 支持门控的较大杠杆，故单列、低优先。

## 1. P1.5 — full-param 替换 LoRA（仍是 LoRA 的家族）

- **surface**：`configs/model/diffusion/sd3_5/medium.yaml:14`、`wan_2_1/{14b.yaml:11, i2v_14b.yaml:14, 1_3b.yaml:12}`、`wan_2_2/{a14b.yaml:14, i2v_a14b.yaml:18}` 仍 `use_lora: true`。
- **why**：全参训练是 LoRA 之外**最大的非-FP8 吞吐/质量杠杆**（GEMM 不再被 adapter 旁路稀释）。cosmos predict2.5 已是全参基线。
- **gate**：受显存/多卡门控——全参 14B 在单卡放不下，需多卡或更大卡。先确认目标家族 + 硬件再动。

## 2. P3 — FP8/FP4 精度轴

- **surface**：`vrl/config/precision.py:35` `_CANONICAL = ("fp32", "bf16", "fp16")` 无 fp8/fp4 token。
- **why**：FP8 是 Hopper/Blackwell 上的下一档吞吐杠杆。
- **状态**：用户 **2026-06-14 已 park**，分析保留在 [[SPRINT_gemm_utilization]]。需 kernel/硬件支持 + 数值稳定性验证（与现有 logprob drift guard `vrl/trainers/online/trainer.py:909-915` 对齐）才解冻。

## 3. 非目标

- 不重开 gemm 已落的 profiler / QKV / compile 结论。
- 不在无 FP8 kernel 支持的硬件上强行加 fp8 token（会变成假旋钮）。

## 相关
- [[SPRINT_gemm_utilization]]（`done/`，父 sprint）
- `vrl/scripts/perf/gemm_projection_breakdown.py`、`vrl/scripts/perf/compile_benchmark.py`（已落地的测量工具）
