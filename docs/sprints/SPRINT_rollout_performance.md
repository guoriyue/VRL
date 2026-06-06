# SPRINT: Rollout Performance

状态：proposed（基于 SD3.5 OCR GRPO fp32 profiling 数据，2026-06-06）。

## Profiling 结论

硬件：RTX 5090 32GB，SD3.5 Medium，fp32 full precision，10-step denoise，8 images/prompt。

### CUDA Kernel 时间分布（181s total GPU time，1 training epoch）

```text
GEMM / Linear layers:    86.0s   47.5%
Attention (SDPA):        61.6s   34.0%
Element-wise ops:        23.1s   12.8%
LayerNorm + other:       10.3s    5.7%
```

### 关键发现

1. **所有 kernel 都在跑 fp32。** Attention 用的是 `fmha_cutlassF_f32`（memory-efficient attention fp32 path），不是 FlashAttention。GEMM 用的是 `cutlass_80_tensorop_s1688gemm` fp32 tile。这是因为 config 明确设置了 `mixed_precision: "no"`、`bf16: false`（理由是 "SD3.5 OCR replay is fp16-fragile"）。

2. **Attention backward 比 forward 贵。** Forward attention 28.6s，backward attention 33.0s。这只影响 training（replay），不影响 rollout。

3. **torch.compile 被关掉了。** 理由是 "dynamic batch sizes from zero-advantage filtering"。但这只影响 training microbatch，不影响 rollout（rollout batch size 固定）。

4. **diffusers 不是瓶颈。** fp32 → bf16 转换、torch.compile、FlashAttention-2 都可以在 diffusers transformer 上直接做，不需要 native executor。

### Trainer 阶段时间分布（phase_events）

```text
backward (replay + gradient):  108.3s   58.3%
evaluate (reference forward):   77.4s   41.7%
optim_step:                      0.1s    0.0%
```

72 次 evaluate + 72 次 backward = 8 prompts × 8 images × ~0.9 timesteps 的 replay。

## diffusers 是否挡路

不是。当前性能瓶颈全部来自 precision/compile 配置，不来自 diffusers module 结构。

diffusers 已经支持：

```text
bf16/fp16 推理                         ← 只是当前 config 关了
attn_implementation="flash_attention_2" ← 只要输入是 fp16/bf16
torch.compile()                        ← diffusers transformer 可直接编译
set_attn_processor()                   ← SD3.5 已经用了这个 hook
```

diffusers 不支持但当前不是瓶颈的：

```text
fused QKV（separate to_q/to_k/to_v）    ← 节省一次 kernel launch，不是主要开销
自定义 memory layout                    ← fp32→bf16 后不再是问题
block-level mixed precision             ← 当前整体 fp32，先切 bf16 再说
Triton/FlashInfer 级别 kernel           ← 标准 FA2 + compile 已经够
```

结论：native transformer executor（`SPRINT_diffusion_native_transformer_executor`）的 ROI 取决于 bf16 + compile 启用后 attention 是否仍然是瓶颈。当前不需要。

## 优化路径（按优先级排序）

### P0：Rollout bf16 dtype

当前 rollout 和 training 共享同一个 precision 设置。rollout 不算梯度，不需要 fp32 数值稳定性。

改动：

```text
vrl/config/precision.py     — 增加 rollout_dtype 轴，独立于 trainer precision
vrl/scripts/common/online.py — rollout model 用 rollout_dtype 加载
configs/base/actor.yaml      — 默认 rollout_dtype: bf16
```

预期收益：

```text
Attention: fmha_cutlassF_f32 → fmha_cutlassF_f16  ≈ 2x
GEMM:      cutlass fp32 → cutlass fp16/bf16         ≈ 1.5-2x
同时解锁 FlashAttention-2（只支持 fp16/bf16）
```

不影响 training replay 精度。

### P1：torch.compile for rollout

当前 compile 被整体关掉（`torch_compile.enable: false`），理由是 training 的 zero-advantage filtering 导致动态 batch size。但 rollout batch size 固定（`sample_batch_size: 8`），shape 是静态的。

改动：

```text
分离 rollout compile flag 和 trainer compile flag
rollout model 在 from_spec 里 compile
确认 rollout forward 不触发 recompile
```

预期收益：

```text
Element-wise ops 和 GEMM 融合   ≈ 15-30% forward speedup
减少 kernel launch overhead
```

### P2：Multi-GPU data-parallel rollout

当前单 GPU colocated。rollout 不需要 FSDP，只需要 data parallelism：两个 worker 各生成半个 batch。

改动：

```text
configs/experiment 里改 distributed 设置
distributed.resources.rollout.num_workers=2
distributed.resources.rollout.num_gpus=2
```

预期收益：rollout wall clock ≈ 2x（线性扩展，无通信开销）。

需要：第二张 GPU。

### P3：减少 denoise steps

当前 training rollout 10 步，eval 40 步。每步是一次完整 transformer forward。

可能的改动：

```text
尝试 5-step scheduler（LCM、consistency、Lightning）
评估 reward quality vs step count tradeoff
```

预期收益：线性——步数减半，rollout 时间减半。需要验证 image quality 和 reward signal 不劣化。

### P4：Stage pipeline overlap

```text
text-encode → [denoise × N] → VAE decode → reward
```

text-encode 可以和前一个 batch 的 reward scoring 重叠。但 text-encode 相对 10× transformer forward 很便宜，收益可能很小。

前置条件：profile rollout 内部各阶段的绝对时间。`SPRINT_diffusion_rollout_system` 已经讨论了这个问题，结论是需要 profiling 数据再决定。

### 暂不做

```text
Native transformer executor    — attention 占 34% 但可以通过 bf16+FA2 优化，不需要重写 transformer
Fused QKV projections          — 节省一次 kernel launch，不是主要开销
Custom Triton kernels           — 标准 FA2 + compile 已经够
Context projection cache        — cross-attention KV projection 是 attention 的小部分，不值得复杂度
```

## 预期总收益

```text
                          当前 (fp32)    P0 (bf16)    P0+P1 (bf16+compile)    P0+P1+P2 (2GPU)
Rollout forward/step:     ~18s          ~9s          ~6-7s                   ~3-4s
10-step epoch rollout:    ~180s         ~90s         ~60-70s                 ~30-35s
```

估算基于 SD3.5 Medium 2.5B 参数，512x512，8 images batch。实际数字需要实测验证。

## Completion criteria

```text
P0: rollout 跑 bf16，training replay 保持 fp32，reward quality 不劣化
P1: rollout transformer compile 启用，无 recompile，forward 时间下降 >15%
P2: 2-GPU rollout wall clock 接近 1-GPU 的 50%
每个 phase 都有 before/after profiling 数据对比
```
