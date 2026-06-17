# SPRINT: Rollout Performance

状态：profiling 结果归档 + 决定性优化已落地（2026-06-08~06-10）。D0 substage profiling（record_function ranges in executor.py）、D1 rollout-only transformer compile（rollout.denoise_compile，commit d3581ed，经 capability.supports_torch_compile + torch_compile_model_config 接线，非文中已不存在的 RolloutCompileMetadata 符号）、D2 recompile 观测（TORCH_LOGS=recompiles，commit bae13c0）、D5 sde_step_with_logprob 去 per-step host sync（step_index 参数，commit d3581ed）均已落地；GPM scale bump（vae_decode.tiling + sample_batch_size:16，commit d26dc75）取代旧 b8 默认。D3 clone cleanup（hygiene）、D4 staged rollout、P2 multi-GPU rollout（需第二张卡）按文档明确范围保留为可选/延后项，不计入完成。

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

## Denoise block strategy（不以 dtype 为主杠杆）

目标不是把所有东西用 dtype 一刀切提速，而是把热块本身看清楚、收紧、再编译。
这里的热块是：

```text
run_denoise_steps()
  for each timestep:
    snapshot current latents
    transformer forward
    scheduler / SDE step
    write replay buffers
```

`producer -> queue -> consumer` 是更高层的 rollout/trainer overlap。它能减少 trainer 等 rollout，
但不会让单个 diffusion denoise block 更快。diffusion model 内部的 denoise step 又是顺序依赖：

```text
x_t -> transformer -> scheduler -> x_{t-1}
```

所以不要把每个 denoise step 拆成 SGLang-Omni 式 stage。真正可 staged 的只是在 batch 级别 overlap：

```text
batch A denoise
batch B text encode
batch C VAE decode / reward
```

但这要等 profiling 证明 decode/reward/text-encode 有明显占比。当前主线先优化 denoise block 本身。

### D0：Denoise substage profiling

当前只有 chunk-level `stage_durations["denoise"]`，粒度太粗。D0 不再新增一套
自定义 profiler；直接用 torch profiler trace 里的 `record_function` ranges 读：

```text
generation.latent_snapshot
generation.denoise_forward
generation.scheduler_step
generation.latent_write
generation.trajectory_buffer_write
```

成功标准：

```text
rollout torch profiler trace 能看到 denoise 内部 ranges
能从 trace/key_averages 区分 transformer forward、scheduler/SDE、trajectory buffer write、latent snapshot
不在 executor 里维护第二套 wall-time profiler/counter
```

Implementation: keep `record_function(...)` ranges in the denoise loop and read
them from rollout torch profiler traces. For D0 diagnosis, enable CPU+CUDA
activities on rollout so the custom ranges and CUDA kernels are both visible:

```bash
python -m vrl.scripts.train \
  --config experiment/diffusion/sd3_5/online_grpo_ocr \
  /profile=torch_profiler \
  rollout.torch_profiler.activities='[cpu,cuda]'
```

Trace output is under:

```text
<trainer.output_dir>/torch_profiler/generation/<rollout-worker>/
```

本地 D0 run（SD3.5 OCR，`outputs/sd3_5_ocr_grpo_d0_profile`，1 epoch run 在
rollout profile 写完后中断，未用于训练结论；这些数字来自当时的临时同步 wall-time
counter，后续同类数字应从 torch profiler trace/key_averages 读取）：

| stage | total over 8 rollout batches | mean / batch | note |
| --- | ---: | ---: | --- |
| `encode` | 0.499s | 0.062s | prompt encode |
| `prepare_latent` | 0.035s | 0.004s | latent init |
| `denoise` | 78.664s | 9.833s | 10 denoise steps |
| `denoise.forward_s` | 78.573s | 9.822s | 99.884% of denoise |
| `denoise.scheduler_s` | 0.075s | 0.009s | 0.096% of denoise |
| `denoise.trajectory_write_s` | 0.006s | 0.001s | 0.008% of denoise |
| `denoise.latent_snapshot_s` | 0.002s | 0.000s | 0.002% of denoise |
| `denoise.latent_write_s` | 0.002s | 0.000s | 0.002% of denoise |
| `decode` | 0.571s | 0.071s | VAE decode |

### How to read `denoise.forward_s`

`denoise.forward_s`/`generation.denoise_forward` 不是严格意义上的“GPU 每一毫秒都
100% 满载”。它是围住 `model.forward_step(state, step_idx)` 的 forward envelope。
用 torch profiler trace 读它时，看的是这个 range 下的 CPU duration、CUDA time
和内部 kernel 分布。

计时代码范围：

```python
with record_function("generation.denoise_forward"):
    step_output = model.forward_step(state, step_idx)
```

所以这个 range 可以回答：

```text
denoise loop 的等待时间几乎都在 model.forward_step
scheduler / clone / write 基本不是 wall-clock 瓶颈
```

但它不能证明：

```text
每一毫秒 GPU SM 都 100% occupied
没有 kernel launch gap
没有 memory stall
没有 CPU launch overhead
没有 attention/GEMM 内部 under-utilization
```

`model.forward_step` 里面实际是一串 GPU kernels：GEMM、attention、elementwise、
norm 等。旧 profiler 看到的 kernel 时间大头是：

```text
GEMM / Linear: 47.5%
Attention:     34.0%
Elementwise:   12.8%
```

因此正确读法是：`generation.denoise_forward` 占 denoise envelope 的绝对主导，
说明优化方向必须进入 transformer forward path；但 forward 内部 GPU 是否满载，还要看
torch profiler / Nsight 的 kernel trace、SM occupancy、memory bandwidth 和 kernel gaps。

### D0 interpretation：下一步只打 denoise forward

D0 的决定性信息不是“denoise 很慢”，而是“denoise 内部几乎只有 forward 慢”：

```text
denoise.forward_s         99.884% of denoise
denoise.scheduler_s        0.096% of denoise
denoise.trajectory_write_s 0.008% of denoise
denoise.latent_snapshot_s  0.002% of denoise
denoise.latent_write_s     0.002% of denoise
```

所以 sprint 不能再把 clone/write/scheduler cleanup 当成性能主线。它们可以作为 hygiene work，
但不会改变 rollout wall clock。主线必须直接作用在 `model.forward_step(...)` 调的 transformer forward：

```text
当前优化靶点：diffusers SD3 transformer forward
当前非靶点：scheduler/SDE、latent clone、trajectory buffer write、VAE decode、text encode
```

### U0：Utilization-first profiling（2026-06-07）

这次 profiling 要回答的不是“denoise 是否慢”，而是“慢的时候 GPU 是否已经把可用计算资源吃满”。
结论是：**GPU 不是 idle，但也没有接近峰值算力利用**。

粗粒度 `nvidia-smi` 采样看到 GPU utilization 100%、功耗约 575W、显存约 31GB，说明 workload
确实把 GPU 保持在 busy 状态。但 Nsight Compute 的 Compute(SM) / Memory / DRAM 指标显示，
主 kernel 离峰值还有明显距离。因此这里的“not fully used”含义是：

```text
不是：GPU 大量空闲，Python/HF overhead 是唯一瓶颈
而是：GPU 很忙，但主 kernel 没有吃满 SM / tensor core / memory peak
```

本地 NCU 简单采样结果：

| probe | Compute(SM) | Memory | DRAM | note |
| --- | ---: | ---: | ---: | --- |
| GEMM sample avg | 42.8% | 62.3% | 21.7% | 19 个 fp32 SIMT SGEMM samples |
| representative larger GEMM | 58.3% | 48.4% | 5.1% | 920us，waves/SM 1.22 |
| Attention samples | 41.9% | 24.4% | 4.4% | `fmha_cutlassF_f32` samples |

Duration-weighted NCU 读法更可靠，因为它不会让很多短小 GEMM 把平均值拉偏：

| probe | sampled duration | Compute(SM) | Memory | DRAM | Waves/SM | note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fp32 SIMT SGEMM | 8.758ms | 51.2% | 62.3% | 10.2% | 4.83 | 19 samples |
| fp32 attention forward | 17.190ms | 41.9% | 24.4% | 4.3% | 21.48 | 3 `fmha_cutlassF_f32` samples |

Nsight Systems 的时间分布也支持同一个判断：rollout/training 热点主要是 fp32 GEMM 和
fp32 memory-efficient attention，而不是 scheduler/write/clone：

```text
fp32 SIMT SGEMM:          40.000s   52.8%   51,013 instances
elementwise/copy/cat:     13.631s   18.0%  188,243 instances
fp32 attention forward:   11.496s   15.2%    2,883 instances
tensorop GEMM:             5.748s    7.6%   24,835 instances
fp32 attention backward:   2.293s    3.0%      185 instances
```

CUDA API 也显示很多 launch/sync 调用：

```text
cudaLaunchKernel:      228,357 calls, 33.569s total CPU time
cuLaunchKernel:         75,715 calls, 14.345s total CPU time
cudaStreamSynchronize:  12,119 calls, 21.290s total CPU time
```

但这不等于 denoise 内部有大量可 staged 的 GPU 空洞。把 GPU kernel intervals 按 `>=100ms`
的大 gap 切成 dense segments 后，主段内部几乎是连续发 kernel：

| dense segment span | GPU kernel busy | internal gap | busy ratio |
| ---: | ---: | ---: | ---: |
| 13.771s | 13.135s | 0.636s | 95.4% |
| 10.447s | 10.063s | 0.384s | 96.3% |
| 10.346s | 10.232s | 0.115s | 98.9% |
| 10.324s | 10.208s | 0.116s | 98.9% |
| 10.295s | 10.184s | 0.111s | 98.9% |
| 10.250s | 10.137s | 0.112s | 98.9% |
| 10.206s | 10.091s | 0.116s | 98.9% |

所以 staging 的结论要分层：它可以隐藏阶段之间的等待，但不能让 denoise dense segment 里的
fp32 GEMM / attention kernel 自己从 40-50% Compute(SM) 变成 90%。当前更像 kernel path /
precision / shape 利用率问题，不是 denoise 内部 producer-consumer pipeline 问题。

Attention 的 NCU 样本要谨慎读：这次 attention-only NCU 捕到的是中段 kernel sample
（单个 sample 约 4-6ms），Nsight Systems 的 aggregate 更适合判断全局时间占比。NCU 在这里主要用于看
kernel utilization characteristic，不用于替代 nsys 的全局时间占比。

BF16 rollout parity calibration（compute/replay fp32，rollout bf16，`n=2`，1 prompt group）失败：

```text
output:
outputs/diffusion_util_bf16_rollout_parity/

precision:
compute=fp32, rollout=bf16, math=fp32, frozen=bf16

guard:
mode=warn
worst_timestep=8
logprob_abs_diff_max=1.016e-02
ratio_abs_dev_max=1.011e-02
limit=1.000e-03
```

这说明 `fp32 replay + bf16 rollout` 不能直接作为默认性能路径；它会让 collection-time
old logprob 和 replay fresh logprob 偏离，破坏 GRPO ratio≈1 的假设。BF16/FA2 仍然可能是性能主线，
但必须走 parity gate：要么 replay 也切同 dtype 并证明 SD3.5 数值稳定，要么引入明确的 correction /
容差策略，不能静默开启。

保存的 profile artifacts：

```text
torch profiler trace:
outputs/diffusion_util_torch_profile_enabled/torch_profiler/generation/rollout-0/

nsys report:
outputs/diffusion_util_nsys/sd3_5_ocr_rollout.nsys-rep
outputs/diffusion_util_nsys/sd3_5_ocr_rollout.sqlite

ncu roofline reports:
outputs/diffusion_util_ncu/sd3_5_ocr_gemm_attention_roofline.ncu-rep
outputs/diffusion_util_ncu/sd3_5_ocr_attention_mid_roofline.ncu-rep

bf16 rollout parity calibration:
outputs/diffusion_util_bf16_rollout_parity/training_debug.jsonl
outputs/diffusion_util_bf16_rollout_parity/metrics.csv
```

### U1：bf16/fp16 forward micro-benchmark（2026-06-07）

U0 之后所有 SD3.5 实验已默认切到 `precision: bf16`（rollout 与 replay 同 dtype，
解除了 U0 里 `fp32 replay + bf16 rollout` 的 parity 失败——两边 dtype 一致，ratio≈1 成立）。
为量化"低精度到底带来多少显存余量和多少 forward 提速"，对 SD3.5 transformer forward
（D0：占 denoise 99.88%）做隔离实测。

硬件 RTX 5090 31.3GB；真实 rollout shape：latent `16×64×64`（512px），text_seq 256，
eager（未开 compile/FA2 强制）；median over 20 iters，`max_memory_allocated` 读峰值，
`max_batch` 为 forward 不 OOM 的最大 batch（step=8）：

| dtype | weights | B=8 latency | peak mem | max batch (forward-only) |
| --- | ---: | ---: | ---: | ---: |
| fp32 | 8.36GB | 467.6ms | 9.95GB | ~192 |
| bf16 | 4.18GB | 152.2ms | 5.01GB | ~376 |
| fp16 | 4.18GB | 140.1ms | 5.01GB | ~376 |

关键读法：

```text
fp16 vs bf16：weights / peak mem / max_batch 三项完全相同（都是 16-bit, 2 bytes/elem）。
              latency 140 vs 152ms 是 run-to-run 噪声，不是 fp16 的系统性优势。
              → fp16 相对 bf16 在显存和吞吐上零收益；增大 batch 的红利来自 16-bit vs fp32。
fp32 → bf16：weights 砍半（8.36→4.18GB）、peak 砍半（9.95→5.01GB）、
              forward 快 ~3x（467→152ms，Tensor Core 替掉 fp32 SIMT，优于 U0 预测的 2x）、
              forward-only max_batch 接近翻倍（192→376）。
```

注意 `max_batch≈376` 是**纯 transformer forward** 的余量，不是完整 rollout 上限。
完整 rollout 还要叠加 T5 text encoder、VAE、GRPO trajectory replay buffer（存 10 个 denoise step
的 latent）、CFG 2x（cfg_4_5 → cond+uncond）。所以可用 `sample_batch_size` 会低于 376，
但 fp32→bf16 约 2x 的**相对**余量成立——当前 `sample_batch_size: 8` 有明确上调空间。

测量脚本为一次性产物（`_scratch_precision_bench.py`），数字记录于此后即删除。

#### Full rollout profile（bf16，2026-06-07）

micro-benchmark 之后又跑了完整 SD3.5 OCR rollout profile，使用当前默认 `precision: bf16`，
同一个 10-step denoise / rollout batch=8 shape，关闭 eval，未开启 compile。

first-step parity 通过：

```text
output:
outputs/diffusion_util_low_precision_torch_profile_bf16/training_debug.jsonl

compute=bf16, rollout=bf16, math=fp32, frozen=bf16
abs_diff_mean=1.70e-08
abs_diff_max=1.19e-07
ratio_min=1.000000000
ratio_max=1.000000119
```

torch profiler 直接对比：

| range/op | fp32 CUDA total | bf16 CUDA total | speedup |
| --- | ---: | ---: | ---: |
| `engine.forward_chunk` | 14.026s | 5.303s | 2.64x |
| `generation.denoise_step` | 13.569s | 4.824s | 2.81x |
| `generation.denoise_forward` | 13.568s | 4.824s | 2.81x |
| `aten::linear` | 9.078s | 3.859s | 2.35x |
| `aten::addmm` | 7.235s | 1.832s | 3.95x |
| `aten::scaled_dot_product_attention` | 2.187s | 0.339s | 6.45x |

Kernel path 已按预期切换：

```text
fp32 attention:
aten::_scaled_dot_product_efficient_attention
fmha_cutlassF_f32_aligned_64x64_rf_sm80

bf16 attention:
aten::_scaled_dot_product_flash_attention
pytorch_flash::flash_fwd_kernel<..., bfloat16_t, ...>

fp32 GEMM:
cutlass_80_simt_sgemm_*

bf16 GEMM:
cutlass_80_tensorop_bf16_s16816gemm_*
```

Nsight Systems bf16 report：

```text
outputs/diffusion_util_low_precision_nsys_bf16/sd3_5_ocr_rollout_bf16.nsys-rep
outputs/diffusion_util_low_precision_nsys_bf16/sd3_5_ocr_rollout_bf16.sqlite
```

这个 nsys run 是 120s SIGINT 采样，覆盖 rollout + replay/backward；bf16 在同样 wall time
内推进了更多训练工作，所以不要拿 total kernel seconds 和 fp32 做端到端 speedup。它用于判断 kernel mix：

| class | fp32 nsys share | bf16 nsys share | interpretation |
| --- | ---: | ---: | --- |
| fp32 SIMT SGEMM | 52.8% | ~0.0% | 已基本消失 |
| bf16 tensorop GEMM | 0.0% | 42.8% | 主 GEMM path 已切换 |
| fp32 attention forward | 15.2% | 0.1% | 已基本消失 |
| flash attention forward | 0.0% | 9.0% | attention path 已切换 |
| elementwise/copy/cat | 18.0% | 42.3% | 低精度后成为更显眼的残余瓶颈 |

CUDA API 也显示 launch/sync 仍然很多：

```text
cudaLaunchKernel:      976,983 calls, 47.688s total CPU time
cuLaunchKernel:        237,317 calls, 10.702s total CPU time
cudaStreamSynchronize:  15,672 calls, 17.558s total CPU time
```

Nsight Compute bf16 roofline samples：

```text
outputs/diffusion_util_low_precision_ncu_bf16/sd3_5_ocr_bf16_top_gemm_roofline.ncu-rep
outputs/diffusion_util_low_precision_ncu_bf16/sd3_5_ocr_bf16_flash_roofline.ncu-rep
```

| kernel | duration | Compute(SM) | Memory | DRAM |
| --- | ---: | ---: | ---: | ---: |
| `tensorop_bf16_s16816gemm_relu_bf16_256x128...` | 0.522ms | 42.2% | 33.8% | 7.2% |
| `flash_fwd_kernel<..., bfloat16_t, ...>` | 0.675-1.020ms | 44.6% | 27.9% | 13.3% |

读法：低精度后单 kernel 的 Compute(SM) 百分比仍不接近 90%，但这是按更高 bf16/tensor
peak 计算的百分比；真正重要的是热路径 wall time 已经明显下降。bf16 top GEMM 比 fp32
representative large GEMM（0.920ms）更快，即使 Compute(SM)% 看起来更低。

新的性能主线不是重写 GEMM/attention，而是低精度后暴露出来的
`elementwise/copy/cat`、launch/sync 和 shape/compile 问题。

决策：这组数据**不支持现在重写 native diffusion transformer / custom Triton backend**。
它支持先做更低风险、直接作用在同一热路径上的工作：

```text
1. 先落地真正的 rollout-only transformer compile 边界（不要复用 `model.torch_compile`）
2. fp32 → bf16 已落地（rollout==replay 同 dtype，parity gate 解除，见 U1）；下一步是 FA2 + compile
3. larger batch 或 multi-GPU 只解决总吞吐，不解决单 GPU kernel Compute(SM) 偏低
4. 上面都做完仍明显 under-utilized，再评估 native/Triton boundary
```

### D1：Rollout-only transformer compile（当前主线）

不要复用 `model.torch_compile.enable`。那个开关现在同时影响 rollout 和 trainer replay，
而 trainer replay 会因为 zero-advantage filtering 出现动态 microbatch。rollout denoise shape 相对固定，
应该有独立策略：

当前代码状态（2026-06-07）：`RuntimeBuildSpec.torch_compile` 只读 `model.torch_compile`，
`build_sd3_5_runtime_bundle()` 和 `build_sd3_5_replay_runtime_bundle()` 都会消费它。因此现在不能用
`model.torch_compile.enable=true` 做 rollout-only 实验；那会把 replay compile / recompile / warmup
混进结果。正确落点是：diffusion rollout family 在 registry 里声明 compile metadata，
Ray generation launch payload 按这个 metadata 把 `rollout.denoise_compile` 覆盖到 rollout worker 的
`model_config["torch_compile"]`，driver replay bundle 继续保持 `model.torch_compile=false`。

```yaml
rollout:
  denoise_compile:
    enable: true
    target: transformer
    mode: reduce-overhead
```

第一版只 compile transformer，不 compile whole denoise loop。原因是 scheduler/SDE/logprob/buffer writes
有 Python state mutation，whole-loop compile 风险高；transformer forward 才是重块。

落点：

```text
compile target: rollout runtime 的 SD3 transformer module
do not compile: trainer replay path
do not compile: scheduler/SDE/logprob/buffer write loop
```

成功标准：

```text
warmup 后 `generation.denoise_forward` CUDA time 下降 >15%
没有持续 recompile
torch profiler trace 仍能看到 generation.denoise_forward / scheduler_step / trajectory_buffer_write
reward/output 分布无明显异常
trainer replay 仍然可以保持 uncompiled 或使用自己的 compile 策略
```

2026-06-07 已落地：

```text
configs/base/rollout/diffusion.yaml
  rollout.denoise_compile: {enable: false, target: transformer, mode: default}

vrl/generation/ray/launcher.py
  Generic Ray worker launch payload applies entry.compile metadata and overrides
  only launch_contract.model_build.model_config["torch_compile"].

vrl/rollouts/families/registry.py
  Diffusion entries declare RolloutCompileMetadata(
      config_path="rollout.denoise_compile",
      target="transformer",
      model_config_key="torch_compile",
  ).

vrl/models/utils.py
  load_weights_into unwraps torch.compile OptimizedModule via _orig_mod before
  strict trainable-key matching, so rollout LoRA weight sync keeps using the
  uncompiled trainer's transformer.* keys.
```

适用范围：

```text
This is now a shared diffusion rollout strategy, not an SD3.5-only switch.

Covered online diffusion families:
  sd3_5
  wan_2_1 / wan_2_1_i2v
  cosmos-predict2
  cosmos-predict2.5
  cosmos-predict2-anima

Not covered:
  offline diffusion DPO: no Ray rollout worker
  AR families: no denoise transformer target; they need a separate profiled
  prefill/decode compile strategy, not the diffusion denoise_compile switch
```

同口径 torch profiler（bf16, `skip_first=1`, one rollout chunk, 10 denoise steps）：

```text
no compile:
  outputs/diffusion_util_no_compile_torch_profile_bf16_skip1/

rollout-only compile, mode=reduce-overhead:
  outputs/diffusion_util_rollout_compile_torch_profile_bf16/
```

| metric | no compile | rollout-only compile | read |
| --- | ---: | ---: | --- |
| `engine.forward_chunk` CUDA total | 5.158s | 2.363s | 2.18x faster |
| `generation.denoise_step` CUDA total | 4.797s | 2.001s | 2.40x faster |
| `aten::copy_` CUDA total / calls | 842.1ms / 10602 | 26.1ms / 606 | copy pressure mostly removed |
| `aten::to` CUDA total / calls | 637.2ms / 14068 | 0.72ms / 772 | dtype/device copy path mostly removed |
| `aten::_to_copy` CUDA total / calls | 637.2ms / 9632 | 0.72ms / 386 | same |
| `aten::cat` CUDA total / calls | 126.7ms / 801 | not top-level visible | fused/hidden by compiled graph |
| `cudaLaunchKernel` calls | 34905 | 3669 | ~90% fewer |
| `cuLaunchKernel` calls | 9746 | 966 | ~90% fewer |

读法：compile 解决的是 launch / elementwise / copy / cat overhead，不是把 GEMM 本身
推到 90% Compute(SM)。这正好验证了之前的判断：先做 PyTorch path 内的 compile，比直接
重写 native/Triton 更有证据。

记忆点：

```text
torch.compile 的收益来源不是让 CUTLASS GEMM 更接近峰值，而是把 eager PyTorch 的
大量 Python/ATen 边界、逐 op kernel launch、dtype/device/layout copy、cat/pointwise
临时张量合并进 compiled graph。reduce-overhead 模式还会用 CUDA Graph replay 降低
CPU launch path 成本。

所以这次看到的是:
  launch/copy/elementwise overhead 明显下降
  denoise chunk wall time 明显下降
  GEMM Compute(SM) 没有变成 80-90%

这说明当前瓶颈不是“5090 没开满所以换硬件”，而是 diffusers/PyTorch 的 shape 和执行路径。
```

Batch sweep NCU（bf16 no compile, top GEMM target）：

```text
outputs/diffusion_util_batch_sweep_ncu_bf16/
outputs/diffusion_util_low_precision_ncu_bf16/
```

| chunk batch | representative longest top GEMM | Compute(SM) | Memory | DRAM |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 522.9us | 42.2% | 33.8% | 7.2% |
| 16 | 539.5us | 44.3% | 34.3% | 7.1% |
| 24 | 806.3us | 44.0% | 34.1% | 6.9% |
| 32 | 420.0us | 42.3% | 33.8% | 6.4% |

读法：把 `rollout.n/sample_batch_size` 扫到 16/24/32，没有把主力 GEMM 推到高 SM 占用；
大 kernel 仍在 42-44% Compute(SM) 区间。batch 增大能改善 per-sample overhead，但不是
“把 5090 算力单元吃满”的解法。

Compile + batch NCU 结果：

```text
outputs/diffusion_util_batch_compile_ncu_bf16/
```

| run | result |
| --- | --- |
| b24 + `reduce-overhead` | OOM in VAE decode; CUDA Graph private pools ~3.44GiB, then decode needed another 3GiB |
| b16 + `reduce-overhead` | OOM in VAE decode; CUDA Graph private pools ~2.71GiB, then decode needed another 4GiB |
| b24 + `default` | OOM in VAE decode; no graph pool, but NCU/compile run left only ~4.69GiB free before a 6GiB allocation |
| b8 + `reduce-overhead`, `Kernel2` first samples | valid report, but captures tiny WMMA kernels; weighted Compute(SM) 15.0% |
| b8 + `reduce-overhead`, `Kernel2 --launch-skip 100` | valid report; captured mid-size WMMA kernels, weighted Compute(SM) 31.2%, max sample 48.1% |

读法：compile 和 larger batch 不能直接叠加到 b16/b24/b32；32GB 卡上容量边界先出现。
可测到的 compiled `Kernel2` samples 也没有接近 90% Compute(SM)。所以当前结论不是
“硬件坏了/5090 不行”，而是 PyTorch/diffusers shape + launch path 的实际限制：compile 能显著
减少 overhead，但 GEMM roofline 仍没有变成满 SM。

#### Runtime block smoke：b16 denoise + VAE decode micro-batch（2026-06-08）

这次 smoke 不是新的同口径 profiler baseline，而是验证 `rollout.blocks` 能否把
denoise batch 和 VAE decode micro-batch 分开。它直接覆盖上面 `compile + larger batch`
实验暴露的问题：larger denoise batch 先在 VAE decode 附近撞到容量边界。

命令：

```bash
timeout 900 python -u -m vrl.scripts.train \
  --config experiment/diffusion/sd3_5/online_grpo_ocr \
  trainer.total_epochs=1 \
  trainer.output_dir=outputs/sd3_5_ocr_block_policy_b16_decode1_20260608_222142 \
  trainer.save_freq=999999 \
  eval.enable=false \
  trainer.precision_drift_guard.mode="'off'" \
  trainer.debug.first_step=true \
  actor.gradient_accumulation_steps=1 \
  rollout.n=16 \
  rollout.rollout_batch_size=1 \
  rollout.blocks.denoise.batch_size=16 \
  rollout.blocks.decode_latents.batch_size=1
```

Resolved config / request 链路：

```text
request sample_batch_size: 16
executor sample_batch_size: 16
runtime_blocks.decode_latents.batch_size: 1
legacy rollout.sample_batch_size: 8
```

同配置 b8 control：

```bash
timeout 900 python -u -m vrl.scripts.train \
  --config experiment/diffusion/sd3_5/online_grpo_ocr \
  trainer.total_epochs=1 \
  trainer.output_dir=outputs/sd3_5_ocr_block_policy_b8_decode1_control_20260608_222142 \
  trainer.save_freq=999999 \
  eval.enable=false \
  trainer.precision_drift_guard.mode="'off'" \
  trainer.debug.first_step=true \
  actor.gradient_accumulation_steps=1 \
  rollout.n=16 \
  rollout.rollout_batch_size=1 \
  rollout.blocks.denoise.batch_size=8 \
  rollout.blocks.decode_latents.batch_size=1
```

真实结果（both bf16, no compile, one prompt group, `rollout.n=16`, `decode_latents.batch_size=1`）：

| metric | b8 control | b16 denoise | read |
| --- | ---: | ---: | --- |
| chunks | 2 × 8 | 1 × 16 | b16 reduces chunk count |
| chunk keys | `0:8`, `8:16` | `0:16` | request chunking followed block policy |
| `diffusion_decode_batch_size` | 1 | 1 | VAE decode micro-batch applied in both |
| `diffusion_rollout_transformer_dtype` | `bfloat16` | `bfloat16` | current 16-bit default path |
| encode total | 0.177s | 0.145s | b16 saves one prompt-encode boundary |
| prepare total | 0.010s | 0.005s | small |
| denoise total | 7.109s | 7.272s | b16 denoise is slightly slower here |
| decode total | 0.464s | 0.642s | b16 decode wall is higher despite same decode micro-batch |
| recorded stages total | 7.760s | 8.064s | b16 is 3.9% slower in this smoke |
| recorded stages / sample | 0.485s | 0.504s | no standalone throughput win |
| denoise / sample | 0.444s | 0.455s | b16 is 2.3% slower in denoise/sample |
| peak memory | 16,875 MiB | 17,690 MiB | b16 costs about +815 MiB |
| first-step logprob diff max | 1.68e-05 | 0.0 | parity held for this smoke |

Performance 读法：

```text
What this proves:
  runtime block policy is wired through config -> request sampling -> Ray executor -> diffusion runtime
  denoise batch can be raised to 16 while VAE decode is capped at batch 1
  VAE decode micro-batch avoids the old "larger batch dies at decode allocation" failure mode
  decode is still not the main wall-clock stage in this successful run

What this does not prove yet:
  standalone throughput speedup from b8 -> b16
  compiled b16 steady-state performance
  that b24/b32 also fit once decode is micro-batched
```

结论：**b16 + decode1 是容量可行性改进，不是当前 eager/no-compile 路径的性能改进。**
在这次同配置 control 下，b16 虽然把 16 samples 合成一个 chunk，但 per-sample wall
反而略慢。说明当前 b8 的 chunk boundary overhead 已经不大，larger denoise shape 带来的
kernel/activation/decode 代价抵消了少跑一个 chunk 的收益。

对策略的影响：

```text
1. 不要把 denoise batch 默认从 8 提到 16；当前 smoke 没有支持这个默认值变更。
2. 保留 decode_latents.batch_size 作为容量旋钮，因为它解开了 denoise batch 和 VAE decode allocation。
3. 它不能替代 rollout-only compile；compiled b8 的 profiler 数字仍明显强于 no-compile b16。
4. 下一步只测能改变结论的组合：
   b16 decode1 reduce-overhead compile
   b24 decode1 no-compile
   b24 decode1 reduce-overhead compile
5. 对每组只比较这些派生指标：
   denoise_s / diffusion_sample_batch_size
   (encode_s + prepare_s + denoise_s + decode_s) / diffusion_sample_batch_size
   peak_memory_mb
   OOM / recompile status
```

#### Batching follow-up：b4/b8/b16 steady-state（2026-06-08）

目的：把 "bigger batch 是否更快" 和 "compile warmup 是否污染计时" 分开。当前 main
没有 `rollout.blocks.decode_latents.batch_size` 入口，本轮只测 legacy
`rollout.sample_batch_size`，不混入 VAE decode micro-batch。

先修了一个 instrumentation gap：Ray worker 的 `runtime_debug.ray_chunks[*]` 之前只返回
chunk metadata，没有把 `DiffusionChunkResult.stage_durations / engine_counters /
peak_memory_mb` 带回 trainer debug record。现在 debug chunk metrics 会包含：

```text
stage_durations_s
engine_counters
peak_memory_mb
```

测试：

```bash
pytest -q tests/generation/execution/test_worker_debug_metrics.py \
          tests/rollouts/runtime/test_runtime_inputs.py
```

普通 run（`rollout.n=32`，bf16，single-GPU colocated，`eval.enable=false`）：

```text
outputs/sd35_batching_metrics_off_b4_n32_20260608_231408
outputs/sd35_batching_metrics_off_b8_n32_20260608_231408
outputs/sd35_batching_metrics_on_b4_n32_20260608_231408
outputs/sd35_batching_metrics_on_b8_n32_20260608_231408
```

注意：`stage_durations_s` 是 executor 里的 CPU wall ranges，没有 CUDA synchronize。
它适合看 chunk 形状、首 chunk warmup 和容量趋势；不要把它当精确 GPU kernel time。

CPU wall range summary（drop first = 去掉首 chunk warmup / first decode allocation）：

| run | chunks | all total/sample | drop-first total/sample | drop-first denoise/sample | max peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| b4 no-compile | 8 x 4 | 0.465s | 0.453s | 0.443s | 19,702 MiB |
| b8 no-compile | 4 x 8 | 0.464s | 0.447s | 0.442s | 23,577 MiB |
| b4 compile | 8 x 4 | 0.572s | 0.332s | 0.322s | 19,702 MiB |
| b8 compile | 4 x 8 | 0.547s | 0.292s | 0.287s | 23,577 MiB |

CUDA profiler run（`rollout.n=16`，`rollout.torch_profiler.skip_first=1`，
只 profile 第二个 steady-state chunk，避免首 chunk compile warmup）：

```text
outputs/sd35_batching_cuda_off_b4_n16_20260608_232130
outputs/sd35_batching_cuda_off_b8_n16_20260608_232130
outputs/sd35_batching_cuda_on_b4_n16_20260608_232130
outputs/sd35_batching_cuda_on_b8_n16_20260608_232130
```

CUDA total summary：

| run | profiled chunk | engine CUDA / sample | denoise_step CUDA / sample | cudaLaunchKernel calls | compile graphs |
| --- | ---: | ---: | ---: | ---: | --- |
| b4 no-compile | 4 | 0.598s | 0.548s | 34,874 | n/a |
| b8 no-compile | 8 | 0.642s | 0.597s | 34,875 | n/a |
| b4 compile | 4 | 0.433s | 0.379s | 3,638 | unique=1, breaks=0 |
| b8 compile | 8 | 0.376s | 0.331s | 3,639 | unique=1, breaks=0 |

读法：

```text
1. compile 仍然是有效主线：launch 数从约 34.9k 降到约 3.6k。
2. compiled b8 是当前单卡 steady-state 最好点；b4 更省显存但更慢。
3. no-compile 下 b8 没有稳定胜过 b4，说明 "bigger batch" 本身不是可靠收益来源。
4. compiled b8 的 dynamo unique_graphs=1、graph_breaks=0，第二个 chunk 没有 recompile。
```

b16 容量结果：

```text
outputs/sd35_batching_off_b16_n32_20260608_230246
  no-compile b16: first 16-sample chunk passed, second chunk OOM in VAE decode.
  allocation: tried 4.00 GiB, only 1.89 GiB free.

outputs/sd35_batching_on_b16_n32_20260608_230246
  compile b16: first 16-sample chunk OOM in VAE decode.
  allocation: tried 4.00 GiB, only 3.20 GiB free.
  CUDA Graph private pools: 2.32 GiB.
```

Batching decision：

```text
Keep sample_batch_size=8 as the SD3.5 single-GPU default.
Do not promote b16: it is not faster in earlier smoke and fails under repeated chunks / compile.
Do not treat VAE micro-batch as a speed feature; it is only a capacity isolation knob.
For performance comparisons, use compiled b8 steady-state as the baseline.
```

2026-06-10 update：该 default decision 被 GPM scale bump 取代。旧 b16 failure 的
capacity boundary 是 VAE decode；当前 `configs/model/diffusion/sd3_5/medium.yaml`
已启用 `model.memory.vae_decode.tiling=true`，且 chunk 执行器仍保留 CUDA-OOM split
回退。因此默认实验入口已升到 `rollout.n=16` / `rollout.sample_batch_size=16`，live
gate 继续看是否真的带来端到端吞吐收益；若 tiling 后仍 OOM 或变慢，再回退到本节的
compiled b8 baseline。

#### Compile-mode A/B：default vs reduce-overhead（2026-06-09）

目的：确认 compile mode 是不是性能杠杆。同一 session、同一卡（RTX 5090 32 GiB）背靠背重跑
两种 mode，b8 / `rollout.n=16` / profiler 只抓第 2 个 steady-state chunk（`skip_first=1`），
完全镜像 6-08 的 CUDA baseline，消除隔天方差。

```text
outputs/sd35_compilemode_default_b8_n16_20260609_160402
outputs/sd35_compilemode_reduce-overhead_b8_n16_20260609_160402
```

| run | denoise_step CUDA total | /sample | cudaLaunchKernel | **Self CUDA total** | CUDA graphs | peak alloc (b8) |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| default（今天） | 2.654s | 0.332s | 4029 | **2.591s** | no | 23,576 MiB |
| reduce-overhead（今天） | 2.017s | 0.252s | 3639 | **2.573s** | yes (10) | 23,576 MiB |
| reduce-overhead（6-08 baseline） | 2.645s | 0.331s | 3639 | **2.574s** | yes | 23,577 MiB |

读法（重要）：

```text
1. 真实 GPU 工作量持平：Self CUDA total = default 2.591s vs RO 2.573s，差 0.7%，在噪声内。
   compile mode 不改变 kernel 工作量。
2. headline 指标 denoise_step CUDA/sample 不可靠：同一个 RO 配置 6-08=0.331、6-09=0.252，
   隔天摆 24%。==> 这个指标 <20% 的差都别信；只有 compile-vs-no-compile（launch 10x、
   CUDA total ~2x）那种大差是真的。先前 "b8 比 b4 快 13%" 落在这条噪声带里，定性排序
   (compile >> no-compile) 成立，但 per-config 细排序不成立。
3. RO 唯一的机械优势是 launch 更少（3639 vs 4029），但 denoise 是 GPU-bound（~2.58s kernel
   时间），多出的 ~390 次 launch（每次 2-6us、可重叠）换不成 wall-clock，所以不提速。
4. 内存：b8 allocated peak 两者相同（23,576 MiB），但 RO 额外 reserve 一块 CUDA Graph
   private pool（b16 时实测 2.32 GiB），torch max_memory_allocated 不计它、却吃掉 free 显存。
   这正是 6-08 compile-b16 在第 1 个 chunk VAE decode OOM、而 no-compile-b16 能过第 1 个
   chunk 的根因。
```

Compile-mode decision：

```text
Keep denoise_compile.mode=default (it is already the config default in
configs/base/rollout/diffusion.yaml). Do NOT switch the default to reduce-overhead.
- Throughput: default == reduce-overhead on real GPU work (Self CUDA within 0.7%).
- Memory: default has NO CUDA-graph private pool, so it keeps ~2.3 GiB more free
  headroom — strictly better for capacity (e.g. unblocking larger denoise/VAE batches).
reduce-overhead only pays off when a workload is CPU launch-bound; SD3.5 denoise is
GPU-bound, so it is the wrong mode here.
Methodology fix: report Self CUDA total + cudaLaunchKernel count, not denoise_step
CUDA/sample (that inclusive-region metric is run-to-run noisy under CUDA graphs).
```

Follow-up worth one run（未做）：default mode 没有 graph private pool，compile-b16 可能在
default 下 fit（6-08 的 b16 compile OOM 用的是 reduce-overhead）。但 no-compile b16 已证明
per-sample 不更快，所以这只是容量问题、非性能问题，优先级低。

### D2：Recompile observability（用 torch 内置，零 repo 代码）

想知道 compile 在 steady-state 是否还在帮忙、有没有静默 recompile 吃掉收益——但**不为这个
往代码里加任何观测脚本/模块**。torch 自己就提供 recompile 可观测性，零 repo 代码：

```bash
# 每次 recompile 打印一行，含触发它的 guard（哪个 shape/值变了）
TORCH_LOGS=recompiles python -m vrl.scripts.train \
  --config experiment/diffusion/sd3_5/online_grpo_ocr \
  rollout.denoise_compile.enable=true rollout.denoise_compile.mode=reduce-overhead

# 也可加 graph_breaks 看图是否被切碎
TORCH_LOGS=recompiles,graph_breaks ...
```

读法：

```text
warmup 后日志里不再出现 "Recompiling function ..." -> 编译稳定，D1 的收益在 steady-state 成立
持续出现 recompiles + guard 原因 -> 有 shape/guard 漂移，那条 rollout 没拿到稳定收益
```

可选的 CI / fail-fast 守门（也是 torch 内置，不是自写脚本）：

```python
import torch._dynamo as dynamo
dynamo.config.error_on_recompile = True   # steady-state 再 recompile 直接报错
```

**不做的**（曾经做过一版、已撤回）：
- 不写自定义 `compile_metrics.py` / 不往 `engine_counters` 里塞 `diffusion_compile_*` 计数。
  那是「为一个 config 往代码里加观测脚本」，正是不想要的。torch 的 `TORCH_LOGS` 已经够。
- 不做 warmup move（worker 跑合成 forward 强制编译）——它会**主动多触发一次编译**，是额外
  compiler 动作，不加。第一个 chunk 天然含 compile cost，跑 profile 时丢掉首 chunk 即可。

#### 验证结果（2026-06-09，单卡 5090，✅ 通过）

```bash
TORCH_LOGS=recompiles ... rollout.n=32 rollout.sample_batch_size=8 \
  rollout.denoise_compile.enable=true rollout.denoise_compile.mode=reduce-overhead
# outputs/sd35_compile_recompile_check_20260609
```

| chunk | denoise CPU wall |
| ---: | ---: |
| 0 | 33.36s（编译 warmup）|
| 1 | 2.29s |
| 2 | 2.29s |
| 3 | 2.29s |

判定：
- `TORCH_LOGS=recompiles` 全程零输出（注意：该 log 只在 **re**compile 时出声，初次编译静默——
  所以单看零输出不够，须结合首 chunk 33s 证明编译确实发生）。
- chunk 1-3 耗时逐毫秒一致 = CUDA graph 稳定 replay，无 shape/guard 漂移。
- steady-state 2.29s/chunk vs 此前 no-compile b8 ≈3.5s/chunk → ~1.5x（CPU wall 口径；
  CUDA-time 口径见 D1 的 2.4x）。
- 训练数值正常（reward_mean 0.376，metrics.csv 完整）。

**结论：D1 的 rollout-only compile 在 steady-state 成立，无 recompile。D2 关闭。**

### D3：Latent clone cleanup（hygiene，不是性能优先级）

当前每步先 clone：

```python
latents_ori = state.latents.clone()
...
buffers.observations[:, step_idx].copy_(latents_ori.detach())
```

可以改成先写 observation buffer，再 forward/scheduler：

```python
buffers.observations[:, step_idx].copy_(state.latents.detach())
```

但 D0 显示 `denoise.latent_snapshot_s` 只有 0.002% of denoise。这个改动可以减少一次
per-step full-latent clone allocation，属于代码/内存 hygiene，不应当作为性能 sprint 的主路径。

成功标准：

```text
trajectory observations 与旧逻辑数值一致
denoise.latent_snapshot_s / memory allocation 有下降
peak memory 不上升
不把它包装成 rollout wall-clock improvement
```

### D4：Only then consider batch-level staged rollout

如果 D1/D2 后 `generation.denoise_forward` 不再接近全局主导，才考虑 staged rollout。当前 D0 里：

```text
encode 0.499s total
decode 0.571s total
denoise 78.664s total
```

encode/decode 加起来也不到 denoise 的 1.5%，所以现在做 batch-level staged rollout 没有足够收益。
只有当 `decode_latents`、`reward_score` 或 text encode 占比明显，再考虑：

```text
denoise worker -> decode/reward worker
```

不要先上完整 SGLang-Omni runtime。SGLang-Omni 的价值是多阶段异构 serving：
每个 stage 独立 scheduler、inbox/outbox、shared-memory relay。当前 RL diffusion rollout 更像
“一个重 denoise stage + reward”，先做 denoise block optimization 更直接。

### D5：Remove per-step host syncs in `sde_step_with_logprob`（已落地，2026-06-07）

D1 的 transformer compile 只覆盖 transformer forward；`sde_step_with_logprob`
（`vrl/math/diffusion/flow_matching.py`）仍在 compile 边界外，每个 denoise step 触发
host-device 同步，是 bf16 nsys 里 `cudaStreamSynchronize: 15,672 calls / 17.558s` 的主
要残余来源，也是把整个 denoise step 塞进单张 CUDA Graph（D1 的下一步）的硬阻塞——graph
capture 无法捕获把 GPU 值读回 host 的操作。

两处 per-step 同步已消除（无签名破坏、对其它 caller 向后兼容）：

```text
1. sigma_max/sigma_min: scheduler.sigmas[1].item() / [-1].item() 每步两次 .item()。
   这两个是整条 schedule 的 loop-invariant 端点，改为保留 0-d device tensor，算术里
   直接用，去掉 .item() host 读回。

2. index_for_timestep: step_index = [scheduler.index_for_timestep(t) for t in timestep]
   内部做 (timesteps == t).nonzero().item() 搜索 → 每步同步。新增可选 step_index 参数，
   rollout loop（executor.py 两个 denoise_mode 分支）把已知的 step_idx 传进来，绕过搜索。
   默认 step_index=None 保留旧搜索行为，replay/evaluator 路径（sde_logprob.py）不受影响。
```

安全性：`state.timesteps` 直接来自 `scheduler.timesteps`（`sd3_5/model.py:266-267`），flow-match
timesteps 严格单调，故 rollout loop 的 `step_idx` 精确等于 `scheduler.index_for_timestep`。

验证（CPU，真实 `FlowMatchEulerDiscreteScheduler`，10 步）：

```text
step_idx == scheduler.index_for_timestep : max_err = 0      （安全性成立）
log_prob / prev_mean  None-path vs step_index : max_err = 0.0  （位级一致）
std_dev_t  tensor-sigma vs float-sigma        : max_err = 0.0  （sigma 改动位级一致）
pytest tests/generation tests/math/test_ar_flow_matching.py
       tests/algorithms/test_grpo.py tests/rollouts : 157 passed
```

未做（仍属同步源，但不在本次 surgical 范围）：native denoise_mode 里 `state.scheduler.step()`
（diffusers 自带方法，内部也 index_for_timestep）；以及 fp32 upcast 的 elementwise（属
fused-kernel 范畴，见上）。下一步应在 compiled trace 上重数 `cudaStreamSynchronize`，确认
SDE-loop 同步已消除、为 whole-step CUDA Graph 解锁。

### Non-goals

```text
不把 rollout bf16 当作这条 denoise-block strategy 的主解
不重写 native transformer executor
不把每个 denoise timestep 拆成独立 distributed stage
不把 clone/write cleanup 当成性能主线
不在 generation.denoise_forward 仍占绝对主导时做 text-encode / VAE / reward overlap
```

## 通用优化路径（不是 denoise-block 专项）

### P0：Rollout bf16 dtype（简单但伤精度，不当主杠杆）

bf16 是最省事的"提速"开关——一个 dtype 标志就拿 attention ≈2x、GEMM ≈1.5-2x、还解锁 FA2。
但它换来的是数值精度，所以排在 denoise-block strategy 之后，**不作为主杠杆**。

**关键风险：GRPO 的 ratio==1 不变式。** 采集时的 rollout logprob 必须和 replay 时的 logprob 一致
（saved_noise 重放，collection 点的 importance ratio = 1）。若 rollout 跑 bf16、replay 跑 fp32，
两边 logprob 数值不同 → 采集点 ratio 偏离 1 → 偏置/破坏 GRPO。所以 rollout bf16 **不是免费的**：

```text
要么 rollout 与 replay 用同一 dtype  → SD3.5 又回到训练侧 fp16-fragile 的数值问题
要么先实测并量化 ratio 漂移          → 确认在容差内再开
```

适用范围：base 默认已是 bf16；只有 SD3.5 实验族 override 成 fp32（ocr/geneval/pickscore），
所以本条只对这些 fp32 实验有意义。

前置：先做 D0（substage profiling）确认 transformer forward 真是占比大头，且通过上面的
ratio 一致性验证后，才考虑动 dtype。

改动（仅在 ratio 一致性验证通过后）：

```text
vrl/config/precision.py     — 增加 rollout_dtype 轴，独立于 trainer precision
vrl/scripts/common/online.py — rollout model 用 rollout_dtype 加载
configs/experiment 的 fp32 实验 — 显式声明 rollout_dtype（base 仍跟随 bf16）
```

潜在收益（若 ratio 一致性成立）：

```text
Attention: fmha_cutlassF_f32 → fmha_cutlassF_f16  ≈ 2x
GEMM:      cutlass fp32 → cutlass fp16/bf16         ≈ 1.5-2x
解锁 FlashAttention-2（只支持 fp16/bf16）
```

注意：不再声称"不影响 replay 精度"——rollout bf16 会改变 replay 必须复现的 logprob（见上 ratio 风险）。

### P1：torch.compile for rollout（general version）

当前 compile 被整体关掉（`torch_compile.enable: false`），理由是 training 的 zero-advantage
filtering 导致动态 batch size。但 rollout batch size 固定（`sample_batch_size: 8`），shape 是静态的。
具体实现应按 D1 的 `rollout.denoise_compile` 做 rollout-only 策略，不复用 trainer compile 开关。

改动：

```text
分离 rollout compile flag 和 trainer compile flag
rollout model 只 compile transformer forward
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

text-encode 可以和前一个 batch 的 reward scoring 重叠。但 D0 已经给出内部阶段绝对时间：
encode+decode 约 1.07s，denoise 约 78.66s。即使完全隐藏 encode/decode，也只是在当前
rollout wall clock 上拿很小的收益。除非后续 D1/D2 把 generation.denoise_forward 大幅压低，否则 P4 暂不做。

### 暂不做

```text
Native transformer executor    — attention 占 34% 但可以通过 bf16+FA2 优化，不需要重写 transformer
Fused QKV projections          — 节省一次 kernel launch，不是主要开销
Custom Triton kernels           — 标准 FA2 + compile 已经够
Context projection cache        — cross-attention KV projection 是 attention 的小部分，不值得复杂度
```

## 预期总收益

```text
D0 measured rollout denoise:       78.664s / 8 rollout batches
D0 measured denoise.forward_s:      78.573s / 8 rollout batches
D1 target (>15% forward reduction): <=66.786s forward over same shape
P2 target with 2 rollout GPUs:      wall clock close to 50% of 1-GPU rollout
```

不要再用粗略的 "~180s epoch rollout" 当验收数字；它混了旧 profiler 的 whole-epoch GPU time。
本 sprint 的验收数字以 torch profiler trace / key_averages 里的
`generation.denoise_forward` before/after 为准。

## Completion criteria

```text
D0: rollout torch profiler trace 有 generation.* denoise ranges，已能解释 denoise 内部占比
D1: rollout transformer compile 启用，无持续 recompile，generation.denoise_forward CUDA time 下降 >15%
D2: recompile observability 用 torch 内置 `TORCH_LOGS=recompiles`（零 repo 代码），不写自定义观测脚本、不加 warmup 强制编译。steady-state 无 recompile 日志即 D1 收益成立。
D3: clone cleanup 若执行，只作为 memory/code hygiene，不作为性能验收主线
P0 bf16: 只有 ratio==1 parity 验证通过后才允许启用，不把 bf16 当默认完成项
每个性能 phase 都有 before/after profile 文件路径和同 shape 数字对比
```
