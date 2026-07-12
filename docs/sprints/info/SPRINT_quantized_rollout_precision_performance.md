# SPRINT: FP8 / NVFP4 rollout 精度与性能实测归档

状态：**info measurement archive（2026-07-11）**。本页归档 RTX 5090、当时代码快照与
SD3.5 Medium checkpoint 的可复现实测；这些数字不是随后 rebase 后 HEAD 的重新测量，也
不是待办清单。实现状态与后续 gate 仍分别由
`planned/SPRINT_nvfp4_rollout.md` 和 `done/SPRINT_fp8_rollout_gemm_kernel.md` 维护。

## 0. 核心结论

1. **linear premise 成立**：默认 FP8 rowwise 在 24 个 DiT linear shape 上几何平均
   `1.34x`；当前 FP4 只量 production 会命中的 12 个 MLP shape，几何平均 `2.33x`。
2. **不能把 linear 数字外推整模**：真实 SD3.5 Medium checkpoint、512²、CFG 后
   effective batch `32`、`torch.compile(mode="default")` 下：

   | 路径 | BF16 | quantized | transformer speedup | steady peak allocated |
   |---|---:|---:|---:|---:|
   | FP8 rowwise，attention+MLP | 378.248 ms | 245.007 ms | **1.544x** | 5.588 → **4.147 GiB** |
   | NVFP4，MLP-only | 379.012 ms | 273.413 ms | **1.386x** | 5.590 → **4.580 GiB** |

   这是一个 transformer forward，不含 prompt encode、scheduler/SDE、VAE、reward、Ray 与
   replay。按 10 个相同 shape 的 denoise forward 线性相乘只是约 `3.78→2.45s` /
   `3.79→2.73s` 的 forward 投影，不是端到端 rollout 实测。
3. **compile 是决定性条件**：eager 的 clean pairwise 里 FP8 为 `0.92x/0.91x`
   （B=2/B=32，变慢），FP4 只有 `1.062x/1.096x`；compile 后才分别达到
   `1.544x/1.386x`。因此“量化已接”不等于“production 已快”，swap 必须发生在 compile 前。
4. **精度有损但有限**：真实 checkpoint compiled B=32 noise prediction 对 BF16：
   FP8 normalized-L1=`1.295%`、NRMSE=`0.921%`；FP4 MLP-only normalized-L1=`3.640%`、
   NRMSE=`2.376%`。这不是 FID、reward 或 SDE-logprob parity。
5. **修正后的 synthetic correction-path gate 不截断**：production cap `2.0` 下，FP8 与
   FP4 的逐 timestep TIS clip / RS masked 都是 `0%`。旧 FP4 `15.2%@cap1.5` 是把
   35 步 ratio 相乘后的反事实压力值，**不符合 trainer 的逐 timestep loss 语义，已作废**。
6. **真正 P2 仍未完成**：还没有用 MLP-only FP4 真 rollout 生成 observations/actions/
   old-logprob，再让 BF16 replay 对同一 SDE trajectory 逐 timestep 打分。当前不能宣称 FP4
   production-ready，也不能从 noise-pred drift 推导 reward 不退化。

## 1. 环境与代码指纹

| 项 | 值 |
|---|---|
| 本地时间 | 2026-07-11，America/Los_Angeles |
| GPU | NVIDIA GeForce RTX 5090，SM 12.0，31.33 GiB |
| driver | 580.159.03 |
| PyTorch / CUDA runtime | 2.11.0+cu128 / 12.8 |
| Triton / cuDNN | 3.8.0 / 9.19.0 |
| Python | 3.12.2 |
| Git | measurement base `0d20d3970cec` + 当时未提交的 FP4/FP8 工作树 |
| 后续落盘 | production `a5da3d9e`；validation runners `f03ae7a1`（rebase 后未重跑本页全部 SD3 profile） |
| checkpoint | `stabilityai/stable-diffusion-3.5-medium`，local snapshot `b940f670f0eda2d07fbb75229e779da1ad11eb80` |
| linear seed / timing | seed 0；25 warmup + 100 CUDA-event iterations |
| model timing | Python GC disabled；eager 20 warmup + 30 samples；compiled 10 warmup + 20 samples |
| model shape | 512×512 latent；CLIP 77 + T5 128 = 205 text tokens；effective B=2/32 |
| ownership | 默认 drop quantized masters，匹配当前 SD3 LoRA rollout；另报 full-sync kept storage |
| compile environment | `TORCHINDUCTOR_COMPILE_THREADS=1`；本机 dev Triton 的并行 worker fork-unsafe |

compiled A/B 的 active 采样区间达到 97–100% SM，功耗最高 576–578W，核心温度最高
66–69°C；linear 与 model 数字均来自真实 CUDA kernel，不是 vendor headline。

## 2. Production-module linear benchmark

命令：

```bash
python -m vrl.scripts.perf.quantized_linear_benchmark
```

计时包含每次 forward 的动态激活量化、量化 GEMM、bias、reshape 与输出 dtype 处理；不包含
首次权重量化、weight-sync 后重量化、完整 transformer、compile 或通信。drift 公式是：

```text
mean(abs(quantized - bf16)) / mean(abs(bf16))
```

它是 normalized-L1，不是逐元素 mean relative error。

| projection | shape (M,K,N) | BF16 ms | FP8 ms / speedup / nL1 | FP4 ms / speedup / nL1 |
|---|---:|---:|---:|---:|
| qkv | 4096,1536,4608 | 0.3325 | 0.2332 / 1.43x / .0368 | — |
| attn_out | 4096,1536,1536 | 0.1354 | 0.1288 / 1.05x / .0368 | — |
| mlp_up | 4096,1536,6144 | 0.4130 | 0.3080 / 1.34x / .0368 | 0.2217 / 1.86x / .1397 |
| mlp_down | 4096,6144,1536 | 0.4847 | 0.4878 / 0.99x / .0368 | 0.2315 / 2.09x / .1397 |
| qkv | 8192,1536,4608 | 0.5758 | 0.4864 / 1.18x / .0368 | — |
| attn_out | 8192,1536,1536 | 0.2526 | 0.2193 / 1.15x / .0368 | — |
| mlp_up | 8192,1536,6144 | 0.8196 | 0.6311 / 1.30x / .0368 | 0.5686 / 1.44x / .1393 |
| mlp_down | 8192,6144,1536 | 0.9535 | 0.9398 / 1.01x / .0368 | 0.4657 / 2.05x / .1396 |
| qkv | 4096,2048,6144 | 0.5211 | 0.3719 / 1.40x / .0368 | — |
| attn_out | 4096,2048,2048 | 0.2091 | 0.1854 / 1.13x / .0368 | — |
| mlp_up | 4096,2048,8192 | 0.7643 | 0.4994 / 1.53x / .0368 | 0.3621 / 2.11x / .1397 |
| mlp_down | 4096,8192,2048 | 0.8415 | 0.7601 / 1.11x / .0368 | 0.3790 / 2.22x / .1394 |
| qkv | 8192,2048,6144 | 1.0841 | 0.7777 / 1.39x / .0368 | — |
| attn_out | 8192,2048,2048 | 0.4207 | 0.3225 / 1.30x / .0368 | — |
| mlp_up | 8192,2048,8192 | 1.3986 | 0.9662 / 1.45x / .0368 | 0.7025 / 1.99x / .1397 |
| mlp_down | 8192,8192,2048 | 1.7515 | 1.4533 / 1.21x / .0368 | 0.6837 / 2.56x / .1392 |
| qkv | 4096,4096,12288 | 2.1138 | 1.2570 / 1.68x / .0367 | — |
| attn_out | 4096,4096,4096 | 0.8142 | 0.5608 / 1.45x / .0367 | — |
| mlp_up | 4096,4096,16384 | 2.7379 | 1.5660 / 1.75x / .0367 | 0.8916 / 3.07x / .1392 |
| mlp_down | 4096,16384,4096 | 3.3423 | 2.0972 / 1.59x / .0367 | 0.9003 / 3.71x / .1393 |
| qkv | 8192,4096,12288 | 3.9448 | 2.3661 / 1.67x / .0367 | — |
| attn_out | 8192,4096,4096 | 1.4878 | 1.0859 / 1.37x / .0367 | — |
| mlp_up | 8192,4096,16384 | 5.0223 | 3.0396 / 1.65x / .0367 | 1.9011 / 2.64x / .1396 |
| mlp_down | 8192,16384,4096 | 5.4809 | 4.0978 / 1.34x / .0367 | 1.8093 / 3.03x / .1392 |

汇总：

| 路径 | rows | speedup geomean / range | nL1 mean / max | dense-equivalent TFLOP/s geomean |
|---|---:|---:|---:|---:|
| FP8 rowwise，attention+MLP | 24 | **1.34x** / 0.994–1.748x | .0368 / .0368 | 239.3 |
| NVFP4，MLP-only | 12 | **2.33x** / 1.441–3.712x | .1395 / .1397 | 420.0 |
| BF16 reference | 24 | 1.00x | — | 179.1 |

TFLOP/s 用 `2*M*K*N / wall` 计算，且 wall 已包含激活量化，所以只是 dense-equivalent
吞吐；不能与 vendor sparse AI TOPS 混称。

### 2.1 相同 MLP shape 的格式隔离

上表原始汇总的 target scope 不同。只取 FP8/FP4 都实际执行的同一组 12 个 MLP shape，
逐行用同一个 BF16 latency 做比值后再取几何平均：

| 相同 12 个 MLP shape | vs BF16 geomean / range | FP4 vs FP8 |
|---|---:|---:|
| FP8 rowwise | **1.335x** / 0.994–1.748x | reference |
| NVFP4 | **2.326x** / 1.441–3.712x | **1.743x** / 1.110–2.329x |

所以 kernel-level 的答案与旧整模表相反：在相同 MLP Linear 上，NVFP4 明显快于 FP8。
旧整模 FP4 `1.386x` 小于 FP8 `1.544x` 的主要变量是 target coverage，而不是 FP4 GEMM
失败。该表仍是 isolated production modules，不代替下节的真实 transformer forward。

## 3. 真实 checkpoint transformer profile

长期 runner：`vrl/scripts/perf/quantized_sd3_forward_profile.py`。它加载真实 transformer
权重、安装 `SD3JointAttentionProcessor`，使用当前 512² / max_sequence_length=128 shape。

### 3.1 Compiled production shape（主结论）

```bash
TORCHINDUCTOR_COMPILE_THREADS=1 \
python -m vrl.scripts.perf.quantized_sd3_forward_profile \
  --compile --schemes bf16 fp8 --batches 32 --warmup 10 --iters 20

TORCHINDUCTOR_COMPILE_THREADS=1 \
python -m vrl.scripts.perf.quantized_sd3_forward_profile \
  --compile --schemes bf16 fp4 --batches 32 --warmup 10 --iters 20
```

| scheme | median | p10–p90 | speedup | cold compile | noise-pred nL1 / NRMSE |
|---|---:|---:|---:|---:|---:|
| BF16（FP8 pair） | 378.248 ms | 377.708–378.610 | 1.000x | 25.13s | reference |
| FP8 rowwise | **245.007 ms** | 244.864–245.147 | **1.544x** | 40.71s | .01295 / .00921 |
| BF16（FP4 pair） | 379.012 ms | 378.655–379.355 | 1.000x | 5.44s（cache warm） | reference |
| NVFP4 MLP-only | **273.413 ms** | 273.206–273.734 | **1.386x** | 33.29s | .03640 / .02376 |

编译输出的精度不是逐位等于 eager（Inductor 会改变融合/归约），所以每个 compiled quantized
输出都与同一 compiled BF16 pair 比较。cold compile 不进入 steady rollout throughput。

### 3.2 Eager 对照与热状态敏感性

GC-disabled、cool-start pairwise：

| effective batch | BF16 | FP8 rowwise | speedup | FP4 MLP-only | speedup |
|---:|---:|---:|---:|---:|---:|
| 2 | 47.72 ms | 51.85 ms | **0.92x** | 44.73 ms | **1.062x** |
| 32 | 555.61 ms | 612.83 ms | **0.91x** | 503.57 ms | **1.096x** |

5090 是桌面显示卡，长串行 A/B 会把板卡从 42–52°C 推到 77–83°C / 550–578W；固定
`BF16→FP8→FP4` 顺序曾产生 `0.55x` 到 `1.87x` 的互相矛盾假数。根因有两个：

- Python GC 会暂停 eager forward 内的 host launch；CUDA event 仍会把 stream 空洞计入 wall。
- 后跑 scheme 处于不同功耗/温度状态。

runner 现在在 timing window 禁用 GC，并要求 pairwise 运行；文档主判据使用 p10–p90 极窄的
compiled 结果。热浸敏感性不会改变方向：FP8 eager 没有净收益，production compile 才有。

### 3.3 Matched-scope transformer runner

runner 现支持显式 `--target-profile mlp_only|attention_mlp`。同一 profile 会传给 FP8/FP4，
并在计时前逐项比较 `(module path, in_features, out_features)` manifest；count 相同但路径不同
也会 fail。直接 `FP4 vs FP8` speedup 只在 profile 与 manifest SHA256 都相同时输出。

本地 SD3.5 Medium meta 预检：

| target profile | linears | weights | FP4 alignment miss | manifest SHA256 |
|---|---:|---:|---:|---|
| `mlp_only` | 94 | 887,095,296 | 0 | `4268f541aabbcf4ac6232a204d817334bff63f7f8e8ac9fb231eec3fd3803197` |
| `attention_mlp` | 337 | 1,460,404,224 | 0 | `90e4abed332d83562c6d7fac837f4407a1ff0ab021981dd17452e8ae37b9ff31` |

真实 compiled B=32 matched run 尚在队列中：唯一 RTX 5090 正被独立的 300-epoch SANA
训练占用；并发运行会污染 CUDA-event latency 并有 OOM 风险，不能写成有效数字。GPU 空闲并
冷却后执行：

```bash
TORCHINDUCTOR_COMPILE_THREADS=1 \
python -m vrl.scripts.perf.quantized_sd3_forward_profile \
  --target-profile mlp_only --compile --schemes bf16 fp8 fp4 \
  --batches 32 --warmup 20 --iters 30

TORCHINDUCTOR_COMPILE_THREADS=1 \
python -m vrl.scripts.perf.quantized_sd3_forward_profile \
  --target-profile attention_mlp --compile --schemes bf16 fp8 fp4 \
  --batches 32 --warmup 20 --iters 30
```

## 4. 存储与 allocator 高水位

unique storage 按底层 storage 指针去重，包含参数与 buffer，不包含输入、activation、allocator
cache 或 compile workspace。真实 checkpoint 有 2,243,171,520 个参数。

| scheme | swapped linears / weights | 参数覆盖 | master kept | master dropped（当前 LoRA ownership） |
|---|---:|---:|---:|---:|
| BF16 | 0 / 0 | 0% | 4.600 GiB | 4.600 GiB |
| FP8 rowwise | 337 / 1,460,404,224 | 65.10% | 5.963 GiB（+29.6%） | **3.243 GiB（−29.5%）** |
| NVFP4 MLP-only | 94 / 887,095,296 | 39.55% | 5.065 GiB（+10.1%） | **3.412 GiB（−25.8%）** |

这解释了两个常见误读：

- full-parameter base-weight sync 必须保留 BF16 master，量化反而增加常驻 storage；只有 LoRA/
  frozen rollout 的 master-drop 才真正省模型显存。
- “FP4”不等于整个 transformer 缩成四分之一：当前只覆盖 MLP 的 39.55% 参数。

compiled B=32 allocator：

| scheme | steady allocated | peak allocated | incremental workspace | peak reduction vs BF16 |
|---|---:|---:|---:|---:|
| BF16 | 4.698 GiB | 5.588–5.590 GiB | 912–913 MiB | — |
| FP8 rowwise | 3.340 GiB | **4.147 GiB** | 827 MiB | **−25.8%** |
| NVFP4 MLP-only | 3.509 GiB | **4.580 GiB** | 1,096 MiB | **−18.1%** |

`memory_reserved()` 分别约 6.50–7.87 GiB，是 allocator cache，不等同于 live model footprint。

## 5. Precision correction-path probe

```bash
python -m vrl.scripts.perf.quantized_rollout_drift_probe --scheme fp8
python -m vrl.scripts.perf.quantized_rollout_drift_probe --scheme fp4
```

该 probe 用生产量化 module，但模型只是 synthetic categorical head。它回答“量化误差经过
stats/guard/TIS/RS 后如何表现”，不回答真实 diffusion SDE accuracy。

| metric | FP8 rowwise | FP4 full-head synthetic |
|---|---:|---:|
| 单批 abs logprob diff mean / max | .03105 / .11650 | .10378 / .39601 |
| 单批 ratio deviation mean / max | .03098 / .11494 | .10431 / .48589 |
| 35×256 独立 sample-step ratio dev mean / max | .02977 / .15843 | .10766 / .81094 |
| finite | true | true |
| production TIS cap | 2.0 | 2.0 |
| sample-step importance weight > cap | 0% | 0% |
| TIS clip / RS masked | 0% / 0% | 0% / 0% |
| production catastrophic guard | pass | pass |
| counterfactual 35-step ratio-product >2.0 | 0% | 8.6%（diagnostic only） |

真实 trainer 在 `OnlineTrainer` 中逐 timestep 调 evaluator/GRPO，然后用 `1/T` 归一累计梯度；
`TrajectorySignalBuilder` 也只切出当前 timestep 的 old-logprob。因此不能用
`exp(sum_t(replay_t-old_t))` 驱动 TIS gate。修正后的回归测试会把同一 timestep 复制两次，断言
GRPO gradient norm 不翻倍，防止该错误回来。

## 6. Gate 状态

| gate | FP8 | NVFP4 MLP-only |
|---|---|---|
| production-module linear net speed | PASS：1.34x | PASS：2.33x |
| real checkpoint compiled transformer B=32 | PASS：1.544x | PASS：1.386x |
| production compile executes | PASS | PASS |
| master-dropped peak memory | PASS：−25.8% | PASS：−18.1% |
| synthetic per-step correction path | PASS：0% TIS/RS | PASS：0% TIS/RS |
| real rollout→BF16 replay SDE-logprob | historical FP8 live run PASS | **NOT MEASURED for current FP4 profile** |
| reward / generated quality curve | historical FP8 live run GO | **NOT MEASURED** |

所以当前决策是：FP8 已有 production evidence；FP4 的 kernel、compiled forward 和 correction
mechanism 都成立，但真实 MLP-only SDE trajectory/reward gate 未签字，仍是显式实验路径。

## 7. 本轮修正的测量问题

1. linear benchmark 的 FP8 从历史硬编码 `tensorwise` 改为 loader 默认 `rowwise`。
2. FP4 linear verdict 只汇总 live targeting 会替换的 MLP rows，不再把 QKV/attention 算进去。
3. drift probe 从错误的 35-step ratio product 改为 trainer 的逐 timestep loss，并读取 production
   cap/guard typed source；ratio product 仅作明确标注的 counterfactual。
4. `build_precision_split_safety_configs()` 成为 runtime builder 与 probe 的单一 typed 默认来源，
   避免 cap/guard 再漂移。
5. model runner 禁用 timing-window Python GC、分开 cold compile、报告 p10/median/p90，并用
   pairwise A/B 避免固定顺序热污染。
6. 本机并行 Inductor worker 报 `Could not find an active GPU backend`；本地 PyTorch source
   明确建议失败时用 `TORCHINDUCTOR_COMPILE_THREADS=1`。最小 CUDA compile 与完整 SD3 均已验证通过。

## 8. Architecture hygiene 与非目标

应该改变且已改变：测量 recipe、target profile、TIS timestep 语义、共享安全配置来源与真实
checkpoint runner。

应保持不变：

- `formats.py` 的 ALL_CAPS 是 E2M1/E4M3/alignment/swizzle 的硬件格式协议常量，保留。
- `targeting.py::MLP_PATH_SEGMENTS` 是刻意隔离的模型路径 taxonomy，FP4/测试共同消费，保留。
- family base 的 `quantize_rollout_fp8/fp4` 薄方法提供 cross-family dispatch 一致性，保留。
- 旧 `fp8_*` perf 文件是公共 CLI 兼容 facade，保留；canonical runner 仍是 scheme-neutral。

非目标：不合并 FP8/FP4 kernel、不量化训练/replay、不把 full-head synthetic 结果冒充 MLP-only
真实 SDE、不放宽 TIS/RS 阈值来制造通过、不把 transformer speedup 包装成端到端 speedup。

## 9. 复现与 one-shot artifact 生命周期

决定性 stdout 在写入本页前做了 SHA256：

| output | SHA256 |
|---|---|
| linear canonical | `c04a78fce55d8e9d883fe83b5a102c7bcb3115703f3af7367ef09222b12894d0` |
| FP8 drift | `5ce00cac7140a37645a23b8a6f5afb3411130d8ef13106bac939a316da0f5123` |
| FP4 drift | `fcb10f54ff2a5b03ffa08efd7985d2c027fe4c9ec05f59d02d608c15bea65b17` |
| eager BF16/FP4 pair | `8f56fcc5e8c0878d07d6888b929cc03efeca8218aa0440eb414d6ec379af0749` |
| compiled BF16/FP8 pair | `63a5ec98fce073181f3b3e4c45edb75f61cbdcc26b1cddea44a8954826c160d6` |
| compiled BF16/FP4 pair | `3a9a592efcce5342ee3dc645e34009458fef0fd98357d43323ca0b374a716062` |

stdout/dmon 属一次性验证产物；本页已保存完整判决、关键原始值和复现命令，所以 `/tmp` 输出不进入
仓库。长期资产是三个 canonical runner 与其 tests。

## 10. 回归验证

测量工作树当时按相互不重叠的三组相关测试执行，合计 **435 passed**：

| 组 | 覆盖 | 结果 |
|---|---|---:|
| quantization / correction | FP4、FP8、profile runner、逐 timestep drift、online precision bridge、logprob mismatch、GRPO / FlowDPPO | 191 passed |
| config / model integration | precision/schema/experiments、AR rollout quantization、diffusion FP4 targeting、LoRA build | 193 passed |
| runtime guard | Ray runtime config、worker/debug/version slots、precision drift guard | 51 passed |

另有以下终检：

- 本轮涉及文件的 scoped Ruff check 与 Ruff format check：PASS。
- `python -m compileall -q vrl`：PASS。
- `git diff --check`：PASS。
- 15/15/1 条 warning 分别来自 requests、Torch JIT deprecation 与 Ray FutureWarning，没有新增
  failure。
- 全库 `ruff check vrl tests` 仍会被本轮范围外的既有工作树问题挡住：
  `vrl/generation/ray/runtime.py` 的 unused import / import order，以及
  `vrl/ray/actor_group.py`、`vrl/rollouts/orchestration/continuous/producer.py`、
  `vrl/rollouts/orchestration/strict_on_policy.py` 的 import order。本轮没有改写这些用户改动。

### Rebased publish verification（2026-07-12）

代码落盘到上述两个提交后，发布前重新执行：

- quantization core / GPU parity：`57 passed`；
- config + model + online integration：`198 passed`；
- scheme-neutral runner / compatibility tests：`15 passed`；
- complete config suite：`213 passed`，`python -m vrl.config.lint` PASS；
- canonical linear hardware gate：FP8 `1.34x`、FP4 MLP-only `2.37x`，均通过 `1.05x`；
- synthetic drift hardware gate：FP8 / FP4 均 exit `0`，production guard / TIS / RS 全通过；
- changed-file Ruff check / format check、`compileall`、`git diff --check` 全 PASS。

这次只重跑 linear 与 synthetic correction-path gate；没有重跑 SD3.5 real-checkpoint
transformer profile，因此 §0/§4 的 SD3 数字继续作为 2026-07-11 measurement archive，
不外推为 rebased HEAD 的新测量。

## References

- `vrl/scripts/perf/quantized_linear_benchmark.py`
- `vrl/scripts/perf/quantized_rollout_drift_probe.py`
- `vrl/scripts/perf/quantized_sd3_forward_profile.py`
- `vrl/scripts/perf/common/timing.py`
- `vrl/scripts/perf/common/fp8_math.py`
- `vrl/nn/quantization/base.py`
- `vrl/nn/quantization/fp8.py`
- `vrl/nn/quantization/fp4.py`
- `vrl/nn/quantization/fp4_kernels.py`
- `vrl/nn/quantization/formats.py`
- `vrl/nn/quantization/targeting.py`
- `vrl/config/builders.py`
- `vrl/models/loader.py`
- `vrl/trainers/online/trainer.py`
- `vrl/rollouts/evaluators/trajectory.py`
- `vrl/rollouts/evaluators/diffusion/sde_logprob.py`
- `tests/scripts/perf/test_quantized_rollout_drift_probe.py`
- `tests/scripts/perf/test_quantized_sd3_forward_profile.py`
