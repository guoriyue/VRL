# SPRINT: rollout 优化层 — 架构决策 + patch 收拢 + 剩余 wiring (done)

状态：**done（2026-07-02）**。§3 全部收口：P0 假旋钮守卫（06-19）、recipe config-wire（07-02）、blockwise×compile 探针（07-02，NEGATIVE→加守卫）、samples_per_chunk A/B（07-02，1.35x 确认）。fp8 kernel 已落地并 live 验证（见 [[SPRINT_fp8_rollout_gemm_kernel]]）；本 sprint 记录"rollout 怎么优化"的**架构决策**（别再 re-litigate）、可选的一致性收拢、以及剩余的功能 wiring。2026-06-27 NCU 复核确认 §1 架构决策。

> Current config note (2026-07-12): FP8 is selected with
> `precision.rollout.quantization.format=fp8`; its recipe lives at
> `precision.rollout.quantization.recipe`. The ordinary rollout dtype remains
> separate at `precision.rollout.dtype`. Older flat `precision.rollout` and
> `precision.rollout_recipe` spellings below are historical implementation
> evidence, not launch instructions.

> **2026-06-27 NCU 硬件复核 reinforces §1**（全文 [[SPRINT_lossless_diffusion_rl_research]]）：§1 用 `nvidia-smi dmon`（SM 100%）定的"compute-bound、引擎级无用、patch 即可"被 NCU tensor-pipe SOL 实锤——cosmos GEMM 45% ≈ 最优方阵 GEMM 47% = bf16+fp32 累加硬件上限,**不是有头空间没榨**。推论:① fp8 是唯一越 bf16 上限的杠杆但**有损**(只能离 policy path);② §1 否决重写/引擎的结论加强,不是减弱;③ §1.附带决策"小请求用静态 batch 不用 continuous batching"经本轮 stepwise-batching 探针证伪 continuous batching 后更稳。**判 MFU 用 `gpu_preflight` 实测峰值,别用 vendor 419(会把饱和误诊成 51%)。**

## 0. 来历

fp8 swap 接进三家 rollout builder 后，出现一个反复出现的疑问：rollout 优化是不是该**重写 diffusion forward / 建推理引擎**，而不是 patch。这次用 profile + cosmos-rl 源码把它定死，并顺带理清 patch 层的整洁问题。

## 1. 架构决策（已定，证据在此，别再 re-litigate）

**rollout 优化 = patch（module swap），不重写 forward，不建引擎。** 三条证据：

- **profile（`outputs/perf/cosmos25_gen_trace.json`）**：cosmos predict2.5 rollout（512p×93f）**compute-bound**——`nvidia-smi dmon` 实测 **SM 100% / MEM 22–27%**，GEMM 主导，48% 已被 `torch.compile` 融成一个 FX 图。结论：kernel 级 patch（fp8）正中靶心；引擎级（continuous batching / paged cache）**无用武之地**（SM 已满，没有空 SM 给 batch 填）。
- **cosmos-rl 源码**：它的 diffusion rollout 是**裸 PyTorch**——`cosmos_rl/rollout/wfm_rollout/wfm_rollout.py:137` 原话 *"WFM rollout directly use Pytorch model as the naive forward. it doesn't need inference framework like vllm or trtllm."* 它的 fp8 monkey-patch 只在 **LLM/vLLM** 那半边（`vllm_rollout/monkey_patch_for_fp8.py`，patch vLLM 的 `Fp8LinearOp`）。**NVIDIA 自己都没重写 diffusion forward。**
- **vLLM 没有 diffusion 形状的融合 kernel**：它的 fused QKV+RoPE / SwiGLU MLP / RMSNorm 是 **LLM 形状**；diffusion DiT 是 adaLN 调制 + LayerNorm + GELU + joint/cross attention，对不上。能 reuse 的是**单个 fp8 GEMM 路径**：默认 live path 已用 rowwise `torch._scaled_mm` module-swap；vLLM 的 `w8a8_triton_block_scaled_mm` 已作为 `recipe="blockwise"` 实现并 config-wire（当前为 `precision.rollout.quantization.recipe`），但 §3.2 实测它 graph-break torch.compile → eager-only opt-in，永不做 live 默认。重写拿不到新 kernel，只能自己写融合，而那正是 `torch.compile` 已自动生成的。

**重写的成本**：own sd3.5/wan/cosmos 每家的 forward + 永远和 diffusers 数值 bit-match（自有 memory：EDM sigma 域 bug、scheduler logprob parity bug、predict2 GRPO parity），换一个不确定且很小的融合增量。**否决。**

**附带决策**：小请求 under-utilized（SD3.5 小图 batch=1 实测 SM 78%）的解是**静态 batch（`rollout.sample_batch_size`）**，不是 continuous batching——后者是 AR 的变长 KV-cache 专属，diffusion 同形状同步数，静态 batch 就够（batch=4 实测 SM→98%）。

## 2. patch 收拢（一致性 cleanup，可选）

**fp8 本身不乱**（`apply_rollout_quantization` 一个 rollout-scoped helper，一行）。真正的 smell 是 builder 之间**同操作两套写法**：

| | rollout builder | replay builder |
|---|---|---|
| lora | `model.apply_lora(build)`（模型方法） | `apply_lora_to_transformer(model, build)`（loader 函数）|
| compile | `model.torch_compile_transformer(m)` | `compile_transformer(model, m)` |

（cosmos rollout builder `predict2/runtime.py:68`，replay `:128`；sd3_5/wan 同构。）

**收法**：一个按 role 参数化的 prep 函数，统一 6 个 builder（3 家 × rollout/replay）：

```python
def prepare_transformer(model, build, *, quantize_rollout: bool) -> None:
    apply_lora_to_transformer(model, build) if build.use_lora else enable_transformer_full_finetune(model)
    if quantize_rollout:                       # 只有 rollout=True；fp8 仍 rollout-scoped
        apply_rollout_quantization(model, build)
    compile_transformer_if_enabled(model, build)
```

rollout builder 调 `quantize_rollout=True`，replay 调 `False`。重复的 prep 序列收成一处，rollout/replay 唯一区别变一个布尔，helper 不一致顺手统一。

**架构原则（这条要守）**：按 **scope/lifecycle 分组，不按"位置相邻"分组**。fp8 = rollout-scope，compile/attention = 两边都用 —— **不能**把它们打包成 `apply_rollout_optimizations`（混 scope，是"为凑近而抽象"的反模式；attention backend 当前压根不在 builder 这一层）。

**可做可不做**：这是 taste/一致性重构。本项目惯例 "fix correctness first, don't refactor working architecture" + "consistency over cleanup"（cleanup 重构被 revert 过），而现状的**显式 step-list 也可读、好 debug**。倾向 leave it；要动就只动这个 prep 收拢，不扩大。

## 3. 剩余 wiring（真功能缺口，按优先级）

0. ✅ **P0（已落地 2026-06-19）：堵 fp8 假旋钮 + 补 predict2_5/anima wiring**：
   - **补接**：`predict2_5/runtime.py`、`anima/runtime.py` 两家 rollout builder 在 compile 前调 `apply_rollout_quantization(model, build)`（replay builder 不碰）。
   - **Two guards**: `apply_rollout_quantization` rejects a zero-match swap, and
     `assert_rollout_quantization_applied` inspects the worker's real policy model
     after build. The backstop now requires both `QuantizedLinear` and an exact
     `quantization_scheme == requested format`, so an FP8 module cannot satisfy an
     NVFP4 request.
   - **Scheme-neutral ownership**: `QuantizedLinear` owns the shared cache/master
     lifecycle; concrete subclasses publish their precise scheme identity.
   - **测试**：apply 0-swap 报错 + backstop（scheme 参数化 fp8/fp4、有/无 `QuantizedLinear`、含 compiled unwrap、无量化 noop）。283 regression 过。
1. ✅ **config-wire fp8 recipe（已落地 2026-07-02）**：当时新增的 flat
   `precision.rollout_recipe` 现已迁移到
   `precision.rollout.quantization.recipe`（默认缺省 = scheme 默认 `rowwise`；
   `blockwise`/`tensorwise` opt-in）。当前链路：`precision.py` 解析进
   `policy.rollout.quantization.recipe` → `resolve_model_build` 填
   `ModelBuild.rollout.quantization_recipe` →
   `apply_rollout_quantization` passes it to `quantize_rollout_fp8(recipe=...)`.
   The torch-free `QuantizationPolicy` format table is now the single recipe
   compatibility/default source consumed again by `RolloutBuildOptions`. Tests:
   config 解析/拒绝/walker + loader recipe 透传（tests/config/test_precision.py、
   tests/nn/quantization/test_fp8.py）。
2. ✅ **blockwise + torch.compile 交互（已测 2026-07-02，NEGATIVE→加守卫）**：真实 SD3.5-medium DiT（batch 8 = CFG×chunk4，512p，5090）三档对测：

   | recipe | graph breaks | eager ms | compiled ms | compile 加速 |
   |---|---|---|---|---|
   | bf16 | 0 | 141.8 | 103.9 | 1.36x |
   | rowwise | 0 | 137.7 | **62.2** | **2.22x** |
   | blockwise | **45** | 119.8 | **1210.7** | **0.10x（慢 10 倍）** |

   - **rowwise 和 compile 完全干净**（0 break，单图；compiled rowwise 62ms vs compiled bf16 104ms = compile 下 fp8 仍赚 1.67x）。
   - **blockwise graph-break 是结构性的**：断点不在 triton kernel 本身，而在 vLLM wrapper——`per_token_group_quant_fp8` 每次调用都过 `functools.lru_cache` 包装的 `is_deep_gemm_e8m0_used`（dynamo 不能 trace lru_cache，gb0177），`w8a8_triton_block_scaled_mm` 的 config 查找又碰 pynvml ctypes 调用（gb0156）；另外 dynamo 对该 kernel 的 mutation 分析失败（"assuming every input is mutated"），融合全毁。45 个 break 的 guard/重入开销让 compiled 反而比 eager 慢 10 倍。
   - **处置**：`apply_rollout_quantization` 加 build-time 守卫——`recipe='blockwise'` + `model.torch_compile` 直接 `ValueError`（静默 10 倍减速比 crash 更糟）；fp8.py/precision.py 文档同步注明 blockwise = eager-only。**rowwise 维持默认且是唯一 compile 兼容 recipe**；blockwise 仅在 compile-off 时有意义（eager 120ms 仍是最快 eager，但输给 compiled rowwise 62ms —— compile 默认开的家族里 blockwise 严格劣势）。测试：blockwise+compile 拒绝、blockwise 无 compile / rowwise+compile 放行（test_fp8.py）。
3. ✅ **小请求静态 batch A/B（已测 2026-07-02，CONFIRMED ~1.25–1.35x）**：真实 SD3.5-medium 完整 rollout 循环（10-step CFG 4.5 去噪 + VAE decode，512p，5090；旋钮现名 `rollout.samples_per_chunk`，旧 `sample_batch_size` 已改名）：

   | chunk | eager ms/样本 | 加速 | compiled ms/样本 | 加速 | 峰值显存 |
   |---|---|---|---|---|---|
   | 1 | 510.0 | 1.00 | 387.3 | 1.00 | 5.4 GB |
   | 2 | 408.7 | **1.25** | 312.5 | 1.24 | 5.8 GB |
   | 4 | 377.8 | **1.35** | 286.5 | 1.35 | 6.8 GB |
   | 8 | 365.4 | 1.40 | 265.8 | 1.46 | 8.8 GB |

   预期的 ~1.25x 在 chunk=2 就兑现，chunk=4 到 1.35x，之后边际递减；显存 8.8GB 远未顶 32GB。**结论：图像轻负载 `samples_per_chunk>=4`（live OCR config 已是 16，无需动）；视频被显存逼成小 chunk 的仍不适用。**与 cosmos 视频侧实测互相印证（sbs 1→4 = 1.40x、1→8 ≈ 1.95x，见 [[SPRINT_diffusion_rollout_stage_pipeline]]）。

   > 环境注记：两个探针都在 conda base（torch 2.11.0+cu128 / vllm 0.21.0 / dev triton 3.8）跑；probe 3 需 `TORCHINDUCTOR_COMPILE_THREADS=1`（dev triton fork-unsafe，见 triton env memory）。graph-break 计数是 dynamo 结构性结论，与 triton 版本无关。

## 非目标

- 不重写 diffusion forward（§1 已否决，证据在案）。
- 不建 continuous batching / paged 引擎（AR 专属，diffusion 用静态 batch）。
- 不动 TIS / drift guard / precision policy（[[SPRINT_fullparam_and_fp8_precision]] 已落）。
- 不把 fp8 + compile + attention 按位置打包（混 scope）。

## 相关

- [[SPRINT_fp8_rollout_gemm_kernel]]（fp8 kernel + live 验证，本 sprint 的前置）
- [[SPRINT_fullparam_and_fp8_precision]]（precision policy / TIS / drift guard 地基）
- profile trace：`outputs/perf/cosmos25_gen_trace.json`（compute-bound 证据）
- 决策证据：`~/Desktop/cosmos-rl/cosmos_rl/rollout/wfm_rollout/wfm_rollout.py:137`（diffusion rollout 裸 PyTorch）
- patch 实现：`vrl/nn/quantization/fp8.py`、`vrl/models/loader.py:apply_rollout_quantization`
