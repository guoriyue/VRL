# SPRINT: rollout 优化层 — 架构决策 + patch 收拢 + 剩余 wiring (planned)

状态：**planned（2026-06-19）；2026-06-27 NCU 复核确认 §1 架构决策**。fp8 kernel 已落地并 live 验证（见 [[SPRINT_fp8_rollout_gemm_kernel]]）；本 sprint 记录"rollout 怎么优化"的**架构决策**（别再 re-litigate）、可选的一致性收拢、以及剩余的功能 wiring。

> **2026-06-27 NCU 硬件复核 reinforces §1**（全文 [[SPRINT_lossless_diffusion_rl_research]]）：§1 用 `nvidia-smi dmon`（SM 100%）定的"compute-bound、引擎级无用、patch 即可"被 NCU tensor-pipe SOL 实锤——cosmos GEMM 45% ≈ 最优方阵 GEMM 47% = bf16+fp32 累加硬件上限,**不是有头空间没榨**。推论:① fp8 是唯一越 bf16 上限的杠杆但**有损**(只能离 policy path);② §1 否决重写/引擎的结论加强,不是减弱;③ §1.附带决策"小请求用静态 batch 不用 continuous batching"经本轮 stepwise-batching 探针证伪 continuous batching 后更稳。**判 MFU 用 `gpu_preflight` 实测峰值,别用 vendor 419(会把饱和误诊成 51%)。**

## 0. 来历

fp8 swap 接进三家 rollout builder 后，出现一个反复出现的疑问：rollout 优化是不是该**重写 diffusion forward / 建推理引擎**，而不是 patch。这次用 profile + cosmos-rl 源码把它定死，并顺带理清 patch 层的整洁问题。

## 1. 架构决策（已定，证据在此，别再 re-litigate）

**rollout 优化 = patch（module swap），不重写 forward，不建引擎。** 三条证据：

- **profile（`outputs/perf/cosmos25_gen_trace.json`）**：cosmos predict2.5 rollout（512p×93f）**compute-bound**——`nvidia-smi dmon` 实测 **SM 100% / MEM 22–27%**，GEMM 主导，48% 已被 `torch.compile` 融成一个 FX 图。结论：kernel 级 patch（fp8）正中靶心；引擎级（continuous batching / paged cache）**无用武之地**（SM 已满，没有空 SM 给 batch 填）。
- **cosmos-rl 源码**：它的 diffusion rollout 是**裸 PyTorch**——`cosmos_rl/rollout/wfm_rollout/wfm_rollout.py:137` 原话 *"WFM rollout directly use Pytorch model as the naive forward. it doesn't need inference framework like vllm or trtllm."* 它的 fp8 monkey-patch 只在 **LLM/vLLM** 那半边（`vllm_rollout/monkey_patch_for_fp8.py`，patch vLLM 的 `Fp8LinearOp`）。**NVIDIA 自己都没重写 diffusion forward。**
- **vLLM 没有 diffusion 形状的融合 kernel**：它的 fused QKV+RoPE / SwiGLU MLP / RMSNorm 是 **LLM 形状**；diffusion DiT 是 adaLN 调制 + LayerNorm + GELU + joint/cross attention，对不上。能 reuse 的是**单个 fp8 GEMM 路径**：默认 live path 已用 rowwise `torch._scaled_mm` module-swap；vLLM 的 `w8a8_triton_block_scaled_mm` 已作为 `recipe="blockwise"` 实现,但还没 config-wire / live 默认。重写拿不到新 kernel，只能自己写融合，而那正是 `torch.compile` 已自动生成的。

**重写的成本**：own sd3.5/wan/cosmos 每家的 forward + 永远和 diffusers 数值 bit-match（自有 memory：EDM sigma 域 bug、scheduler logprob parity bug、predict2 GRPO parity），换一个不确定且很小的融合增量。**否决。**

**附带决策**：小请求 under-utilized（SD3.5 小图 batch=1 实测 SM 78%）的解是**静态 batch（`rollout.sample_batch_size`）**，不是 continuous batching——后者是 AR 的变长 KV-cache 专属，diffusion 同形状同步数，静态 batch 就够（batch=4 实测 SM→98%）。

## 2. patch 收拢（一致性 cleanup，可选）

**fp8 本身不乱**（`apply_rollout_quantization` 一个 rollout-scoped helper，一行）。真正的 smell 是 builder 之间**同操作两套写法**：

| | rollout builder | replay builder |
|---|---|---|
| lora | `model.apply_lora(spec)`（模型方法） | `apply_lora_to_transformer(model, spec)`（loader 函数）|
| compile | `model.torch_compile_transformer(m)` | `compile_transformer(model, m)` |

（cosmos rollout builder `predict2/runtime.py:68`，replay `:128`；sd3_5/wan 同构。）

**收法**：一个按 role 参数化的 prep 函数，统一 6 个 builder（3 家 × rollout/replay）：

```python
def prepare_transformer(model, spec, *, quantize_rollout: bool) -> None:
    apply_lora_to_transformer(model, spec) if spec.use_lora else enable_transformer_full_finetune(model)
    if quantize_rollout:                       # 只有 rollout=True；fp8 仍 rollout-scoped
        apply_rollout_quantization(model, spec)
    compile_transformer_if_enabled(model, spec)
```

rollout builder 调 `quantize_rollout=True`，replay 调 `False`。重复的 prep 序列收成一处，rollout/replay 唯一区别变一个布尔，helper 不一致顺手统一。

**架构原则（这条要守）**：按 **scope/lifecycle 分组，不按"位置相邻"分组**。fp8 = rollout-scope，compile/attention = 两边都用 —— **不能**把它们打包成 `apply_rollout_optimizations`（混 scope，是"为凑近而抽象"的反模式；attention backend 当前压根不在 builder 这一层）。

**可做可不做**：这是 taste/一致性重构。本项目惯例 "fix correctness first, don't refactor working architecture" + "consistency over cleanup"（cleanup 重构被 revert 过），而现状的**显式 step-list 也可读、好 debug**。倾向 leave it；要动就只动这个 prep 收拢，不扩大。

## 3. 剩余 wiring（真功能缺口，按优先级）

0. ✅ **P0（已落地 2026-06-19）：堵 fp8 假旋钮 + 补 predict2_5/anima wiring**：
   - **补接**：`predict2_5/runtime.py`、`anima/runtime.py` 两家 rollout builder 在 compile 前调 `apply_rollout_quantization(model, spec)`（replay builder 不碰）。
   - **两层守卫**：① `apply_rollout_quantization`（`vrl/models/loader.py`）在 fp8 请求但 swap 命中 0 个 linear 时直接 `RuntimeError`（exclude/min_features 过度过滤的早失败）；② **family- + scheme-agnostic backstop** `assert_rollout_quantization_applied(model, spec)` 在 rollout worker `vrl/generation/execution/worker.py:_build_executor`（worker 侧 build 后、所有家族必经的唯一点）数 transformer 里的 **`QuantizedLinear`**（fp8 今、fp4/int8 后；unwrap `torch.compile` 的 `_orig_mod`），任何 `precision.rollout` 量化 scheme 请求但 0 个 → 启动显式失败。
   - **scheme 通用化**：新增 marker 基类 `vrl/nn/quantization/base.py:QuantizedLinear`，`Fp8Linear` 继承它；守卫 `isinstance(m, QuantizedLinear)` 而非 hardcode `Fp8Linear`，**以后加 fp4/int8 scheme 只要继承基类就自动被守卫覆盖**（不用改 guard、不留静默缺口）。比"builder 自己记 `runtime_caps.swapped_count`"更稳：直接 inspect 真实模型，漏接也照样炸。
   - **测试**：apply 0-swap 报错 + backstop（scheme 参数化 fp8/fp4、有/无 `QuantizedLinear`、含 compiled unwrap、无量化 noop）。283 regression 过。
1. ✅ **config-wire fp8 recipe（已落地 2026-07-02）**：新增 `precision.rollout_recipe`（默认缺省 = scheme 默认 `rowwise`；`blockwise`/`tensorwise` opt-in）。链路：`precision.py` 解析进 `PrecisionPolicy.rollout_recipe`（非量化 rollout 上设置直接 raise，堵假旋钮）→ `extract_runtime_spec` 填 `RuntimeBuildSpec.rollout_quantization_recipe`（asdict 自动过 Ray launch contract）→ `apply_rollout_quantization` 传给 `quantize_transformer_fp8(recipe=...)`。recipe 词表唯一真源仍是 `Fp8Linear`（config 层不重复维护值校验）。测试：config 解析/拒绝/walker + loader recipe 透传（tests/config/test_precision.py、tests/nn/quantization/test_fp8.py）。
2. **blockwise + torch.compile 交互**：vLLM triton kernel 会不会让 compile graph-break、损失多少融合 —— rowwise(`_scaled_mm`)和 compile 干净，blockwise 要 live 测（[[SPRINT_fp8_rollout_gemm_kernel]] §3.5 开放项）。
3. **小请求静态 batch A/B**：SD3.5 小图 `sample_batch_size=1` 时 SM 78%。显存够的轻负载调大 `sample_batch_size` 跑 wall-clock A/B（预期 ~1.25x）。视频被显存逼成 1，不适用。

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
