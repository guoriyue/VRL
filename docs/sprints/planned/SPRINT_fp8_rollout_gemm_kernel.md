# SPRINT: fp8 rollout GEMM kernel — 把 fp8 真正接进生成引擎前向 (planned)

状态：**planned（2026-06-18 从 [[SPRINT_fullparam_and_fp8_precision]] §3 第 2 步拆出）**。

## 0. 来历

[[SPRINT_fullparam_and_fp8_precision]] 落地了 fp8 的**精度-修正地基**（precision policy 的 fp8/fp4 token + `rollout` 轴、drift guard 复用、TIS、验证 probe），但 fp8 rollout **不能 live 跑**——`vrl/scripts/common/online.py` 里对 `precision.rollout=fp8/fp4` 直接 `NotImplementedError`。本 sprint 收剩下唯一的活：**让生成引擎的 diffusion 前向真正用 fp8 GEMM**。地基零改动，接上即生效。

## 1. 根因（实测）

普通 `nn.Linear` 在 float8 上：
```
NotImplementedError: "addmm_cuda" not implemented for 'Float8_e4m3fn'
```
fp8 不是 drop-in dtype（bf16/fp16 有原生自动 dispatch 的 GEMM，fp8 没有）。fp8 GEMM 必须显式：
1. 激活按 amax 量化成 fp8 + scale；
2. 权重量化成 fp8 + scale；
3. `torch._scaled_mm(a_fp8, w_fp8, scale_a, scale_b, out_dtype=bf16)`，**高精度累加**；
4. 非矩阵乘（norm / softmax / 残差 / SDE-logprob 数学）留 **bf16 master**。

参考 recipe 已在仓库里：`vrl/scripts/perf/fp8_rollout_drift_probe.py` 的 `_fp8_matmul` / `_amax_scale`（per-tensor dynamic amax，真 `_scaled_mm`）。

## 2. 落点（已查证的代码锚）

| 环节 | 文件:行 | 说明 |
|---|---|---|
| weight_dtype 入口 | `vrl/scripts/common/online.py:662-683` | `rollout_precision` 解析（662）；fp8/fp4 时**当前是 `NotImplementedError` 拦截**（672 起）|
| 模型加载（按 dtype） | `vrl/models/diffusion/sd3_5/model.py:110-143`（`from_spec`）、`wan_2_1/model.py:142-149`、`cosmos/predict2/model.py:109-143` | 三家都 `Pipeline.from_pretrained(torch_dtype=spec.dtype)` 加载 **diffusers transformer** |
| **swap 缝（核心）** | `vrl/models/diffusion/base.py:244-249` `torch_compile_transformer` | in-place 包 transformer 的现成位;fp8 swap 做成**同级方法** `quantize_transformer_fp8(recipe)` |
| 调用点（每家） | `sd3_5/runtime.py:56-78`、`wan_2_1/runtime.py`、`cosmos/predict2/runtime.py` | `from_spec` → LoRA/full-finetune → `torch_compile`;fp8 swap 插在 **compile 之前**（LoRA 之后）|
| 前向（被 swap 的 Linear 在这跑） | `vrl/models/diffusion/sd3_5/model.py:285-332` `forward_step` → `common/backbone.py:84-150` | DiT forward;实际 `nn.Linear`（to_q/k/v、ff）在 **diffusers transformer 内**,需遍历 module tree swap |
| rollout 去噪循环（主成本） | `vrl/generation/diffusion/executor.py:672-746`，`forward_step` 调用在 **:680**，`no_grad` 下逐步重复 | fp8 加速的就是这段 |
| 现有 fp8 基建 | `vrl/scripts/perf/fp8_rollout_drift_probe.py:54-64` | `_scaled_mm` 参考实现;**无 torchao / transformer_engine 依赖**(grep 确认) |

## 3. 工作分解

1. **选量化路径**：优先 **torchao `Float8Linear`**（module swap，和仓库已用的原生 `_scaled_mm` 一脉、维护成本低）；备选 TransformerEngine、或自研 `Fp8Linear`（直接复用 probe 的 `_fp8_matmul`）。先做一个 spike A/B：torchao swap vs 自研，比 sm_120 上的正确性 + 速度。
2. **加 swap 方法**：在 `vrl/models/diffusion/base.py` 加 `quantize_transformer_fp8(self, recipe)`（与 `torch_compile_transformer` 同级，base.py:244 旁），遍历 `self.transformer` 的 module tree,按 §4 的 include/exclude 名单把目标 `nn.Linear` 替换成 fp8 linear。每家 runtime builder 在 `torch_compile` 前调用（`sd3_5/runtime.py:75` 之前）。
3. **拆 weight_dtype 语义**：`online.py:662-683` —— fp8/fp4 时模型 **storage 仍 bf16**（master），不再 `resolve_torch_dtype(fp8)→float8` 把模型存成 float8;量化在 GEMM 内部做。把 fp8 意图作为**单独的 `rollout_quantization` 信号**经 `build_ray_generation_inputs_for_family` → runtime spec 传给 swap;**删掉 672 起的 `NotImplementedError`**，换成"runtime 报告支持 fp8 才放行"的能力检查。
4. **scaling recipe**：先 per-tensor dynamic amax（probe 现成）;不够再 per-row 或 128×128 per-block（slime 式）。静态 vs 动态在这步定。
5. **torch.compile 交互**：compile 在 swap **之后**（让 inductor 看到 fp8 linear）。已知坑（Wan LoRA 实测）：compile `mode=default` 与 PEFT 兼容，`reduce-overhead`/CUDAGraphs 撞 LoRA + grad-ckpt;fp8 linear + compile default 要单独验。
6. **LoRA 交互（开放）**：`use_lora` 配置下,base 量化 fp8 + adapter 留 bf16 的顺序与正确性需验（先在 full-finetune cosmos 上做，绕开 LoRA）。

## 4. include / exclude 名单（fp8 只碰大 GEMM）

- **INCLUDE**（策略 GEMM）：attention 的 `to_q`/`to_k`/`to_v`/`to_out`、cross-attn 投影、feed-forward 的两个 `nn.Linear`。
- **EXCLUDE**（小且数值敏感，留 bf16）：patch/latent embedding、timestep/condition 投影、最后的 noise-pred 输出头、所有 norm。
- **绝不碰**：SDE / logprob 数学（`sde_step_with_logprob` 等,精度轴 `math` 恒 fp32）。

## 5. 验收

1. **正确性**：`fp8_rollout_drift_probe.py` 换真实 DiT 跑,`ratio_dev` 分布落在可控范围;drift guard 设 `warn` 一轮不报灾难。
2. **修正联动**：据实测 `ratio_dev` 校准 `trainer.precision_correction` 的 cap,确认 TIS `tis_clip_fraction` 合理（不是大面积截）。
3. **吞吐**：`vrl/scripts/perf/gemm_projection_breakdown.py` / `compile_benchmark.py` 量 rollout 段实际加速,再按 colocated 串行 Amdahl 折端到端（先测 rollout 占 cycle 比例）。
4. **门控**：`online.py` 的 `NotImplementedError` 已换成能力检查,`precision.rollout=fp8` live run 能起。

## 6. 非目标

- 不碰训练侧（replay forward 恒 bf16/fp32;fp8 只在 rollout）。
- 不在本 sprint 改 TIS / drift guard / precision policy（[[SPRINT_fullparam_and_fp8_precision]] 已落,这里只把 kernel 接上让它们生效）。
- 不追求 FP4（先把 fp8 跑通验证;fp4 的 token 已在,但精度风险更高,单列）。

## 相关
- [[SPRINT_fullparam_and_fp8_precision]]（父 sprint,§3 是本 sprint 的来源）
- [[SPRINT_gemm_utilization]]（`done/`,profiler / compile 测量工具的出处）
- `vrl/scripts/perf/fp8_rollout_drift_probe.py`（fp8 GEMM + 漂移测量参考实现）
- `vrl/scripts/perf/gemm_projection_breakdown.py`、`compile_benchmark.py`（吞吐测量工具）
