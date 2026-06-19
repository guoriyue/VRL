# SPRINT: fp8 rollout GEMM kernel — 把 fp8 真正接进生成引擎前向

状态：**kernel + swap + 全链路 wiring 已落地，并在 5090 上真实 SD3.5 GRPO live run 验证通过（2026-06-19）**。从 [[SPRINT_fullparam_and_fp8_precision]] §3 第 2 步拆出。

## 验证结论（5090，已落地）

- **效率（确认 fp8 真的更快）**：`fp8_linear_benchmark.py` 实测 DiT shapes，fp8 `_scaled_mm` linear（含激活动态量化开销）vs bf16 nn.Linear，**geomean 1.40x**（1.1–1.75x）。✅
- **正确性**：`Fp8Linear` 单 GEMM 与 bf16 <6%；真实 cosmos 权重量化漂移 **2.2%**（比合成随机的 3.7% 更低）；真实 arch 前向漂移 cosmos 3.6% / sd3.5 1.1%。✅
- **LIVE RUN（关键，2026-06-19）**：真实 SD3.5-medium OCR GRPO，`precision={forward:bf16,rollout:fp8}` + TIS，跑满 3 epoch 无崩。**首次 live run 抓到并修了一个真 bug**：rowwise `_scaled_mm` 不收 fp32 输出，而 DiT adaLN/pooled 路径喂 fp32 → `Fp8Linear` 改成 bf16 累加再 cast 回（commit 01dd8e7）。fp8 还和 **LoRA 共存**（337 个 base linear 在 adapter 里被 swap）。
- **RL 精度 go/no-go（这才是判据）**：3 epoch metrics（reward_std≈0.31–0.46，信号健康）：
  | | ratio_abs_dev mean | ratio_abs_dev max | logprob_abs_diff mean | tis_clip_fraction |
  |---|---|---|---|---|
  | ep0–2 | **0.61–0.76%** | **4.8–5.7%** | 0.62–0.78% | 0（没触发）|
  - **fp8 importance-ratio 漂移 ~0.68% mean / 5.4% max，而 advantage 量级 O(1)（reward_std≈0.36 归一化）→ fp8 噪声比信号小 ~100x。GO。**
  - **TIS 根本没 engage**（max ratio≈1.05 ≪ cap 2.0）——fp8 漂移天然在容差内，TIS 是没用上的保险。
- **门控**：fp8 ungate；fp4 仍 gated（e4m3 only）。

### cosmos predict2 也跑了（full fine-tune，2026-06-19）

真实 Cosmos-Predict2-2B-Video2World GRPO（240p_33f，kling reward），`rollout=fp8`+TIS。**第二个 live bug 抓到并修了**：cosmos 是**全参**，trainer 每步把 base 权重 sync 给 rollout（`load_trainable_state` 要 `.weight` key），而 swap 后只有 `weight_fp8` → `Fp8Linear` 改成**留 bf16 master `weight` + sync 后 requantize**（cosmos-rl 同款，commit 365c62c）。SD3.5 没撞到是因为 LoRA sync 的是 adapter。修完 8 个 fp8 视频正常生成、full-finetune sync 干净。

cosmos go/no-go（step 0）：

| | ratio_dev mean | ratio_dev max | mismatch_kl | reward_std | tis_clip |
|---|---|---|---|---|---|
| SD3.5 | 0.68% | 5.4% | 0.0069 | 0.36 | 0 |
| **cosmos** | **0.30%** | **1.0%** | 0.0030 | 0.07 | 0 |

**两个家族都 GO**：fp8 ratio 漂移 ≤1%，advantage 归一化后 O(1) → fp8 噪声比信号小 100–300x，TIS 都没触发。cosmos 漂移反而更小（35 步去噪 vs SD3.5 10 步，per-step 误差摊薄），尽管前向漂移更大。

caveat：cosmos reward_std=0.07 是**弱信号 task**（advantage 归一仍 O(1)，所以 fp8 判据不变；但任务本身学不学得动是另一回事，与 fp8 无关）。结果存 `outputs/fp8_validation/{sd35,cosmos_pred2}_*_drift.csv`。

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

1. ✅ **量化路径（自研 `Fp8Linear`）**：`vrl/nn/quantization/fp8.py`。torch 原生 `_scaled_mm`、e4m3、bf16 master + bf16 累加、weight 构造时量化一次、activation 每步动态量化。`rowwise`（per-token / per-output-channel，抗激活 outlier）或 `tensorwise`。没引 torchao/TE（保持零依赖）。
2. ✅ **swap 方法**：`base.py` `quantize_transformer_fp8(recipe)`（与 `torch_compile_transformer` 同级）→ `swap_linears_to_fp8` 遍历 module tree，按 exclude 子串 + `min_features` 只换大 attention/MLP linear。
3. ✅ **拆 weight_dtype 语义 + ungate**：`online.py` fp8 时 storage = bf16 master（`policy.compute`）、删 `NotImplementedError`（fp4 仍 gated）；`rollout_quantization` 信号由 `extract_runtime_spec` 从 `precision.rollout` 派生进 `RuntimeBuildSpec`；三家 rollout builder（sd3_5/wan/cosmos）compile 前调 `loader.apply_rollout_fp8(model, spec)`，replay builder 不碰。
4. ✅ **scaling recipe + 精度 profile**：rowwise（默认）+ tensorwise 实现并验。`vrl/scripts/perf/fp8_recipe_accuracy.py`（fake-quant，对齐过真 `_scaled_mm`）量四档漂移：
   - **clean 激活**：四档≈ 3.7%（e4m3 floor，没 outlier 可吃）。
   - **outlier channels（真实情形）**：**block-1x128 最低 0.028** > rowwise 0.033 > tensorwise 0.036；block 比 tensorwise 少 ~22% 漂移。**MX-1x32 反而不帮**（e8m0 幂二 scale 丢 value 精度，抵消细粒度）。
   - **结论**：rowwise（已发，cu128 上有真 kernel）已吃掉 tensor→block 改善的 ~60%；**block-1x128 是 slime 同款最优，但真 kernel 要 CUDA≥12.9**（本机 cu128=12.8 只能 fake-quant）。已查证的正确调用：`scale_a` 列优先 `(M,K//128)`、`scale_b` 行优先 `(K//128,N)`、`out_dtype=bf16`、K%128==0 —— CUDA 升级后直接接。
5. ⏳ **torch.compile 交互**：swap 在 compile 前（已保证顺序）。`mode=default` + fp8 linear 的 inductor 行为要在 live run 验（已知坑：`reduce-overhead`/CUDAGraphs 撞 LoRA + grad-ckpt）。
6. ⏳ **LoRA 交互（开放）**：当前 `apply_rollout_fp8` 在 LoRA/full-finetune 之后调；full-finetune（cosmos predict2）是首选验证路径。LoRA 下 swap 命中 PEFT `base_layer` 的正确性留 live 验。

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
