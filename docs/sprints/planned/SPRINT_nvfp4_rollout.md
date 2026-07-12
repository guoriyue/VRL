# SPRINT: NVFP4 rollout（fp4 生成 + bf16 训练重放）

状态：**P1 已通过（2026-07-11）：当前 MLP-only production target 的 12 个 linear shape
几何平均 `2.33x` bf16；真实 SD3.5 Medium checkpoint、B=32、`torch.compile` transformer
forward 为 `1.386x`。P2 仍缺真实 SDE trajectory。** 旧 full-head synthetic probe 把 35 个
denoise step 的 log-ratio 相乘，并用非生产 cap `1.5` 得出 `15.2%`，该 gate 与 trainer
逐 timestep loss 语义不符，已删除。修正后 production cap `2.0` 下 FP4 synthetic
sample-step TIS clip/RS masked 均为 `0%`；这只验证 correction path，不代替真实 MLP-only
rollout→BF16 replay 的 SDE-logprob 验收。完整实测见
`info/SPRINT_quantized_rollout_precision_performance.md`。

> 来由：外部 SOTA `FP4 Explore, BF16 Train`（arXiv:2604.06916）同款契约（FP4 只跑
> rollout、logprob 漂移交给重要性校正）报 ~2.5-3x rollout / 1.5-2x 端到端（H100 数，
> 5090 需本机重测——P0 已重测，见 §3）。姊妹 sprint
> `SPRINT_fp4_off_policy_reward_vae.md`（reward/VAE 上 FP4）**挂起**：reward ≈ rollout
> 的 14% 且已被 sleep-offload/异步隐藏，fp4 加速被 overlap 藏掉的时间端到端≈0；
> 等 VAE decode 占比实测（那边 P0）再议。

## 0. 一句话

rollout（denoise 生成）的大 GEMM 上 NVFP4，训练重放保持 bf16；fp8 rollout 的全部
seam（precision token、swap 机制、weight-sync 重量化、TIS/RS、drift guard）原样复用，
`Fp4Linear` seam 与 fused 动态激活量化已通过 production linear 性能门；接下来必须
验证真实 logprob 漂移与 TIS 截断率，才能进入端到端配方验收。

## 1. 训练侧 fp4 判定（记录，防止下次再议）

"native NVFP4 training"（前后向 GEMM 都 fp4）拆三种形态：

- **非对称（fp4 replay vs bf16 rollout）— 死**。fresh logprob 带 fp4 噪声污染 ratio。
  当前实现只允许相反方向：FP4 rollout + BF16 replay。修正后的 synthetic probe 在 production
  cap=2.0 下 FP8/FP4 的逐 timestep TIS clip 都是 `0%`；是否可接受仍由真实 SDE trajectory
  决定，不能用 35 步 ratio 乘积代替。
- **对称（rollout 和 replay 同一个 fp4 forward）— 数学成立，但当前不落地**。
  θ=θ_old 处 ratio≡1，ppo_epochs=1 下退化为干净 REINFORCE，量化只影响梯度质量。
  挡它的三道坎：① 梯度信噪比——NVFP4 预训练靠 stochastic rounding + Hadamard +
  2D scaling 压梯度噪声，且是万亿 token 预训练的容忍度；本仓库 RL 微调是
  grad_norm ~2e-4 的小信号政权（cosmos+Kling flat-curve 尸检），fp4 反向噪声大概率
  制造 within-noise 假曲线；② sm_120 无训练级生态——TE NVFP4 训练配方对准 sm_100，
  torchao mx 训练原型同样要 sm_100，torch 原生无 fp4 autograd，dgrad/wgrad + SR cast
  要自写 Triton；③ ROI——训练段占 cycle 一部分 × ~1.6-1.8x ≈ 端到端 ~10%，过不了
  MFU 北极星的 lossy-off-policy-path 规则。**上 B200/TE 后对称形态可重启**，验收门
  就是现成的 parity 探针（bit-exact 验证 ratio≡1）。
- **fp8 训练（torchao float8 rowwise，消费卡有生态）**——若真要碰量化训练，先试这
  个而不是 fp4；同样要过"对称 or TIS"的 ratio 契约。非本 sprint 目标。

## 2. 插座清单（全部现成，fp4 token 已流经六层）

| 层 | 状态 | 位置 |
|---|---|---|
| config 解析/校验（fp4 合法 rollout token + rollout_recipe） | ✅ | `vrl/config/precision.py:51,93` |
| spec 传递到 Ray worker | ✅ | `vrl/models/runtime_config.py:53` |
| 模块 swap 调度（注释明言 "fp4/int8 as siblings"） | ✅ | `vrl/models/loader.py:apply_rollout_quantization` |
| worker 端量化生效断言 | ✅ | `vrl/generation/execution/worker.py:654` |
| drift guard + TIS/RS + mismatch stats（精度无关） | ✅ | precision_guard / logprob_mismatch |
| online storage bridge（fp8/fp4 均加载 train-dtype master） | ✅ | `vrl/scripts/common/online.py:_rollout_weight_dtype` |

**当前缺件**：MLP-only 与 full-attention profile 的真实模型 drift 验收及端到端 recipe。
synthetic drift probe 的 fp4 支路只验证 correction plumbing，不能判定 full-policy FP4；
`Fp4Linear`、fused quantize kernel、model facade、loader dispatch、production-size compile
和 benchmark 均已通过 P1。P2 前 fp4 仍是用户显式选择的实验路径。

## 3. P0 硬件门结果（2026-07-11，5090 sm_120，torch 2.11.0+cu128）

`torch._scaled_mm` 的 nvfp4 路径（`float4_e2m1fn_x2` 操作数 + e4m3 1x16 block scale
**blocked-swizzled 布局** + fp32 两级 tensor scale）**原生可跑**：

| shape (M,K,N) | bf16 ms | fp8 ms | fp4 ms | fp4/bf16 | 输出漂移(rel L1) |
|---|---|---|---|---|---|
| (4096,2048,8192) | 0.912 | 0.621 | 0.268 | **3.40x** | 0.134 |
| (4096,2048,2048) | 0.226 | 0.175 | 0.083 | **2.72x** | 0.134 |
| (8192,2048,6144) | 1.561 | 0.784 | 0.428 | **3.64x** | 0.134 |

- 数字是 **GEMM 裸时延**（激活+权重预量化）；生产要像 Fp8Linear 一样每 forward 动态
  量化激活。纯 torch 原型曾只有 `0.35x` bf16；换成 fused Triton block-scale +
  nearest-even pack 后，同一 24 组 DiT shape 净 speedup 为 `1.47x–3.50x`、几何平均
  当前 MLP-only 12-shape geomean `2.33x`，P1 linear 性能门通过。
- 漂移 0.134 是朴素 RTN、随机数据的数字——**比 fp8 tensorwise 大一个量级**，这正是
  §5 验收门要在真 logprob 路径上量的。
- 关键 API 知识（P1 直接用）：scale 必须过 cuBLAS blocked swizzle
  （128-row × 4-col tile，`view(nrb,128,ncb,4).permute(0,2,1,3).reshape(-1,4,32,4)
  .transpose(1,2).reshape(-1,32,16).flatten()`）；两级缩放 tensor_scale =
  amax/(448·6)，block_scale(e4m3) = block_amax/6/tensor_scale，输出乘回
  tensor_scale_a·tensor_scale_b。probe 脚本为一次性 scratch（未入库），配方以此段
  为准。

## 4. Phase plan

- **P1 — `Fp4Linear` + swap**：**DONE（2026-07-11）**。落地：
  `vrl/nn/quantization/fp4.py`（`Fp4Linear`/`quantize_nvfp4`/`swap_linears_to_fp4`/
  swizzle；bf16 master + 非持久 fp4 cache + sync 重量化 + `drop_master`，与
  Fp8Linear 同构）；`drop_fp8_masters` 泛化为 base.py `drop_quantized_masters`
  （QuantizedLinear 契约，scheme 无关）；`QuantizedLinear._apply` 统一保护 packed
  cache，model `.to(device, dtype=...)` 只移动/cast master 并重新量化，避免 FP4 shell
  dtype copy 失败和 FP8 cache 被静默 cast；diffusion/ar base 的
  `quantize_rollout_fp4`；loader dispatch（fp4 拒收 fp8 recipe 名，并在 mutation 前用
  `nvfp4_available` 拒绝非 Blackwell CUDA target）；online storage
  bridge（fp8/fp4 同走 train-dtype master）；benchmark 加 fp4 列。
  `targeting.py` 保守地将 FP4 live swap 收窄为 MLP-only；旧 synthetic trajectory-product
  gate 已被证明不是这一选择的有效证据，但 attention/head 比 MLP 更直接影响策略分布，真实
  SDE gate 前不扩大覆盖。FP8 仍使用 attention+MLP profile。
  对齐规则：量化块沿 K 每 16 个值共享 scale；当前 packed `_scaled_mm` 额外要求
  `in_features % 32 == 0` 且 `out_features % 16 == 0`，不满足的 Linear 直接跳过
  （无粗粒度回退配方）。测试 `tests/nn/quantization/test_fp4.py`：CPU 量化数学
  （on-grid bit-exact 往返、独立 packed decoder、nearest-even midpoint、误差界）+
  swap 定位 + master/sync/dtype-move 契约 + loader/spec 接线，GPU parity（vs 独立
  去量化参照 rel<5e-3，本机实跑通过）。
  CUDA 热路径通过 Triton 将 1x16 block amax、E4M3 scale、nearest-even E2M1 pack 与
  cuBLAS scale swizzle 融合；CPU 版本只保留为可读 bit reference。production module
  benchmark 包含动态激活量化成本，FP4 MLP-only 几何平均 `2.33x` bf16；真实
  SD3.5 checkpoint compiled B=32 transformer forward 为 `1.386x`，production-size
  `1024x1024` `torch.compile(fullgraph=True)` 也通过。§3 的 GEMM-only scratch 是
  **one-shot feasibility artifact**，其答案已记录，不恢复为长期脚本；
  `quantized_linear_benchmark.py` 是**长期验收资产**，分别汇总 FP8/FP4 并让负结果
  返回失败。
- **P2 — 漂移验收门（IN PROGRESS）**：`quantized_rollout_drift_probe.py --scheme fp4`
  已走生产 `Fp4Linear` + 与 runtime 同源的 guard/TIS/RS 配置。35×256 个独立
  sample-step ratio_dev mean/max=`0.1077/0.8109`，production cap=2.0 下 TIS clip=`0%`、
  RS masked=`0%`、finite=true；35 步 ratio 乘积 mean/max=`0.562/3.902` 只保留为
  counterfactual，不进入 gate。live targeting 仍保守保持 **MLP-only FP4，attention/head/
  embedding 为 bf16**；下一门是在真实 diffusion recipe 上测 rollout 轨迹逐 timestep SDE
  logprob，不能继续用 synthetic categorical head 代替。
- **P3 — 端到端**：真 recipe（SD3.5 512p 或 cosmos 240p）rollout wall-clock 与
  reward 曲线 vs fp8/bf16 基线；`old==fresh` 契约按对称口径验证。

## 5. 验收

- P1：✅ swap 命中数 >0、weight-sync 后重量化 parity；包含动态激活量化的 MLP-only
  linear benchmark 几何平均 `2.33x`（门槛 `1.05x`）；真实 SD3.5 compiled transformer
  forward `1.386x`；production-size compile 通过。
- P2：MLP-only 真模型逐 timestep TIS 截断率 ≤ `10%`，RS masked fraction < `5%`，且
  finite=true；当前 full-policy categorical synthetic 既不是 SDE 模型，也不是 live target，
  不再承担通过/失败判决。
- P3：P1/P2 通过后，rollout wall-clock 实测下降（净目标 ≥1.5x vs bf16）且
  eval reward 曲线不退化。

## 6. 非目标

- **训练/replay 上 fp4**（§1 记录在案；B200/TE 前不重启）。
- reward/VAE fp4（挂起，见姊妹 sprint 与其 P0 门）。
- 不与稀疏注意力/compile 新变量同实验叠加（一次一变量）。

## 引用

- 插座：`vrl/config/precision.py`、`vrl/models/runtime_config.py:53`、
  `vrl/models/loader.py`、`vrl/generation/execution/worker.py:654`、
  `vrl/scripts/common/online.py:772`
- kernel 基座：`vrl/nn/quantization/fp8.py`（Fp8Linear：master/cache/重量化/drop_master）
- NVFP4 fused quantize：`vrl/nn/quantization/fp4_kernels.py`；CPU bit reference 与
  module ownership：`vrl/nn/quantization/fp4.py`
- 验收工具：`vrl/scripts/perf/quantized_linear_benchmark.py`、
  `vrl/scripts/perf/quantized_rollout_drift_probe.py`（均调用生产模块）
- 外部：FP4 Explore BF16 Train https://arxiv.org/abs/2604.06916 ；
  姊妹 sprint `SPRINT_fp4_off_policy_reward_vae.md`
