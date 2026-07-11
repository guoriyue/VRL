# SPRINT: NVFP4 rollout（fp4 生成 + bf16 训练重放）

状态：**planned，P0 硬件门已过（2026-07-11 本机实测，见 §3）**。性质：把已落地的
fp8-rollout 契约（lossy rollout + bf16 replay + TIS/drift guard）扩展到 NVFP4，
GEMM 裸上限 ~2x fp8 / ~3.4x bf16。下一道门是 §5 的 logprob 漂移验收——fp4 的
量化漂移比 fp8 大一个量级，TIS 截断率能不能接住是成败所在。

> 来由：外部 SOTA `FP4 Explore, BF16 Train`（arXiv:2604.06916）同款契约（FP4 只跑
> rollout、logprob 漂移交给重要性校正）报 ~2.5-3x rollout / 1.5-2x 端到端（H100 数，
> 5090 需本机重测——P0 已重测，见 §3）。姊妹 sprint
> `SPRINT_fp4_off_policy_reward_vae.md`（reward/VAE 上 FP4）**挂起**：reward ≈ rollout
> 的 14% 且已被 sleep-offload/异步隐藏，fp4 加速被 overlap 藏掉的时间端到端≈0；
> 等 VAE decode 占比实测（那边 P0）再议。

## 0. 一句话

rollout（denoise 生成）的大 GEMM 上 NVFP4，训练重放保持 bf16；fp8 rollout 的全部
seam（precision token、swap 机制、weight-sync 重量化、TIS/RS、drift guard）原样复用，
只差一个 `Fp4Linear` kernel 和它的漂移验收。

## 1. 训练侧 fp4 判定（记录，防止下次再议）

"native NVFP4 training"（前后向 GEMM 都 fp4）拆三种形态：

- **非对称（fp4 replay vs bf16 rollout）— 死**。fresh logprob 带 fp4 噪声污染 ratio。
  fp8 已实测单步 ratio_dev ~0.14、T=35 累积 0.87、cap=1.5 截断 1.2%；fp4 粗 4-8 倍，
  截断率不可接受。
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
| 入口 fail-loud 门（kernel 落地后拆） | ✅ | `vrl/scripts/common/online.py:772` |

**缺件**：`Fp4Linear`（`vrl/nn/quantization/`，`Fp8Linear` 同构：bf16 master +
非持久 fp4 cache，`_load_from_state_dict` 重量化，`drop_master`）+ 各家 model 的
`quantize_rollout_fp4` + loader dispatch 分支 + benchmark/drift probe 的 fp4 支路。

## 3. P0 硬件门结果（2026-07-11，5090 sm_120，torch 2.11.0+cu130）

`torch._scaled_mm` 的 nvfp4 路径（`float4_e2m1fn_x2` 操作数 + e4m3 1x16 block scale
**blocked-swizzled 布局** + fp32 两级 tensor scale）**原生可跑**：

| shape (M,K,N) | bf16 ms | fp8 ms | fp4 ms | fp4/bf16 | 输出漂移(rel L1) |
|---|---|---|---|---|---|
| (4096,2048,8192) | 0.912 | 0.621 | 0.268 | **3.40x** | 0.134 |
| (4096,2048,2048) | 0.226 | 0.175 | 0.083 | **2.72x** | 0.134 |
| (8192,2048,6144) | 1.561 | 0.784 | 0.428 | **3.64x** | 0.134 |

- 数字是 **GEMM 裸时延**（激活+权重预量化）；生产要像 Fp8Linear 一样每 forward 动态
  量化激活，净加速会低于此上限。
- 漂移 0.134 是朴素 RTN、随机数据的数字——**比 fp8 tensorwise 大一个量级**，这正是
  §5 验收门要在真 logprob 路径上量的。
- 关键 API 知识（P1 直接用）：scale 必须过 cuBLAS blocked swizzle
  （128-row × 4-col tile，`view(nrb,128,ncb,4).permute(0,2,1,3).reshape(-1,4,32,4)
  .transpose(1,2).reshape(-1,32,16).flatten()`）；两级缩放 tensor_scale =
  amax/(448·6)，block_scale(e4m3) = block_amax/6/tensor_scale，输出乘回
  tensor_scale_a·tensor_scale_b。probe 脚本为一次性 scratch（未入库），配方以此段
  为准。

## 4. Phase plan

- **P1 — `Fp4Linear` + swap**：nvfp4 两级缩放 + `_scaled_mm`；`quantize_rollout_fp4`
  按 `quantize_rollout_fp8` 同构；loader dispatch；拆 online.py fail-loud 门；
  `fp8_linear_benchmark.py` 加 fp4 支路（量含激活量化开销的净加速）。
- **P2 — 漂移验收门（成败门）**：`fp8_rollout_drift_probe.py` 加 fp4 支路——单步
  ratio_dev、T=35 累积、TIS cap=1.5 截断率。fp8 基线：0.14 / 0.87 / 1.2%。fp4 若
  截断率 >~10% 或 guard 灾难档触发 → 回退：只量化部分层（MLP 上 fp4、attention 保
  fp8/bf16）或 1x16 改保守 exclude 清单，再不行判 NEGATIVE 归档。
- **P3 — 端到端**：真 recipe（SD3.5 512p 或 cosmos 240p）rollout wall-clock 与
  reward 曲线 vs fp8/bf16 基线；`old==fresh` 契约按对称口径验证。

## 5. 验收

- P1：swap 命中数 >0、weight-sync 后重量化 parity、compile 兼容性结论（fp8 blockwise
  的 graph-break 前科 → fp4 kernel 同样要测 compile 交互，不行就 loader guard 拒绝组合）。
- P2：TIS 截断率 ≤ 目标带（对齐 fp8 的 1.2% 量级为佳，上限 ~10%）；drift guard
  auto 档不触发灾难判定。
- P3：rollout wall-clock 实测下降（GEMM 裸上限 2.7-3.6x，净目标 ≥1.5x vs bf16）且
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
- 验收工具：`vrl/scripts/perf/fp8_linear_benchmark.py`、
  `vrl/scripts/perf/fp8_rollout_drift_probe.py`（均已改调生产模块，加支路即可）
- 外部：FP4 Explore BF16 Train https://arxiv.org/abs/2604.06916 ；
  姊妹 sprint `SPRINT_fp4_off_policy_reward_vae.md`
