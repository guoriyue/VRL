# SPRINT: GEMM 利用率 / 吞吐优化（planned）

状态：proposed / planned。这份 sprint 回答一个具体问题——**rollout/train 前向里 GEMM 是最大切片（~48-51% kernel 时间），怎么把它做得更快**。结论先行：不是"写个更快的 matmul"（cuBLAS/CUTLASS 已接近峰值），而是**喂给 tensor core 形状更大、精度更省、更不碎片化的活**。所有数字来自本仓自己的 profiling（`docs/sprints/info/SPRINT_{rollout,cosmos,cross_model}_performance.md`）。

> 方法：对 3 份带 torch-profiler / Nsight / NCU trace 的 perf 文档做了交叉读取，并回到 `vrl/` 与已装 diffusers 0.37.1 逐条核实每个杠杆的代码现状（已做 / 一键开 / 真要写）。硬件经 `nvidia-smi` 实测确认。

---

## 1. 核心结论 (TL;DR)

**为什么 GEMM 比 attention 大——这是 DiT 的结构，不是 bug。** DiT 的 FLOP 压在 Linear 上：FFN/MLP 两个大 matmul（中间维 ~4× hidden）+ QKV 投影 + out-proj + AdaLN/modulation 投影。attention 的 `softmax(QK^T)V` 是 O(seq²)，**只有序列很长才占优**；你的负载序列短（cosmos 240p "序列太短 attention~0"；SD3.5 512px bf16 后 attention 仅 9%），所以 FFN 这类 GEMM 自然占 ~48-51%。**GEMM 大 = 在这个分辨率下就是 GEMM-bound 模型。**

**真正的问题不是"kernel 慢"，是"喂不饱"。** NCU 实测 GEMM `Compute(SM)` 只有 **42-58%**、SM occupancy **18-26%**，定性为"**小 shape / fragmented / launch-bound 指纹**"。cuBLAS/CUTLASS 的 GEMM kernel 本身已接近峰值——**所以"手写更快的 GEMM kernel"不在选项内**。能做的只有两类：

- **(A) 换更高的硬件 peak** → FP8/FP4。硬件实测是 **RTX 5090 / Blackwell `sm_120`（compute_cap 12.0）**，FP8/FP4 tensor core 在，但精度档还停在 bf16，**完全没用上**——这是唯一没开采的硬件 peak。
- **(B) 去碎片化 / 做大 M** → full-param 剔除 LoRA 瘦 GEMM、融合 QKV/gate_up 投影、做大 batch×tokens。这是大部分 headroom，且基本是配置/架构改动，不是写 kernel。

**现实天花板**：在"单卡 32GB + bf16 + 小 batch + diffusers shape"约束下，GEMM `Compute(SM)` 上限约 42-58%、occupancy 18-26%。文档自己反复说"**不是 5090 不行，是 PyTorch/diffusers 的 shape 和执行路径**"。突破这条线只有三选一——**FP8（换精度）/ 做大 M（被显存卡死）/ 去碎片化（全参+融合）**——没有一条免费。

---

## 2. 背景数字（来自本仓 profiling）

- **kernel 分布**：SD3.5 fp32 `GEMM 47.5% / Attention 34% / elementwise 12.8%`；bf16 后 attention path 几乎消失（fp32 attn 15.2% → 0.1% + flash 9.0%），GEMM 升到 nsys 42.8%，elementwise 42.3%。cosmos 训练 step `GEMM 250.4s/50.7% / elementwise 231.9s/46.9% / attention ~0%`。
- **GEMM Compute(SM)**：fp32 duration-weighted 51.2%（representative large 58.3%）；bf16 top tensorop GEMM **仅 42.2%**；batch sweep b8/16/24/32 **死钉 42-44%**；compiled WMMA weighted 15-31%。
- **占用**（GPM active 段）：SM occupancy 18-26%、tensor core 21-31%、DRAM 27-38%——"小 shape / fragmented kernel 指纹"。
- **launch-bound**：fp32 228k `cudaLaunchKernel`/epoch；cosmos 一个 step 发 1.3M+ kernel，训练 span 只有 ~64% 在忙。
- **elementwise 风暴 ~70% 来自 LoRA plumbing**：280 个 PEFT 层的双 mm（lora_A/lora_B 瘦 GEMM）+ fp32/bf16 cast + scaling mul，把 GEMM 之间塞满碎 kernel。
- **Amdahl**：cosmos 训练占 epoch 63%，所有调度类优化合计端到端上限 ~15%。

---

## 3. 杠杆清单——按"已做 / 一键开 / 真要写"排

| 杠杆 | 效果 | 工作量 | 代码现状 |
|---|---|---|---|
| **P0 逐-projection GEMM 拆分**（FFN vs QKV vs AdaLN vs out-proj 各几秒） | 不提速，但**决定 FP8/融合先打哪类 GEMM**——文档最细只到 `aten::addmm`，从没拆到 per-projection | 极小（trace 已有） | **空白，最该先做** |
| **full-param 替 LoRA**（sd3.5 / wan / predict2.5） | 干掉 ~47% elementwise + lora_A/lora_B 瘦 GEMM → 每个 linear 一个大 dense GEMM | 配置 `use_lora:false`（+显存/多卡） | 路径已有（`enable_full_finetune`）；**cosmos predict2 已是全参**，故只对仍在 LoRA 的家族有用 |
| **torch.compile 开在 cosmos** | 融合 elementwise epilogue、削 launch | 一行 `torch_compile.enable:true`（predict2_2b.yaml） | **全链路已接线（rollout + train），只是默认关**；但 `fullgraph=False` + grad-ckpt 挡住 CUDA-graph，launch-bound 下提升有限 |
| **融合 QKV / gate_up 投影** | 3 个瘦 GEMM 拼成 1 个大 GEMM，提 M/N 利用率 + 减 launch | 小（全参 SD3/Wan）；**Cosmos 无 fuse API 要自定义** | **vrl 从没调过** `fuse_qkv_projections()`（diffusers 有）；**与 LoRA 的 `target_modules=to_q/k/v` 冲突**（fuse 后变 `to_qkv`，旧 adapter 悬空）。文档当年判"低 ROI"是 fp32 时代仅凭"省一次 launch"，**今天 launch-bound 已确诊，值得重测** |
| **rollout 侧 merge LoRA → dense** | 35 步×CFG 的推理前向变单个大 GEMM | 真要写（merge/unmerge 要跟 colocated 训练 + 权重同步配合） | 只接了 `disable_adapter`（给 KL ref 用），**没有 `merge_adapter`** |
| **FP8/FP4 线性层**（torchao float8 / TransformerEngine） | **唯一没碰的硬件 peak，~2× bf16 throughput** | 大 + 风险高 | **完全没做**：`precision.py` 是封闭的 `(fp32,bf16,fp16)`；要新精度轴 + float8 替换 + dtype plumbing；**必须过 rollout-vs-train logprob parity 红线**（`trainer.py:603` 均差≤0.01）；DiT 的 modulation/最终投影**必须 FQN 过滤否则毁图**；调参尾巴是多天 |
| 加大 batch/分辨率（做大 M） | 直接对症 occupancy | 配置 | **做透了**：b8→32 GEMM 死钉 42-44%；32GB 容量墙先到（predict2 512p93f sbs=8 OOM）——**单卡不是解** |

---

## 4. 执行顺序（按杠杆排）

- [ ] **P0 — 逐-projection GEMM 拆分**。这是最便宜、信息量最大的一步：在现有 trace 上按 module FQN 把 `aten::addmm` / `aten::linear` 的耗时归类到 FFN / QKV / out-proj / AdaLN-modulation（或加一个按 module 名打标的 profiler 钩子）。不知道哪类 GEMM 占大头，谈 FP8 和融合都是盲打。**先做这个，再决定下面投哪。**
- [ ] **P1 —（仅 LoRA 家族 sd3.5/wan/predict2.5）full-param + `fuse_qkv_projections()`**。两者组合：`use_lora:false` 砍掉 ~47% LoRA 碎片化基数（每个 linear 回到单个大 dense GEMM），再对 SD3/Wan 调 `fuse_qkv_projections()` / `fuse_projections()` 把 QKV 拼成一个大 GEMM。这是比在已全参的 cosmos 上折腾更大的结构性收益。代价：显存 / 多卡。
- [ ] **P2 —（cosmos predict2，已全参）翻 `torch_compile.enable:true` 顺手测**。几乎免费、全链路已接线；但别期待大：launch-bound + `fullgraph=False` + grad-ckpt 决定了 compile 削不动 GEMM 的 `Compute(SM)`。当成"几乎零成本顺手做"，不是主攻。
- [ ] **P3 — FP8/FP4（放最后，真正的天花板）**。新增 fp8 精度轴 + float8 linear 替换（抄 cosmos-rl 的 fp8 path / scaling learnings Tier2 规划）。**前置门槛**：(a) 必须 A/B 过 logprob parity 红线（`trainer.py:603`，均差≤0.01）——rollout 与 train replay 的 fp8 数值要一致；(b) DiT 的 AdaLN/modulation/timestep/最终输出投影必须 FQN 过滤出 fp8，否则毁图；(c) 用 FID/样本质量 A/B，不能只看 loss。FP8 提的是 throughput 不是 occupancy%。

> P0 是无条件先做；P1/P2/P3 互相独立，可按显存与时间分别推进。P1 对 LoRA 家族、P2 对 cosmos、P3 对全部。

---

## 5. 约束 / 红线

- **logprob parity 红线**：`trainer.py:603-610` 一旦 rollout-vs-train 首步 log-prob 均差 >0.01 就报警并判 "GRPO ratios untrustworthy"（注释点名 Predict2 sigma bug 曾差到 ~115），外加 `precision_drift_guard`。**任何改变 rollout 前向数值的杠杆（FP8、融合后数值漂移、rollout 单独 merge LoRA）都必须保证两侧走等价数值路径。** 这就是为什么 SD3 在 rollout 和 train 两侧装同一个 attention processor、LoRA 包后再重装（`sd3_5/model.py:88,418,99,428`）。
- **单一调用点**：rollout `forward_step` 与 train `replay_forward` 共用 `common/backbone.py:152` 的 `_call_transformer → self.transformer(**kwargs)`——任何 fuse/compile/merge 作用在 `self.transformer` 上会**同时影响两侧**（既是便利也是 parity 约束）。
- **compile 限制**：`torch.compile(..., fullgraph=False)`；reduce-overhead/CUDAGraphs 与 PEFT LoRA + grad-ckpt 冲突（见项目记忆 Wan 那条），所以 compile 削 launch 但换不成 CUDA-graph 的 launch 消除。
- **容量墙**：单卡 32GB。做大 M 的路被 OOM 先堵（predict2 512p93f sbs=8 OOM，b16 在 VAE decode OOM 后靠 VAE tiling 回到 16）。

---

## 6. 非目标 (KEEP / 不做)

- **不手写 GEMM kernel**：cuBLAS/CUTLASS tensor-core GEMM 对给定 shape 已接近峰值；kernel 级唯一能提的只有降精度（FP8），不是重写 matmul。
- **不做 attention-kernel 类优化**：bf16 切换后 attention 已 ~0-9%（cosmos 240p ~0%），自定义 attention kernel 对这些配置无关（见 `docs/sprints/info/SPRINT_cosmos_performance.md`）。
- **不再刷 batch/分辨率求 occupancy**：已实测到头，单卡被容量墙卡死。
- **不重写 native diffusion transformer executor**：perf 文档已结论"这组数据不支持现在重写 native / custom Triton backend；上面都做完仍明显 under-utilized 再评估"。本 sprint 的 P1-P3 正是"上面"。

---

## 关键文件引用

**Profiling（证据，info/ 下的测量档，勿删）**
- `docs/sprints/info/SPRINT_rollout_performance.md:11-16,230-238,413-419,438,551-580,589-598,1138-1141` — kernel 分布 / GEMM Compute(SM) / batch sweep / compile / "fused QKV 暂不做"
- `docs/sprints/info/SPRINT_cosmos_performance.md:278-291,92-94,493` — cosmos 训练 GEMM 50.7%/elementwise 46.9%/attn~0、launch-bound、Amdahl
- `docs/sprints/info/SPRINT_cross_model_performance.md:22-24,88-97,116-120` — elementwise ~70% 来自 LoRA plumbing、occupancy 18-26%、predict2 512p OOM
- 硬件：`nvidia-smi` → RTX 5090, compute_cap 12.0 (Blackwell `sm_120`), 32GB

**代码现状（决定杠杆可用性）**
- `vrl/models/diffusion/common/backbone.py:152` — `_call_transformer: self.transformer(**kwargs)`（rollout+train 共用调用点）
- `vrl/models/diffusion/base.py:196-201` + `vrl/models/loader.py:117-124` — rollout 与 train 的 `torch.compile(..., fullgraph=False)`，同一 `model.torch_compile.enable` 门控
- `vrl/models/diffusion/cosmos/predict2/runtime.py:80-94,148-156` — `use_lora` 分支（`apply_lora` vs `enable_full_finetune`）+ compile gate
- `configs/model/diffusion/cosmos/predict2_2b.yaml` — "LoRA removed 2026-06-10: full-param"、`torch_compile.enable:false`
- `configs/model/diffusion/sd3_5/medium.yaml:14-30` — `use_lora:true`、`target_modules: to_k/to_q/to_v`、`torch_compile.enable:true`
- `vrl/config/precision.py:35` — `_CANONICAL=("fp32","bf16","fp16")`（无 fp8 档）
- `vrl/trainers/online/trainer.py:603-610,700-709` — logprob parity 红线 + `precision_drift_guard`
- `vrl/rollouts/evaluators/diffusion/sde_logprob.py:78,108-109` — train 走 `replay_forward`；KL ref 走 `disable_adapter()`（仅 toggle，非 merge）
- diffusers 0.37.1：`transformer_sd3.py:217 fuse_qkv_projections` / `transformer_wan.py:224 fuse_projections` / `attention.py:248 Attention.fuse_projections`（vrl 从未调用）；`transformer_cosmos.py:554 CosmosTransformer3DModel` 无 fuse 方法

**FP8 参考**
- `docs/sprints/reading/SPRINT_cosmos_rl_scaling_learnings.md:73-86,104` — FP8 Tier2 规划、torchao `convert_to_float8_training` rowwise、DiT modulation/最终投影必须 FQN 过滤、"省吞吐不省显存"、"数值调参尾巴多天"
- `docs/sprints/parked/SPRINT_low_precision_tis.md:24,97` — fp8 rollout 是激活 TIS 修正的触发条件
- cosmos-rl 本地源（`~/Desktop/cosmos-rl`）`vllm_rollout/monkey_patch_for_fp8.py` — 现成 fp8 path 可抄
