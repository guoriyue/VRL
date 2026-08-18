# SPRINT: GEMM 利用率 / 吞吐优化（done）

状态：**done（profiler + QKV A/B + torch.compile 已落地，2026-06-18 归档至 done/；P1.5 全参替换 + P3 FP8 已拆分到 [[SPRINT_fullparam_and_fp8_precision]]）**。P0 逐-projection GEMM profiler（f446776，gemm_projection_breakdown.py）+ P1 QKV 融合 A/B（实测低 ROI，--fuse-qkv 仅留 profiler、不落 runtime）+ P2 torch.compile（31f6843，predict2_2b.yaml torch_compile.enable:true，1.37×/1.25×）均已完成。剩 P1.5（LoRA 家族 sd3.5/wan use_lora:false 全参替换，待显存/多卡）+ P2 logprob drift guard 下次真实 run 复绿；P3 FP8 按用户 2026-06-14 决定暂缓。

> 方法：对 3 份带 torch-profiler / Nsight / NCU trace 的 perf 文档做了交叉读取，并回到 `vrl/` 与已装 diffusers 0.37.1 逐条核实每个杠杆的代码现状（已做 / 一键开 / 真要写）。硬件经 `nvidia-smi` 实测确认。

---

## 1. 核心结论 (TL;DR)

**为什么 GEMM 比 attention 大——这是 DiT 的结构，不是 bug。** DiT 的 FLOP 压在 Linear 上：FFN/MLP 两个大 matmul（中间维 ~4× hidden）+ QKV 投影 + out-proj + AdaLN/modulation 投影。attention 的 `softmax(QK^T)V` 是 O(seq²)，**只有序列很长才占优**；你的负载序列短（cosmos 240p "序列太短 attention~0"；SD3.5 512px bf16 后 attention 仅 9%），所以 FFN 这类 GEMM 自然占 ~48-51%。**GEMM 大 = 在这个分辨率下就是 GEMM-bound 模型。**

**Corrected interpretation (2026-07-12): low occupancy is not proof that the
GPU is under-fed.** Tensor-core GEMMs commonly have low occupancy because their
register and shared-memory use limits resident warps. NCU tensor-pipe SOL must be
compared with a same-machine square-GEMM baseline; that comparison, not the
`18-26%` occupancy value, supports the conclusion that the main GEMMs are near
the achievable kernel ceiling. See
`docs/sprints/done/SPRINT_gpu_saturation_and_colocation_decision.md`.

The measured optimization classes were:

- **(A) 换更高的硬件 peak** → FP8/FP4。硬件实测是 **RTX 5090 / Blackwell `sm_120`（compute_cap 12.0）**，FP8/FP4 tensor core 在，但精度档还停在 bf16，**完全没用上**——这是唯一没开采的硬件 peak。**注：FP8 已按用户决定暂缓（2026-06-14）**；P0 显示它本可安全覆盖 ~95% 的 GEMM，分析保留在 §4 P3 供日后复活。代价是最大的 FFN（~50%）在不上 FP8 时没有杠杆。
- **(B) Reduce fragmentation or change shape** → full-param removes thin LoRA
  GEMMs, compile fuses launch-heavy pointwise work, and batching changes M. Each
  lever needs its own throughput and memory A/B; occupancy does not predict its
  gain.

**Evidence boundary:** the reported `42-58%` NCU `Compute(SM)` and `18-26%`
occupancy values are descriptive measurements, not universal ceilings. The
same-machine GEMM comparison and the actual compile/QKV/batch A/B results decide
the conclusions for the tested shapes. FP8, larger M, and fragmentation reduction
remain distinct levers with different correctness and memory costs.

---

## 2. 背景数字（来自本仓 profiling）

- **kernel 分布**：SD3.5 fp32 `GEMM 47.5% / Attention 34% / elementwise 12.8%`；bf16 后 attention path 几乎消失（fp32 attn 15.2% → 0.1% + flash 9.0%），GEMM 升到 nsys 42.8%，elementwise 42.3%。cosmos 训练 step `GEMM 250.4s/50.7% / elementwise 231.9s/46.9% / attention ~0%`。
- **GEMM Compute(SM)**：fp32 duration-weighted 51.2%（representative large 58.3%）；bf16 top tensorop GEMM **仅 42.2%**；batch sweep b8/16/24/32 **死钉 42-44%**；compiled WMMA weighted 15-31%。
- **Active-window GPM counters**: SM occupancy 18-26%, tensor activity 21-31%,
  and DRAM activity 27-38%. These values describe the run and do not establish
  recoverable throughput.
- **launch-bound**：fp32 228k `cudaLaunchKernel`/epoch；cosmos 一个 step 发 1.3M+ kernel，训练 span 只有 ~64% 在忙。
- **elementwise 风暴 ~70% 来自 LoRA plumbing**：280 个 PEFT 层的双 mm（lora_A/lora_B 瘦 GEMM）+ fp32/bf16 cast + scaling mul，把 GEMM 之间塞满碎 kernel。
- **Amdahl**：cosmos 训练占 epoch 63%，所有调度类优化合计端到端上限 ~15%。

---

## P0 实测结果（2026-06-14，bf16，RTX 5090，已完成）

用 `vrl/scripts/perf/gemm_projection_breakdown.py`（config-init 随机权重 + 生产维度；GEMM 时间由 shape 决定，故分布忠实）实测 per-projection GEMM **CUDA self-time** 占比（batch=2 = cond+uncond，eager，LoRA off，单次前向；wan 用 12 层、生产宽度，比例与层数无关）：

| 类别 | SD3.5-medium | Cosmos Predict2 | Wan 2.1 |
|---|---|---|---|
| **ffn** | **58.1%** | **47.9%** | **42.9%** |
| **qkv** | 25.8% | 32.9% | 39.9% |
| **attn_out** | 11.3% | 14.9% | 16.3% |
| adaln（modulation） | 4.2% | 4.2% | 0%* |
| io_embed | 0.6% | 0.1% | 0.9% |

\*Wan modulation 是 `scale_shift_table` 参数表（非 Linear），不计入 GEMM——符合预期。三家 `other` 桶全空（覆盖完整）。

三个关键结论：

1. **FFN 是单一最大 GEMM 类（43–58%），且无法融合。** SD3/cosmos 的 FFN 是 `net.0.proj`（升维）→`net.2`（降维）**串联**，不是并联，拼不成一个 GEMM——它的体量是固有的。
2. **QKV 第二大（26–40%），是并联投影，可融合。** `fuse_qkv_projections()` 把 to_q/k/v 拼成一个宽 GEMM（self-attn 三个全可融；cross-attn 只能融 k+v，q 输入不同）。**这是结构性融合（P1）唯一有意义的靶子。** attn_out（11–16%）单个 out-proj 无法融。
3. **数值最敏感的部分恰好最便宜：adaln + io_embed 合计仅 ~5%（wan <1%）。** 这意味着量化时把这 5% 留在高精度、对吞吐几乎零损失。

### P1 验证：QKV 融合 A/B（实测 2026-06-14，确认低 ROI，不落地）

用 `--fuse-qkv`（`model.fuse_qkv_projections()`）在同一生产维度模型上做 fuse off vs on 的 A/B（bf16，GPU，active=10）：

| | qkv GEMM 时间 | 总 GEMM | qkv launch 次数 | 数值正确性 |
|---|---|---|---|---|
| **SD3.5** | 64,378 → 58,438 µs（**−9.2%**） | 249,029 → 244,500 µs（**−1.8%**） | 2880 → 960（3× 更少） | 等价（rel diff 0.6%，bf16 噪声） |
| **Wan 2.1** | 336,283 → 345,926 µs（**+2.9%**，更差） | 847,680 → 857,377 µs（+1.1%） | 1440 → 720（2× 更少） | 等价（rel diff 0.55%） |
| **Cosmos** | — | — | — | 无 `fuse_qkv_projections()`，不支持 |

**结论：融合数值正确，但收益边际、且方向不一致**——SD3.5 因 QKV 投影小（hidden 1536）属 utilization-bound，融成大 GEMM 省 ~9% qkv / ~2% 总 GEMM；Wan 的 QKV 已经宽（hidden 5120）属 compute-bound，融合不提利用率，反而略差。Cosmos（flagship）根本无 public fuse API。**这实测证实了 perf 文档旧"低 ROI"判断**。叠加约束（需 full-param、破 LoRA `target_modules=to_q/k/v`、最大的 FFN 完全不受影响），**不值得为 ~2%/0% 在 runtime 里加这个 config-gated 特性**——故不落地，仅在 profiler 里保留 `--fuse-qkv` 作为度量手段。

> **由此得到的 sprint 总结论（FP8 暂缓前提下）：GEMM 已基本到这套栈的实践地板。** 最大的 FFN（43–58%）唯一杠杆是 FP8（已暂缓）；QKV 融合实测只值 ~2%（SD3.5）/0%（Wan）/不支持（Cosmos）；adaln/io <5% 不值得碰。剩下能动的都不在"GEMM kernel 本身"：full-param 削 LoRA 碎片化、compile 削 launch/elementwise——它们改善的是 GEMM *之间* 的开销，不是 GEMM 本身。**compile 那条实测见下方 P2，收益比先验大。**

---

## P2 实测结果（2026-06-15，bf16，RTX 5090，已完成）

翻 `torch_compile.enable:true`（`torch.compile(transformer, mode=default, fullgraph=False)`，与 `base.py:200` 运行时调用一字不差）在 cosmos-predict2 上做的 compile A/B。复用 P0 同一套 config-init synthetic transformer（28 层生产维度，batch=2=cond+uncond；compile 的融合/guard 只看 op 图与 shape，不看权重值，故 config-init 忠实，无需下 checkpoint），量两条与真实代码对齐的路径：**rollout**（forward-only `no_grad`，35 步×CFG denoise 就是这个前向的重复）和 **train**（forward+backward，开 gradient checkpointing，对齐 `cosmos/train.py:131` 默认 True）。工具：`vrl/scripts/perf/compile_benchmark.py`。

| 路径 | compile | 每步延迟 | 每步 kernel launch | 峰值显存 | 加速 |
|---|---|---|---|---|---|
| **rollout**（denoise forward） | off | 59.9 ms | 2763 | 3891 MB | — |
| | **on** | **43.6 ms** | **947** | 3925 MB | **1.37×** |
| **train**（fwd+bwd, grad-ckpt） | off | 225.2 ms | 10098 | 7778 MB | — |
| | **on** | **179.8 ms** | **3838** | 7789 MB | **1.25×** |

三个结论，且**修正了本 sprint 的悲观先验（"别期待大 / 提升有限"）：**

1. **compile 是真·收益。** rollout 1.37×、train 1.25×，不是噪声。来源是 launch 数被砍 ~2.6–2.9×（rollout 2763→947、train 10098→3838）——inductor 把成串 elementwise epilogue 融进 Triton kernel。先验"换不成 CUDA-graph 的 launch 消除 → 提升有限"低估了这套负载有多 launch-bound：一次 rollout forward 就发 2763 个 kernel，**即便没 CUDA-graph，光靠 fusion 削 launch 就值 1.25–1.37×**。
2. **显存几乎不变**（rollout +34 MB、train +11 MB）——compile 不吃容量预算，不与 §5 的 32GB 容量墙冲突。
3. **代价确实接近零**：一行 config flag，全链路已接线；唯一一次性成本是首次编译的 graph capture，RL 定形状、跨 epoch 摊销到 ~0、无 recompile。

**caveat（诚实记下）：**
- **grid 依赖**：实测用 cosmos T2I backbone 维度 + 中等视频 latent grid（t=1, 44×80）。满长视频（93f）序列长约两个量级 → 每层 compute 占比上升 → launch-bound 比例下降 → **compile 加速会收窄**（短 clip/图像条件接近实测，满长视频更接近 ~1.05–1.1×）。compile 对定形状前向最差中性、不会变负，故"默认开"安全。
- **parity 红线未现场确认**：compile 同时作用在 rollout 与 train 的 `self.transformer`（同一调用点 `backbone.py:152`、同 mode），两侧走等价编译图 → 按构造 drift 不应变化（§5）。但 `trainer.py:603` 的 logprob drift guard 仍需在下次真实 cosmos run 上确认保持绿，才算完全收尾。
- dynamo 对 diffusers `attention_dispatch` 的 `functools.lru_cache` + grad-ckpt `_gradient_checkpointing_func` 报 "potential silent incorrectness（未观察到）"——观察项，非阻塞。

**决策：`configs/model/diffusion/cosmos/predict2_2b.yaml` 翻 `torch_compile.enable: true`（已落地 2026-06-15）。** 下次真实 cosmos run 确认 drift guard 绿即完全收尾。

### CUDA graph（reduce-overhead）+ 跨家族对比（实测 2026-06-15）

回答本 sprint 一直悬着的"CUDA graph 能不能再救一截"，以及"为什么 cosmos 比 sd3.5 提升大"。同一 `compile_benchmark.py`，加 `--mode reduce-overhead`（torch 的 CUDA-graph 路径）与 `--family sd3_5`：

| | cosmos rollout | cosmos train | sd3.5 rollout | sd3.5 train |
|---|---|---|---|---|
| eager launch/步 | 2763 | 10098 | 1317 | 5300 |
| **default**（fusion） | **1.37×**（947） | **1.25×**（3838） | **1.16×**（673） | **1.11×**（3789） |
| **reduce-overhead**（CUDA graph） | 1.36×（**1**） | 1.35×（**10**） | — | — |

（sd3.5 为**不带 LoRA** 的 apples-to-apples；speedup 各对自家 eager。）

**结论 1：CUDA graph 没被"挡住"，但也不值钱——可以划掉。** `reduce-overhead` 在 cosmos 全参上跑通了，把整张前向录成图、单次重放，launch 直接归零（rollout 947→**1**、train 10098→**10**，grad-ckpt 把图切成 ~10 段）。**但 rollout 延迟 44.6 vs default 43.6 ms 没差**——fusion 之后这步已是 GEMM `Compute(SM)` bound，残余 launch 早被藏在 kernel 执行后面，消掉=零收益。这反过来实证了本 sprint "compile 削不动 `Compute(SM)`"。train 侧 reduce-overhead 1.35× vs default 1.25×（多 ~8%，因 grad-ckpt 在 backward 重算前向、更 launch-bound），但 grad-ckpt+CUDA-graph 数值正确性没验（dynamo 自警 silent incorrectness），为 8% 担风险不值。**净：留 `mode=default`，CUDA graph 不做。**

**结论 2：compile 帮所有家族（程度不同），cosmos 最大——是可融碎片多寡，非机制不同。** 三家走同一套 `torch.compile(mode=default, fullgraph=False)`（cosmos/sd3.5 经 `base.py:200`；wan 经自家 `wan_2_1/model.py:217` 遍历 expert，flags 相同），都是 inductor elementwise fusion：

| family（同一套 compile） | rollout | train | eager launch/步 |
|---|---|---|---|
| **cosmos**（28 层，全参） | **1.37×** | **1.25×** | 2763 |
| **sd3.5**（24 层，no-LoRA） | 1.16× | 1.11× | 1317 |
| **wan 2.1**（12 层，no-LoRA） | 1.14× | 1.10× | 818 |

cosmos 最大是因为 eager 碎得多（层更深 + AdaLN-LoRA modulation 每块多串小 linear，P0 的 156 个 adaln 调用，+ 3D 视频 patchify/rope）→ 可融多。**三家全是正收益、compile 不会把任何一家变慢**。注意全是 **no-LoRA**；真实 sd3.5/wan 带 LoRA（§ "~70% elementwise 来自 280 PEFT 层"）→ 可融更多 → 真实收益更高，这正是它们部分配置默认 `enable:true` 的原因。

> **关键反混淆**：P1 表里"wan 融合后 +1.1% 更慢 / sd3.5 −1.8% 几乎没动"是 **QKV 融合**（`fuse_qkv_projections()`，把 to_q/k/v 拼成一个宽 GEMM——改 GEMM 形状），**不是 compile**。两者机制无关：compile 削 launch（帮 wan 1.14×），QKV 融合改 GEMM 宽度（wan 已经够宽 → compute-bound → 融了反而 +1.1%）。**wan 慢只在 QKV 融合那条，compile 这条 wan 是快的。**

#### 复测更正（2026-08-16，同一 RTX 5090、同一 `compile_benchmark.py`）

上面「CUDA graph 跑通但零收益」的结论**方向对、幅度错**。当时只测了 cosmos 的
`reduce-overhead` 一条腿（44.6 vs 43.6ms，判为「没差」）。这次把 4 个 family 的
两条腿都测全，直接比**两个 compiled 臂**（eager baseline 在 GPU 竞争下会漂，
compiled 臂才是生产实际跑的东西）：

| family | fusion (default) | CUDA graph (reduce-overhead) | graph 相对 fusion |
|---|---|---|---|
| sd3_5（24 层） | 29.37 ms | 28.35 ms | **+3.5%** |
| wan_2_1（12 层） | 174.94 ms | 173.43 ms | +0.9% |
| **cosmos-predict2（28 层）** | **50.61 ms** | **80.01 ms** | **−58%（大幅变慢）** |
| **cosmos-predict2.5（28 层）** | **40.57 ms** | **76.19 ms** | **−88%（大幅变慢）** |

四个 family 的 `reduce-overhead` 都把 launch 打到 **1**，捕获全部成功。但：

- **cosmos 两个家族上 CUDA graph 比 fusion 慢 1.6–1.9 倍。** 旧记录的「没差」
  低估了代价 —— `reduce-overhead` 为了保证 replay 安全会插入额外的输入拷贝，
  而 cosmos 的 fusion 收益本来就最大（本次复测 default 2.06×/2.47×，高于
  2026-06 记录的 1.37×），graph 把这部分吃掉了。
- **sd3_5 / wan 上是微小正收益（+3.5% / +0.9%）**，不是零。所以
  「CUDA graph 对 diffusion 一律无用」这个说法**不能外推**，它是 family-specific 的。

**净结论不变（不做 CUDA graph），但理由要换**：不是「消 launch 零收益」，而是
**收益 family 相反且最大受益家族反受其害**，同时安全前提未满足（见下）。

#### 为什么即使收益为正也不能开（2026-08-16 审计，旧 sprint 未查）

旧 sprint 只问了「快不快」，没问「对不对」。CUDA graph replay 的三条硬前提：

| 前提 | 判定 | 证据 |
|---|---|---|
| 权重地址稳定 | ✅ 满足 | `load_state_dict` 原地 copy（`weight_utils.py:65`），版本化 slot 同样是 copy 语义；全仓 `assign=True` 零命中 |
| graph 活过 sleep/wake | ❌ **硬阻断** | parking 强制 `empty_cache()`（`cuda_memory.py:300-305`）且残留 >256MiB 硬报错（`cuda_memory.py:25-27`）。CuMem 的「virtual addresses stay valid」只覆盖 `pool.building()` 作用域内的分配，而 graph 是首次 warmup forward 才录的，那时 building 已关闭 → graph pool 来自 torch caching allocator，拿不到保护。代码里**没有任何 graph 失效钩子**（`graph_pool` / `cuda.graph` / `make_graphed` 在 `vrl/` 下零命中） |
| 捕获区内无 host 控制流 | ❌ 违反 | TeaCache 的 `.item()`（`teacache.py:59`）决定 forward 跑不跑；`disable_adapter()` 的 reference 分支在同一 module 上跑两种结构的 forward |

第三条已经在生产里踩过并留了注释：

```yaml
# Disable torch.compile on the transformer — DPO's twin policy/ref forwards
# under no_grad collide with reduce-overhead CUDA graphs.
```
`vrl/config/presets/experiment/wan_2_1/offline_dpo_pickapic.yaml:36-37`

**要开 CUDA graph，必须先给 `WorkerMemoryParking.sleep()` 加 graph 失效钩子** ——
那部分代码今天不存在。为 sd3_5 的 +3.5% 写这个不划算。

**结论 3：vLLM/SGLang 靠 CUDA graph 吃饭、我们不行的根因。** 它们录的是 decode（每步 1 token = memory-bound 小 GEMV，时间几乎全是 launch 开销）→ CUDA graph 1.5–2×；做法是对 bucket 过的 batch size 各录一张图、运行时 pad 到最近桶再重放、变长 paged attention 走 piecewise eager。**我们每步是大块 DiT GEMM（compute-bound），launch 只占一小条** → 同样消 launch，他们赚 2×、我们赚 0。CUDA graph 只在"极低分辨率 / 极短序列"这种真正 launch-bound 的区间才对扩散有意义，生产视频/图像分辨率不在其中。

---

## 3. 杠杆清单——按"已做 / 一键开 / 真要写"排

| 杠杆 | 效果 | 工作量 | 代码现状 |
|---|---|---|---|
| **P0 逐-projection GEMM 拆分**（FFN vs QKV vs AdaLN vs out-proj 各几秒） | 不提速，但**决定 FP8/融合先打哪类 GEMM**——文档最细只到 `aten::addmm`，从没拆到 per-projection | 极小（trace 已有） | **空白，最该先做** |
| **full-param 替 LoRA**（sd3.5 / wan / predict2.5）**【rollout-only 收益，训练段慢 26%，见 §4 P1.5 的补测】** | 干掉 ~47% elementwise + lora_A/lora_B 瘦 GEMM → 每个 linear 一个大 dense GEMM | 配置 `use_lora:false`（+显存/多卡） | 路径已有（`enable_full_finetune`）；**cosmos predict2 已是全参**，故只对仍在 LoRA 的家族有用 |
| **torch.compile 开在 cosmos**【已实测 2026-06-15，落地】 | 融合 elementwise epilogue、削 launch → **实测 1.37× rollout / 1.25× train**（launch 数砍 2.6–2.9×） | 一行 `torch_compile.enable:true`（predict2_2b.yaml） | **已落地默认开**；全链路已接线（rollout + train）。纯 inductor fusion 削 launch 已值 1.25–1.37×——比先验"提升有限"大（见 P2 实测）。（注："`fullgraph=False` + grad-ckpt 挡住 CUDA-graph" 是**已被推翻的旧说法**，捕获其实能跑通；见 P2 的 2026-08-16 复测更正） |
| **融合 QKV 投影**【已实测 2026-06-14，低 ROI】 | 3 个瘦 GEMM 拼成 1 个大 GEMM，减 launch | 小（全参 SD3/Wan）；**Cosmos 无 fuse API** | **实测确认低 ROI**：SD3.5 −2% 总 GEMM、Wan ~0、Cosmos 不支持（无 `fuse_qkv_projections()`）。数值等价（rel<0.6%），但需 full-param（破 LoRA `target_modules=to_q/k/v`）、不碰 FFN。**不落地 runtime**；`--fuse-qkv` 留在 profiler 作度量 |
| **rollout 侧 merge LoRA → dense** | 35 步×CFG 的推理前向变单个大 GEMM | 真要写（merge/unmerge 要跟 colocated 训练 + 权重同步配合） | 只接了 `disable_adapter`（给 KL ref 用），**没有 `merge_adapter`** |
| **FP8/FP4 线性层**（torchao float8 / TransformerEngine）**【暂缓 2026-06-14】** | **唯一没碰的硬件 peak，~2× bf16 throughput**；P0 证实可安全覆盖 ~95% GEMM | 大 + 风险高 | **用户暂缓，未做**：`precision.py` 是封闭的 `(fp32,bf16,fp16)`；要新精度轴 + float8 替换 + dtype plumbing；**必须过 rollout-vs-train logprob parity 红线**（`trainer.py:603` 均差≤0.01）；DiT 的 modulation/最终投影**必须 FQN 过滤否则毁图**；调参尾巴是多天 |
| Larger batch/resolution (larger M) | Changes shape and amortization; requires throughput/memory A/B | Config | Tested through b8→32: GEMM `Compute(SM)` stayed 42-44% and the 32 GB capacity limit arrived first (predict2 512p93f sbs=8 OOM); occupancy is not the decision metric |

---

## 4. 执行顺序（按杠杆排）

- [x] **P0 — 逐-projection GEMM 拆分（已完成 2026-06-14，见上方"P0 实测结果"）。** 结论：FFN 43–58%（不可融，只能降精度）、QKV 26–40%（可融）、attn_out 11–16%、敏感的 adaln+io 仅 ~5%。

> **范围决定（2026-06-14）：FP8 暂不做（用户决定）。** 直接后果要诚实记下：P0 显示**最大的那块 FFN（43–58%）在不上 FP8 时没有任何杠杆**——它不可融合、cuBLAS/CUTLASS 已接近峰值、加 batch 推不动 `Compute(SM)`。所以**不上 FP8 的可寻址范围基本只剩 QKV（融合）+ LoRA 碎片化（全参/merge）+ launch overhead（compile）**，FFN 那 ~50% 的体量维持现状。换句话说：非-FP8 路径的天花板明显低于含-FP8 路径，这是接受 FP8 暂缓的代价。

非-FP8 活跃路径（按杠杆排）：

- [x] **P1 — QKV 融合：已实测，确认低 ROI，不落地（见上方"P1 验证"）。** SD3.5 ~−2% 总 GEMM、Wan ~0、Cosmos 不支持；需 full-param、破 LoRA targeting、不碰 FFN。数值正确但收益不抵复杂度——**不在 runtime 加这个特性**，`--fuse-qkv` 仅留在 profiler 作度量。
> **⚠️ 2026-08-17 补测 + break-even 收口：P1.5 作为速度杠杆已死，无生产
> lane 能到 break-even。** rollout 侧增益有硬上界 = 折叠曲线（480p 2.7%），
> 训练侧代价 c ∈ [10%, 26%]，结构比 R = CFG/(3·tf) ≤ 2.67 < break-even
> 3.7–9.6。完整推导：`info/SPRINT_train_phase_gap_hunt.md` §2.1。
> 保留 P1.5 的唯一理由是质量/容量，不是速度。
> 本条只算了 kernel 数（LoRA 909 vs 全参 440 每次 replay 迭代，方向正确），
> 但 wall time 相反：**全参训练步 43.01 ms vs LoRA 34.19 ms，慢 26%**。
> 两个原因叠加——反向要为 340M 参数算梯度而不是 4.7M（replay 慢 9%）；
> optimizer/clip/EMA 要扫 340M 而不是 4.7M（非-replay 占比 0.9% → 14%）。
> **动手前先量该 run 的 rollout/训练时间配比**，不要当作无条件杠杆。
> 数据：`info/SPRINT_train_phase_gap_hunt.md` §2。

- [ ] **P1.5（仅 LoRA 家族 sd3.5/wan/predict2.5）— full-param 替 LoRA**：`use_lora:false`，砍掉 ~47% 的 LoRA elementwise + lora_A/lora_B 瘦 GEMM 碎片化。注意这改善的是 GEMM *之间* 的碎片化/elementwise，不是 GEMM kernel 本身。代价：显存 / 多卡。cosmos predict2 已是全参。**这是非-FP8 路径里剩下最大的一条**，但前提是显存/多卡可用。
- [x] **P2 —（cosmos predict2，已全参）翻 `torch_compile.enable:true`：已实测 + 落地（见上方"P2 实测结果"）。** 结果比先验好：rollout 1.37× / train 1.25×，靠 launch 数砍 2.6–2.9×（fullgraph=False 削不动 `Compute(SM)` 是对的，但削 launch 这条本身就值钱）。显存不变。config 默认已翻 true，唯一未结：下次真实 run 确认 logprob drift guard 绿。

- [ ] **P3 — FP8/FP4（暂缓，用户决定 2026-06-14；保留分析供日后复活）**。P0 实测**强化**了 FP8 的吸引力：可全量 FP8 的 FFN+QKV+attn_out ≈ **95%**（wan 99%），而必须 FQN 过滤留高精度的 adaln/timestep/最终投影**只占 ~5%**——即"最大覆盖 + 最低数值风险"。日后若重启：新增 fp8 精度轴 + float8 linear 替换（抄 cosmos-rl `vllm_rollout/monkey_patch_for_fp8.py` / scaling learnings Tier2）；**前置门槛**：(a) A/B 过 logprob parity 红线（`trainer.py:603`，均差≤0.01）；(b) adaln/modulation/timestep/最终输出投影 FQN 过滤出 fp8；(c) 用 FID/样本质量 A/B，不只看 loss。FP8 提的是 throughput 不是 occupancy%。

> P0 / P1 / **P2 已完成**（P1 低 ROI 不落地；P2 实测 1.25–1.37×、已落地默认开）。当前活跃只剩 P1.5（LoRA→全参，需显存/多卡）。P3 FP8 按用户决定暂缓。

---

## 5. 约束 / 红线

- **logprob parity 红线**：`trainer.py:603-610` 一旦 rollout-vs-train 首步 log-prob 均差 >0.01 就报警并判 "GRPO ratios untrustworthy"（注释点名 Predict2 sigma bug 曾差到 ~115），外加 `precision_drift_guard`。**任何改变 rollout 前向数值的杠杆（FP8、融合后数值漂移、rollout 单独 merge LoRA）都必须保证两侧走等价数值路径。** 这就是为什么 SD3 在 rollout 和 train 两侧装同一个 attention processor、LoRA 包后再重装（`sd3_5/model.py:88,418,99,428`）。
- **单一调用点**：rollout `forward_step` 与 train `replay_forward` 共用 `common/backbone.py:152` 的 `_call_transformer → self.transformer(**kwargs)`——任何 fuse/compile/merge 作用在 `self.transformer` 上会**同时影响两侧**（既是便利也是 parity 约束）。
- **compile 限制**（已被 P2 CUDA graph A/B 修正）：运行时用 `torch.compile(..., mode=default, fullgraph=False)`。`reduce-overhead`（CUDA-graph）实测**在 cosmos 全参上能跑通、把 launch 消到 1**——并非被 PEFT LoRA / grad-ckpt 完全挡住（旧说法过强）。真正不用它的原因（**2026-08-16 复测更正**）：收益 family 相反 —— sd3_5/wan 上是 +3.5%/+0.9% 的微小正收益，但 **cosmos 两个家族上比 fusion 慢 1.6–1.9 倍**；且 graph 活不过 worker 的 sleep/wake（parking 强制 `empty_cache`，无失效钩子）。详见上方"P2 实测结果 → CUDA graph A/B → 复测更正"。
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
- `docs/sprints/done/SPRINT_low_precision_tis.md:24,97` — fp8 rollout 是激活 TIS 修正的触发条件
- cosmos-rl 本地源（`~/Desktop/cosmos-rl`）`vllm_rollout/monkey_patch_for_fp8.py` — 现成 fp8 path 可抄
