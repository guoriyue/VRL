# SPRINT: Cosmos video DiT MFU — NCU 复核关闭 kernel 杠杆

状态：**DONE / 收敛为 gpu_preflight + NCU 复核（2026-06-27）**。per-op 分解、GEMM 峰值、SDPA 后端、NCU tensor SOL 四条证据把单卡无损 kernel 杠杆逐个证伪：融合 AdaLN（compile 已融）、Blackwell GEMM（bf16+fp32 累加已到消费卡有效上限）、FA-3（Hopper 专属，Blackwell 无）、SDPA 后端切换（flash≈cuDNN 在噪声内）。**cosmos video 的主 compute kernel 已按本机 bf16 上限饱和；无单卡无损 kernel 杠杆。** 唯一落地交付 = `vrl/scripts/perf/gpu_preflight.py`（perf-only，MFU probe 显式调用，不接入训练启动，根治 419 误诊）。原文（下方 §1.x/§2/§3 的"杠杆"框架）保留作推演记录，结论以 §0 为准。
> 旧状态：planned / 经两次 probe 大幅收缩（2026-06-26）。原计划"融合 AdaLN-Zero + 融合注意力"两个无损杠杆，先被 per-op/GEMM probe 收缩，又被 2026-06-27 NCU 复核收敛为"不做 kernel"。**下方 §1 的 51% MFU、§2/§3 的 AdaLN/FA-3/GEMM 设计都是旧推演，不再是执行计划。**
> ⚠️ 正确判饱和口径：解析 MFU 只能做筛查；真正判断必须看 NCU `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`，并和同机方阵 GEMM 对照，不和 vendor headline 对照。
> Tool lifecycle note (2026-08-10): the one-shot GEMM/MFU/op-breakdown commands cited below have been retired. Their measurements remain historical evidence; `gpu_preflight.py` is the maintained calibration entrypoint.

## 0. 一句话（NCU 复核后：主 compute kernel 已饱和，剩配置自检）

**2026-06-27 NCU 复核确认：45% tensor SOL 不是"半个 tensor core 闲着"，而是 RTX 5090 上 bf16+fp32 累加的有效上限区间。**

- **修正 1（P0 per-op,§1.5）**:compile 已把 AdaLN/RoPE/modulation 从 20.5% 融到 3.5% → 手写融合 AdaLN 不是杠杆,砍。
- **修正 2（GEMM 峰值,§1.6）**:之前的 "51% MFU / 60% 时间在 Ampere GEMM = 头空间" 是**拿错峰值(419)算的**。已退役的 `gemm_peak_probe` 当时测得：4096³=162、8192³=215、12288³=232、16384³=233 TFLOPS。419 是 fp8/fp4 headline，不是 bf16 dense 分母。
- **修正 3（NCU tensor SOL 复核）**:8192³ 方阵 bf16 GEMM = 47.48% tensor SOL / 11.59% DRAM；cosmos 主 GEMM = 45.29-45.33% tensor SOL / 7.66-7.69% DRAM；batch=4 flash attention = 43.42% tensor SOL / 2.35% DRAM。**cosmos 主 GEMM 跟方阵 GEMM 同一上限区间，`cutlass_80` 不是慢 kernel。**
- **注意力杠杆也死了(P3 实测 + NCU)**:① **FA-3 是 Hopper(sm_90)专属,Blackwell sm_120 根本没有 FA-3 build**;flash-attn 最新(2.8.3)是 FA-2,torch 已内置同款,装了≈零增益。② SDPA 后端 flash≈cuDNN(189 vs 182 TFLOPS,胜负随热态/warmup 翻转)，batch=4 flash attention tensor SOL 已达方阵 GEMM 的 91%。**没有稳健的注意力杠杆。**
- **唯一真正交付 = `vrl/scripts/perf/gpu_preflight.py`(perf-only + 测试过)**:MFU probe 显式 log 本机真实 bf16 峰值(MFU 正确分母)、arch 匹配、最快 SDPA 后端。这把整段误诊的根因(用 vendor headline 419 当峰值)根治了,并把所有 MFU probe 的默认峰值改成实测值。**这是本 sprint 收敛后唯一落地的东西——cosmos 已饱和,没有单卡无损 kernel 杠杆。**

## 1. 旧解析结果（cosmos-predict2.5, 已退役 video_dit_mfu_probe；已被 §0/§1.6 推翻）

```
frames  vid_tok  attn%   ms/fwd        eager 42% → compile 51% MFU (1.23x)
1       880      11%     38            对比 SD3.5 image: compile 94% (1.37x)
4       3520     21%     116           对比 wan 14B:     compile 56% (1.17x)
8       7040     32%     232
16      14080    47%     596  ← 近 attn-dominated
```

旧读法是：① compile 只 1.23x(image 1.37x)→ 大量未融合 bandwidth 算子；② attention 占比随帧数升到 32-47% → 注意力成为大头。**这个读法现在只保留为历史推演；NCU 已确认主 GEMM/attention compute kernel 并不存在 45% 级别的无损头空间。**

## 1.5 P0 实测：per-op 时间分解（已退役 `video_op_breakdown_probe`，cosmos 8 帧）

torch.profiler 把 compile 前后的 GPU self-time 按 kernel 分桶：

```
              EAGER              COMPILED
MATMUL        112ms (49.1%)      112.6ms (60.2%)   cutlass_80_tensorop_bf16_s16816gemm (Ampere SM_80)
ATTENTION      65ms (28.5%)       67.8ms (36.3%)   pytorch_flash::flash_fwd (FlashAttention-2)
NORM_ELEM      47ms (20.5%)        6.5ms ( 3.5%)   ← compile 已融掉(triton_red_fused_layer_norm…)
OTHER         4.5ms ( 2.0%)        0.0ms
```

**三条结论,直接定优先级：**
1. **融合 AdaLN 不是杠杆**：compile 已把 NORM_ELEM 从 20.5% 融到 3.5%。AdaptiveLoad 的 3.2x 是 AdaLN *kernel* 自身,折到端到端只剩 ~2-3%。**砍掉原 P1。**
2. **attention 是大时间切片，但不是稳健杠杆**:占 compiled 时间 36%,跑的是 FlashAttention-2(`pytorch_flash::flash_fwd`)；后续实测 FA-3 不支持 Blackwell，flash≈cuDNN，batch=4 tensor SOL 已接近方阵上限。
3. **GEMM 不是杠杆**:占 60%,dispatch 名称里有 **Ampere SM_80** cutlass kernel(`cutlass_80_tensorop`)，但 NCU 显示主 GEMM 45.3% tensor SOL，和同机方阵 GEMM 47.5% 同区间。这里不是没用上 Blackwell，而是 bf16+fp32 累加到有效上限。

## 1.6 GEMM 峰值实测：推翻"60% 时间在慢 GEMM"（已退役 `gemm_peak_probe`）

已退役的 `vrl/scripts/perf/gemm_peak_probe.py` 当时使用 cuBLAS bf16 dense fp32 累加：

```
4096³ 162 | 8192³ 214 | 12288³ 232 | 16384³ 231 TFLOPS  → 5090 bf16 真实峰值 ~232
torch arch_list 含 sm_120(已为 Blackwell 编译);cosmos GEMM 实测 ~245 TFLOPS = 到顶
```

**之前 51% MFU 是拿 peak=419 算的,而 419 是 fp8/稀疏 "AI TOPS",bf16 dense(消费卡 fp32 累加半速)实测 ~232。** 按 232 重算 cosmos = 218 TFLOPS = **~94% 饱和**。`cutlass_80` 的 s16816 MMA 跑到了 bf16 上限——**不是用了旧 kernel,是 bf16 到硬件极限**。**GEMM 无损杠杆不存在**;想超过 232 只能 fp8/fp4(有损,离 policy path)。MFU 分母必须每台机器通过当前 `gpu_preflight.py` 实测,别用 vendor headline——这是 51% 误诊的根因。

## 2. 旧结构诊断（保留作推演记录，非执行计划）

### 2.1 旧假设：AdaLN 未融合（已被 compile 后 NORM_ELEM=3.5% 证伪）
`CosmosTransformerBlock.forward`（transformer_cosmos.py:408-440）每块 3 次 AdaLN-Zero：

```python
norm_hidden_states, gate = self.norm1(hidden_states, embedded_timestep, temb)  # CosmosAdaLayerNormZero
attn_output = self.attn1(norm_hidden_states, image_rotary_emb=image_rotary_emb)
hidden_states = hidden_states + gate * attn_output
# norm2 + attn2(cross) + gate, norm3 + ff + gate 同构
```

`CosmosAdaLayerNormZero.forward`（:128-148）：
```python
shift, scale, gate = embedded_timestep.chunk(3, dim=-1)
hidden_states = self.norm(hidden_states)                 # LayerNorm(no affine)
hidden_states = hidden_states * (1 + scale) + shift       # 调制:独立 elementwise
return hidden_states, gate                                # gate 之后又一次 elementwise: gate * attn
```

**28 块 × 3 = 84 个 AdaLN-Zero**,每个 `LayerNorm → (1+scale)*x+shift → gate*out` 是三段独立 bandwidth-bound op,中间激活全物化。AdaptiveLoad 的融合 LayerNorm-Modulate kernel(统计量留寄存器、只输出最终结果)正是打这个:报 AdaLN 自身激活 -61.9%、forward-kernel 3.2x。

### 2.2 旧假设：注意力 + RoPE 未融合（已被 FA-3/SDPA/NCU 复核收敛）
`CosmosAttnProcessor2_0/2_5`（:151/:215）：q/k 投影后**独立** `apply_rotary_emb(query)` / `apply_rotary_emb(key)`,再 SDPA。video 长度下 attention 占 32-47%,且 RoPE 的 elementwise + SDPA 分开。杠杆:把 RoPE 融进 attention,或上 FA-3(FP16/BF16 exact path;**FP8 path 是 lossy,禁止上 policy path**)。

### 2.3 旧集成草案（未执行）
`self.transformer` 是 diffusers `CosmosTransformer3DModel`(model.py:123),repo 已有 module-swap 先例:`swap_linears_to_fp8(self.transformer)`(`vrl/nn/quantization/fp8.py`)、SD3 的 `install_sd3_joint_attention_processor`。融合 AdaLN/attention 走**同款 swap**,作用在 `self.transformer` → rollout builder 和 replay builder(两者 compile 同一 transformer)都自动吃到。

## 3. 旧设计草案（已取消：无 kernel swap）

### 旧杠杆 ①：FA-3 / Blackwell-native 注意力（已取消）
- 现状:`pytorch_flash::flash_fwd`(FlashAttention-2)。RoPE 在 SDPA 前独立 `apply_rotary_emb`(transformer_cosmos.py:184/251)。
- 动作:① 上 FA-3 BF16 path(exact)替换当前 FA-2;② 把 RoPE 融进 attention q/k 路径去掉物化。
- 接入:`CosmosAttnProcessor` 是可替换的 processor(diffusers 标准),走 SD3 `install_sd3_joint_attention_processor` 同款 swap → 装一个 FA-3 processor。
- **正确性**:FA-3 **BF16/FP16 path 是 exact**(研究已验证);**FP8 path 是 lossy,禁止上 policy path**(污染 old_log_prob)。
- caveat:研究的 FA-3 1.5-2x 是 **H100** 数,5090(Blackwell)要自测;且要确认 5090 上有可用的 FA-3 build(否则先解决"为什么 dispatch 到 FA-2 而非更新 kernel")。

### Deliverable ②：per-machine 配置自检（已保留并落地）
GEMM 杠杆不存在(§1.6 bf16 到顶),但 51% 误诊暴露了真正的工程风险:**MFU 分母用错、或机器 torch build 不含本机 SM**。加 `vrl/scripts/perf/gpu_preflight.py` 做 perf 自检:
- 断言 `f"sm_{cap}" in torch.cuda.get_arch_list()`(否则 PTX-JIT 退化/失败);
- 跑一次当前 `gpu_preflight.py` 拿本机真实 bf16 峰值,所有 MFU probe 用它当分母(别硬编码 419);
- log flash-attn 是否可用 + SDPA 选了哪个后端。

### 砍掉的杠杆
- **融合 AdaLN-Zero kernel**:P0 实测 compile 已融到 3.5%,端到端 ~2-3%,不做。
- **Blackwell GEMM kernel**:§1.6 实测 bf16 已到 232 硬件峰值,`cutlass_80` 不是慢,不做。fp8/fp4 GEMM 有损,只能离 policy path(repo 已有 fp8 rollout)。

### 旧 swap 接入（未执行）
- attention:装 flash-attn(为 sm_120 编译)后,确认 SDPA dispatch 到它/FA-3;若需显式 processor,走 SD3 `install_sd3_joint_attention_processor` 同款 swap,作用 `self.transformer` → rollout + replay 都吃到。
- 默认 off,proof-gated;**FA-3 FP8 path 禁止**(lossy)。

## 4. 正确性契约（旧 kernel swap 的保留标准）

- 融合 kernel 必须**算法级 exact**:同数学、fp32 reduction、无近似复用。这是它能上 policy path 的前提(与量化/feature-cache 的本质区别——后者改输出分布、污染 `old_log_prob`,见 [[SPRINT_lossless_diffusion_rl_research]] §2.2 的 verl 铁律)。
- swap 作用同一 `self.transformer` → rollout 采样 forward 和 replay 训练 forward 用**同一融合算子**,训推差异不增(同 [[SPRINT_train_inference_alignment]] 的"单一共享 forward")。
- parity 门（若未来重新引入任何 kernel swap 才启用）:`compile_benchmark` 同款数值检查(rollout `max_rel_out`、train `max_abs_grad` 在阈内)对比融合 vs 未融合。
- RL 门（同上）:短 dry-run,`ratio_abs_dev` / TIS-RS 触发率 / reward 曲线与未融合基线一致。

## 5. Phase plan（已关闭：只保留 preflight）

- **P0 — per-op 分解 ✅**:MATMUL 60% / ATTN 36% / NORM_ELEM 3.5% → 砍融合 AdaLN。
- **GEMM 峰值 ✅**:bf16 真实峰值 ~232,cosmos GEMM 到顶 → 砍 GEMM 杠杆,cosmos ~94% 饱和。
- **P1 — 配置自检(`gpu_preflight.py`)**:arch 匹配断言 + 本机 bf16 峰值实测 + flash-attn/SDPA 后端 log。**这是现在最高价值、最便宜的交付**——防止跨机器误诊。
- **P2/P3/P4/P5 — 取消**:FA-3/attention/GEMM/AdaLN kernel 杠杆均已证伪；没有要 swap 进 policy path 的新 kernel。

> 期望收益已从"~45% 头空间"收敛到"配置正确性"。bf16 算力已到顶,大头空间只在 fp8/fp4(有损,离 policy path)和多卡 pipeline。

## 6. 验收（已完成）

- 已退役 `gemm_peak_probe` 的历史结果：RTX 5090 bf16 dense achieved peak = 233 TFLOPS；torch build 包含 `sm_120`。
- NCU 方阵 GEMM:8192³ bf16 GEMM tensor SOL 47.48%，这是同机对照上限。
- NCU cosmos 主 GEMM:45.29-45.33% tensor SOL，DRAM 7.66-7.69%，和方阵同区间。
- NCU batch=4 flash attention:43.42% tensor SOL，DRAM 2.35%，没有大内存瓶颈或 45% 级别无损头空间。
- `gpu_preflight`:perf 自检真实 bf16 峰值、arch 匹配、SDPA 后端，防止 419 headline 再污染 MFU。

## 7. 非目标

- 不上 FP8/int attention 或 feature-cache 到 policy path(lossy,污染 old_log_prob)。
- 不碰 image 家族 compile(SD3.5 已 94% 饱和,融合 AdaLN 只剩 ~6% 残余)。
- 不和 FSDP2 同开 compile(`strategy.py:480` 硬门);融合 kernel 本身与 FSDP 兼容性单独验。
- 不重写 attention 数学;只融合已有算子(RoPE+SDPA)。
- 不追 novelty;AdaptiveLoad/FA-3 是 prior art,本 sprint 的实际价值是**用 NCU 关闭错误 kernel 方向 + 落地 gpu_preflight 防误诊**。

## 8. 关键文件

- `vrl/scripts/perf/gpu_preflight.py`：本 sprint 唯一保留交付，perf/probe 路径实测 bf16 峰值、arch 匹配、SDPA 后端。
- `vrl/scripts/perf/gemm_peak_probe.py`：已退役；历史同机 bf16 GEMM peak 来源。
- `vrl/scripts/perf/video_dit_mfu_probe.py`：已退役；历史解析 FLOP/time 筛查来源。
- `vrl/scripts/perf/video_op_breakdown_probe.py`：已退役；历史 op-time attribution 来源。
- `compile_benchmark.py`：仅当未来重新引入 kernel swap 时作为 parity 门。
- Triton 基建：`~/Desktop/moemoekit`（记忆:Triton kernel 复用,CPU 需 naive fallback）
- 证据:记忆 `project_lossless_diffusion_rl_research`、`project_rollout_bound_class_probe`
