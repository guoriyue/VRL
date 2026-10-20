# SPRINT：rollout 与 reward 吞吐——PyTorch Conference 2026 带来的候选项

状态：**planned（2026-10-20）**。每一项都先测量再决定，彼此独立，一项一个 commit。

来源：PyTorch Conference NA 2026 Day 1：068 SGLang-Diffusion、060 NVFP4 four-over-six、073 VLM token 成本、096 CUDA graph 可观测性；海报 P18、P115。

不重复以下已有计划：
- `SPRINT_wan22_grpo_throughput_weight_placement.md`：权重放置；
- `SPRINT_rollout_finalize_overlap_ga.md`：finalize 与 copy-stream；
- `SPRINT_continuous_three_stage_pipeline_program.md`：三段流水线；
- `SPRINT_triton_map_and_perf_ledger.md`：Triton 账本。

## 0. 结论先行

**SGLang-Diffusion 不替换 VRL 的 rollout。** 它已经有 `/rollout` 接口和权重更新 API，但有三个问题：不支持推理过程中换权重；默认档位带 kernel fusion，会破坏 parity；相对 vLLM-Omni 只快 1.07–1.17 倍。而 VRL 自己的路径和 vLLM-Omni 只差约 1.2 倍（`docs/flux_qwen_naive_vs_vllm_omni_profiling.md`）。

可以借鉴的是它的几项优化。下面按"预期收益 / 改动量"排序。

## 1. 当前代码现实（HEAD a914644c6 核对）

- **CFG**：`predict_with_cfg`（`vrl/models/steps/denoise/common/cfg.py:10-56`）。
  - Wan、SANA、SD3.5 把 cond 和 uncond 拼成 2 倍 batch，做一次前向。
  - Qwen-Image(-2.1)、Krea2、MOSS、Cosmos2/2.5/3 做两次前向。
  - FLUX 和 H3 没有 uncond 分支（guidance 已蒸馏进模型）。
  - 没有跨卡的 CFG parallel。
- **prompt 编码**：`vrl/generation/bindings/full_sequence/executor.py:280-287, 489-494`。
  - 每个 sample batch 都重新编码 prompt 和 negative prompt。negative prompt 是常量，也每批重编。
  - batch 不跨 prompt（`sample_batches.py:321-349`），所以每个 prompt 要编码 ceil(G / `samples_per_generation_batch`) 次。
  - 没有 text-encoder 缓存。
  - `stage_durations` 只在 `runtime_debug` 时输出，而且不做 CUDA sync。
- **NVFP4**：`vrl/nn/quantization/fp4.py:93-138` 和 `fp4_kernels.py:61-65`，是两级纯 max scaling：block scale = `block_amax/6/tensor_scale`。只量化 MLP（`targeting.py`）。
- **CUDA graph**：全仓没有用。TeaCache 每步一次 `.item()`（`teacache.py:54`），但默认关闭，也没有 preset 开启它。
- **Qwen3-VL judge**：每个 criterion 单独一行，每行都带上全部图像，一次前向（`vrl/rewards/models/qwen3vl_preference.py:588-608`）。所以视觉 token 每个问题都要重算一遍。`judge_max_side=1024`。没有 vLLM 服务，也没有前缀缓存。
- **媒体交接**：pipelined 模式下，每个 batch 先 `ray.put` 一次（`vrl/generation/ray/worker.py:106`），finalizer 合并后再 `ray.put` 一次（`finalizer.py:62-70`），对象存储里有两份拷贝。没有交接耗时的指标。
- **阶段计时**：rollout 的墙钟阶段一直开着（`rollout.collect_s`、`rollout.weight_sync_s`）。trainer 的 GPU 阶段（advantage、evaluate、backward、optim_step）只在 `trainer.profile` 时记录（`trainer.py:1759-1766`）。
- **SP rollout**：目前只有 SD3 支持。每个 rank 复制整套流水线，rank0 统一 RNG（`worker.py:475-500`），所以噪声天然与 SP 并行度无关，**不需要**另加不变性测试。

## 2. 候选项

### T1. 缓存 negative prompt 的编码（收益确定、改动小）

- 做法：同一个 rollout request 内，常量 negative prompt 只编码一次，按 batch 复用。正向 prompt 在 `samples_per_generation_batch` 小于组大小时，也按 prompt 复用一次编码。
- 先测：在 `runtime_debug` 下读 `stage_durations["encode"]` 的占比，测之前加 CUDA sync。
- 验收：rollout 的轨迹和 logprob 逐位不变（编码是确定性的）；encode 阶段耗时下降。

### T2. Qwen3-VL judge 的图像只编码一次（中等收益、改动中等）

- 先测：judge 在 GRPO 每步中的耗时占比。
- 做法：B 个问题共享同一组图像，视觉编码和图像前缀的 KV 只计算一次，再分给各个问题行。实现前先看 HF Qwen3-VL 能否传入预计算的 `pixel_values` 特征或 `past_key_values`。
- 验收：所有问题的 Yes 概率和改动前一致（容差需实测）；judge 耗时下降。

### T3. NVFP4 four-over-six 校准（改动小、需要验证）

- 做法：每个 block 在"amax 映射到 6"和"amax 映射到 4"两种 scale 中，选重构误差更小的那个。060 在 Nemotron 3 Ultra 上，中位 MSE 降 16.4%。
- 改动点：Triton kernel 和 torch 参考实现都在 `fp4.py`、`fp4_kernels.py`。
- 验收：对比改动前后的权重重构 MSE、rollout 与 replay 的 `mismatch_kl`、reward。漂移没有下降就不合入。

### T4. 跨卡 CFG parallel（收益依场景而定）

- 适用：做两次前向的家族（Cosmos、Qwen-Image、Krea2、MOSS）在多卡 rollout 时，cond 和 uncond 分到两张卡并行。数值不变。
- 触发：这些家族出现多卡 rollout 的需求，并且实测 rollout 是瓶颈。
- 改动点：`cfg.py` 的 `batched=False` 路径，加上 rank_group 的通信。

### T5. 用 CUDA graph 跑去噪循环（探针）

- 先测：用 `capture_torch_trace` 看小模型（SANA 512px）在 5090 上去噪步之间的 GPU 空闲间隙。间隙小于步时的 15% 就停止。
- 探针要验证三点：shape 固定的 DiT 前向用 `torch.cuda.graph` 捕获后，rollout 和 replay 是否逐位一致；与 CuMem parking 的内存池是否冲突；对 TeaCache 的 `.item()` 是否需要 breakable graph。
- 观测：上了 graph 之后 `profile_range` 会失效，对应方案（`torch.cuda.graph_annotations`）需要 torch 2.15，见升级 sprint。

### T6. 媒体交接去掉一次 `ray.put`，并补上交接计时（小）

- 先加指标：rollout 端 `ray.put` 的字节数和耗时，reward 端 `MediaReference.resolve` 的耗时。
- 交接占步时 5% 以上时，finalizer 直接转发 batch 的 ObjectRef 列表，不再合并后重新 put。

### T7. trainer 阶段计时常开（小）

- 依据：海报 P115 StageFrontier。各 rank 只记录 host 端的粗粒度时间，在已有的同步边界（collect、reward、train、sync 都是 Ray RPC 或 collective 边界）上，取所有 rank 中最大的累计时间。
- 做法：`trainer.py:1759-1766` 在不做 CUDA sync 的情况下也记录墙钟阶段，写进每步的 metrics。`profile=True` 时仍然做 sync 并写 `phase_events`。
- 验收：常开模式下步时开销 < 1%，阶段之和约等于步时。

## 3. 不做

- 不接入 SGLang-Diffusion 或 vLLM-Omni 作为 rollout 后端；它们只用作性能基准。
- 不做流水线化的 Ulysses all-to-all：只有 SD3 有 SP，`sequence_parallel.py` 的 docstring 说明图像 token 量级下 all-to-all 受延迟限制。等出现视频家族的 SP rollout 需求再说。
- 不做"精确 / 无损 / 近似"三级质量档：目前没有两种近似同时启用的场景。
