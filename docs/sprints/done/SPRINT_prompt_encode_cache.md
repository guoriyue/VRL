# SPRINT：prompt embedding 缓存 —— 实测否决，不实施

状态：**done / 不实施（2026-08-17）**。KILL-RISK 门没过：收益上限
**0.4–0.6%**，而且**本仓早就直接测过这个干预本身**。本文保留为否决记录。

基线 main @ `abb8e4da`。

## 0. 结论先行

前提全部成立，收益不成立。

- text encoder 确实冻结 ✅
- 同一 prompt 确实跨 chunk 重复编码 ✅（`plan()` 逐行确认，见 §1）
- **但 encode 只占 rollout 的 0.63%** ❌

结构性原因：**encode 每个 chunk 只发生一次，denoise 每个 chunk 发生
`steps × CFG` 次**。缓存最多消掉 encode 那一次，所以收益天花板就是 encode 的
占比。这个比值对 video 家族只会**更小**（步数更多、序列更长，分母涨得比分子快）。

## 1. 前提确认（都是对的，保留供引用）

**text encoder 冻结** —— 四条独立证据：
`pipeline.text_encoder.requires_grad_(False)`（`vrl/models/families/wan_2_1/model.py:196`）；
不在 `trainable_modules`（同文件 :357-359）；不在 `policy_cores`
（`vrl/models/steps/denoise/base.py:433-443` 只返回 transformer）；
weight sync 只推 trainable state（`vrl/trainers/weight_sync.py:112-127`）。

**重复编码确实发生** —— `GenerationSampleBatch` 是 prompt-major 的
`(prompt_index, sample_start, sample_count)`
（`vrl/generation/execution/sample_batches.py:254-277`），而 `plan()`
（同文件 :299-331）对每个 prompt 切出 `ceil(samples_per_prompt /
max_samples_per_batch)` 个 batch，**全部带同一个 `prompt_index`**。
`_forward_chunk` 每个 batch 调一次 `encode_prompt_for_batch`
（`vrl/generation/bindings/full_sequence_denoise/executor.py:272-280`）。
所以 `samples_per_prompt=12, max_samples_per_batch=1` → 同一 prompt 编码 12 次。

## 2. 为什么还是不做

### 2.1 本仓已经直接测过这个干预

`docs/sprints/info/SPRINT_rollout_performance.md:686-700` 的 b8 vs b16 对照，
**其效果正是本 sprint 想要的**（chunk 数减半 → prompt encode 次数减半）：

| metric | b8 control | b16 | 读法 |
|---|---:|---:|---|
| encode total | 0.177s | 0.145s | 原文注：`b16 saves one prompt-encode boundary` |
| recorded stages total | 7.760s | 8.064s | — |

省下 **0.032s / 7.76s = 0.4%**，而且被 denoise 自身的 2.3% 变慢完全淹没。

### 2.2 绝对占比

同文档 :136-150 的 SD3.5 OCR D0 run（8 个 rollout batch）：

| stage | total | 占 denoise |
|---|---:|---:|
| `encode` | 0.499s | **0.63%** |
| `prepare_latent` | 0.035s | 0.04% |
| `denoise` | 78.664s | — |

同文档 :991 的原话已经下过结论：

> encode/decode 加起来也不到 denoise 的 1.5%

### 2.3 video 家族只会更小，不会更大

有人会问「SD3.5 是图像，wan/cosmos 的 umT5-XXL / T5 更重，是不是不一样？」
—— 方向相反：

- 分子（一次文本前向）≈ `2 × N_text_params × text_tokens`。umT5-XXL 5B ×
  256 token ≈ 2.6 TFLOP。
- 分母（denoise）≈ `steps × CFG × 2 × N_dit_params × video_tokens`。wan 1.3B ×
  ~32k video token × 35 步 × CFG ≈ 数千 TFLOP。

video 家族的 denoise 分母比 SD3.5 涨得多得多，**比值反而降到 ~0.04% 量级**。
文本编码器再大也只跑一次。

## 3. 什么条件下应该重开

**只有一个**：极少步数的蒸馏家族（causvid 这类 causal/distilled Wan，
3–4 步而不是 35–50 步）。那时 `steps × CFG` 这个放大因子塌掉，encode 的占比
可能升到有意义的量级。

重开前置：**先在那个家族上测出 `stage_durations["encode"]` 与
`stage_durations["denoise"]` 的实际比值**（telemetry 已在生产里，
`executor.py:273` 的 `profile_range("generation.prompt_encode")`），比值
不到 5% 就不要开工。

## 4. 顺带记下的设计难点（若将来重开）

即使重开，缓存 key 不像看上去那么简单：

- `ReferenceConditionedBatches.encode_prompt_for_batch`
  （`executor.py:488-505`）**同时编码一张 per-chunk 的参考图**
  （`_reference_image_for_chunk`），那不是文本常量，不能进文本 key。
  Cosmos V2W / Wan I2V 走这条路径。
- 基础版把整个 `request=video_request` 传进 `model.encode_prompt`
  （`executor.py:498-504`），家族实现从中读什么没有统一约束 —— key 要覆盖
  哪些字段必须逐家族核对（18 个家族都实现了 `encode_prompt`）。
- GPU 常驻缓存会计入 parking 的 256 MiB 残留预算
  （`vrl/utils/cuda_memory.py` 的 `CUDA_RUNTIME_RESIDUAL_BYTES_LIMIT`），
  `sleep()` 必须显式清理，否则 parking 直接硬报错。

三条加起来，工作量远超 0.4% 的收益。

## 5. 相关

- 直接测量：`docs/sprints/info/SPRINT_rollout_performance.md:686-700`（b8/b16）
  与 :136-150（D0 stage 表）
- 同为「实测后否决」：`docs/sprints/done/SPRINT_train_step_sync_audit.md`
- 父 program：`docs/sprints/planned/SPRINT_train_phase_efficiency_program.md`
