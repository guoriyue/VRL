# SPRINT: 接入 Mochi-1 T2V 家族

状态：planned（2026-07-01）。性质：新增 T2V flow-matching 家族，套 5D 视频 seam。
> 拆自 [[SPRINT_thin_model_seam_and_ten_model_expansion]] §3 第 7 项。

## 0. 一句话

**Mochi-1 是 10B AsymmDiT flow-matching T2V（Apache-2.0）——T2V 三个里调度器最干净的一个
（真 flow-matching + 真 CFG），无 KILL-RISK 门，wan 模板直接套。**
建议 T2V 侧接入顺序：HunyuanVideo（单分支+guidance）→ **Mochi-1（双分支基线）** → CogVideoX（待门 A）。

## 1. 模型事实（落地第一步核对 diffusers 类名）

- checkpoint：`genmo/mochi-1-preview`（10B AsymmDiT：视觉流参数远大于文本流）。
- 预期 diffusers 类：`MochiPipeline` / `MochiTransformer3DModel` / `AutoencoderKLMochi`
  （diffusers ≥0.32；本机核对签名）。
- 文本编码器：T5-XXL 单编码器（embeds + mask，无 pooled）。
- 调度器：flow-matching（官方即 rectified flow）→ 算法层零改动。
- 潜变量：5D，AsymmVAE（空间 8x / 时间 6x 压缩）→ 套 wan 视频布局 + `ChunkedLatentDecoder`。
- 真 CFG 双分支（官方默认 guidance ~4.5）→ runner 双分支 `batched_cfg`，wan 模板。

## 2. 技术点

1. seam 工作量 ≈ "wan 的单 transformer 版"：无多变体、无 boundary_ratio、单 transformer →
   rollout/replay 侧**预期都能委托共享 builder**（对比 wan 因多 transformer 自建）。
2. T5 mask 线程照 qwen_image；VAE tiling 必开（Mochi VAE 高分辨率下重）。
3. 官方建议 fp8/bf16 混合——先 bf16 LoRA，量化留给 rollout quantization 现有开关。

## 3. 落地文件

同 [[SPRINT_hunyuan_video_t2v]] §3 形状：`vrl/models/diffusion/mochi/` 四件 + registry +
`configs/model/diffusion/mochi/preview.yaml` + tests + wiring 参数化。

## 4. 验收

同 [[SPRINT_hunyuan_video_t2v]] §4：CPU 结构绿 + GPU 短视频 parity + 短 GRPO 曲线。

## 5. 非目标

- 不做 Mochi 的 fine-tune 官方脚本路径（我们走自己的 LoRA seam）；不接后续 Mochi 变体。

## 参考

- https://huggingface.co/genmo/mochi-1-preview
- diffusers：https://huggingface.co/docs/diffusers/api/pipelines/mochi
- 模板：`vrl/models/diffusion/wan_2_1/`（5D 布局）
