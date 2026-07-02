# SPRINT: 接入 HunyuanImage-2.1 T2I 家族（大模型，LoRA-only）

状态：planned（2026-07-01）。性质：新增大体量 T2I DiT 家族，套薄 seam，LoRA-only。
> 拆自 [[SPRINT_thin_model_seam_and_ten_model_expansion]] §3 第 4 项。

## 0. 一句话

**HunyuanImage-2.1 是 ~17B flow-matching DiT，双文本编码器（MLLM + byT5 字形支）+ 32x 高压缩 VAE，
强中文/文字渲染——是 10 个里 T2I 侧最重的一个，放在 SANA/Lumina 之后接。**

## 1. 模型事实（落地第一步核对；此模型的 diffusers 支持最需要验证）

- checkpoint：`tencent/HunyuanImage-2.1`（~17B DiT，2048px）。
- 预期 diffusers 类：`HunyuanImagePipeline` / `HunyuanImageTransformer2DModel`——**加入时间较新，
  本机 0.37.1 是否含、签名如何，是本 sprint 第一个动作**；若无则等 diffusers 升级或走官方仓库代码
  （后者显著加重，见 §5）。
- 文本编码器：**双支**——MLLM（Qwen2.5-VL 系）出主 embeds + **byT5-small 字形支**（文字渲染）。
  `encode_prompt` 要出两组 embeds，chunk executor 的 repeat 逻辑覆盖两组。
- 调度器：flow-matching → 算法层零改动。
- VAE：32x 高压缩（DC-AE 同类）→ decode 照 SANA 经验。
- **~17B → LoRA-only**（全参不进单卡；对齐 qwen_image 20B 的先例）。

## 2. 技术点

1. 双 embeds 线程：`encode_prompt` 返回 `{"prompt_embeds", "glyph_embeds", masks...}`，
   `build_chunk_encoded` repeat 两组——这是它区别于 SANA/Lumina 的主要 seam 工作量。
2. 冻结编码器 CPU offload：MLLM 编码器体量大，照 flux 落地实测经验（[[SPRINT_flux_t2i]] §0.1.1）
   `from_spec` 放 CPU、`encode_prompt` 在编码器设备跑。
3. refiner / distilled 变体一律不接（§5）。

## 3. 落地文件

同 [[SPRINT_sana_t2i]] §3 形状：`vrl/models/diffusion/hunyuan_image/` 四件 + registry +
`configs/model/diffusion/hunyuan_image/2_1.yaml`（`use_lora: true` 强制）+ tests + wiring 参数化。

## 4. 验收

- CPU 结构绿；GPU parity 对照官方 pipeline（同 prompt/seed；含中文文字渲染 prompt 一条）。
- 短 GRPO：LoRA rank 32 起步，reward 曲线判据同 index sprint §5。
- 显存预算：单卡 bf16 LoRA 可跑通 rollout+replay（不行则记录并降分辨率/加 offload，不静默放弃）。

## 5. 非目标

- 不接 HunyuanImage-3.0（统一理解+生成 MoE，需新 seam，见 [[SPRINT_model_family_coverage]] Tier-3）。
- 不接 refiner 流水线与蒸馏 few-step 变体。
- diffusers 无支持时**不 vendor 官方推理仓库**——推迟本 sprint 而不是引入一坨外部代码。

## 参考

- https://huggingface.co/tencent/HunyuanImage-2.1
- 模板：[[SPRINT_sana_t2i]]（32x VAE）+ [[SPRINT_flux_t2i]] §0.1（大编码器 offload 经验）
