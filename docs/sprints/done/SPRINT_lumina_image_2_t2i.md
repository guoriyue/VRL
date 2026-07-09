# SPRINT: 接入 Lumina-Image 2.0 T2I 家族

状态：**DONE（2026-07-08）——随 [[SPRINT_thin_model_seam_and_ten_model_expansion]] Phase 1 落地并
真权重 rollout 验证（replay parity 0.0e+00，摄影级输出）。短 GRPO 曲线未单独跑（战役按 rollout
验证关账，详见 index sprint 文件头验证记录）。**性质：新增 T2I flow-matching DiT 家族，套薄 seam。
> 拆自 [[SPRINT_thin_model_seam_and_ten_model_expansion]] §3 第 3 项。

## 0. 一句话

**Lumina-Image 2.0 是 2.6B flow-matching DiT + Gemma-2-2B 文本编码器——和 SANA 同一形状类
（单编码器 + mask + 无 pooled + flow-matching），SANA 落地后照抄其模板即可，增量成本最低。**
建议排在 SANA 之后接，两者共享全部技术点。

## 1. 模型事实（落地第一步核对 diffusers 类名）

- checkpoint：`Alpha-VLLM/Lumina-Image-2.0`（2.6B Next-DiT）。
- 预期 diffusers 类：`Lumina2Pipeline` / `Lumina2Transformer2DModel`（diffusers ≥0.32 加入；本机 0.37.1
  预期在，仍须 `inspect.signature` 核对 encode_prompt / forward 签名）。
- 文本编码器：Gemma-2-2B，出 embeds + mask，无 pooled。
- 调度器：flow-matching（`FlowMatchEulerDiscreteScheduler`）→ 零算法层改动。
- VAE：SD3 同款 16ch KL-VAE（8x）→ `decode_latents` 照 sd3_5。

## 2. 技术点

1. 全部复用 SANA 的 mask 线程 + batched_cfg runner；差异仅在 transformer forward kwargs 命名
   （落地时以 `inspect.signature` 为准）。
2. Lumina2 的 system-prompt 约定（官方 pipeline 在 prompt 前拼固定 system prefix）——`encode_prompt`
   必须与官方 pipeline 逐字一致，否则 GPU parity 过不了。

## 3. 落地文件

同 [[SPRINT_sana_t2i]] §3 形状：`vrl/models/diffusion/lumina_image_2/` 四件（runner 若与 SANA 仅差
kwargs 命名，可评估共享一个 mask-runner——但**默认每家族一份薄 runner**，保持四文件模式的跨家族一致性）
+ registry + `configs/model/diffusion/lumina_image_2/2_0.yaml` + tests + wiring 参数化。

## 4. 验收

同 [[SPRINT_sana_t2i]] §4：CPU 结构绿 + GPU 生成 parity（对照官方 `Lumina2Pipeline`）+ 短 GRPO 曲线。

## 5. 非目标

- 不接 Lumina-Next / Lumina-T2X 旧版；不做 Lumina 的多分辨率外推特性。

## 参考

- https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0
- diffusers Lumina2：https://huggingface.co/docs/diffusers/api/pipelines/lumina2
- 模板：[[SPRINT_sana_t2i]]（形状类相同）
