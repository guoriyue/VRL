# SPRINT: 接入 Qwen-Image T2I 家族（套现有 diffusion seam）

状态：**parked：LANDED（代码）/ BLOCKED（单卡 naive 实跑，2026-06-21）**。
触发条件：可用显存足以容纳约 40GB transformer 的多卡/大卡，或原生分片/fp8 加载路径落地。四文件 + registry + model yaml +
smoke recipe + train entrypoint 全部落地；CPU 单测绿（`tests/models/diffusion/qwen_image/`，含
separate-CFG 不等长序列与 norm-rescale）。落地修复同 FLUX 的两处真实 seam 缺口（CPU encoder offload
+ 动态 shifting `mu`）。**单 GPU naive 实跑被显存挡住**：Qwen-Image transformer ~20B，bf16/fp16 权重
~40GB > 32GB，编码器已 offload 仍放不下 transformer 本身（fp32 更甚；fp8 需先以 bf16 载入 40GB 再
swap，载入即 OOM）。结论：Qwen-Image 的单 32GB 卡 naive 前向对所有精度结构性 blocked，需 ≥2 卡或
权重分片/原生 fp8 载入。详见 [`docs/flux_qwen_naive_vs_vllm_omni_profiling.md`](../../flux_qwen_naive_vs_vllm_omni_profiling.md)。性质：新增一个 T2I DiT 家族。

> 拆自 [[SPRINT_model_family_coverage]]（竞品覆盖路线图）的 Tier-1 第 2 项。
> 相关：[[SPRINT_flux_t2i]]（同档同构的另一个 T2I 家族）、
> [[SPRINT_flow_grpo_recipe_parity]]（GRPO 配方）、记忆 `project_first_trustworthy_curve`（判据）。

## 0. 一句话

**Qwen-Image 是 VeRL-Omni 的旗舰已发布模型、当前开源 T2I SOTA 之一（Apache-2.0、强文字渲染），我们没有。**
它是标准 flow-matching MMDiT，**有真正的 CFG**，seam 比 FLUX 还顺；本 sprint 的主要约束是**体量大（~20B）**——
LoRA-only 起步 + VAE tiling，和 `encode_prompt` 走 Qwen2.5-VL 文本编码器（与 SD3 的 "prompt+pooled" 形状不同）。

## 1. 模型事实（接入前必须对齐的形状）

- **Qwen-Image**：~20B MMDiT（Apache-2.0），flow-matching，强中英文字渲染。
- **文本编码器 = Qwen2.5-VL 系**：`encode_prompt` 产出 **序列 `prompt_embeds` + attention mask**，
  **无 SD3 那种 `pooled_prompt_embeds`**——transformer 只吃 `encoder_hidden_states` + mask，
  这是与 sd3_5 实现的主要差异点。
- **有真 CFG**：diffusers pipeline 用 `true_cfg_scale`（cond/uncond 双分支），→ 直接套现有
  `cfg_mode="batched_cfg"`、`cfg_base="uncond"`，**不需要像 FLUX 那样新增单分支模式**。
- **体量**：~20B → 首版 **LoRA-only**，VAE decode 开 tiling/slicing。

## 2. 落地：要动的文件

1. **Registry 一条目**（`vrl/rollouts/families/registry.py`，照 sd3_5 第 132-143 行）：
   ```python
   register_rollout_family(_diffusion_entry(
       family="qwen_image", task="t2i", aliases=(),
       executor_cls="vrl.models.diffusion.qwen_image.runtime:QwenImageChunkExecutor",
       runtime_builder="vrl.models.diffusion.qwen_image.runtime:build_qwen_image_runtime_bundle",
       model_build_resolver="vrl.models.diffusion.qwen_image.runtime:resolve_qwen_image_model_build",
       request_prefix="qwen_image", default_task_type="text_to_image",
   ))
   ```
2. **家族目录** `vrl/models/diffusion/qwen_image/`（四文件）：
   - `model.py`：`QwenImageModel(DiffusionModelBase)` + `QwenImageReplayModel`。实现 `base.py:41-96` 的
     `encode_prompt`（Qwen2.5-VL 出序列 embeds + mask，**无 pooled**）/ `prepare_sampling` /
     `forward_step` / `decode_latents`（tiled VAE）+ replay 三件套。
   - `runner.py`：`QwenImageDiffusionBackboneRunner`——**套现有 `batched_cfg` 双分支**，
     `build_branch` 出 cond/uncond，`encoder_hidden_states=prompt_embeds`、`extra_kwargs` 传 attention mask
     （**不带 pooled_projections**）。
   - `runtime.py`：`build_qwen_image_runtime_bundle` / `build_qwen_image_replay_runtime_bundle` /
     `resolve_qwen_image_model_build(... task_variant="t2i")` / `QwenImageChunkExecutor`，
     复用 `vrl/models/loader.py` 的 `load_diffusers_transformer / load_flow_match_scheduler /
     apply_lora_to_transformer / apply_rollout_quantization`。
   - `__init__.py`：导出。
3. **Config** `configs/model/diffusion/qwen_image/base.yaml`（照 `sd3_5/medium.yaml`）：
   `family: qwen_image`、`path: Qwen/Qwen-Image`、`use_lora: true`、`lora.target_modules`（MMDiT 的
   `to_q/to_k/to_v/to_out.0` + img/txt 双流的 add 投影）、`memory.vae_decode.tiling: true`、`torch_compile`。
4. **共享层不改**：`common/*`、`flow_matching.py`、`loader.py` 直接复用。

## 3. 验收

- **推理**：`qwen_image/base.yaml` 经 registry 解析→构建 runtime bundle→生成 N 张图（含一张**含文字**的
  prompt 验证文字渲染）、tiled VAE decode 正常，与 diffusers 官方 `QwenImagePipeline`（同 prompt/seed/steps/
  `true_cfg_scale`）做视觉 + 数值 sanity 对照。
- **训练**：套 [[SPRINT_flow_grpo_recipe_parity]] 配方，固定 prompt 集跑短 GRPO（LoRA），BLOCK test 判读
  reward >2σ 单调抬升（见 `project_first_trustworthy_curve`）；注意 20B + group≥16 的显存预算，
  必要时降分辨率/microbatch（见 `project_two_level_async` 的 streaming/microbatch 手段）。
- **测试**：照 `tests/models/diffusion/` 加家族注册 + `from_build/encode_prompt/forward_step/decode_latents`
  结构性用例（不断言配置字面值，见 `feedback_no_exact_config_tests`）。

## 4. 非目标

- **不做 full fine-tune**：20B 首版 LoRA-only；FT 另议（见 [[SPRINT_fullparam_and_fp8_precision]]）。
- **不做 Qwen-Image-Edit / 图像条件变体**：本 sprint 只做纯 T2I。
- **不改 diffusion 共享层**：新家族应只靠 model/runner/runtime 适配。

## 参考

- diffusers QwenImagePipeline：https://huggingface.co/docs/diffusers/api/pipelines/qwenimage
- Qwen-Image：https://huggingface.co/Qwen/Qwen-Image
- 接入面证据：`vrl/rollouts/families/registry.py:92-143`、`vrl/models/diffusion/base.py:41-96`、
  `vrl/models/diffusion/sd3_5/{model,runner,runtime}.py`、`configs/model/diffusion/sd3_5/medium.yaml`
