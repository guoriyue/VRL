# SPRINT: 接入 SANA T2I 家族（Phase-0 薄 seam 首个样例）

状态：**code-landed（2026-07-07）——首个 descriptor-born 家族；GPU parity 待跑（§4 清单）**。
性质：新增 T2I DiT 家族，套 Phase 0 薄化后的 diffusion seam。

> **落地形态（比 §3 计划更薄——runner 合并与 descriptor 化之后）**：model.py（含 backbone-runner
> 协议 + replay 投影）+ runtime.py 仅 executor + registry 一条 descriptor + 2 个 yaml + tiny 真
> transformer parity 测试 ×4。**零 builder 函数**。batched_cfg 判断修正：cond/uncond 同 pad 到
> max_seq=300 可以 batch（sprint 原文说照 qwen_image 的 separate_cfg 是错的，qwen 是变长才分开）。
> `complex_human_instruction` 暴露为 sampling kwarg 默认关；parity 跑两边须一致（diffusers 默认开）。
> 拆自 [[SPRINT_thin_model_seam_and_ten_model_expansion]] §3 第 1 项。前置：Phase 0 共享 builder 已合入
> （`vrl/models/diffusion/build.py`，分支 phase0-thin-model-seam）。

## 0. 一句话

**SANA 是 Phase 0 之后第一个"薄接"样例：linear-DiT + DC-AE，1.6B 极省显存，cosmos-rl 已支持，
flow-matching 完全吻合现有 seam。** 模板选 **qwen_image**（同为"单编码器 + attention mask + 无 pooled"），
不是 sd3_5。

## 1. 模型事实（已核对，2026-07-01，本机 diffusers 0.37.1）

- checkpoint：`Efficient-Large-Model/Sana_1600M_1024px_diffusers`（1.6B linear-attention DiT）。
- diffusers 类**已确认存在**：`SanaPipeline` / `SanaTransformer2DModel` / `AutoencoderDC`（DC-AE，
  **32x 压缩**，vs SD3/FLUX 的 8x）/ `FlowMatchEulerDiscreteScheduler`。
- 文本编码器：**Gemma-2**（单编码器，出 `prompt_embeds + prompt_attention_mask`，**无 pooled**）。
  `SanaPipeline.encode_prompt` 返回 `(prompt_embeds, prompt_attention_mask, negative_*, negative_*_mask)`。
- transformer forward：`(hidden_states, encoder_hidden_states, timestep, guidance=None,
  encoder_attention_mask=None, ...)`——已用 `inspect.signature` 核对。
- 调度器：flow-matching → 直接复用 `sde_step_with_logprob` / `load_flow_match_scheduler`，零算法层改动。

## 2. 技术点

1. **mask 线程**：`encoder_attention_mask` 要一路穿过 runner → chunk executor（repeat 到 chunk batch）→
   replay tensors（照 qwen_image 的 mask 处理）。
2. **DC-AE decode**：`AutoencoderDC` 无 `shift_factor`，scaling 语义与 KL-VAE 不同；`decode_latents`
   照 SanaPipeline 原实现对齐（latent 通道 32）。
3. **CFG**：SANA 有真 CFG 双分支（非 guidance-distilled）→ 复用 `batched_cfg` 模式，runner 照 qwen_image。
4. 复杂度低是选它当首个样例的原因：无 packed latent（对比 flux）、无三编码器（对比 sd3_5）。

## 3. 落地文件（Phase 0 薄形状）

```text
vrl/models/diffusion/sana/model.py      # SanaModel(LoraModelMixin, DiffusionModelBase) + SanaReplayModel
vrl/models/diffusion/sana/runner.py     # SanaDiffusionBackboneRunner（mask + batched_cfg，照 qwen_image）
vrl/models/diffusion/sana/runtime.py    # 薄 stub（委托共享 builder，transformer_classname="SanaTransformer2DModel"）
                                        #   + SanaChunkExecutor（build_chunk_encoded：repeat embeds + mask）
vrl/models/diffusion/sana/__init__.py
vrl/rollouts/families/registry.py       # +1 条 _diffusion_entry(family="sana", task="t2i", ...)
configs/model/diffusion/sana/1600m.yaml # 照 sd3_5/medium.yaml；lora target_modules 按 SANA 线性注意力命名核对
tests/models/diffusion/sana/            # 结构性用例（不断言配置字面值）
```

wiring 测试参数化追加 `build_sana_replay_runtime_bundle`（照 tick 2 给 flux/qwen_image 的加法）。

## 4. 验收

- **CPU 结构**：wiring / config-resolve / registry round-trip 绿（无权重可跑）。
- **GPU parity（必须，见 index sprint §6.3.1）**：真权重跑 1 次生成，与 diffusers `SanaPipeline`
  同 prompt/seed/steps 数值 sanity；`debug.first_step` 断言 rollout logprob == replay logprob。
- **训练信号**：固定 prompt 集短 GRPO，reward >2σ 抬升（判据见记忆 `project_first_trustworthy_curve`）。

## 5. 非目标

- 不接 SANA-Video / SANA 4K / SANA-Sprint（few-step 蒸馏版，轨迹太短不适配 GRPO）。
- 不动 `common/*` / 算法层；mask 线程若逼改共享层，显式记录。

## 参考

- https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_diffusers
- diffusers SanaPipeline：https://huggingface.co/docs/diffusers/api/pipelines/sana
- 模板：`vrl/models/diffusion/qwen_image/{model,runner,runtime}.py`；共享 builder：`vrl/models/diffusion/build.py`
