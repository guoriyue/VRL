# SPRINT: 接入 HunyuanVideo T2V 家族（套 Wan/Cosmos 5D 视频 seam）

状态：**DONE（2026-07-08）——随 [[SPRINT_thin_model_seam_and_ten_model_expansion]] Phase 1 落地并
真权重 rollout 验证（replay parity 0.0e+00；13B + tiled decode）。短 GRPO 曲线未单独跑（战役按
rollout 验证关账，详见 index sprint 文件头验证记录）。**
性质：新增 T2V 视频扩散家族，套现有 5D 潜变量视频 seam。
> 拆自 [[SPRINT_thin_model_seam_and_ten_model_expansion]] §3 第 5 项；
> [[SPRINT_model_family_coverage]] Tier-2 已点名。

## 0. 一句话

**HunyuanVideo 是 13B flow-matching T2V DiT——能套我们最厚的一侧（Wan/Cosmos 视频 seam），
主要技术点是它像 FLUX 一样是"embedded guidance"单分支，runner 照 flux 而非 wan。**

## 1. 模型事实（落地第一步核对 diffusers 类名）

- checkpoint：`tencent/HunyuanVideo`（13B DiT，dual-stream→single-stream 结构）。
- 预期 diffusers 类：`HunyuanVideoPipeline` / `HunyuanVideoTransformer3DModel` /
  `AutoencoderKLHunyuanVideo`（diffusers ≥0.32；本机核对签名）。
- 文本编码器：**LLaVA-LLaMA-3-8B（MLLM，出序列 embeds + mask）+ CLIP-L（出 pooled）**。
- **embedded guidance（蒸馏 CFG）**：官方推理 `guidance_scale=1.0` + `embedded_guidance_scale=6.0`，
  单分支 + guidance 标量 → **runner 照 flux 的 single+guidance 模式**，不是 wan 的双分支。
- 调度器：flow-matching（动态 shift 同 FLUX 家族风格）→ 算法层零改动；shift 处理照 flux 落地经验
  （[[SPRINT_flux_t2i]] §0.1.2 的 `mu` 延后设置）。
- 潜变量：5D `[B,C,T,H,W]`，3D causal VAE → 套 wan 的视频 latent 布局与 `ChunkedLatentDecoder` 视频路径。

## 2. 技术点

1. **单分支 + guidance 的视频版**：flux runner 模式 × wan 的 5D 布局，两个现成件的组合。
2. 13B + 8B 文本编码器 → **LoRA-only + 编码器 CPU offload**（flux 经验直接搬）。
3. 显存：视频 rollout 是大头；对齐 [[SPRINT_video_rollout_stage_overlap]] 的分阶段 offload 纪律，
   起步用小分辨率/短帧数配置（如 61 帧 → 起步 17/33 帧）。

## 3. 落地文件

```text
vrl/models/diffusion/hunyuan_video/{model,runner,runtime,__init__}.py   # 模板：wan_2_1 的 5D + flux 的 runner
vrl/rollouts/families/registry.py                                       # +1 条（default_task_type="text_to_video"）
configs/model/diffusion/hunyuan_video/13b.yaml                          # use_lora: true 强制
tests/models/diffusion/hunyuan_video/                                    # 结构性用例 + wiring 参数化
```

注意：**replay builder 大概率照 wan 自建**（视频家族多组件），rollout 侧评估能否直接委托共享
`build_diffusion_runtime_bundle`——wan 没委托是因为多 transformer 变体，HunyuanVideo 单 transformer，
预期可以委托（这会是第一个委托共享 builder 的视频家族，验证 seam 通用性）。

## 4. 验收

- CPU 结构绿；GPU parity：真权重 1 段短视频（同 prompt/seed/steps）对照官方 pipeline 数值 sanity +
  `debug.first_step` logprob 一致。
- 短 GRPO（视频 reward 用现有 video reward 栈）曲线判据同 index sprint §5。

## 5. 非目标

- 不接 HunyuanVideo-I2V / Avatar 变体；不做 FP8 官方权重路径（先 bf16 LoRA）。
- 不动视频共享层；帧数/分辨率的显存问题用配置解决，不改架构。

## 参考

- https://huggingface.co/tencent/HunyuanVideo
- diffusers：https://huggingface.co/docs/diffusers/api/pipelines/hunyuan_video
- 模板：`vrl/models/diffusion/wan_2_1/`（5D 布局）+ `vrl/models/diffusion/flux/runner.py`（single+guidance）
