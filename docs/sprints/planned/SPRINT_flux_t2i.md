# SPRINT: 接入 FLUX.1 [dev] T2I 家族（套现有 diffusion seam）

状态：**LANDED（2026-06-21）** — 四文件 + registry + model yaml + smoke recipe + train entrypoint
全部落地；CPU 单测绿（`tests/models/diffusion/flux/`）；**单 GPU(RTX 5090 32GB) naive 前向跑通**
（bf16/fp16：256² 4 步，~0.23s/step，peak ~22.9GB，82% compute-bound）。落地中发现并修复两处
真实 seam 缺口（见下 §0.1）。fp32 naive OOM（transformer 48GB > 32GB，已记录）。vLLM-Omni 对照见
[`docs/flux_qwen_naive_vs_vllm_omni_profiling.md`](../../flux_qwen_naive_vs_vllm_omni_profiling.md)。性质：新增一个 T2I DiT 家族，套现有四文件模式，不碰算法/rollout/loader 共享层。

## 0.1 落地实测发现（超出原 plan 的两处真实修复）

1. **大文本编码器必须 CPU offload 才能单卡跑**：FLUX.1-dev 的 T5-XXL(~9.4GB) + transformer(~24GB,bf16)
   合计 ~33GB > 32GB，全驻留单卡 OOM。修复：`from_spec` 把冻结编码器放 CPU，`encode_prompt` 在编码器
   设备上跑再把 embeds 搬到 GPU（`enable_model_cpu_offload` 纪律）。这是"单卡端到端跑通"的前置必需，
   不是可选优化。
2. **FlowMatch 动态 shifting 需要按分辨率算 `mu`**：FLUX 调度器 `use_dynamic_shifting=True`，
   `set_timesteps` 必须传分辨率派生的 `mu`（build 期不知道分辨率）。修复：`set_num_steps` 对动态
   shifting 调度器延后设置，`prepare_sampling` 先建 latents 再用 `calculate_shift(image_seq_len,...)`
   设 timesteps（复用 diffusers 原函数，保证 parity）。

> 拆自 [[SPRINT_model_family_coverage]]（竞品覆盖路线图）的 Tier-1 第 1 项。
> 相关：[[SPRINT_flow_grpo_recipe_parity]]（flow_grpo 母体已含 **Flux** 配方，配方风险已被验证）、
> 记忆 `project_first_trustworthy_curve`（diffusion GRPO 判据）。

## 0. 一句话

**FLUX.1 是社区 T2I 事实基线、竞品都支持、我们没有；接它 ≈ 1 条 registry + 4 个文件 + 1 个 yaml。**
唯一的结构性差异是 **`FLUX.1-dev` 是 guidance-distilled——没有真正的 CFG 双分支**，靠一个 `guidance` 标量
做条件注入，所以 `runner.py` 需要一个单分支 + guidance 的新 `cfg_mode`，这是本 sprint 的主要技术点。

## 1. 模型事实（接入前必须对齐的形状）

- **FLUX.1-dev**：12B rectified-flow transformer（Black Forest Labs），flow-matching，16 通道 VAE，
  **packed latent**（2×2 patchify 成 token 序列，非 SD3 的 `[B,C,H,W]` 直接卷）。
- **双文本编码器**：CLIP（出 `pooled_prompt_embeds`）+ T5-XXL（出 `prompt_embeds` 序列）。
  `encode_prompt` 要同时产出这两支，喂给 transformer 的 `encoder_hidden_states`（T5）+ `pooled_projections`（CLIP）。
- **guidance-distilled**：dev 版把 CFG 蒸进了单次前向，**没有 negative 分支**；diffusers pipeline 用
  `guidance` 标量（默认 3.5）作为额外条件输入。→ 我们的 `DiffusionBackboneRunner` 默认
  `cfg_mode="batched_cfg"`/双分支在这里不适用。

## 2. 主要技术点：单分支 + guidance 的 runner

现有 `SD3DiffusionBackboneRunner`（`vrl/models/diffusion/sd3_5/runner.py:46-91`）是
`cfg_mode="batched_cfg"`、`cfg_base="uncond"`、`build_branch` 出 cond/uncond 两支。

FLUX 要新增一个 **单分支** 模式：
- `cfg_mode = "single"`（无 uncond 分支）；
- `build_branch` 只构 cond 分支，并在 `extra_kwargs` 注入 `guidance`（`torch.full((B,), guidance_scale)`）；
- `finalize_noise_pred` 直接返回 cond 输出（不做 `uncond + s*(cond-uncond)` 合成）。

> 若 `common/cfg.py`/backbone 的 batching 假设了"必有两支"，优先在 runner 层用单分支绕过；
> **确实绕不过再动 `common/`，但要在本 sprint 显式记录是哪个假设逼着改共享层**（见非目标）。

## 3. 落地：要动的文件

1. **Registry 一条目**（`vrl/rollouts/families/registry.py`，照 sd3_5 第 132-143 行）：
   ```python
   register_rollout_family(_diffusion_entry(
       family="flux", task="t2i", aliases=(),
       executor_cls="vrl.models.diffusion.flux.runtime:FluxChunkExecutor",
       runtime_builder="vrl.models.diffusion.flux.runtime:build_flux_runtime_bundle",
       runtime_spec_extractor="vrl.models.diffusion.flux.runtime:extract_flux_runtime_spec",
       request_prefix="flux", default_task_type="text_to_image",
   ))
   ```
2. **家族目录** `vrl/models/diffusion/flux/`（四文件）：
   - `model.py`：`FluxModel(DiffusionModelBase)` + `FluxReplayModel`。实现 `base.py:41-96` 的
     `encode_prompt`（双编码器，出 T5 序列 + CLIP pooled）/ `prepare_sampling`（packed latent 初始化）/
     `forward_step`（注入 guidance）/ `decode_latents`（unpack→VAE decode）+ replay 三件套。
   - `runner.py`：`FluxDiffusionBackboneRunner`——**单分支 + guidance（本 sprint 核心）**。
   - `runtime.py`：`build_flux_runtime_bundle` / `build_flux_replay_runtime_bundle` /
     `extract_flux_runtime_spec(... task_variant="t2i")` / `FluxChunkExecutor`，
     复用 `vrl/models/loader.py` 的 `load_diffusers_transformer / load_flow_match_scheduler /
     apply_lora_to_transformer`。
   - `__init__.py`：导出。
3. **Config** `configs/model/diffusion/flux/dev.yaml`（照 `sd3_5/medium.yaml`，31 行）：
   `family: flux`、`path: black-forest-labs/FLUX.1-dev`、`use_lora: true`、`lora.target_modules`
   （Flux 的 `to_q/to_k/to_v/to_out.0` + `add_q_proj/add_k_proj/add_v_proj/to_add_out`）、`memory`（VAE tiling）、
   `torch_compile`。
4. **共享层不改**：`common/*`、`flow_matching.py`、`loader.py` 直接复用（除非 §2 的单分支被 batching 假设逼改）。

## 4. 验收

- **推理**：`flux/dev.yaml` 经 registry 解析→构建 runtime bundle→生成 N 张图、VAE decode 正常，
  与 diffusers 官方 `FluxPipeline`（同 prompt/seed/steps/guidance）做视觉 + 数值 sanity 对照。
- **训练**：套 [[SPRINT_flow_grpo_recipe_parity]] 配方（`eps_clip=1e-3`、KL 开、`global_std`、group≥16、
  lr=1e-4），固定 prompt 集跑短 GRPO，BLOCK test 判读 reward >2σ 单调抬升（不把噪声当 learning，
  见 `project_first_trustworthy_curve`）。
- **测试**：照 `tests/models/diffusion/` 加家族注册 + `from_spec/encode_prompt/forward_step/decode_latents`
  结构性用例（不断言配置字面值，见 `feedback_no_exact_config_tests`）。

## 5. 非目标

- **不接 `FLUX.1-schnell`**：4 步 timestep-distilled，GRPO 轨迹太短不适配；只接 `FLUX.1-dev`。
- **不做 Flux 的 image-edit / Kontext / ControlNet 变体**：本 sprint 只做纯 T2I。
- **不改 diffusion 共享层**：新家族应只靠 model/runner/runtime 适配；§2 若被迫动 `common/`，须显式记录原因。

## 参考

- diffusers FluxPipeline：https://huggingface.co/docs/diffusers/api/pipelines/flux
- FLUX.1-dev：https://huggingface.co/black-forest-labs/FLUX.1-dev
- 接入面证据：`vrl/rollouts/families/registry.py:92-143`、`vrl/models/diffusion/base.py:41-96`、
  `vrl/models/diffusion/sd3_5/{model,runner,runtime}.py`、`configs/model/diffusion/sd3_5/medium.yaml`
