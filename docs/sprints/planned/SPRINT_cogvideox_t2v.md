# SPRINT: 接入 CogVideoX T2V 家族（⚠️ 带调度器 KILL-RISK 门）

状态：planned（2026-07-01）。性质：新增 T2V 家族（Zhipu/GLM 系），**与 PixArt-Σ 共享同一个架构门**。
> 拆自 [[SPRINT_thin_model_seam_and_ten_model_expansion]] §3 第 6 项。

## 0. 一句话 + KILL-RISK 门

**CogVideoX 是 v-prediction + DDIM 家族调度器，不是 flow-matching**——与 PixArt-Σ 撞同一堵墙：
我们的 logprob 层只有 flow-matching（证据同 [[SPRINT_pixart_sigma_t2i]] §0）。二选一：

- **门 A（扩展）**：等/共享 [[SPRINT_pixart_sigma_t2i]] 门 A 的 DDIM-SDE logprob 数学，
  再加 v-prediction → epsilon 换算分支（v-pred 的 logprob 参数化要单独推导+测试）。
- **门 B（换模型）**：换成 flow-matching T2V。候选：**LTX-Video 2**（但 echo 已是 LTX 系，重复度高）、
  **StepVideo / Allegro**（核对调度器后定）。选门 B 则本 sprint 关闭并在 index 记录。

**决策顺序建议：先做 PixArt 的门 A（图像侧便宜），成了 CogVideoX 跟进；不成两个一起换。**

## 1. 模型事实（落地第一步核对）

- checkpoint：`THUDM/CogVideoX-5b`（5B，Zhipu/GLM 系——与 GLM-Image 同门，用户点名 GLM 生态）。
- 预期 diffusers 类：`CogVideoXPipeline` / `CogVideoXTransformer3DModel` / `AutoencoderKLCogVideoX`
  （老牌支持，仍须核对）。
- 文本编码器：T5-XXL（embeds + mask，无 pooled）。
- **调度器：v-prediction + CogVideoX DDIM/DPM**（KILL-RISK 根源）。
- 潜变量：5D，3D causal VAE（temporal 压缩 4x）→ 套 wan 视频布局。

## 2. 技术点（过门后）

1. v-pred logprob：门 A 产出上加 `prediction_type="v_prediction"` 分支，parity 测试对 epsilon 版回归。
2. runner：真 CFG 双分支（wan 模板）；mask 线程照 qwen_image。
3. 3D VAE tiling：CogVideoX VAE 显存重，`memory.vae_decode` tiling 必开。

## 3. 落地文件

同 [[SPRINT_hunyuan_video_t2v]] §3 形状（`vrl/models/diffusion/cogvideox/` 四件 + registry + yaml + tests），
另依赖门 A 的 `vrl/math/diffusion/ddim.py`（v-pred 分支）。

## 4. 验收

- 门 A（v-pred 分支）单独 parity 验收后才开工。
- 其余同 [[SPRINT_hunyuan_video_t2v]] §4（CPU 结构 + GPU 视频 parity + 短 GRPO）。

## 5. 非目标

- 不接 CogVideoX1.5-I2V；不接 CogVideoX-2b（5b 是主线）。
- 门 B 被选时本 sprint 关闭，替换模型另开 sprint。

## 参考

- https://huggingface.co/THUDM/CogVideoX-5b
- KILL-RISK 证据：同 [[SPRINT_pixart_sigma_t2i]] 参考节
- 模板：`vrl/models/diffusion/wan_2_1/`
