# SPRINT: 接入 PixArt-Σ T2I 家族（⚠️ 带调度器 KILL-RISK 门）

状态：planned（2026-07-01）。性质：新增 T2I DiT 家族，**但有一个必须先过的架构门**。
> 拆自 [[SPRINT_thin_model_seam_and_ten_model_expansion]] §3 第 2 项。

## 0. 一句话 + KILL-RISK 门（先过门再动手）

**PixArt-Σ 是 epsilon-prediction 扩散（DDPM/DPM-Solver 家族），不是 flow-matching。** 而我们的
logprob 层只有 flow-matching：`DiffusionSDELogProbEvaluator` 自述 "for flow-matching diffusion models"
（`vrl/rollouts/evaluators/diffusion/sde_logprob.py:18`），数学层只有 `vrl/math/diffusion/flow_matching.py`，
**没有 DDIM/epsilon SDE logprob 分支**。所以二选一：

- **门 A（扩展）**：先落一个 `vrl/math/diffusion/ddim.py`（DDPO/DanceGRPO 式 DDIM-SDE logprob）+
  evaluator 分支。这是真实但有界的算法层扩展，**必须独立验证**（DDIM logprob parity 测试）后本 sprint 才开工。
- **门 B（换模型）**：若不想开算法层，把本条目换成一个 flow-matching T2I（候选：Lumina2 已在列表；
  或 SANA 变体），本 sprint 关闭。

**先做门 A/B 决策，再进 §3。不允许"先接上再说"——没有 logprob 数学，接上也训不了。**

## 1. 模型事实（落地第一步核对 diffusers 类名）

- checkpoint：`PixArt-alpha/PixArt-Sigma-XL-2-1024-MS`（0.6B DiT，T5-XXL 文本编码器）。
- 预期 diffusers 类：`PixArtSigmaPipeline` / `PixArtTransformer2DModel`（老牌支持，仍须本机核对）。
- 文本编码器：T5-XXL，出 `prompt_embeds + prompt_attention_mask`，无 pooled → mask 线程照 SANA/qwen_image。
- **调度器：DPM-Solver / DDPM epsilon-prediction**（KILL-RISK 根源，见 §0）。
- KV 压缩注意力（Σ 的效率点）对 seam 无影响（transformer 内部）。

## 2. 技术点（过门后）

1. DDIM-SDE logprob 数学 + `sde_type="ddim"` evaluator 分支（门 A 的产出，独立 sprint 化亦可）。
2. runner：真 CFG 双分支 `batched_cfg`，mask 线程照 qwen_image。
3. VAE 是标准 SD KL-VAE（8x）→ `decode_latents` 照 sd3_5。

## 3. 落地文件

同 [[SPRINT_sana_t2i]] §3 形状（`vrl/models/diffusion/pixart_sigma/` 四件 + registry + yaml + tests），
另加门 A 的 `vrl/math/diffusion/ddim.py` + `tests/math/diffusion/test_ddim_logprob.py`。

## 4. 验收

- 门 A 单独验收：DDIM logprob 与解析解/参考实现 parity 测试绿。
- 其余同 [[SPRINT_sana_t2i]] §4（CPU 结构 + GPU 生成 parity + 短 GRPO 曲线）。

## 5. 非目标

- 不接 PixArt-α（被 Σ 取代）；不做 ControlNet/LCM 变体。
- 门 B 被选中时本 sprint 直接关闭并在 index 记录替换。

## 参考

- https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS
- KILL-RISK 证据：`vrl/rollouts/evaluators/diffusion/sde_logprob.py:16-22`、`vrl/math/diffusion/flow_matching.py`
- DDPO/DanceGRPO 的 DDIM logprob 先例：https://github.com/kvablack/ddpo-pytorch
