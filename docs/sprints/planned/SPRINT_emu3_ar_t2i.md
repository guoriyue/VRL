# SPRINT: 接入 Emu3 AR T2I 家族（GLM-Image 的前置压力测试）

状态：planned（2026-07-01）。性质：新增纯离散 AR T2I 家族，AR seam 的第二个 discrete 样例。
> 拆自 [[SPRINT_thin_model_seam_and_ten_model_expansion]] §3 第 9 项。
> 战略定位：**GLM-Image 之前先接它**——最干净的 next-token AR，把 ARModelBase + decode loop 在
> janus 之外的第二个家族上验证一遍，GLM-Image 的混合形态风险就只剩 decoder 部分。

## 0. 一句话

**Emu3 是"纯 next-token 预测"的 8B AR T2I（BAAI）：单一 transformer + MoVQ 视觉 tokenizer，
无 diffusion、无混合头——ar_discrete seam（janus_pro 模板）几乎原样套。**

## 1. 模型事实（落地第一步核对 transformers 类名）

- checkpoint：`BAAI/Emu3-Gen`（8B，LLaMA 系架构）。
- 预期 transformers 原生类：`Emu3ForConditionalGeneration` + `Emu3Processor`（transformers ≥4.48 加入；
  本机核对版本与签名——若本机 transformers 过旧，升级评估是第一动作）。
- 视觉 tokenizer：SBER-MoVQGAN（离散 codebook，图像 → token 序列 → 图像）。
- 生成：标准 categorical next-token + CFG（cond/uncond 双序列，janus 同模式）→ per-token logprob
  即 GRPO `old_log_prob`，`ar_discrete_family_capability` 直接用。

## 2. 技术点

1. decode loop：`ARDecodeLoop` + 新 `Emu3ARModelRunner`（照 `janus_pro/runner.py`）；
   tokenizer_key 新增 `emu3`。Emu3 的图像 token 序列含结构 token（行分隔等）——token_mask 要把
   结构 token 排掉或并入（对齐官方 logits processor 的约束解码，核对后定，记录进本 sprint）。
2. 约束解码：官方用 prefix-constrained logits processor 保证 token 网格合法——RL 采样时同样要挂，
   否则 rollout 会产出非法图（这是 Emu3 区别于 janus 的主要点）。
3. `Emu3Model(ARModelBase)` + LoRA 打 LM trunk（`ARModelBase` 的 `disable_adapter` 走
   `self.language_model`，命名对齐）。
4. VQ decode = MoVQ decoder，位置照 janus 的 `engine.vq_decode` 段。

## 3. 落地文件

```text
vrl/models/ar/emu3/{model,runner,runtime,__init__}.py   # 模板：ar/janus_pro 全套
vrl/rollouts/families/registry.py                       # +1 条（collector kind="ar_discrete"）
configs/model/ar/emu3/gen_8b.yaml
tests/models/ar/emu3/ + wiring 参数化
```

前置改进同 [[SPRINT_glm_image_ar_t2i]] §4：先折 `build_ar_runtime_bundle`。

## 4. 验收

- CPU 结构绿；GPU parity：同 seed 对照官方 `generate`（含约束解码）出图一致 + logprob 合理性。
- 短 GRPO：geneval/OCR reward 曲线判据同 index sprint §5。

## 5. 非目标

- 不接 Emu3-Chat / 理解方向（我们只要生成 RL）；不接 Emu3.5。
- 不改共享 `ARDecodeLoop`——约束解码通过 runner 的 logits 处理挂入，逼改共享层则显式记录。

## 参考

- https://huggingface.co/BAAI/Emu3-Gen ；https://github.com/baaivision/Emu3
- transformers Emu3：https://huggingface.co/docs/transformers/model_doc/emu3
- 模板：`vrl/models/ar/janus_pro/`、`vrl/models/ar/base.py`
