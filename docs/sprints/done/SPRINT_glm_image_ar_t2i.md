# SPRINT: 接入 GLM-Image AR T2I 家族（⭐ 用户点名；混合 AR+decoder）

状态：**DONE（2026-07-08）——随 [[SPRINT_thin_model_seam_and_ten_model_expansion]] Phase 1 落地并
真权重验证（transformers 5.13 升级后落地；原生 1024px CPU 全程验证——1280 prior token 采样 +
20 步 DiT 解码，全战役最佳画质）。短 GRPO 曲线未单独跑（战役按 rollout 验证关账，详见 index
sprint 文件头验证记录）。**
性质：新增混合 AR 家族——10 个里战略权重最高、也是 AR 侧最重的一个。
> 拆自 [[SPRINT_thin_model_seam_and_ten_model_expansion]] §3 第 8 项。**用户点名的 GLM AR 模型。**
> 前置：[[SPRINT_emu3_ar_t2i]] 先落地（纯离散 AR 压力测试 seam），再接本 sprint 的混合形态。

## 0. 一句话

**GLM-Image = 9B AR（GLM-4-9B 底座，出语义 token）+ 7B diffusion decoder（+Glyph Encoder，出像素）。
RL 只训 AR 段（LoRA），decoder 全程 frozen——logprob 打在 AR 语义 token 上，形状最贴 nextstep_1
（AR + 生成 head），不是纯离散 janus_pro。**

## 1. 模型事实（落地第一步核对加载路径——本模型 unknown 最多）

- checkpoint：`zai-org/GLM-Image`（16B 总量：9B AR + 7B diffusion decoder；Glyph Encoder 强文字渲染，
  CVTG-2K Word Accuracy 0.9116）。
- **加载路径待核对**：transformers 原生类 or `trust_remote_code` or zai 官方 SDK——第一个动作是把
  官方推理代码跑通并盘点 tensor contract（哪层出语义 token、decoder 输入是什么）。
- AR 段：GLM-4-9B 架构 → `ARModelBase` 直接继承（Phase 0 已备好），LoRA 打在 LM trunk。
- decoder 段：7B diffusion，**frozen**——对齐 [[SPRINT_frozen_component_preservation]] 的冻结组件纪律
  （CPU offload / 精度独立）。

## 2. KILL-RISK 门（两个，先过再动手）

1. **语义 token logprob 可达性**：RL 需要 AR 段的 per-token logprob。若官方代码的生成循环不暴露
   logits/token id（如只给端到端 pipeline），要评估 fork 生成循环的成本——照 janus/nextstep 的
   `runner.py` 模式重写 decode loop 是预期路径，但若语义 token 不是标准 categorical（如连续 latent），
   数学要重推（届时对齐 nextstep_1 的连续 flow-head logprob）。
2. **decoder frozen 的 reward 通路**：reward 打在 decoder 输出的像素上，梯度只回 AR 段——确认
   trajectory 结构里 decoder 是纯 `vq_decode` 式后处理（janus 的 VQ decode 位置），不进 replay。

## 3. 技术点（过门后）

1. `GLMImageModel(ARModelBase)`：AR 段 LoRA + frozen decoder 挂载（不进 `trainable_modules`）。
2. decode loop 照 `ar/janus_pro/runner.py` + `ARDecodeLoop` 驱动；tokenizer_key 新增 `glm_image`。
3. capability：语义 token 若离散 → `ar_discrete_family_capability("glm_image", "ar_t2i")`；
   若连续 → `ar_continuous_*`（门 1 的核对结果定）。
4. 16B 显存：AR 9B LoRA + decoder 7B frozen(可 offload) → 单卡 bf16 预算可行，rollout 时 decoder
   只在 decode 段驻留。

## 4. 落地文件

```text
vrl/models/ar/glm_image/{model,runner,runtime,__init__}.py   # 模板：nextstep_1（混合头）+ janus_pro（decode loop）
vrl/rollouts/families/registry.py                            # +1 条完整 RolloutFamilyEntry（AR 用全构造）
configs/model/ar/glm_image/16b.yaml                          # use_lora: true 强制
tests/models/ar/glm_image/ + wiring 参数化
```

前置改进（建议先做）：折叠 `build_ar_runtime_bundle(spec, entry)`——见 index sprint 薄化审计，
janus/nextstep 的 bundle 组装 ~18 行逐字重复，接 3 个 AR 新家族前折掉，每家族省一份。

## 5. 验收

- 门 1/2 的 contract 盘点记录进本 sprint（token 类型、logprob 通路、decoder 接口）。
- CPU 结构绿；GPU parity：官方推理 vs 我们的 decode loop 同 seed 出图一致 + per-token logprob 合理性
  （对数和有限、无 NaN、温度响应正确）。
- 短 GRPO：文字渲染 reward（OCR reward 栈现成，见 [[SPRINT_ocr2_reward_backend]]）曲线抬升——
  GLM-Image 的强项正好配我们的 OCR reward。

## 6. 非目标

- 不训 diffusion decoder（LoRA 也不打）；不接 image-to-image 编辑模式；不做 Glyph Encoder 微调。
- 不为它开新 rollout seam——AR 段必须套进现有 ar_discrete/ar_continuous 之一，套不进则回到 index 记录并降级。

## 参考

- https://github.com/zai-org/GLM-Image ；https://huggingface.co/zai-org/GLM-Image
- 模板：`vrl/models/ar/nextstep_1/`、`vrl/models/ar/janus_pro/runner.py`、`vrl/models/ar/base.py`
