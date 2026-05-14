# SPRINT：各 family 最小 replay module 加载分析

状态：planned。

## 核心目标

这份 sprint 专门回答：

```text
每个模型在 trainer replay 阶段到底需要加载哪些 module？
哪些 module 只属于 rollout / reward / decode，不应该出现在 trainer 进程？
```

它接在 `SPRINT_sd3_trainer_memory_root_fix.md` 后面做。前一个 sprint 已经准备 shared infra：

```text
ReplayPolicy protocol
RuntimeBundle metadata
trainable-only weight sync
host memory guard
```

这份 sprint 才实现 family-specific minimal replay loader。

## 统一原则

rollout policy 和 trainer replay policy 不是同一个职责：

```text
rollout policy:
  prompt encode
  sampling / decode loop
  VAE / VQ decode
  reward-visible artifact generation
  trajectory emission

trainer replay policy:
  restore state from TrajectoryBatch / TrainingView
  recompute trainable forward
  produce logprob / logits / model_pred for loss
  load trainable state pushed from trainer
```

最终目标：

```text
Ray rollout worker keeps full generation policy.
Trainer process loads minimal replay policy.
```

## 不该做的事

不能把 generation-only module 用 lazy 属性藏在 trainer policy 里：

```text
text encoder
VAE
VQ decoder
safety checker
image processor
full diffusers pipeline
full upstream generation wrapper
```

如果某个 family replay 真的需要 scheduler math，也应该优先存成 lightweight scheduler config / timestep tensors，而不是把完整 pipeline 留在 trainer。

## Family 分析矩阵

### SD3.5

当前 full policy：

```text
vrl/models/families/sd3_5/policy.py
vrl/models/families/sd3_5/runtime.py
```

当前 trainer 会加载：

```text
StableDiffusion3Pipeline
transformer
text_encoder / text_encoder_2 / text_encoder_3
VAE
scheduler
image processor
```

trainer replay 理论上只需要：

```text
transformer
LoRA adapter / trainable transformer params
restore_eval_state(...)
forward_step(...)
disable_adapter(...)
load_trainable_state(...)
```

trajectory / replay 已经应该携带：

```text
observations / actions / timesteps
prompt_embeds
pooled_prompt_embeds
negative_prompt_embeds
negative_pooled_prompt_embeds
old_log_prob
mask
```

候选新增文件：

```text
vrl/models/families/sd3_5/replay_policy.py
vrl/models/families/sd3_5/replay_runtime.py
tests/models/test_sd3_5_replay_policy.py
```

风险点：

- 需要确认 `flow_matching` evaluator 不再依赖完整 scheduler object。
- `restore_eval_state` 需要从 replay tensors 和 batch context 重建足够状态。
- fp32 LoRA replay 路径不能退化，当前 SD3.5 OCR recipe 已经依赖 full precision。

### Wan 2.1

当前 full policy：

```text
vrl/models/families/wan_2_1/diffusers_policy.py
vrl/models/families/wan_2_1/runtime.py
```

当前 full policy 主要拥有：

```text
WanPipeline
transformer
text_encoder
VAE
scheduler
```

trainer replay 理论上只需要：

```text
transformer
LoRA adapter / trainable transformer params
prompt_embeds / negative_prompt_embeds from trajectory
video latent observations / actions / timesteps
```

不应该在 trainer 持有：

```text
text_encoder
VAE
decode_video path
full WanPipeline
```

候选新增文件：

```text
vrl/models/families/wan_2_1/replay_policy.py
vrl/models/families/wan_2_1/replay_runtime.py
tests/models/test_wan_2_1_replay_policy.py
```

风险点：

- Wan video latent shape 是 5D，replay policy 必须复用现有 trajectory axis 语义。
- 如果 scheduler object 只用于 timestep/sigma 查表，应改成 replay context tensor/config。

### Cosmos Predict2

当前 full policy：

```text
vrl/models/families/cosmos/predict2/policy.py
vrl/models/families/cosmos/predict2/runtime.py
```

当前 full policy 可能拥有：

```text
Cosmos2VideoToWorldPipeline
transformer
text encoder
VAE / decoder
safety checker / guardrail components
conditioning helpers
```

trainer replay 理论上只需要：

```text
transformer
LoRA adapter / trainable transformer params
prompt embeddings
init_latents
conditioning masks / indicators
fps / sigma conditioning
trajectory latents / timesteps
```

不应该在 trainer 持有：

```text
safety checker
text encoder
VAE decode path
full pipeline object
```

候选新增文件：

```text
vrl/models/families/cosmos/predict2/replay_policy.py
vrl/models/families/cosmos/predict2/replay_runtime.py
tests/models/test_cosmos_predict2_replay_policy.py
```

风险点：

- Cosmos conditioning 比 SD3/Wan 更复杂，不能把 conditioning 隐式留在 full policy object。
- `TrajectoryBatch.context` / segment replay inputs 必须显式携带 replay 所需 tensor。

### Cosmos Predict2.5

当前 full policy：

```text
vrl/models/families/cosmos/predict2_5/policy.py
vrl/models/families/cosmos/predict2_5/runtime.py
```

trainer replay 理论上只需要：

```text
transformer / diffusion model
LoRA adapter / trainable params
DiffusionNFT replay inputs when enabled
prompt / conditioning tensors from trajectory
latents / timesteps / masks
```

不应该在 trainer 持有：

```text
text encoder when replay tensors already contain prompt conditioning
VAE / video decode path
full pipeline object
```

候选新增文件：

```text
vrl/models/families/cosmos/predict2_5/replay_policy.py
vrl/models/families/cosmos/predict2_5/replay_runtime.py
tests/models/test_cosmos_predict25_replay_policy.py
```

风险点：

- Predict2.5 需要先审清 `skip_text_encoder` 和 DiffusionNFT path。
- 不能把 NFT reference prediction 隐式依赖藏在 full pipeline 里。

### Janus-Pro

当前 full policy：

```text
vrl/models/families/janus_pro/policy.py
vrl/models/families/janus_pro/runtime.py
```

当前 full policy 主要拥有：

```text
multimodal causal LM
gen_head / image token projection
VQ model
processor / tokenizer
vision encoder / aligner
decode_image_tokens(...)
```

trainer replay 理论上只需要：

```text
language model trainable path
gen_head / image token logits path
prompt token ids / masks from trajectory
sampled image token ids
old logprob / mask
disable_adapter(...)
load_trainable_state(...)
```

不应该在 trainer 持有：

```text
VQ decoder
image decode path
understanding vision encoder, unless replay explicitly needs image-reference encoding
```

候选新增文件：

```text
vrl/models/families/janus_pro/replay_policy.py
vrl/models/families/janus_pro/replay_runtime.py
tests/models/test_janus_pro_replay_policy.py
```

风险点：

- Janus KV-cache rollout optimization 是另一个 sprint，不能和 minimal replay loader 混在一起。
- replay policy 需要保留 image token logits，而不是只支持 text logits。

### Janus-Pro-R1

Janus R1 和 Janus-Pro 共享一部分模型，但 replay scope 更复杂：

```text
first_gen segment
compre / self-check segment
final_gen segment
```

trainer replay 理论上需要：

```text
image-token logits for generation segments
text-token logits for comprehension/self-check segment
per-segment mask / old_logprob / advantage scope
```

不应该在 trainer 持有：

```text
VQ decode path
reward image decode-only helpers
rollout-only sampling state
```

风险点：

- 不能把 R1 multi-segment 重新塞回 `extra["r1_segments"]`。
- replay policy 必须通过 `TrajectoryBatch` segment refs 恢复每段训练视图。

### NextStep-1

当前 full policy：

```text
vrl/models/families/nextstep_1/policy.py
vrl/models/families/nextstep_1/runtime.py
```

当前 full policy 主要拥有：

```text
language_model
image_head / flow head
image_in_projector
VAE
processor
upstream pipeline wrapper
```

trainer replay 理论上只需要：

```text
language_model
image_head / flow head
image_in_projector
prompt ids / masks from trajectory
continuous image tokens
saved_noise
old logprob / mask
```

不应该在 trainer 持有：

```text
VAE decode path
processor-only prompt encode path
full upstream pipeline wrapper
```

候选新增文件：

```text
vrl/models/families/nextstep_1/replay_policy.py
vrl/models/families/nextstep_1/replay_runtime.py
tests/models/test_nextstep_1_replay_policy.py
```

风险点：

- 当前 NextStep upstream binding 还有 TODO，minimal replay loader 可能要等 `_load_pipeline` / projector / VAE decode 边界稳定后再落。
- replay 需要 continuous token / saved noise，不是 Janus 那种 discrete token logprob。

## 实施顺序

建议顺序：

```text
1. SD3.5 minimal replay loader
2. Janus-Pro minimal replay loader
3. Wan 2.1 minimal replay loader
4. Cosmos Predict2 / Predict2.5 minimal replay loader
5. Janus-Pro-R1 multi-segment replay loader
6. NextStep-1 minimal replay loader
```

理由：

- SD3.5 是当前真实 OOM 的来源，优先级最高。
- Janus-Pro 可以验证 AR family 的最小 replay policy。
- Wan/Cosmos 是 video diffusion，conditioning 和显存/内存压力更复杂。
- R1 和 NextStep 涉及 multi-segment / continuous token，适合在基础 family 跑通后做。

## 每个 family 的完成标准

每个 family 完成时都必须满足：

```text
trainer bundle metadata:
  runtime_role: minimal_replay_policy
  loads_full_generation_modules: false
  requires_minimal_replay_loader: false

rollout worker bundle metadata:
  runtime_role: full_generation_policy
  loads_full_generation_modules: true
```

测试必须覆盖：

- replay policy 不暴露 decode-only module。
- replay policy 不持有 full pipeline / upstream wrapper。
- evaluator 能用 replay policy 从 trajectory 复算 signal。
- weight sync payload 只包含 trainable params。
- rollout worker 仍能加载 full generation policy。
- colocated strict memory guard 对 minimal replay policy 不报错。

## 第一轮只分析什么

第一轮不直接写所有 family loader。先做：

```text
1. 审每个 family policy/runtime，列出 full-only module 和 replay-needed module。
2. 审每个 family trajectory replay inputs，确认 replay 是否缺 tensor。
3. 为 SD3.5 写具体 implementation plan。
4. 为 Janus-Pro 写 AR replay loader plan。
5. 标出 Wan/Cosmos/NextStep 的 blocker。
```

第一轮完成后，再开具体 implementation sprint。
