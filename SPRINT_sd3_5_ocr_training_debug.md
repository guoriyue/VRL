# SPRINT: SD3.5 OCR GRPO 与 Flow-GRPO 剩余差异收口

## 1. 当前结论

SD3.5 OCR GRPO 的核心训练 math 已经基本对齐 Flow-GRPO：

- SD3.5 Medium
- LoRA rank=32, alpha=64
- AdamW lr=3e-4, beta=(0.9, 0.999), weight_decay=1e-4
- num_steps=10
- train timesteps=9
- guidance_scale=4.5
- noise_level=0.7
- n=8
- rollout_batch_size=8
- gradient_accumulation_steps=4
- 2 optimizer updates / epoch
- KL beta/init_kl_coef=0.04
- eps_clip=1e-4
- adv_clip_max=5
- global_std=true
- per_prompt_stat_tracking=true
- rollout old logprob 与 replay fresh logprob parity 成立
- Ray worker 与 driver LoRA parity 成立
- denoiser transformer 已改为 fp32-like Flow-GRPO 路径

之前最大的硬差异是 SD3 denoiser fp16/autocast path。full-precision smoke 已证明该差异会明显影响训练信号量级，因此它不再只是理论风险。

## 2. 仍然不同的地方

这些差异还没有完全收口：

### 2.1 Seed

```text
Flow-GRPO 默认 seed = 42
当前 YAML trainer.seed = 0
```

影响：

- 会改变 prompt sampling 顺序、初始 latent 随机性、训练曲线。
- 不应该单独造成 200 epoch 完全不涨，但会影响和 Flow-GRPO 曲线逐点对比。

### 2.2 Eval 覆盖面

```text
Flow-GRPO eval 跑 test dataloader
当前 eval.max_prompts = 16
```

影响：

- 当前 eval 太小，曲线噪声大。
- train reward 可以看训练链路，但不能作为泛化效果结论。

### 2.3 Prompt sampling order

```text
Flow-GRPO 使用 DistributedKRepeatSampler
当前使用 sample_prompt_indices(...)
```

影响：

- 两边都保证每个 prompt 有 group samples，但采样顺序不是 byte-for-byte 一样。
- 若要严格复现实验，需要让 SD3 train script 复用 DistributedKRepeatSampler 语义。

### 2.4 EMA cadence

```text
Flow-GRPO 在 training microbatch loop 里调用 ema.step(...)
当前在 optimizer step 后调用 ema.step(...)
```

影响：

- 训练权重本身不受 EMA 影响，主要影响 eval/checkpoint 的 EMA 权重。
- 如果 eval 默认使用 EMA，则曲线可能不可直接比较。

### 2.5 OCR reward implementation

```text
Flow-GRPO 使用 flow_grpo.ocr.OcrScorer
当前使用 vrl.rewards.ocr.OCRReward
```

影响：

- 当前行为很接近，但不是 byte-for-byte 同一个 class。
- PaddleOCR 调用、异常处理、图像格式转换、target text 解析都需要继续保持对齐。

### 2.6 Runtime / logging / checkpoint

```text
Flow-GRPO: Accelerate local training loop
当前: trainer + Ray rollout + weight sync + VRL checkpoint
```

影响：

- logging/checkpoint/Ray launch 实现不同。
- 在已验证 worker/driver parity 的前提下，这些不应该改变 loss math。
- 仍然需要通过 debug hash/version parity 保持 fail-fast。

## 3. 下一步收口

优先顺序：

1. 把 SD3 OCR experiment seed 改成 42，用于和 Flow-GRPO 对齐。
2. 增大 eval 覆盖面，至少提供一个 full-test 或 large-test eval preset。
3. 跑 fp32-like 20-60 epoch 对照，观察 reward、grad_norm、kl_penalty 是否持续高于旧 fp16 run。
4. 若曲线仍不涨，再检查 OCRReward 与 Flow-GRPO OcrScorer 的 byte-level 差异。
5. 最后再考虑是否复刻 DistributedKRepeatSampler 和 EMA cadence。

## 4. 判断标准

短跑 smoke 通过标准：

- `mixed_precision = no`
- `autocast_enabled = false`
- driver/worker parameter dtype 都是 fp32 denoiser
- driver/worker trainable hash 一致
- first-step old/fresh logprob diff 接近 0
- grad_norm 不再停留在旧 fp16 的 `~5e-4` 量级

中跑判断标准：

- 20-60 epoch 内 train reward 或 eval reward 至少出现可解释趋势。
- 若 reward 不涨，必须能从 OCR debug samples 判断是 reward 噪声、prompt 难度、还是 policy 更新无效。

## 5. 相关文件

```text
configs/experiment/sd3_5_ocr_grpo.yaml
vrl/scripts/sd3_5/train.py
vrl/trainers/online.py
vrl/models/families/sd3_5/policy.py
vrl/models/families/sd3_5/builder.py
vrl/engine/diffusion/denoise.py
vrl/rewards/ocr.py
vrl/trainers/data.py
vrl/trainers/checkpointing.py
```
