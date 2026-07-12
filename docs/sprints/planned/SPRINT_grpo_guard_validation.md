# SPRINT: GRPO-Guard 正确性验证实验（flux + PickScore，planned）

状态：planned（2026-06-21）。性质：**算法正确性验证跑**——为新加的 `grpo_guard`
（FlowGRPO + ratio-mean-bias 修正 + 跨步幅度归一）设计一次可判读的 learning 验证。
模型 flux、数据/奖励对齐母体 Flow-GRPO 的人类偏好任务：**`pickscore` reward + PickScore prompts**
（learned preference reward model，Flow-GRPO 三大 task 之一）。排除 sd3.5 / cosmos。

> 来源：实现 `vrl/algorithms/grpo/continuous.py:321`（`GRPOGuard`）+ recipe
> `configs/recipe/online/flow_matching_grpo_guard.yaml` + 母体论文
> `docs/papers/diffusion-flow-rl/flow-grpo-online-flow-matching-rl.pdf`。
> **注**：GRPO-Guard 无独立 PDF（verl-omni 合成）；判据从算法契约 + FlowGRPO 母体推导。
> 相关：[[SPRINT_flow_dppo_validation]]（同读 proposal mean，但走"丢样本"而非"软修正"）。

## 0. Core Decision（先看这一段）

**GRPO-Guard 与 Flow-DPPO 处理同一个问题（current-vs-rollout 均值漂移），但走相反的设计：
保留每个样本，把漂移软性折进 ratio 指数 + 跨去噪步归一 loss 幅度。** 所以正确性判据：

1. **能把 reward 学起来**（修正项没把梯度弄坏）。
2. **保留全部样本**——不像 Flow-DPPO 会 mask；GRPO-Guard 的 `clip_fraction` 来自标准 ratio
   clip（`clip_ratio=1e-4`），`kl_penalty` 指标在这里记的是 `ratio_mean_bias.mean()`
   （`continuous.py:372`），应**有限且小**。
3. **跨步幅度归一生效**：loss 除以 `sqrt_dt_mean²`（`continuous.py:360`），让早/晚 timestep
   的梯度量级可比——这是 "Guard" 的稳定性来源。判别性实验：在**会放大漂移的设置**下
   （更高 lr / 跨全 timestep 训练），GRPO-Guard 应比 plain GRPO 更稳。

硬前置：同 Flow-DPPO，读 `old_prev_sample_mean`，recipe 必须 `return_prev_sample_mean: true`
（`flow_matching_grpo_guard.yaml`），否则 `_require_trust_region_signals` fail-fast。

## 1. 算法实锤

### 1.1 ratio-mean-bias + step-scale norm（核心，`continuous.py:347-360`）

```python
log_ratio = signals.log_prob - signals.old_log_prob
sqrt_dt_mean = signals.dt.mean()
scale = sqrt_dt_mean * signals.std_dev_t.mean()
mean_diff_sq = (prev_sample_mean - old_prev_sample_mean).pow(2).mean(non_batch)
ratio_mean_bias = mean_diff_sq / (2 * scale.pow(2))          # 把均值漂移投到 log-ratio 尺度
ratio = torch.exp((log_ratio + ratio_mean_bias) * scale)     # 软修正后再 clip
clipped_ratio = torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio)
per_sample_loss = torch.maximum(-adv * ratio, -adv * clipped_ratio)
policy_loss = per_sample_loss.mean() / sqrt_dt_mean.pow(2).clamp_min(1e-12)   # per-step norm
```

- **保留 clipped surrogate**（与 GRPO 同），但 ratio 先被 `ratio_mean_bias` 软修正再 clip。
- `dt` 由 `_require_trust_region_signals` 保证存在（**无静默 fallback-to-1**，否则 step-scale
  归一会被悄悄抹掉，`continuous.py:348-349`）。
- guard 项**不引入新超参**：全部由每步扩散 scale（`std_dev_t`、`sqrt_dt`）派生
  （`GRPOGuardConfig` 注释，`continuous.py:313-318`）。

## 2. 实验设计（data / reward / model）

| 维度 | 选择 | 理由 |
|---|---|---|
| 模型 | **flux/dev** LoRA bf16 256² | 与另两个 GRPO 变体同母体，可三方横比。显存吃紧退 `wan_2_1/1_3b`。 |
| 数据 | **`/dataset/pickscore_sfw`** | Flow-GRPO 的人类偏好 prompt 集（SFW），有 `eval_manifest`（test.txt）可做固定 eval（`configs/dataset/pickscore_sfw.yaml`）。 |
| 奖励 | **`/reward/pickscore`**（CLIP-ViT-H-14 + `PickScore_v1`，fp32，全本地） | Flow-GRPO 的偏好 reward model，**直接对应论文 PickScore task**。比 OCR 重（要下 CLIP-H + PickScore 权重）但全本地、无需接线（`vrl/rewards/functions/pickscore.py`）。 |
| 超参 | `clip_ratio=1e-4`（recipe）、`kl_coef=0`、`global_std=true`、`G=16`（Flow-GRPO 用 24；单卡折中）、`num_steps=10`、`noise_level=0.7`、`lr=1e-4`、`return_prev_sample_mean=true`（recipe 已设） |
| 对照 | plain `flow_matching_grpo`，**同 lr 与 timestep_fraction=1.0**（跨全步训练、放大漂移；Flow-GRPO PickScore 配方 `kl_coef=0.01`） | GRPO-Guard 的卖点是在漂移大时更稳；对照需制造漂移才能体现差别 |

## 3. 落地

新建 `configs/experiment/diffusion/flux/online_grpo_guard_pickscore_validation.yaml`：

```yaml
# GRPO-Guard correctness-validation run: ratio-mean-bias + per-step scale norm.
defaults:
  - /recipe/online/flow_matching_grpo_guard   # sets return_prev_sample_mean: true
  - /model/diffusion/flux/dev
  - /sampling/image/512
  - /sampling/denoise/10_step_cfg_4_5
  - /reward/pickscore          # <- Flow-GRPO human-preference task (local CLIP-H + PickScore_v1)
  - /dataset/pickscore_sfw
  - _self_

precision:
  training:
    dtype: bf16
sampling: { height: 256, width: 256, num_steps: 10, max_sequence_length: 64 }

actor:
  optim: { lr: 1.0e-4 }
  gradient_checkpointing: true
  timestep_fraction: 1.0       # train across all denoise steps -> exercises the per-step norm

algorithm:
  clip_ratio: 1.0e-4
  global_std: true

rollout:
  n_samples_per_prompt: 16     # Flow-GRPO uses 24; 16 = single-GPU floor (watch zero_std_ratio)
  rollout_batch_size: 8
  sample_batch_size: 1
  noise_level: 0.7
  return_prev_sample_mean: true        # REQUIRED (recipe default; explicit here)
  sde: { window_range: [0, 10] }

trainer:
  entrypoint: vrl.scripts.diffusion.flux.train:train_flux_grpo
  output_dir: outputs/flux_grpo_guard_pickscore_validation
  total_epochs: 300
  save_freq: 50
  debug: { first_step: true }
  eval: { enabled: true, freq: 25, samples_per_prompt: 2, max_prompts: 32, seed: 20260621 }

model: { torch_compile: { enable: false } }
```

## 4. 判据（finishing criteria）

- **学习信号**：固定 PickScore test 集 `eval_reward_mean` 单调上升 >2σ，over ~200-400 更新。
  锚点：Flow-GRPO PickScore **21.72→~23.3**（~1400 steps）——本验证只验上升方向/相对稳定性。
- **保留全样本**：与 Flow-DPPO 不同，GRPO-Guard 不丢样本——确认 loss 用 `.mean()` 全样本
  （`continuous.py:360`），无 mask 分支。
- **guard 项有界**：`kl_penalty`（= `ratio_mean_bias.mean()`）**有限且小**（不随训练发散）。
  若它爆炸 → `scale`/`dt` 取值或 `old_prev_sample_mean` 链路有问题。
- **step-scale 归一生效**：与对照 plain GRPO（`timestep_fraction=1.0`、同 lr）相比，GRPO-Guard
  的 `approx_kl` / grad_norm 跨训练应**更平稳**（早/晚步幅度被归一）；plain GRPO 在跨全步 +
  漂移下应更抖或更易塌。这是 "Guard" 的判别性证据。
- **dt 硬契约**：删 `return_prev_sample_mean` 或制造缺 `dt` 的输入，应 fail-fast
  （`continuous.py:217`）——证明无静默 fallback-to-1（否则 step-scale 归一被悄悄关掉）。
- **first-step**：step 0 `ratio_mean_bias≈0`（rollout==replay，漂移为 0）、`ratio≈1`。

## 5. 非目标 / Non-Goals

- **不与 Flow-DPPO 比谁更好**——两者是不同设计取舍（软修正 vs 丢样本），本 sprint 只验
  GRPO-Guard 自身正确 + 稳定性卖点成立。
- **不引入新超参**——guard 项由扩散 scale 派生，保持这一性质。
- **不复现论文绝对数值**——验证机制与稳定性**方向**。
- **不做 PickScore reward-hacking 审计**——Flow-GRPO 警告 PickScore 去 KL 会塌成单一风格；本
  sprint 只验 GRPO-Guard 学起来 + 稳定，多样性塌缩另议。

## References
- 实现：`vrl/algorithms/grpo/continuous.py:195-224,312-376`（`_require_trust_region_signals`
  + `GRPOGuard`）、`configs/base/algorithm/grpo_guard.yaml`、
  `configs/recipe/online/flow_matching_grpo_guard.yaml`
- 母体论文：`docs/papers/diffusion-flow-rl/flow-grpo-online-flow-matching-rl.pdf`
- proposal-mean / dt 来源：`vrl/rollouts/evaluators/diffusion/sde_logprob.py`、
  `vrl/generation/diffusion/executor.py:203-209`
- 基线 config：`configs/experiment/diffusion/flux/online_grpo_smoke_single_gpu.yaml`
- 奖励/数据：`vrl/rewards/functions/pickscore.py`、`configs/reward/pickscore.yaml`、
  `configs/dataset/pickscore_sfw.yaml`
