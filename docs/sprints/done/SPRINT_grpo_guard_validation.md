# SPRINT: GRPO-Guard 正确性验证实验（flux + PickScore）

状态：**已完成——仅机制验证（2026-07-18）**。multi-epoch 探针触发了
ratio-mean-bias 与 clipping。短跑没有建立 held-out reward 上升 >2σ 的曲线；该长期结论已
移交可信 reference-curve 计划。

> 来源：实现 `vrl/algorithms/grpo/continuous.py`（`GRPOGuard`）+ recipe
> `vrl/config/presets/recipe/online/flow_matching_grpo_guard.yaml` + 母体论文
> `docs/papers/diffusion-flow-rl/flow-grpo-online-flow-matching-rl.pdf`。
> **注**：GRPO-Guard 无独立 PDF（verl-omni 合成）；判据从算法契约 + FlowGRPO 母体推导。
> 相关：[[SPRINT_flow_dppo_validation]]（同读 proposal mean，但走"丢样本"而非"软修正"）。

## 0. 核心结论（先看这一段）

**GRPO-Guard 与 Flow-DPPO 处理同一个问题（current-vs-rollout 均值漂移），但走相反的设计：
信赖域本身不按 KL 丢样本，而是把漂移软性折进 ratio 指数 + 跨去噪步归一 loss 幅度。**
正确性分成机制闭环与
长期学习两个层次；本 sprint 只关闭机制层：

1. **机制正确性（本 sprint 已关闭）：**在默认 precision-correction 设置下保留所有样本，
   要求 rollout proposal mean 与 `dt`，在 drift 下触发 ratio-mean-bias，并在
   multi-epoch reuse 中触发 clipped surrogate。
2. **学习有效性（本 sprint 未关闭）：**固定 held-out PickScore 必须在长跑中上升 >2σ；短探针
   没有建立该结论。

默认 precision correction 关闭时，GRPO-Guard 的信赖域本身不像 Flow-DPPO 那样 mask。
`clip_fraction` 来自标准 ratio clip（`clip_ratio=1e-4`）；`kl_penalty` 在这里记录
`ratio_mean_bias.mean()`，应**有限且小**。可选的共享 TIS/RS precision correction 仍可拒绝
精度漂移样本；这不是 GRPO-Guard 信赖域的样本丢弃语义。

**跨步幅度归一**：loss 除以 `sqrt_dt_mean²`，让早/晚 timestep 的梯度量级可比——这是
"Guard" 的稳定性来源。判别性实验应在会放大漂移的设置下比较 plain GRPO，但该长期对照没有
在本 sprint 完成。

硬前置：同 Flow-DPPO，读 `old_prev_sample_mean`，recipe 必须 `return_prev_sample_mean: true`
（`flow_matching_grpo_guard.yaml`），否则 `_require_trust_region_signals` fail-fast。

## 1. 算法实锤

### 1.1 ratio-mean-bias + step-scale norm（核心，`GRPOGuard.compute_loss`）

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
  归一会被悄悄抹掉）。
- guard 项**不引入新超参**：全部由每步扩散 scale（`std_dev_t`、`sqrt_dt`）派生
  （见 `GRPOGuardConfig`）。

## 2. 实验设计（data / reward / model）

| 维度 | 选择 | 理由 |
|---|---|---|
| 模型 | **flux/dev** LoRA bf16 256² | 与另两个 GRPO 变体同母体，可三方横比。显存吃紧退 `wan_2_1/1_3b`。 |
| 数据 | **`/dataset/pickscore_sfw`** | Flow-GRPO 的人类偏好 prompt 集（SFW），有 `eval_manifest`（test.txt）可做固定 eval（`vrl/config/presets/dataset/pickscore_sfw.yaml`）。 |
| 奖励 | **`/reward/pickscore`**（CLIP-ViT-H-14 + `PickScore_v1`，fp32，全本地） | Flow-GRPO 的偏好 reward model，**直接对应论文 PickScore task**。比 OCR 重（要下 CLIP-H + PickScore 权重）但全本地、无需接线（`vrl/rewards/functions/pickscore.py`）。 |
| 超参 | `clip_ratio=1e-4`（recipe）、`kl_coef=0`、`global_std=true`、`G=16`（Flow-GRPO 用 24；单卡折中）、`num_steps=10`、`noise_level=0.7`、`lr=1e-4`、`return_prev_sample_mean=true`（recipe 已设） |
| 对照 | plain `flow_matching_grpo`，**同 lr 与 timestep_fraction=1.0**（跨全步训练、放大漂移；Flow-GRPO PickScore 配方 `kl_coef=0.01`） | GRPO-Guard 的卖点是在漂移大时更稳；对照需制造漂移才能体现差别 |

## 3. 可复现入口与观测结果

当前维护的实验配置是
`vrl/config/presets/experiment/flux/online_grpo_guard_pickscore_validation.yaml`。
它选择 GRPO-Guard recipe、PickScore + `pickscore_sfw`、全 timestep 训练、
`ppo_epochs=4`、group size 16 和 `return_prev_sample_mean=true`。

短探针显示了预期的前后差异：

- `ppo_epochs=1`：ratio-mean-bias 与 clip fraction 都为零；
- `ppo_epochs=4`：ratio-mean-bias 为 `0.00028-0.00047`，clip fraction 为
  `0.025-0.095`。

reward 曲线仍在短跑噪声内，且没有完成成对的 plain-GRPO 长期稳定性曲线。这些测量只证明机制
触发，不证明收敛或比较质量。

## 4. 关闭判据与结论

### 4.1 机制闭环——已完成

- 没有 proposal-mean drift 且 scale=1 时，GRPO-Guard 近似共享 Flow-GRPO objective；
  drift 增大时 ratio-mean-bias 随之增大。
- 缺少 `old_prev_sample_mean` 或 `dt` 会 fail fast，typed dispatch 选择 `GRPOGuard`。
- trust-region 算法拒绝 strict on-policy `ppo_epochs=1`，但允许 multi-epoch reuse。
- 短探针使 ratio-mean-bias 与 clipping 都变为非零，同时 bias 保持有限且小。

这些 true/false 路径由 `tests/algorithms/test_flow_dppo_grpo_guard.py` 和
`tests/trainers/online/test_trust_region_engages.py` 固定。

### 4.2 原始完整曲线判据——未完成；已移交

原计划要求固定 held-out PickScore 在约 200-400 次 update 后上升 >2σ，并用成对
plain-GRPO 对照证明 KL/gradient 更平稳。短探针没有建立任一长曲线结论；这些证据归可信
reference-curve 计划所有，而不属于已经关闭的机制 sprint。

## 5. 非目标 / Non-Goals

- **不与 Flow-DPPO 比谁更好**——两者是不同设计取舍（软修正 vs 丢样本），本 sprint 只验
  GRPO-Guard 自身机制；长期稳定性对比未在这里完成。
- **不引入新超参**——guard 项由扩散 scale 派生，保持这一性质。
- **不复现论文绝对数值**——只验证机制，不把短探针解释为长期稳定性证据。
- **不做 PickScore reward-hacking 审计**——Flow-GRPO 警告 PickScore 去 KL 会塌成单一风格；
  本 sprint 不对长期学习、稳定性或多样性作结论。

## References
- 实现：`vrl/algorithms/grpo/continuous.py`（`_require_trust_region_signals`
  + `GRPOGuard`）、`vrl/config/presets/base/algorithm/grpo_guard.yaml`、
  `vrl/config/presets/recipe/online/flow_matching_grpo_guard.yaml`
- 母体论文：`docs/papers/diffusion-flow-rl/flow-grpo-online-flow-matching-rl.pdf`
- proposal-mean / dt 来源：`vrl/rollouts/evaluators/denoise/sde_logprob.py`、
  `vrl/generation/bindings/full_sequence_denoise/executor.py`
- 当前维护的实验：
  `vrl/config/presets/experiment/flux/online_grpo_guard_pickscore_validation.yaml`
- 奖励/数据：`vrl/rewards/functions/pickscore.py`、
  `vrl/config/presets/reward/pickscore.yaml`、
  `vrl/config/presets/dataset/pickscore_sfw.yaml`
- 回归测试：`tests/algorithms/test_flow_dppo_grpo_guard.py`、
  `tests/trainers/online/test_trust_region_engages.py`
