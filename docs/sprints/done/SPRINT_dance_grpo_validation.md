# SPRINT: DanceGRPO 正确性验证实验（flux + aesthetic/DrawBench）

状态：**已完成——仅机制验证（2026-07-18）**。随机 timestep selection 与 strided
对照表现一致，且 clipping 已触发。短跑没有建立 held-out reward 上升 >2σ 的证据；该长期
学习结论已移交可信 reference-curve 计划。

> 来源：论文 `docs/papers/diffusion-flow-rl/dancegrpo-visual-generation.pdf`
> (arXiv 2505.07818) + 实现 `vrl/algorithms/grpo/continuous.py`（`GRPO`）+
> recipe `vrl/config/presets/recipe/online/flow_matching_dance_grpo.yaml` + trainer
> timestep 消费点 `vrl/trainers/online/trainer.py`。
> 相关：[[SPRINT_flow_grpo_recipe_parity]]（同一 FlowGRPO 配方母体）。

## 0. 核心结论（先看这一段）

**DanceGRPO 在本仓库不是新 loss，而是 GRPO loss + trainer 的随机 timestep 子集。** 所以
"正确性"分成两个互相独立的层次。本 sprint 只关闭第一层：

1. **机制正确性（本 sprint 已关闭）：**`dance_grpo` 直接复用 `GRPO.compute_loss`；
   `actor.timestep_selection=random` 选择合法、已排序的随机 timestep 子集。成对短探针必须
   触发 clipping，且 random 不能明显差于 strided 对照。
2. **学习有效性（本 sprint 未关闭）：**固定 held-out eval 必须在长跑中上升 >2σ。random
   与 strided 的短探针都持平，不能建立该结论；证据归可信 reference-curve 计划所有。

唯一的代码区别：`flow_matching_dance_grpo.yaml` 的 `timestep_selection: random` 被
`vrl/trainers/online/trainer.py` 中的 `select_timestep_subset(...)` 消费；typed config 只接受
`strided|random`。

## 1. 算法实锤

### 1.1 loss（与 GRPO 同一份 clipped surrogate）
`GRPO.compute_loss` in `vrl/algorithms/grpo/continuous.py`：

```python
raw_ratio = torch.exp(signals.log_prob - old_log_probs)
clipped_ratio = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)
per_sample_loss = torch.maximum(-advantages * ratio, -advantages * clipped_ratio)
```

`clip_ratio=1e-4`（recipe 覆盖）——flow-matching 每步
ratio 极小、要乘穿 ~10 步，所以 clip 必须比 AR 的 0.2 紧 2000 倍。

### 1.2 论文配方（用于对齐超参与判据）
- **no KL**（论文丢掉 β 项，"minimal performance differences"）→ 本仓库
  `GRPOConfig.kl_coef=0.0` 默认即满足，recipe 未开 KL ✓。
- **shared init noise**：组内 G 个样本共享初始噪声（不同噪声会 reward hacking，Fig.8）。
- **advantage aggregation**：多 reward 时在 advantage 层（非 raw reward）相加——本验证只用单
  aesthetic reward，不触发。
- **timestep subsample τ=0.6**（保留 ~60% 步、丢 ~40%）。
- 验证锚点（FLUX.1-dev，论文 Table 3）：HPS-v2.1 **0.304 → 0.372**，clip ε=**1e-4**，
  lr=**1e-5**（full-param），G=**12**，**25** 去噪步，~300 iters 单调上升。

## 2. 实验设计（data / reward / model）

| 维度 | 选择 | 理由 |
|---|---|---|
| 模型 | **flux/dev**（FLUX.1-dev ~12B），LoRA + bf16，256×256 | 论文原文在 FLUX 上验过 DanceGRPO；t2i 与论文判据一致。256² 省显存（论文用 512²）。显存吃紧时退 `wan_2_1/1_3b`（1.3B，最小，`online_grpo_ocr.yaml` 已跑通）。 |
| 数据 | **`/dataset/drawbench_train_192`** | DanceGRPO/Flow-GRPO 用 DrawBench 做多样 prompt 的质量探针；192 条文本 prompt，单卡够用（`vrl/config/presets/dataset/drawbench_train_192.yaml`）。注意此集**无 `eval_manifest`**——fixed eval 用从 192 里切出的固定 prompt+seed 子集。 |
| 奖励 | **`/reward/aesthetic`**（CLIP ViT-L/14 + MLP head，全本地） | DanceGRPO 主轴是 HPS-v2.1（学习到的偏好/质量分）；repo 最接近的纯质量 reward 是 aesthetic。比 OCR 重（要下 CLIP-L 权重）但全本地、无需额外接线（`vrl/rewards/functions/aesthetic.py`）。**最便宜 fallback 仍是 `/reward/ocr`**。 |
| 超参 | `clip_ratio=1e-4`（recipe）、`kl_coef=0`（DanceGRPO 丢 KL）、`global_std=true`、`G=n_samples_per_prompt=12`（论文 DanceGRPO 组大小）、`num_steps=10`（flow-grpo denoising reduction）、`timestep_selection=random`、`noise_level=0.7`（flow-grpo a=0.7）、`lr=1e-4`（LoRA 比论文 full-param 1e-5 高，见 cosmos 单卡实测） |
| 对照 | 同 config 但 `timestep_selection=strided`（= plain `flow_matching_grpo`） | 复现论文 §3.6：random 应与 strided 终点相当 |

## 3. 可复现入口与观测结果

当前维护的实验配置是
`vrl/config/presets/experiment/flux/online_dance_grpo_aesthetic_validation.yaml`。
它选择 DanceGRPO recipe、aesthetic reward、DrawBench prompts、
`timestep_selection=random` 和 `ppo_epochs=4`。成对对照复用同一配置，只覆盖
`actor.timestep_selection=strided`；不需要第二份配置文件。

短探针得到以下机制证据：

- 最后 3 个 epoch 的 reward 均值：random `4.560`，strided `4.547`；
- clip fraction：random `0.27-0.38`，strided `0.27-0.35`；
- 两条 reward 曲线都在观测噪声内持平，因此这是“同等平”，不是任一实验已经学到的证据。

## 4. 关闭判据与结论

### 4.1 机制闭环——已完成

- random selector 返回大小正确、已排序的子集，并在多次调用间重新采样；typed config 接受
  `random` 并拒绝未知值。
- `dance_grpo` 分派到共享 GRPO loss；其梯度提高高 advantage 样本、降低低 advantage 样本。
- 成对短探针触发了 clipped surrogate，且 random 与 strided 没有实质差异（`4.560` vs
  `4.547`）。

这些 true/false 路径由 `tests/algorithms/test_dance_grpo.py` 和
`tests/algorithms/test_flow_dppo_grpo_guard.py` 固定。

### 4.2 原始完整曲线判据——未完成；已移交

原计划要求固定 prompt+seed eval 在约 200-300 次 update 后上升 >2σ，且 random 与
strided 两条曲线都上升到相近终点。短探针既没有达到、也没有完整检验该 finishing bar；它只
关闭机制问题，学习有效性结论已移交可信 reference-curve 计划。

## 5. 非目标 / Non-Goals

- **不复现论文绝对数值**（base/reward/分辨率均不同）；短探针只验证机制与
  random-vs-strided 相对行为，不验证 reward 上升。
- **不做 multi-reward advantage aggregation**——单 aesthetic reward，避开多奖励归一
  （论文多 reward 时在 advantage 层相加）。
- **不动 loss 代码**——机制已具备，本 sprint 只验证 selector、dispatch 与短探针机制行为。
- **不强求复现 HPS 绝对分**——repo 无 HPS-v2.1，用 aesthetic 作本地替身；本记录不声称
  reward 已上升。

## References
- 论文：`docs/papers/diffusion-flow-rl/dancegrpo-visual-generation.pdf`（Algorithm 1、
  Table 3 FLUX、Table 6 超参、§3.6 + Fig.4b timestep ablation）
- 实现：`vrl/algorithms/grpo/continuous.py`、
  `vrl/config/presets/base/algorithm/dance_grpo.yaml`、
  `vrl/config/presets/recipe/online/flow_matching_dance_grpo.yaml`
- timestep 消费：`vrl/trainers/online/trainer.py`
- 当前维护的实验：
  `vrl/config/presets/experiment/flux/online_dance_grpo_aesthetic_validation.yaml`
- 奖励/数据：`vrl/rewards/functions/aesthetic.py`、
  `vrl/config/presets/reward/aesthetic.yaml`、
  `vrl/config/presets/dataset/drawbench_train_192.yaml`
- 回归测试：`tests/algorithms/test_dance_grpo.py`、
  `tests/algorithms/test_flow_dppo_grpo_guard.py`
