# SPRINT: DanceGRPO 正确性验证实验（flux + aesthetic/DrawBench，planned）

状态：planned（2026-06-21）。性质：**算法正确性验证跑**——为新加的 `dance_grpo`
设计一次可判读的 learning 验证，而非功能移植。模型选 flux（论文原文即在 FLUX.1-dev 上
验过 DanceGRPO），数据/奖励对齐论文的偏好/质量轴：**`aesthetic` reward + DrawBench prompts**
（DanceGRPO 主轴是 HPS-v2.1，repo 里最接近的纯质量/偏好 reward 是 aesthetic）。排除 sd3.5 / cosmos。

> 来源：论文 `docs/papers/diffusion-flow-rl/dancegrpo-visual-generation.pdf`
> (arXiv 2505.07818) + 实现 `vrl/algorithms/grpo/continuous.py:33`（`GRPO`）+
> recipe `configs/recipe/online/flow_matching_dance_grpo.yaml` + trainer timestep
> 消费点 `vrl/trainers/online/trainer.py:688,922`。
> 相关：[[SPRINT_flow_grpo_recipe_parity]]（同一 FlowGRPO 配方母体）。

## 0. Core Decision（先看这一段）

**DanceGRPO 在本仓库不是新 loss，而是 GRPO loss + trainer 的随机 timestep 子集。** 所以
"正确性"有两个互相独立的判据：

1. **它能像 GRPO 一样把 reward 学起来**（loss 路径正确）——`dance_grpo` 直接复用
   `GRPO.compute_loss`（`continuous.py:84`），`configs/base/algorithm/dance_grpo.yaml`
   只是把 `kind` 改名、保持同一组超参。
2. **把 `actor.timestep_selection` 从 `strided` 换成 `random` 不破坏学习**——这是
   DanceGRPO 的定义性技术。论文 §3.6 ablation（Fig.4b）证明 random 30%/60% 的 timestep
   子集与 full 100% 学到的 reward 相当（早期高噪步贡献最大），所以正确实现下
   random 曲线应与 strided 终点相当、且 compute 更省。

唯一的代码区别：`flow_matching_dance_grpo.yaml:20` 的 `timestep_selection: random` 被
`trainer.py:688,922` 的 `select_timestep_subset(num_timesteps, timestep_fraction,
timestep_selection)` 消费；schema 校验在 `core/types.py:330`（只接受 `strided|random`）。

## 1. 算法实锤

### 1.1 loss（与 GRPO 同一份 clipped surrogate）
`vrl/algorithms/grpo/continuous.py:105-124`：

```python
raw_ratio = torch.exp(signals.log_prob - old_log_probs)
clipped_ratio = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)
per_sample_loss = torch.maximum(-advantages * ratio, -advantages * clipped_ratio)
```

`clip_ratio=1e-4`（recipe 覆盖，`flow_matching_dance_grpo.yaml:13`）——flow-matching 每步
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
| 数据 | **`/dataset/drawbench_train_192`** | DanceGRPO/Flow-GRPO 用 DrawBench 做多样 prompt 的质量探针；192 条文本 prompt，单卡够用（`configs/dataset/drawbench_train_192.yaml`）。注意此集**无 `eval_manifest`**——fixed eval 用从 192 里切出的固定 prompt+seed 子集。 |
| 奖励 | **`/reward/aesthetic`**（CLIP ViT-L/14 + MLP head，全本地） | DanceGRPO 主轴是 HPS-v2.1（学习到的偏好/质量分）；repo 最接近的纯质量 reward 是 aesthetic。比 OCR 重（要下 CLIP-L 权重）但全本地、无需额外接线（`vrl/rewards/functions/aesthetic.py`）。**最便宜 fallback 仍是 `/reward/ocr`**。 |
| 超参 | `clip_ratio=1e-4`（recipe）、`kl_coef=0`（DanceGRPO 丢 KL）、`global_std=true`、`G=n_samples_per_prompt=12`（论文 DanceGRPO 组大小）、`num_steps=10`（flow-grpo denoising reduction）、`timestep_selection=random`、`noise_level=0.7`（flow-grpo a=0.7）、`lr=1e-4`（LoRA 比论文 full-param 1e-5 高，见 cosmos 单卡实测） |
| 对照 | 同 config 但 `timestep_selection=strided`（= plain `flow_matching_grpo`） | 复现论文 §3.6：random 应与 strided 终点相当 |

## 3. 落地

新建 `configs/experiment/diffusion/flux/online_dance_grpo_aesthetic_validation.yaml`，从
`flux/online_grpo_smoke_single_gpu.yaml` 派生，只换 recipe + 放大到能出 learning 信号：

```yaml
# DanceGRPO correctness-validation run: FlowGRPO + random timestep selection.
defaults:
  - /recipe/online/flow_matching_dance_grpo   # <- was flow_matching_grpo
  - /model/diffusion/flux/dev
  - /sampling/image/512
  - /sampling/denoise/10_step_cfg_4_5
  - /reward/aesthetic            # <- HPS analog (paper optimizes HPS-v2.1); fallback /reward/ocr
  - /dataset/drawbench_train_192 # <- diverse quality-probe prompts
  - _self_

precision: bf16
sampling: { height: 256, width: 256, num_steps: 10, max_sequence_length: 64 }

actor:
  optim: { lr: 1.0e-4 }
  gradient_checkpointing: true
  timestep_selection: random        # the DanceGRPO knob (recipe default; explicit here)

algorithm: { kl_coef: 0.0, global_std: true }   # no-KL per paper

rollout:
  n_samples_per_prompt: 12          # group size (paper DanceGRPO uses 12)
  rollout_batch_size: 8
  sample_batch_size: 1
  noise_level: 0.7
  sde: { window_range: [0, 10] }

trainer:
  entrypoint: vrl.scripts.diffusion.flux.train:train_flux_grpo
  output_dir: outputs/flux_dance_grpo_aesthetic_validation
  total_epochs: 300
  save_freq: 50
  debug: { first_step: true }
  eval: { enabled: true, freq: 25, samples_per_prompt: 2, max_prompts: 32, seed: 20260621 }

model: { torch_compile: { enable: false } }
```

对照跑：复制一份，`defaults` 改回 `flow_matching_grpo` 且 `actor.timestep_selection: strided`，
其余不变。

## 4. 判据（finishing criteria）

- **学习信号**：固定 prompt+seed eval 网格的 `eval_reward_mean`（aesthetic 分）相对 baseline
  （epoch=-1）**单调上升、幅度 >2σ**，覆盖 ~200-300 次更新。形状参照论文 HPS 曲线（FLUX
  0.304→0.372，SD 0.239→0.365，~300 iters 单调）——本验证换 aesthetic reward，只验**方向/形状**，
  不复现 HPS 绝对值。判读用固定网格、不看训练 `reward_mean`（每 epoch 轮换 prompt，不可读，
  见 [[SPRINT_cosmos_kling_fixed_eval_signal]]）。
- **timestep ablation 复现**：`random` 与 `strided` 两条曲线**都上升且终点相当**（论文 §3.6 的
  核心声明）。若 random 明显学不动而 strided 能学 → timestep 子集采样实现有 bug。
- **诊断在带内**：`approx_kl` 不发散；`clip_fraction` 因 ε=1e-4 会偏高（正常），重点是
  loss/grad 不炸；`tis_clip_fraction` / `rs_seq_masked_fraction` ≈ 0（same-dtype on-policy）。
- **first-step 平价**：step 0 `ratio≈1`、`approx_kl≈0`（on-policy 自洽，`debug.first_step`）。

## 5. 非目标 / Non-Goals

- **不复现论文绝对数值**（base/reward/分辨率均不同；只验证 reward 上升的**形状/方向**与
  timestep ablation 的相对结论）。
- **不做 multi-reward advantage aggregation**——单 aesthetic reward，避开多奖励归一
  （论文多 reward 时在 advantage 层相加）。
- **不动 loss 代码**——机制已具备，本 sprint 只配实验。
- **不强求复现 HPS 绝对分**——repo 无 HPS-v2.1，用 aesthetic 作本地替身，只验上升方向 +
  timestep ablation 的相对结论。

## References
- 论文：`docs/papers/diffusion-flow-rl/dancegrpo-visual-generation.pdf`（Algorithm 1、
  Table 3 FLUX、Table 6 超参、§3.6 + Fig.4b timestep ablation）
- 实现：`vrl/algorithms/grpo/continuous.py:33,84,105-124`、
  `configs/base/algorithm/dance_grpo.yaml`、`configs/recipe/online/flow_matching_dance_grpo.yaml`
- timestep 消费：`vrl/trainers/online/trainer.py:688,922`、`vrl/trainers/core/types.py:276,330`
- 基线 config：`configs/experiment/diffusion/flux/online_grpo_smoke_single_gpu.yaml`、
  `configs/experiment/diffusion/wan_2_1/online_grpo_ocr.yaml`（更省显存的 fallback）
- 奖励/数据：`vrl/rewards/functions/aesthetic.py`、`configs/reward/aesthetic.yaml`、
  `configs/dataset/drawbench_train_192.yaml`（cheap fallback：`configs/reward/ocr.yaml` + `/dataset/ocr`）
