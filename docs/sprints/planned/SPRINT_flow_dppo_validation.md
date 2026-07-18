# SPRINT: Flow-DPPO 正确性验证实验（flux + GenEval，planned）

状态：planned（2026-06-21）。性质：**算法正确性验证跑**——为新加的 `flow_dppo`
（信赖域 GRPO：丢弃高 KL、扩大差距的样本，无 PPO ratio clip）设计一次可判读的 learning 验证。
模型 flux、数据/奖励对齐母体 Flow-GRPO 的头条任务：**`geneval` reward + GenEval prompts**
（组合性 compositional benchmark，Flow-GRPO 的核心 task）。排除 sd3.5 / cosmos。

> 来源：实现 `vrl/algorithms/grpo/continuous.py:239`（`FlowDPPO`）+ recipe
> `configs/recipe/online/flow_matching_dppo.yaml` + 母体论文
> `docs/papers/diffusion-flow-rl/flow-grpo-online-flow-matching-rl.pdf`（FlowGRPO 的
> 高斯 KL 闭式）+ 信赖域参照 `op-grpo-off-policy-flow.pdf`。
> **注**：Flow-DPPO 无独立 PDF；它是 FlowGRPO 配方 + DPPO 信赖域（verl-omni
> `add_kl_coefficient` 分支）的合成，判据从算法契约 + FlowGRPO 母体推导。
> 相关：[[SPRINT_grpo_guard_validation]]（同读 rollout proposal mean 的另一信赖域变体）、
> [[SPRINT_segment_signal_dead_field_cleanup]]（`old_prev_sample_mean` 即为此算法保留）。

## 0. Core Decision（先看这一段）

**Flow-DPPO 用"精确高斯 KL 的不对称信赖域"替换 PPO 的对称 ratio clip。** 它**不 clip
ratio**，而是计算当前 vs rollout proposal mean 的高斯 KL，把**高 KL 且在扩大差距**的样本
整条丢掉（正 advantage 把 ratio 往上推、或负 advantage 把 ratio 往下推），保留所有"拉回旧策略"
的更新。所以正确性有三个判据：

1. **能把 reward 学起来**（信赖域没把信号全 mask 掉）。
2. **`masked_fraction` 落在合理带内**（不为 0、不接近 1）——`kl_mask_threshold` 是唯一旋钮，
   sweep 它应能单调改变 mask 比例与 drift。
3. **不对称性成立**：只有"扩大差距"的更新被丢，"拉回"的永远保留（`continuous.py:288-290`）。

硬前置：Flow-DPPO 读 `signals.old_prev_sample_mean`（rollout 时的 reverse-SDE proposal
mean），所以 recipe **必须** `rollout.return_prev_sample_mean: true`（`flow_matching_dppo.yaml:10`），
否则 `_require_trust_region_signals` 立即 fail（`continuous.py:205-212`）。

## 1. 算法实锤

### 1.1 信赖域 mask（核心，`continuous.py:267-293`）

```python
ratio = torch.exp(signals.log_prob - signals.old_log_prob)
kl_per_sample = compute_kl_divergence(prev_sample_mean, old_prev_sample_mean,
                                      std_dev_t, sqrt_neg_dt=dt)  # add_kl_coefficient=True
high_kl = kl_per_sample >= cfg.kl_mask_threshold
pos_rm = high_kl & (ratio > 1.0) & (advantages > 0)   # 正 adv 把 ratio 推高 → 扩大差距
neg_rm = high_kl & (ratio < 1.0) & (advantages < 0)   # 负 adv 把 ratio 推低 → 扩大差距
keep = (~(pos_rm | neg_rm)).detach()
per_sample_loss = torch.where(keep, -advantages * ratio, 0)   # NO ratio clip
```

- **无 ratio clip**：与 GRPO/DanceGRPO 的关键差别。信赖域完全靠 KL mask。
- `add_kl_coefficient=True`（`FlowDPPOConfig:236`）→ KL 的 sigma 折进每步扩散系数
  `sigma_t = std_dev_t * sqrt_dt`（闭式 `compute_kl_divergence`）；False 则退化为单位方差
  `mean_diff²/2`（对齐 verl-omni 的 `add_kl_coefficient=False` 分支）。
- `kl_mask_threshold=1.0`（默认）= 信赖域边界。

### 1.2 母体闭式（FlowGRPO 论文）
两个等方差高斯的 KL 退化为均值速度差的平方：
`D_KL = (Δt/2)·(σ_t(1−t)/2t + 1/σ_t)²·‖v_θ − v_ref‖²`。Flow-DPPO 把 ref 换成
**rollout 策略**（不是 frozen ref），度量 current-vs-rollout drift——这正是
`old_prev_sample_mean` 这个第三个均值存在的理由。

## 2. 实验设计（data / reward / model）

| 维度 | 选择 | 理由 |
|---|---|---|
| 模型 | **flux/dev** LoRA bf16 256² | 与 [[SPRINT_dance_grpo_validation]] 同一 t2i 母体，便于横比；SDE-logprob 路径 family-agnostic（`factory.py` 按 `kind` 选 evaluator，不挑 family）。显存吃紧退 `wan_2_1/1_3b`。 |
| 数据 | **`/dataset/geneval`** | Flow-GRPO 的组合性头条任务，prompt 带 object/count/color/position 结构；有 `eval_manifest`（test.jsonl）可做固定 eval（`configs/dataset/geneval.yaml`）。 |
| 奖励 | **`/reward/geneval`** | 规则化组合性打分（counting/position/color），论文里**奖励即 metric**。⚠️ **需接线**：`configs/reward/geneval.yaml` 的 `import_path: ""` 留空——必须填入对象检测打分器（Mask2Former 类）的 import 路径才能跑（`vrl/rewards/functions/geneval.py`）。**runnable-today fallback：`/reward/pickscore`（CLIP-H，全本地）或 `/reward/ocr`**。 |
| 超参 | `kl_mask_threshold=1.0`（默认，主 sweep 对象）、`add_kl_coefficient=true`、`kl_coef=0`（无显式 KL 罚项，信赖域靠 mask）、`global_std=true`、`G=16`（Flow-GRPO GenEval 用 24；G<12 在 Flow-GRPO 会塌，单卡折中 16，不稳就靠 grad-accum 升到 24）、`num_steps=10`、`noise_level=0.7`、`lr=1e-4`、`return_prev_sample_mean=true`（recipe 已设） |
| 对照 | (a) plain `flow_matching_grpo`（有 ratio clip，Flow-GRPO GenEval 配方 `kl_coef=0.04`）；(b) Flow-DPPO 自身 sweep `kl_mask_threshold ∈ {0.3, 1.0, 3.0}` | 证明信赖域可控、且能替代 clip 学起来 |

## 3. 落地

> 2026-07-11 配置面纠偏：`reward/geneval` 仍保留为 scorer adapter，但默认
> `import_path: ""` 没有可执行 evaluator。因此不再发布
> `experiment/sd3_5/online_grpo_geneval` 这类必然在运行时失败的 active
> experiment。下面的 Flow-DPPO 配方在真实 GenEval scorer 接入并通过 reward memory
> preflight 前只是一份设计草案；今天可运行的验证继续使用 PickScore/OCR fallback。

新建 `configs/experiment/flux/online_flow_dppo_geneval_validation.yaml`：

```yaml
# Flow-DPPO correctness-validation run: Gaussian-KL trust region (no ratio clip).
defaults:
  - /recipe/online/flow_matching_dppo        # sets return_prev_sample_mean: true
  - /model/flux/dev
  - /sampling/image/512
  - /sampling/denoise/10_step_cfg_4_5
  - /reward/geneval            # <- Flow-GRPO headline task; set reward.kwargs.geneval.import_path!
  - /dataset/geneval
  - _self_

precision:
  training:
    dtype: bf16
sampling: { height: 256, width: 256, num_steps: 10, max_sequence_length: 64 }

# REQUIRED: wire the GenEval object-detector scorer (geneval.yaml ships import_path: "").
# reward: { kwargs: { geneval: { import_path: "<your.module:Scorer>" } } }

actor:
  optim: { lr: 1.0e-4 }
  gradient_checkpointing: true

algorithm:
  kl_mask_threshold: 1.0       # trust-region boundary (the one knob to sweep)
  add_kl_coefficient: true     # fold sqrt(-dt) diffusion coeff into KL sigma
  global_std: true

rollout:
  n_samples_per_prompt: 16     # Flow-GRPO GenEval uses 24; <12 collapses. 16 = single-GPU floor
  rollout_batch_size: 8
  sample_batch_size: 1
  noise_level: 0.7
  return_prev_sample_mean: true        # REQUIRED (recipe default; explicit here)
  sde: { window_range: [0, 10] }

trainer:
  entrypoint: vrl.scripts.train:train_online
  output_dir: outputs/flux_flow_dppo_geneval_validation
  total_epochs: 300
  save_freq: 50
  debug: { first_step: true }
  eval: { enabled: true, freq: 25, samples_per_prompt: 2, max_prompts: 32, seed: 20260621 }

model: { torch_compile: { enable: false } }
```

## 4. 判据（finishing criteria）

- **学习信号**：固定 GenEval test 集 `eval_reward_mean` 单调上升 >2σ，over ~200-300 更新——证明
  **无 ratio clip 也能靠信赖域学起来**。锚点：Flow-GRPO GenEval **0.63→0.95**（baseline GRPO
  应往 ~0.9+ 爬，Flow-DPPO 与之相当）。
- **mask 比例在带内**：`clip_fraction`（此处复用为 `masked_fraction`，`continuous.py:306`）
  应 ∈ ~5%–40%。**≈0** → 信赖域从未触发（`kl_mask_threshold` 太松，等于裸 unclipped policy
  gradient，会不稳）；**≈1** → 几乎全丢（太紧，学不动）。
- **threshold 可控性**：sweep `kl_mask_threshold ∈ {0.3,1.0,3.0}`，mask 比例应**单调下降**、
  `approx_kl`（drift）随阈值放松而**单调上升**——证明旋钮真的在控制信赖域。
- **不对称性自检**（一次性 unit-level）：构造正/负 advantage × 高/低 ratio 的 4 象限输入，
  确认只有 `(high_kl, ratio>1, adv>0)` 和 `(high_kl, ratio<1, adv<0)` 被 mask，另两象限保留
  （直接断言 `continuous.py:288-290` 的 keep 掩码）。
- **前置 fail-fast**：故意不设 `return_prev_sample_mean` 应在第 0 步抛
  `"FlowDPPO needs signals.old_prev_sample_mean"`（`continuous.py:207`）——证明硬契约生效。

## 5. 非目标 / Non-Goals

- **不与 OP-GRPO 的 off-policy buffer 对比**——OP-GRPO 是另一条（序列级 IS 校正 + 截断）路线，
  本仓库未实现，超出范围（仅作信赖域参照）。
- **不调 `compute_kl_divergence` 数学**——闭式已与 FlowGRPO 母体一致。
- **不复现论文绝对数值**——验证信赖域**形状/可控性**，非数字。
- **不实现 GenEval 检测器**——本 sprint 假设 `import_path` 已接好（或用 pickscore/ocr fallback）；
  接线打分器本身归 reward 基建，不在本 sprint。

## References
- 实现：`vrl/algorithms/grpo/continuous.py:195-309`（`_require_trust_region_signals` +
  `FlowDPPO`）、`configs/base/algorithm/flow_dppo.yaml`、
  `configs/recipe/online/flow_matching_dppo.yaml`
- 母体/参照论文：`docs/papers/diffusion-flow-rl/flow-grpo-online-flow-matching-rl.pdf`
  （高斯 KL 闭式）、`docs/papers/diffusion-flow-rl/op-grpo-off-policy-flow.pdf`（信赖域/clip
  fraction 视角）
- KL 闭式：`vrl/math/diffusion/flow_matching.py: compute_kl_divergence`
- proposal-mean 存储：`vrl/generation/diffusion/executor.py:203-209`、`layout.py:62-141`
- 基线 config：`configs/experiment/flux/online_grpo_smoke_single_gpu.yaml`
- 奖励/数据：`vrl/rewards/functions/geneval.py`、`configs/reward/geneval.yaml`（`import_path` 需填）、
  `configs/dataset/geneval.yaml`（fallback：`configs/reward/{pickscore,ocr}.yaml`）
