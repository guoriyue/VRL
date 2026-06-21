# SPRINT: DiffusionNFT 正确性验证实验（flux 移植 + 验证，planned）

状态：planned（2026-06-21）。性质：**先移植、再验证**——DiffusionNFT 当前的 model 接口只在
cosmos/predict2_5 上实现（被排除），所以要在非排除 family 上验证，**必须先把 NFT 接口移植到
一个 t2i family（选 flux），再做 learning 验证**。这是本批 4 个算法里唯一不能纯配置落地的。
排除 sd3.5 / cosmos。

> 来源：论文 `docs/papers/diffusion-flow-rl/diffusion-nft-forward-process-rl.pdf`
> (arXiv 2509.16117, DiffusionNFT) + 实现 `vrl/algorithms/diffusion_nft.py` + 参考接口
> `vrl/models/diffusion/cosmos/predict2_5/model.py:211-240,467,568`。
> 相关：[[SPRINT_cosmos_predict25_rl_paper_parity]]（cosmos NFT 实跑），
> [[SPRINT_frozen_component_preservation]]（adapter 边界）。

## 0. Core Decision（先看这一段）

**DiffusionNFT 是 likelihood-free 的前向过程 RL：不用 log-prob、不算 ratio，从干净 latent +
prompt embeds + 采样 timestep + reward 直接做 reward-weighted flow-matching MSE。** 它把每个
样本按 optimality 概率 `r=0.5+0.5·clip(r_norm/Z,−1,1)` 软分到正/负集，用**一个可训速度场
`v_θ` + 一个冻结的 previous-policy `v_old`** 隐式参数化正/负预测：
`v⁺=(1−β)v_old+β v_θ`、`v⁻=(1+β)v_old−β v_θ`（`diffusion_nft.py:254-255`）。

**关键事实（决定本 sprint 形态）**：NFT 需要 model 暴露三件东西——
`diffusion_nft_prepare_transformer_input(...)`、`sync_previous_policy_adapter(decay=...)`、
一个独立的 `"previous"` LoRA adapter slot——**目前只有 `cosmos/predict2_5/model.py` 实现**
（grep 全仓库唯一命中）。flux/qwen_image/wan 只继承了 base 的 `activate_adapter` + `transformer`，
缺这三件。算法在缺失时硬 fail（`diffusion_nft.py:189-192,305-309`）。

所以正确的落地是两阶段，**不能跳过 Phase A 直接配实验**：
- **Phase A（实现，非配置）**：把 NFT model 接口移植到 flux。
- **Phase B（验证）**：在 flux 上跑 NFT，用三个 NFT 专属判据判读。

> 诚实说明：若接受用 cosmos 验证（已有 `cosmos_predict2_5/online_nft_*.yaml`），则零移植成本——
> 但那违反 user 的 "except cosmos" 约束。本 sprint 按约束走 flux 移植路线。

## 1. 算法实锤

### 1.1 likelihood-free loss（`diffusion_nft.py:254-271`）

```python
positive_prediction = beta * forward_prediction + (1 - beta) * previous_prediction
negative_prediction = (1 + beta) * previous_prediction - beta * forward_prediction
positive_x0 = xt - t_expanded * positive_prediction
negative_x0 = xt - t_expanded * negative_prediction
positive_loss = normalized_mse(positive_x0, x0)        # DMD-style self-normalized
negative_loss = normalized_mse(negative_x0, x0)
policy_loss = (flat_mix*positive_loss/beta + (1-flat_mix)*negative_loss/beta).mean() * advantage_scale
kl_loss = ((forward_prediction - ref_prediction)**2).mean()   # vs disable_adapter ref
loss = policy_loss + kl_coef * kl_loss
```

- **三次 transformer forward**（`:235-239`）：`previous`（no_grad）、`default`（trainable）、
  `disable_adapter` ref（no_grad）。previous 每步由 `after_optimizer_step` 刷新
  （`:301-312`，`sync_previous_policy_adapter(decay=weight_copy_decay)`）。
- **严格 on-policy**：`tolerates_off_policy_staleness=False`（`:51`）——NFT 不算 IS ratio
  纠偏，旧策略数据是**有偏**而非仅有噪，所以 staleness 必须为 0。
- **无 log-prob 评估器**：`uses_evaluator=False`（`:37`），训练数据是 `latents_clean` /
  `prompt_embeds` / `timesteps`（`required_data_keys`，`:42`）。

### 1.2 论文配方与判据锚点
- base **SD3.5-Medium**、512²、LoRA(α64/r32)、G=**24**、**10** 采样步、CFG-free。
- **β≈1 稳**（默认 1.0，0.1 更快）；soft-update η 渐增（η=0 塌、η=0.9 太慢）；adaptive 权重。
- 头条：单 reward **GenEval 0.24→~0.95–0.98，~1k iters，比 FlowGRPO 快 3–25×（wall-clock）**。
- **负分支消融**：去掉 `(1−r)‖v⁻−v‖²` 半 → reward **几乎瞬间崩**（NFT 最干净的实现自检）。

## 2. Phase A：把 NFT 接口移植到 flux（实现）

参考 `cosmos/predict2_5/model.py`，在 `vrl/models/diffusion/flux/model.py`（`FluxModel`）补：

1. **`"previous"` adapter slot**：build 一个与 `default` 同构的 LoRA adapter，从 default 拷权重
   并 freeze（镜像 cosmos `model.py:215-219`）。flux 已用 `LoraModelMixin`，可复用其 adapter 机制。
2. **`sync_previous_policy_adapter(self, *, decay=0.0)`**：把 default 权重按 decay 拷进
   previous（镜像 cosmos `:224-234`）。
3. **`diffusion_nft_prepare_transformer_input(self, *, latents, prompt_embeds,
   prompt_attention_mask, pooled_prompt_embeds, timestep, num_frames, height, width)`**：
   返回 flux transformer 的 forward kwargs dict。这是 family-specific 的核心工作——复用
   `FluxModel.forward_step` / `prepare_sampling`（`flux/model.py:270,334`）已有的 flux 专属
   构造：`img_ids` / `txt_ids` / `guidance` / pooled embeds / `t/1000` 时间嵌入约定
   （`flux/model.py:19,346`），`num_frames=1`（image）。
4. **trainer 入口**：cosmos 用 `train_cosmos_predict25_diffusion_nft`；需要一个 flux 对应的 NFT
   训练入口（或抽出一个 family 无关的 diffusion-NFT 训练 loop）。NFT 走
   `recipe/online/diffusion_nft`，不经 SDE-logprob evaluator（`factory.py` NFT 分支
   `evaluator=None`）。

> 验证 Phase A 完成：`grep diffusion_nft_prepare_transformer_input vrl/models/diffusion/flux/`
> 命中；构造一个 dummy batch 调用三方法不抛 `"DiffusionNFT model must expose …"`。

## 3. Phase B：验证实验设计（data / reward / model）

| 维度 | 选择 | 理由 |
|---|---|---|
| 模型 | **flux/dev**（移植后），LoRA(α64/r32) bf16，256² | 非排除 family 里最便宜的 t2i；NFT 的正/负预测都从 x0 回归，与 t2i 论文判据一致。 |
| 数据 | **`/dataset/geneval`** | DiffusionNFT 的最干净单 reward 头条任务就是 GenEval（0.24→0.95）；有 `eval_manifest`。 |
| 奖励 | **`/reward/geneval`** | 组合性规则打分，论文里**奖励即 metric**、NFT 不需要 log-prob 评估器。⚠️ **需接线**：`configs/reward/geneval.yaml` 的 `import_path: ""` 必须填对象检测打分器。**runnable fallback：`/reward/pickscore`（全本地，也是 NFT 单 reward task 之一）或 `/reward/ocr`**。 |
| 超参 | `nft_beta=1.0`、`advantage_scale=5.0`、`kl_coef=1.0`、`weight_copy_decay=0.0`（η=0 起，必要时渐增）、`global_std=false`、`G=n_samples_per_prompt=16`（论文 24；单卡折中）、`num_steps=10`、`lr=1e-4`（LoRA） | 对齐 `configs/base/algorithm/diffusion_nft.yaml` 默认 |
| 对照 | **负分支消融**：临时把 `negative_loss` 权重置 0（或 `flat_mix≡1`），其余不变 | 论文核心自检：去负分支应 reward 崩 |

落地 config（Phase A 完成后）：从 `cosmos_predict2_5/online_nft_kling_video_reward.yaml`
取结构，换 `/model/diffusion/flux/dev` + `/sampling/image/512` + `/reward/geneval` + `/dataset/geneval`
（GenEval 需先填 `reward.kwargs.geneval.import_path`；否则退 `/reward/pickscore` + `/dataset/pickscore_sfw`），
entrypoint 指向新的 flux NFT 入口，`total_epochs≈256`（论文 256 更新），开 `debug.first_step`
与固定 eval 网格。

## 4. 判据（finishing criteria）

按可信度从高到低（前两个是 NFT 专属、最能区分"实现对/错"）：

1. **lr=0 不变量（advantage-flip 反对称）**：`first_step_invariant_check`
   （`diffusion_nft.py:72-117`）——previous adapter 刚同步时，翻转 advantage 符号 loss 不变
   （`|loss − flipped_loss| ≤ 1e-6`）。这是 NFT 版的 "step-0 parity"，区分"信号弱"与"信号是垃圾"
   （previous 没同步 / scheduler 漂移）。`debug.first_step=true` 自动跑。
2. **负分支消融崩塌**：去掉负分支后 reward **几乎瞬间崩/不升**；带负分支则稳定上升。这是论文
   Fig 强调、NFT 独有的判别性证据——一对配对跑（带/不带负分支）即可。
3. **学习信号**：固定 GenEval test 集 `eval_reward_mean` 单调上升 >2σ（往 ~0.9 方向）。
   论文锚点 GenEval 0.24→~0.95 ~1k iters；本验证只要**方向/形状**对（不复现绝对值）。
4. **隐式参数化自检**：确认 `v_old` 来自 `activate_adapter("previous")` 且 no_grad（无梯度回流，
   `:235-236`）；正/负预测是 `(1∓β)v_old ± β v_θ`（断言 `:254-255`）。
5. **on-policy 守卫**：把 `max_stale>0` 配进 rollout schedule 应 fail-fast（`:51`
   `tolerates_off_policy_staleness=False` 被 `build_rollout_schedule` 消费）。

## 5. 非目标 / Non-Goals

- **不在 cosmos 上验**——已有 `cosmos_predict2_5/online_nft_*`，但 user 约束排除 cosmos；本 sprint
  专做非排除 family 的移植路线。
- **不移植到 wan（video）/ qwen_image**——video NFT 的 `num_frames>1` 与显存远贵于 flux t2i；
  qwen_image(20B) 比 flux(12B) 更重。先在最便宜的 flux 上验证接口与算法，再议扩展。
- **不复现论文 3–25× 效率数字**——那需要与 FlowGRPO 等 wall-clock 对照的完整跑；本 sprint 验
  **正确性**（不变量 + 负分支 + reward 方向），效率对比另立。
- **不做 multi-reward 联合训练**——单 GenEval（或 fallback pickscore），避开论文的 5-reward
  advantage 聚合。
- **不实现 GenEval 检测器**——假设 `import_path` 已接好或用 fallback reward；接线归 reward 基建。

## References
- 论文：`docs/papers/diffusion-flow-rl/diffusion-nft-forward-process-rl.pdf`
  （正/负隐式参数化、β/η/adaptive 权重、负分支消融、GenEval 0.24→0.95）
- 算法：`vrl/algorithms/diffusion_nft.py:37-51,72-117,139-312`、
  `configs/base/algorithm/diffusion_nft.yaml`、`configs/recipe/online/diffusion_nft.yaml`
- **参考接口（移植源）**：`vrl/models/diffusion/cosmos/predict2_5/model.py:211-240,467,568`
- **移植目标**：`vrl/models/diffusion/flux/model.py:68,270,334`（`FluxModel` / `prepare_sampling`
  / `forward_step`）
- factory NFT 分支：`vrl/scripts/common/factory.py`（NFT `evaluator=None`）
- cosmos NFT 实跑 config（结构参考）：
  `configs/experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward.yaml`
- 奖励/数据：`vrl/rewards/functions/geneval.py`、`configs/reward/geneval.yaml`（`import_path` 需填）、
  `configs/dataset/geneval.yaml`（fallback：`configs/reward/pickscore.yaml` + `/dataset/pickscore_sfw`）
