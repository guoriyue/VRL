# SPRINT: flow_grpo 源码研究 + diffusion GRPO 配方对照（borrow what actually learns）

状态：**DONE / superseded（2026-07-09 终局对账；原 planned 2026-06-20）**。
性质：**源码研究 + 逐 knob 差距对照**，落地是一次**配方对齐重跑**，不是功能移植。

> **终局对账（2026-07-09，对 configs/git 实况）**：本 sprint 的三半都已被后续工作覆盖，无独属剩余项：
> - **研究半（§1/§2）**：写文档时即完成（knob 对照表就是产出）。
> - **配方对齐（§3.1）已落地，且比本文方案更结构化**——不是 experiment override，而是 family
>   boundary：`configs/recipe/online/flow_matching_grpo.yaml` 给全 diffusion 族设
>   `clip_ratio: 1.0e-4`（d1547dfe，注释引用 flow_grpo SD3 clip_range）；
>   `configs/recipe/online/cosmos_predict25_grpo.yaml` 视频族收到 `1.0e-3`（= flow_grpo 原值，
>   24682ec4）。`global_std: true` 已进 geneval + 全部新 validation 实验；`kl_coef: 0.004` 已进
>   wan/echo/anima 视频实验、geneval 用 0.04（= flow_grpo geneval 值）；LoRA `lr: 1e-4` + EMA-on +
>   paper-shaped group（8×32）在 predict2.5 GRPO recipe 里落齐。predict2.5 未开 KL 是刻意的：
>   paper 路线用 diffusion-loss 正则替代，open item 归 [[SPRINT_cosmos_predict25_rl_paper_parity]]。
> - **§2 一处判断被实测推翻**：「`ppo_epochs=1` ✓ 不动」——flux 四算法 validation 实测
>   ppo_epochs=1 时 ratio≡1、clip_fraction≡0（trust region 机制死），predict2.5 GRPO recipe
>   已改 `ppo_epochs: 4` 并在 yaml header 记录根因（info/SPRINT_flux_algo_validation_curves.md）。
> - **重跑半（§3.2/3.3）以演化形式执行**：flux 四算法 validation runs + cosmos25 kling 曲线记录
>   （info/SPRINT_cosmos25_kling_reward_curve.md：reward 噪声内持平，根因 = per-step 梯度小 +
>   轮换 prompt 度量不可读，须固定 eval prompt 集）。「>2σ 单调抬升」的 learning verdict 尚未在
>   cosmos 上达成——该判据的所有权已随 [[SPRINT_cosmos_predict25_rl_paper_parity]] /
>   [[SPRINT_cosmos_predict2_2b_trustworthy_curve]] 走，不再由本 sprint 承载。归档 done/。

> 来源：通读了 `~/Desktop/flow_grpo`（SD3/Flux/Wan GRPO 训练库，本仓库 RL 层的母体）的
> `config/grpo.py` 与 `scripts/train_sd3.py` 训练循环，对照本仓库 `vrl/algorithms/grpo/`、
> `vrl/trainers/online/trainer.py`、`vrl/trainers/core/types.py`。
> 相关：[[SPRINT_cosmos_predict25_rl_paper_parity]]（论文配方角度，需多卡）、
> 记忆 `project_first_trustworthy_curve`（Cosmos+Kling GRPO 实跑 NO learning）。

## 0. 一句话

**trainer 不缺功能——缺的是配方。** vrl 的 online GRPO 在算法机制上已经**追平甚至超过**
flow_grpo（zero-advantage filter、global_std、flow-matching KL、timestep_fraction、外加
flow_grpo 没有的 TIS/RS 精度校正都在）。但**默认超参是 LLM-PPO 那一套，不是 diffusion-GRPO
那一套**：`eps_clip=0.2`（flow_grpo 是 `1e-3`，松 200 倍）、KL 默认关、`global_std` 默认关。
这三项默认值的错配，足以解释 `project_first_trustworthy_curve` 那条 noise 级的平curve。
本 sprint 把 flow_grpo 已被验证能学起来的 SD3/Flux 配方逐项抄过来，做一次干净重跑。

## 1. flow_grpo 训练循环实锤（已读源码）

### 1.1 每个去噪步都是一个 PPO 样本

`scripts/train_sd3.py:345`：

```python
num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)
```

`timestep_fraction=0.99`、`num_steps=10` → 每条轨迹拆成 ~10 个 (sample, t) 样本，
内层对 `train_timesteps = range(num_train_timesteps)` 逐 t 算 loss（`train_sd3.py:869-898`）。
**不是单 t**——这点本仓库已实现（`vrl/trainers/online/trainer.py:573,628`），✓ 已对齐。

### 1.2 clipped surrogate + 极紧的 clip

`train_sd3.py:886-898`：

```python
ratio = torch.exp(log_prob - sample["log_probs"][:, j])
unclipped_loss = -advantages * ratio
clipped_loss = -advantages * torch.clamp(ratio, 1.0 - config.train.clip_range, 1.0 + config.train.clip_range)
policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
```

`config.train.clip_range = 1e-3`（`config/grpo.py:56`）。diffusion 的 ratio 是单步去噪分布的
比值、且要乘穿 ~10 步，**每步只允许极小的策略移动**，所以 clip 必须紧。

### 1.3 KL 作为 loss（GRPO 推荐），打在 prev_sample_mean 上

`train_sd3.py:899-902`：

```python
if config.train.beta > 0:
    kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1,2,3), keepdim=True) / (2 * std_dev_t ** 2)
    loss = policy_loss + config.train.beta * kl_loss
```

`beta = 0.004`（sd3）~ `0.04`（geneval/harder，`config/grpo.py:54,106`）。ref 均值由参考模型
no_grad 重算（`train_sd3.py:880-883`）。本仓库等价实现在
`vrl/algorithms/grpo/continuous.py:140-155`（flow-matching 分支用 `compute_kl_divergence`
打在 `prev_sample_mean` 上，✓ 数学一致），但 **`init_kl_coef` 默认 0.0 = 关掉了**。

### 1.4 global_std advantage + zero-std 诊断/掩码

`train_sd3.py:766-787`：`PerPromptStatTracker(config.sample.global_std)`，
advantage = `(r - mean)/(std+1e-4)`；`global_std=True`（geneval，`config/grpo.py:108`）。
zero-std 比例做诊断（`calculate_zero_std_ratio`，`train_sd3.py:133-167`），
全零 advantage 的 prompt 被 mask 掉（`train_sd3.py:802,821`）。
本仓库 `vrl/trainers/online/trainer.py:533` 的 `nonzero_advantage_mask` 注释明确写了
"Flow-GRPO applies this globally..."，✓ 已移植；但 `global_std` 默认 `False`。

### 1.5 死 knob 提醒

`config/grpo.py:67` 的 `config.diffusion_loss = True` 在 `scripts/` 里**无任何消费点**
（grep 证实）——flow_grpo 自己没实现 diffusion-loss 正则。所以**不要**当作"可借的功能"。
diffusion-loss 正则的真实出处是 Cosmos 论文 §4.2.2，归 [[SPRINT_cosmos_predict25_rl_paper_parity]]
管，本仓库 online 侧目前也没有（只有 offline DPO 的 `diffusion_sft_loss`，
`vrl/algorithms/dpo.py:108`）。

## 2. 逐 knob 对照（flow_grpo 参考配方 vs vrl 当前默认）

| knob | flow_grpo (sd3/geneval) | vrl 默认 | 差距 | 动作 |
|---|---|---|---|---|
| clip / `eps_clip` | `1e-3` | `0.2`（`continuous.py:25`）| **松 200×** | 调到 `1e-3` |
| KL `beta` / `init_kl_coef` | `0.004`~`0.04` | `0.0`（KL 关，`continuous.py:26`）| KL 没生效 | 开到 `0.004` 起 |
| `global_std` | `True`（harder task）| `False`（`continuous.py:29`）| 组内 std 噪声大 | 开 |
| lr | `1e-4`（`config/grpo.py:55`）| 见 experiment yaml | 记忆已证 LoRA 要 `1e-4` 非 `1e-5` | 对齐 |
| group size `num_image_per_prompt` | `24`（geneval）| 那次失败跑只有 12 | 组太小 advantage 估计差 | ≥16，争取 24 |
| `num_steps` | `10` | — | — | 对齐 10 |
| `timestep_fraction` | `0.99` | 已实现 | ✓ | 保持 |
| `ppo_epochs`/inner | `1` | `1`（`core/types.py:273`）| ✓（之前怀疑要加 inner，**证伪**——flow_grpo 也是 1）| 不动 |
| EMA | `True` | 有 `vrl/trainers/ema.py` | 确认 online 启用 | 开 |
| grad_accum | `num_batches_per_epoch//2`（每 epoch 更新 2 次）| — | 结构性 | 参照 |

> **纠一个旧判断**：`project_first_trustworthy_curve` 里"需要更多 inner step"的猜测，
> 被源码证伪——flow_grpo `num_inner_epochs=1`。真正的杠杆是 **clip 收紧 + KL 开 +
> global_std + 更大 group**，不是加 inner epoch。

## 3. 落地（一次干净重跑）

1. 在 cosmos predict2.5 的 GRPO experiment yaml 上，把 `eps_clip→1e-3`、`init_kl_coef→0.004`、
   `global_std→True`、group≥16、lr=1e-4 对齐成 flow_grpo 形状（**只改 yaml override，不动
   trainer 代码**——机制已具备）。
2. 复用 runbook §3 的 contended-GPU 可续跑 wrapper（见 `project_cosmos_reward_run_setup`），
   固定 prompt 集、用 BLOCK test 判读（非单 t），避免把 ~1σ 噪声当 learning。
3. 判据：reward 曲线相对 baseline 有 >2σ 且单调的抬升，且 `zero_std_ratio` / clip_fraction /
   approx_kl 三个诊断在合理带内（clip_fraction 不应贴 1，approx_kl 不应炸）。

## 4. 非目标

- **不移植 flow_grpo 的 trainer 代码**——本仓库 online GRPO 已更全（TIS/RS、flow-matching KL、
  filter 都在），抄代码是倒退。借的是**超参配方**，不是实现。
- **不碰 `diffusion_loss` 正则**——flow_grpo 没实现它；要做归 cosmos 论文 parity sprint。
- **不改默认值本身**（先用 experiment yaml override 验证配方有效，确认后再议是否改
  `GRPOConfig` 的 diffusion-RL 默认；改默认会动到所有 family，超出本 sprint 范围）。
