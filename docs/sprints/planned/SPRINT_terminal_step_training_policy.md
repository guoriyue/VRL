# SPRINT：末步训练策略 —— 把「tf<1 不训末步」从 floor 数学的巧合变成决定

状态：**门 1 已过（2026-08-22），等门 2（质量 A/B）**。基线 main @ 本文所引
测量的提交。
单卡 5090 + wan 1.3B LoRA 可完整执行（含质量 A/B）。

## 0. 结论先行

`strided` 选择在 `timestep_fraction < 1` 时**永远不训末步**，但这是
`int(i·T/count)` floor 数学的**副产品，不是决定**。两条配置 lane 会把末步
路进训练：

1. `timestep_fraction = 1.0`（`range(T)` 全训）；
2. `timestep_selection = random`（DanceGRPO 路径，`randperm` 以概率 tf 抽中）。

而末步是整个 schedule 里**唯一病理的一步**：σ = 0.001 → 0 的转移在
`noise_level=1.0` 下 std 只有 9.7e-4,每元素 logprob 梯度尺度是 step 0 的
**×5400**（nl=0.7 下 ×1000）；相邻的 step 33 只有 ×33（良性）。病理集中在
一步,不是平滑上升的坡。

本 sprint 提议:**用一个 noise_scale 下限把近确定性步显式排除出
train_indices**,让排除成为决定并覆盖 `random` 路径 —— 但**默认行为是否改变
必须先过质量 A/B**（§3),因为这改变哪些步拿梯度。

## 1. 证据（全部实测,真实 Wan2.1-T2V-1.3B）

### 1.1 仓库自己的 doctrine 已经支持这个方向

Flash-GRPO 的 sde_window 注释（`vrl/trainers/online/trainer.py:1219-1223`）:

> training any step outside it would put surrogate loss on a **deterministic
> ODE transition that was never a policy action**.

nl=1.0 下末步 std=9.7e-4 —— 一个 99.9% 确定性的转移。同一 doctrine 适用。
`SPRINT_cross_model_performance.md` 也留过同类警告（windowed SDE 启用前必须
过滤 train_indices,否则确定性步的无意义 logprob 会进 loss）。

### 1.2 梯度尺度表（每元素 1/noise_scale,相对 step 0）

| step | σ | nl=0.7 | nl=1.0 |
|---|---:|---:|---:|
| 0 | 1.000 | ×1 | ×1 |
| 17 | 0.500 | ×6 | ×2 |
| 33 | 0.030 | ×33 | ×33 |
| **34（末步）** | **0.001** | **×999** | **×5421** |

### 1.3 parity 分 lane 判决（末步是否被训完全决定 gate 结果）

fp8 rowwise 在真实 35 步链上（门 = mean 与 max ≤ 1e-2）:

| nl | 不含末步的 lane（strided tf<1） | 含末步的 lane（tf=1.0 / random 抽中） |
|---|---|---|
| 0.7（生产） | PASS（max ≤ 2.2e-3） | PASS（max 2.5e-3） |
| 1.0 | tf≤0.5 PASS；tf=0.99 max 2.6e-2 | **FAIL（max 8.8e-1）** |

完整表:`info/SPRINT_quantized_rollout_precision_performance.md` §5.5。
当前没有 preset 占据 FAIL lane,但没有任何机制阻止占据它。

### 1.4 诚实的反证据（为什么不能直接改默认）

- nl=0.7（生产主流）下末步 std=2.2e-2,并非严格确定性;flow-grpo 一系的
  公开实现全步训练且能学 —— **「末步梯度有害」目前是假设,不是事实**。
- 梯度 ×1000 可能被 PPO clip 立刻饱和（ratio 溢出 clip 区间 → 梯度走
  clipped 分支）,净效应或许只是浪费,不是破坏。未测。

## 2. 范围

- `_train_timestep_indices` / `_sample_batch_train_indices` 增加
  noise_scale 下限过滤（从 scheduler σ 表 + noise_level 推,单点实现,
  不加用户 knob —— 阈值若成立就是常量,`sde_window` 路径不动）。
- 覆盖 `random` 选择（当前它能抽中末步）。
- parity gate 自动受益:gate 只量受训步,过滤后 `nl=1.0 ∧ tf=1.0` lane
  对有界 drift 源不再必炸。

## 3. KILL-RISK 门（必须先过,顺序执行）

1. **梯度支配性实测 —— 已执行,CONFIRMED（2026-08-22）。**
   真实 Wan2.1-1.3B、repo LoRA 配置（r=32/α=64、8 投影、23.59M adapter 参数）、
   tf=1.0、240p latent（σ schedule 与分辨率无关）、uniform advantage
   （GRPO advantage 按 per-sample 广播,不改变步间相对尺度）。

   **(a) update-1 每步 adapter 梯度范数（相对均值）**：

   | | s0 | s17 | s30 | s32 | s33 | **s34** |
   |---|---|---|---|---|---|---|
   | nl=1.0 | 0.09× | 0.21× | 0.75× | 1.80× | 3.25× | **21.3×** |
   | nl=0.7 | 0.09× | 0.67× | 1.66× | 2.99× | 3.96× | **4.8×** |

   nl=1.0 是单步尖峰:**末步独占总 grad-norm² 的 96.3%**（s33+s34 = 98.5%）。
   nl=0.7（生产）是**晚段坡**而非尖峰:s32 起 3–4×,末两步占 54.3% ——
   **fix 应按 noise_scale 连续下限 gate,不是"删下标 34"**。

   **(b) update-2 clip 饱和（反假设检验）**:「PPO clip 从 update-2 起自动
   中和末步」**在 nl=0.7 被否决**——lr=1e-4（wan preset）扰动后全链
   |ratio−1| ≤ 7.4e-4,离 clip 带 0.2 差 250×,晚段过权重**永久不被 clip
   拦截**;nl=1.0 也要累计 ~10×lr 漂移才触 clip（s34 |ratio−1|=0.124 仍在
   带内,梯度照流）。

   原判据（<2× 平均 → 证伪）:**两个 noise level 均超过,门 1 通过**。
   原始数据（探针脚本 + 完整 35 步数组）为一次性验证产物,结论已在此,
   未入库。
2. **质量 A/B（真实 shape,嵌入本轮教训）—— 下一步,未执行**:同 seed 双臂
   短 run,比 reward 曲线。门 1 的 (a) 修正了要测的形态:**noise_scale 下限
   过滤（会截掉 nl=0.7 的晚段坡）与只删末步（保守）两个臂都测**。
   无显著差异 → 过滤安全,仅作为 `random`/`tf=1.0` 的保护落地;
   曲线更好 → 改默认并记录。
3. 两门都过才动 `_train_timestep_indices`;先门后码。

## 4. 非目标

- 不动 parity 阈值（放宽 = 对所有 drift 源一起废掉 gate）。
- 不动 `noise_level` 公式与 `sde_window`。
- 不为过滤加用户 config knob（单一实现点,阈值是推导出的常量;
  见 config-layering 纪律）。

## 5. 相关

- 判决表与机制:`docs/sprints/info/SPRINT_quantized_rollout_precision_performance.md` §5.5
- 放大器与选择数学的推导:`docs/sprints/done/SPRINT_rollout_lora_merge.md` §6
- doctrine 出处:`vrl/trainers/online/trainer.py:1219-1223`
