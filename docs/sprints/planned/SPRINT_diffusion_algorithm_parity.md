# SPRINT: 与 verl-omni 的 diffusion-RL 算法对齐（补齐缺失算法）

状态：planned（2026-06-20）。范围：盘点 verl-omni `diffusion_algos.py` 的六个 diffusion-RL 目标，对照 vrl `algorithms/` 现状做诚实差距分析，并为真正缺失的算法（DanceGRPO / Flow-DPPO / GRPO-Guard）给出落点、loss 数学、config kind 与验收。

关联：[[SPRINT_reward_execution]]、[[SPRINT_memory_plan_full]]、[[SPRINT_resolved_struct_field_audit]]

## 0. Core Decision（先看这一段）

六个算法里 **三个已存在，不重实现**：FlowGRPO（= vrl 的 flow_matching `GRPO`，`vrl/algorithms/grpo/continuous.py:31`）、DiffusionNFT（`vrl/algorithms/diffusion_nft.py:28`，含 implicit-negative + previous-policy adapter）、Diffusion-DPO（`vrl/algorithms/dpo.py:36`，离线 trainer 直调）。**三个真正缺失**：DanceGRPO、Flow-DPPO、GRPO-Guard。其中 DanceGRPO 在 loss 层面就是 FlowGRPO（verl-omni 把两者注册到同一个 `FlowGRPOLoss`，`diffusion_algos.py:268-269`），差异只在 collect 端的 timestep dropout 与 multi-reward，所以它是“最便宜”的一个；而 **Flow-DPPO 和 GRPO-Guard 的共同硬阻塞是同一个数据缺口**——它们都需要 rollout 时刻的提案均值 `old_prev_sample_mean`，而 vrl 现在的 SDE evaluator 只算了“当前策略 replay 的 `prev_sample_mean`”和“冻结 ref 的 `ref_prev_sample_mean`”，**根本没采集 rollout 时的那一个**（`vrl/rollouts/evaluators/diffusion/sde_logprob.py:136-137`）。因此本 sprint 的真正工作量不在 loss 公式（几十行），而在打通这条采集链路。

## 1. 已存在，不重实现（三项，附 path:line）

### 1.1 FlowGRPO —— 已存在，不重实现

verl-omni 的 FlowGRPO 是标准 clipped surrogate：

```python
ratio = torch.exp(log_prob - old_log_prob)
unclipped_loss = -advantages * ratio
clipped_loss = -advantages * torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
per_elem_loss = torch.maximum(unclipped_loss, clipped_loss)
```
（`verl_omni/trainer/diffusion/diffusion_algos.py:313-321`）

vrl 现状逐行同构，且多了 TIS 修正：

```python
raw_ratio = torch.exp(signals.log_prob - old_log_probs)
ratio, tis_keep = apply_truncated_importance_weight(raw_ratio, pc)
clipped_ratio = torch.clamp(ratio, 1.0 - cfg.eps_clip, 1.0 + cfg.eps_clip)
per_sample_loss = torch.maximum(unclipped_loss, clipped_loss)
```
（`vrl/algorithms/grpo/continuous.py:93-102`）

vrl 的 advantage 归一化 `(r - mean) / max(std, eps)` 也与 verl-omni 的 `compute_flow_grpo_outcome_advantage` 同义（vrl `group_relative_advantages` 在 `vrl/algorithms/advantages.py:8`；verl-omni 在 `diffusion_algos.py:190-265`）。两边都支持 `global_std`。**结论：FlowGRPO 已存在，不重实现。** kind 已注册为 `"grpo"`（`vrl/config/schema.py:90`、`vrl/config/builders.py:195`、`vrl/scripts/common/factory.py:167`）。

### 1.2 DiffusionNFT —— 已存在，不重实现

verl-omni 的 NFT 正/负分解：

```python
positive_prediction = beta * forward_prediction + (1.0 - beta) * old_prediction
implicit_negative_prediction = (1.0 + beta) * old_prediction - beta * forward_prediction
```
（`diffusion_algos.py:820-821`）

vrl 现状逐行同构（previous = old）：

```python
positive_prediction = beta * forward_prediction + (1.0 - beta) * previous_prediction
negative_prediction = (1.0 + beta) * previous_prediction - beta * forward_prediction
```
（`vrl/algorithms/diffusion_nft.py:250-251`）

reward-prob 映射 `(adv/adv_clip_max)/2 + 0.5` 两边一致（vrl `diffusion_nft.py:245`；verl-omni `_advantage_to_reward_prob` `diffusion_algos.py:943`）。vrl 还额外实现了 previous-policy adapter 的每步刷新（`after_optimizer_step` → `sync_previous_policy_adapter`，`diffusion_nft.py:297-308`）和 ref-KL 项（`diffusion_nft.py:265-267`，对应 verl-omni `ref_kl_coef * ref_kl_loss` `diffusion_algos.py:846`）。**结论：DiffusionNFT 已存在，不重实现。** kind 已注册为 `"diffusion_nft"`（`vrl/config/schema.py:90`、`vrl/scripts/common/factory.py:241`）。

**两个可选差异，记录为 non-goal（见 §4）**：
- verl-omni 用 **自适应权重归一化**（`normalized_mse` 的分母是 `|x0_pred - x0|.mean().clip(min=adaptive_weight_min)`，`diffusion_algos.py:826-841`），vrl 走的是 `vrl/math/diffusion/nft.py` 的 `normalized_mse`（本 run 未读其分母实现，**未验证** 是否同口径）。
- verl-omni 把 advantage→reward_prob 暴露了 `adv_mode`（`positive_only/negative_only/one_only/binary`，`diffusion_algos.py:935-942`），vrl 现在固定线性映射，没有 `adv_mode` 开关。

### 1.3 Diffusion-DPO —— 已存在，不重实现

verl-omni 的 DPO inside-term：

```python
inside_term = -0.5 * beta * (w_diff - l_diff)
dpo_loss = -F.logsigmoid(inside_term).mean()
```
（`diffusion_algos.py:745-747`）

vrl 现状逐行同构：

```python
inside_term = -0.5 * beta * (model_diff - ref_diff)
loss = -F.logsigmoid(inside_term).mean()
```
（`vrl/algorithms/dpo.py:93-94`）

两边都用 winner/loser 的 (model_err − ref_err) 差，都给出 `implicit_acc`。vrl 明确把 DPO 定位为离线偏好学习、不走在线 `Algorithm` 协议（`vrl/algorithms/dpo.py:1-12`），离线 trainer 直调 `diffusion_dpo_loss`（`vrl/trainers/offline/dpo.py`）；verl-omni 的 `DPOLoss` 额外内置了 online pairing（`build_online_dpo_pair_indices`，`diffusion_algos.py:598-625`），这是 vrl 没有的“在线 DPO 配对”路径。**结论：Diffusion-DPO（离线）已存在，不重实现。** kind 已注册为 `"diffusion_dpo"`，且 online recipe 明确拒绝它（`vrl/scripts/common/factory.py:258-261`）。
- **可选差异（non-goal）**：online DPO 配对（top-vs-bottom per uid）在 vrl 缺失；本 sprint 不补，归 §4。

## 2. 真正缺失（三项）

> 共享前置：verl-omni 的所有 reverse-SDE 类 loss 都吃 `prev_sample_mean / std_dev_t / sqrt_dt / old_prev_sample_mean` 四件套（`diffusion_algos.py:365-366`、`470-471`）。vrl 的 `SegmentSignal` 已经有 `prev_sample_mean / ref_prev_sample_mean / std_dev_t / dt` 字段（`vrl/rollouts/evaluators/types.py:23-26`），`sde_step_with_logprob` 也已返回 `prev_sample_mean / std_dev_t / sqrt_neg_dt`（`vrl/math/diffusion/flow_matching.py:192-198`）。**唯一缺的是 `old_prev_sample_mean`**：vrl 的 SDE evaluator 只填了当前策略 replay 的 `result.prev_sample_mean` 和冻结 ref 的 `ref_prev_sample_mean`（`vrl/rollouts/evaluators/diffusion/sde_logprob.py:136-137`），从未存“rollout 时刻产生这条轨迹的策略的提案均值”。这是 Flow-DPPO 和 GRPO-Guard 的共同硬阻塞。

### 2.1 DanceGRPO（最便宜的一个）

**它是什么**：loss 与 FlowGRPO 完全相同。verl-omni 把两者注册到同一个类、同一个 advantage 估计器：

```python
@register_diffusion_loss("flow_grpo")
@register_diffusion_loss("dance_grpo")
class FlowGRPOLoss(DiffusionLossFn): ...
```
（`diffusion_algos.py:268-270`；adv 估计器 `diffusion_algos.py:190-191` 同样双注册）

所以 DanceGRPO ≠ 新 loss。它相对 FlowGRPO 的真实差异是 **collect 端**：(1) timestep dropout —— 每条轨迹只随机抽一部分去噪步参与训练（verl-omni 的 `_select_train_timesteps` 按 `timestep_fraction` 对每行 `[B,T]` 做 randperm 截断，`diffusion_algos.py:946-964`，注意这是挂在 NFT 路径上的，但 DanceGRPO 论文的核心采样技巧同形）；(2) multi-reward 聚合。

**vrl 现状**：vrl 没有 `dance_grpo` kind（`vrl/config/schema.py:90` 的 Literal 不含它）。timestep dropout 在 vrl 里没有对应物——`vrl/config/schema.py:357` 有个 `timestep_fraction: Any = None` 字段但本 run 未见消费者（**未验证** 是否真正接线，疑似 dead/预留）。multi-reward 聚合：vrl 有 reward registry（`vrl/rewards/functions/registry.py`）但本 run 未确认是否支持多 reward 加权聚合到 advantage（**未验证**）。

**落点**：
- loss：**不新增 loss 类**。新增一个 `DanceGRPO(GRPO)` 薄子类放在 `vrl/algorithms/grpo/continuous.py` 末尾（与 `TokenGRPO(GRPO)` 同文件同模式，避免新建 lean 文件），`compute_loss` 直接复用父类；config `DanceGRPOConfig(GRPOConfig)` 增 `timestep_fraction: float` 一个字段。
- collect：timestep dropout 的真正落点在 SDE evaluator / planner 的 timestep 选择处——按 `timestep_fraction` 对每条轨迹的去噪步做无放回抽样。multi-reward 聚合落在 advantage 计算前的 reward 组装层。
- config kind string：`"dance_grpo"`。需加到 `vrl/config/schema.py:90` Literal、`vrl/config/builders.py`（复用 GRPO 分支）、`vrl/scripts/common/factory.py:167` 的 `grpo` 分支（同一 evaluator）。
- 验收：(a) `timestep_fraction=1.0` 时与 `kind="grpo"` 数值逐位等价（lr=0 单步 loss 相等）；(b) `timestep_fraction=0.5` 时实际参与训练的 timestep 数 ≈ 半数且每 epoch 重抽；(c) multi-reward 两路加权和单路退化一致。

### 2.2 Flow-DPPO（exact-Gaussian-KL trust region）

**它是什么**：用**精确高斯 KL 的信任域 mask** 替换 PPO 的 ratio clip。不再 clamp ratio，而是当“KL 超阈值 **且** 更新方向在远离旧策略”时把该样本的梯度置零：

```python
mean_diff_sq = (prev_sample_mean - old_prev_sample_mean).pow(2)
sigma_t = std_dev_t * sqrt_dt                       # add_kl_coefficient 分支
kl_per_elem = mean_diff_sq / (2 * sigma_t.pow(2))   # 精确高斯 KL
high_kl_mask = kl_per_sample >= kl_mask_threshold
pos_rm_mask = high_kl_mask & (ratio > 1.0) & (advantages > 0)
neg_rm_mask = high_kl_mask & (ratio < 1.0) & (advantages < 0)
keep_mask = (~(pos_rm_mask | neg_rm_mask)).detach()
per_elem_loss = torch.where(keep_mask, unclipped_loss, zero)
```
（`diffusion_algos.py:402-421`）

注意它是 **asymmetric** 的：只 mask“正优势把 ratio 推高”和“负优势把 ratio 推低”这两种会扩大偏移的方向，反方向（拉回旧策略）不 mask。这正是与对称 PPO clip 的本质区别。

**vrl 现状**：缺。vrl 的 GRPO 走对称 `torch.clamp(ratio, 1-eps, 1+eps)`（`vrl/algorithms/grpo/continuous.py:99`），没有 KL-mask 路径。vrl 已有精确高斯 KL 的工具 `compute_kl_divergence`（`vrl/math/diffusion/flow_matching.py:201-213`，分母 `2*(std_dev_t*sqrt_neg_dt)**2`，与 verl-omni 的 `2*sigma_t**2` 同形），且 GRPO 在 `init_kl_coef>0` 时已经调用它（`continuous.py:127-132`）——**数学件齐了**。**唯一缺 `old_prev_sample_mean`**：现有 KL 用的是 `prev_sample_mean` vs `ref_prev_sample_mean`（current vs frozen-ref），而 DPPO 要的是 current vs **rollout-old**。

**落点**：
- 数据链路（主要工作量）：在 SDE evaluator 采集 rollout 时刻的提案均值并存入 trajectory replay；在 `SegmentSignal` 增字段 `old_prev_sample_mean`（`vrl/rollouts/evaluators/types.py`），由 `TrajectorySignalBuilder`（`vrl/rollouts/evaluators/trajectory.py`）透传，evaluator 侧从 rollout 缓存读取而非重算（`sde_logprob.py:136` 附近补一路）。
- loss：新增 `FlowDPPO(GRPO)` 子类，落在 `vrl/algorithms/grpo/continuous.py`（同文件，复用 advantage 与 TIS 基础设施）；config `FlowDPPOConfig(GRPOConfig)` 增 `kl_mask_threshold: float`、`add_kl_coefficient: bool`。KL 复用 `compute_kl_divergence`，但分母用 `old_prev_sample_mean` 而非 ref。
- config kind string：`"flow_dppo"`。加到 schema Literal、builders、factory（需用一个带 KL-intermediates 的 evaluator，`SignalRequest(need_kl_intermediates=True)`）。
- 验收：(a) `kl_mask_threshold=+inf` 时 keep_mask 全 1，退化为“无 clip 的 vanilla PG”（`-adv*ratio` 均值），数值可对拍；(b) 构造一批人工 (ratio, adv, mean_diff) 让 pos/neg mask 各命中若干，断言 `masked_fraction` 与手算一致；(c) `old_prev_sample_mean == prev_sample_mean`（首步严格 on-policy）时 KL=0、mask 全保留。

### 2.3 GRPO-Guard（ratio-mean-bias 校正 + 逐步尺度归一）

**它是什么**（arXiv:2510.22319）：在标准 FlowGRPO ratio 上加一个“ratio-mean bias”项，显式惩罚当前策略 reverse-SDE 提案均值相对 rollout 策略的漂移；再用逐步扩散系数 `sqrt_dt*sigma_t` 把它投影到 `log_prob - old_log_prob` 的同尺度，最后整条 loss 乘 `1/sqrt_dt**2` 做跨 timestep 的梯度幅度归一：

```python
scale = sqrt_dt.mean() * std_dev_t.mean()
mean_diff_sq = (prev_sample_mean - old_prev_sample_mean).pow(2).mean(dim=非batch)
ratio_mean_bias = mean_diff_sq / (2 * scale**2)
ratio = torch.exp((log_ratio + ratio_mean_bias) * scale)
# 之后照常 clipped surrogate ...
pg_loss = torch.mean(per_elem_loss) / (sqrt_dt_mean**2)
```
（`diffusion_algos.py:527-549`）

与 Flow-DPPO 的区别：DPPO 用 KL 阈值 **mask 掉**样本（硬信任域）；GRPO-Guard 不丢样本，而是把 mean-drift **加进 ratio 指数**并做尺度归一（软校正 + 跨步幅度一致）。两者都依赖 `old_prev_sample_mean`。

**vrl 现状**：缺。vrl 的 GRPO ratio 是纯 `exp(log_prob - old_log_prob)`（`continuous.py:93`），既无 mean-bias 项、也无 `1/sqrt_dt**2` 的逐步尺度归一。`std_dev_t`、`dt`（= `sqrt_neg_dt`）在 `SegmentSignal` 里已有（`types.py:25-26`），`sde_step_with_logprob(return_dt=True)` 已能产出 `sqrt_neg_dt`（`flow_matching.py:191`）；**同样只缺 `old_prev_sample_mean`**（与 §2.2 共用同一链路修复）。

**落点**：
- 数据链路：与 Flow-DPPO 共享 `old_prev_sample_mean` 采集（做一次，两个算法都受益）。
- loss：新增 `GRPOGuard(GRPO)` 子类，落在 `vrl/algorithms/grpo/continuous.py`；config `GRPOGuardConfig(GRPOConfig)` 复用 `eps_clip`、`adv_clip_max`，无需新增数值阈值字段。
- config kind string：`"grpo_guard"`。加到 schema Literal、builders、factory（同样 `need_kl_intermediates=True` 拿 `sqrt_dt`）。
- 验收：(a) `old_prev_sample_mean == prev_sample_mean` 且 `scale≈1` 时，`ratio_mean_bias=0`、`exp(log_ratio*scale)≈exp(log_ratio)`，loss 退化到接近 FlowGRPO（容差对拍）；(b) 人工注入 mean-drift，断言 `ratio_mean` 随 drift 单调上移；(c) 跨两个不同 `sqrt_dt` 的 timestep，`1/sqrt_dt**2` 归一后两步 loss 量级可比（验证“跨 timestep 一致幅度”这一卖点）。

## 3. 共享前置任务（解锁 §2.2 + §2.3，做一次）

打通 `old_prev_sample_mean` 采集链路，是 Flow-DPPO 与 GRPO-Guard 的公共阻塞，单独列为 P0：

1. rollout（生成）时，在每个 SDE 去噪步把该步的 `prev_sample_mean` 存进 trajectory replay 缓存（生成端已经在算 `prev_sample_mean` 来采样 `prev_sample`，见 `flow_matching.py:158-175`，复用即可，不重算）。
2. `SegmentSignal` 增字段 `old_prev_sample_mean: Any | None = None`（`vrl/rollouts/evaluators/types.py:23-27` 同区）。
3. `TrajectorySignalBuilder.single_segment` 透传该字段（`vrl/rollouts/evaluators/trajectory.py:39/58/84/143` 同模式补一行）。
4. SDE evaluator 从 replay 读取 rollout-old 提案均值填入（`sde_logprob.py:132-143` 的 builder 调用处）。
- 验收：首步严格 on-policy（rollout 策略 == replay 策略、同 dtype）时 `old_prev_sample_mean` 与 replay 的 `prev_sample_mean` 逐位相等（与 logprob 首步 parity 同一招）。

## 4. Non-Goals

- **不重实现 FlowGRPO / DiffusionNFT / Diffusion-DPO**（§1，均已存在并附 path:line）。
- **不补 DiffusionNFT 的 `adv_mode` 开关与自适应权重对齐**（§1.2 两个可选差异）；若后续要严格对齐 verl-omni 数值再单开任务。`vrl/math/diffusion/nft.py` 的 `normalized_mse` 分母口径标注为**未验证**，留待对齐时核。
- **不补 online DPO 配对**（verl-omni `build_online_dpo_pair_indices`，`diffusion_algos.py:598-625`）；vrl 的 DPO 是离线路径，在线偏好配对是独立特性。
- **不引入 verl-omni 的 worker-side `DiffusionLossFn` 注册表 / `DiffusionLossResult` 抽象**（`diffusion_algos.py:50-145`）。vrl 已有自己的 `Algorithm` Protocol + factory 分发（`vrl/algorithms/base.py:13`、`vrl/scripts/common/factory.py`），保持现有约定，新算法以 `GRPO` 子类落地，不另起一套注册机制。
- **不改 advantage 归一化语义**：vrl 的 `group_relative_advantages` 已等价 verl-omni 的 flow_grpo 估计器，复用即可。
- **不做 DanceGRPO 的 multi-reward 大改**：若 vrl reward registry 已支持加权聚合（**未验证**），只接线不重写 reward 层。

## 5. 落地顺序（按解锁成本）

1. DanceGRPO（kind `dance_grpo`）—— loss 复用，只加 timestep dropout + multi-reward 接线，无数据链路阻塞，最先做。
2. 共享前置 §3 —— 采集 `old_prev_sample_mean`。
3. Flow-DPPO（kind `flow_dppo`）+ GRPO-Guard（kind `grpo_guard`）—— 数据链路就绪后并行落地，各自一个 `GRPO` 子类。

每个新 kind 都要同步三处：`vrl/config/schema.py:90` 的 Literal、`vrl/config/builders.py:188-222` 的 `build_algorithm_config` 分发、`vrl/scripts/common/factory.py:160-263` 的 evaluator 配对。

## References

阅读文档 / 代码路径（本 run 实际读取）：
- verl-omni 全部六个 diffusion-RL loss：`/home/mingfeiguo/Desktop/verl-omni/verl_omni/trainer/diffusion/diffusion_algos.py`（FlowGRPO/DanceGRPO 268-358、Flow-DPPO 361-463、GRPO-Guard 466-587、DPO 590-778、DiffusionNFT 781-1010、KL 1013-1051；adv 估计器 190-265）
- vrl FlowGRPO（= flow_matching GRPO）：`/home/mingfeiguo/Desktop/wm-infra/vrl/algorithms/grpo/continuous.py:31-168`
- vrl TokenGRPO（子类模式范本）：`/home/mingfeiguo/Desktop/wm-infra/vrl/algorithms/grpo/token.py:23-128`
- vrl MultiSegmentTokenGRPO：`/home/mingfeiguo/Desktop/wm-infra/vrl/algorithms/grpo/multisegment.py:36-136`
- vrl DiffusionNFT：`/home/mingfeiguo/Desktop/wm-infra/vrl/algorithms/diffusion_nft.py:28-308`
- vrl Diffusion-DPO：`/home/mingfeiguo/Desktop/wm-infra/vrl/algorithms/dpo.py:36-117`
- vrl advantage 归一化：`/home/mingfeiguo/Desktop/wm-infra/vrl/algorithms/advantages.py:8-36`
- vrl TIS / drift：`/home/mingfeiguo/Desktop/wm-infra/vrl/algorithms/logprob_mismatch.py:40-133`
- vrl 算法协议 / adapter / metrics：`/home/mingfeiguo/Desktop/wm-infra/vrl/algorithms/base.py:13`、`trajectory.py:12-63`、`types.py:8-38`
- vrl 信号 schema（prev_sample_mean/std_dev_t/dt 字段）：`/home/mingfeiguo/Desktop/wm-infra/vrl/rollouts/evaluators/types.py:9-27`
- vrl SDE evaluator（缺 old_prev_sample_mean 的决定性证据）：`/home/mingfeiguo/Desktop/wm-infra/vrl/rollouts/evaluators/diffusion/sde_logprob.py:60-143`
- vrl flow-matching 数学（sde_step_with_logprob / compute_kl_divergence）：`/home/mingfeiguo/Desktop/wm-infra/vrl/math/diffusion/flow_matching.py:21-219`
- vrl kind 分发三处：`/home/mingfeiguo/Desktop/wm-infra/vrl/config/schema.py:88-91`、`vrl/config/builders.py:188-222`、`vrl/scripts/common/factory.py:160-263`

未验证项（需后续核实，不可当事实引用）：
- `vrl/math/diffusion/nft.py` `normalized_mse` 分母是否与 verl-omni 自适应权重同口径（本 run 未读）。
- `vrl/config/schema.py:357` `timestep_fraction` 是否有真实消费者（疑似预留/dead）。
- vrl reward registry 是否已支持 multi-reward 加权聚合。
