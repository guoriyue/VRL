# SPRINT: Flow-DPPO 正确性验证实验（flux + PickScore）

状态：**已完成——仅机制验证（2026-07-18）**。mask 非零、随 threshold 单调变化、具有
不对称性，并能控制 drift。短跑没有建立 held-out reward 上升 >2σ 的曲线；该长期结论已移交
可信 reference-curve 计划。

> 来源：实现 `vrl/algorithms/grpo/continuous.py`（`FlowDPPO`）+ recipe
> `vrl/config/presets/recipe/online/flow_matching_dppo.yaml` + 母体论文
> `docs/papers/diffusion-flow-rl/flow-grpo-online-flow-matching-rl.pdf`（FlowGRPO 的
> 高斯 KL 闭式）+ 信赖域参照 `op-grpo-off-policy-flow.pdf`。
> **注**：Flow-DPPO 无独立 PDF；它是 FlowGRPO 配方 + DPPO 信赖域（verl-omni
> `add_kl_coefficient` 分支）的合成，判据从算法契约 + FlowGRPO 母体推导。
> 相关：[[SPRINT_grpo_guard_validation]]（同读 rollout proposal mean 的另一信赖域变体）、
> [[SPRINT_segment_signal_dead_field_cleanup]]（`old_prev_sample_mean` 即为此算法保留）。

## 0. 核心结论（先看这一段）

**Flow-DPPO 用"精确高斯 KL 的不对称信赖域"替换 PPO 的对称 ratio clip。** 它**不 clip
ratio**，而是计算当前 vs rollout proposal mean 的高斯 KL，把**高 KL 且在扩大差距**的样本
整条丢掉（正 advantage 把 ratio 往上推、或负 advantage 把 ratio 往下推），保留所有"拉回旧策略"
的更新。正确性分成机制闭环与长期学习两个层次；本 sprint 只关闭机制层：

1. **机制正确性（本 sprint 已关闭）：**mask 具有不对称性，依赖 rollout proposal mean 与
   `dt`，在 policy drift 下变为非零，随 `kl_mask_threshold` 单调变化，并能控制 drift。
2. **学习有效性（本 sprint 未关闭）：**固定 held-out reward 必须在长跑中上升 >2σ；
   PickScore 短探针没有建立该结论。

硬前置：Flow-DPPO 读 `signals.old_prev_sample_mean`（rollout 时的 reverse-SDE proposal
mean），所以 recipe **必须** `rollout.return_prev_sample_mean: true`，否则
`_require_trust_region_signals` 立即 fail。

## 1. 算法实锤

### 1.1 信赖域 mask（核心，`FlowDPPO.compute_loss`）

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
- `add_kl_coefficient=True`（见 `FlowDPPOConfig`）→ KL 的 sigma 折进每步扩散系数
  `sigma_t = std_dev_t * sqrt_dt`（闭式 `compute_kl_divergence`）；False 则退化为单位方差
  `mean_diff²/2`（对齐 verl-omni 的 `add_kl_coefficient=False` 分支）。
- `kl_mask_threshold=1.0`（默认）= 信赖域边界。

### 1.2 母体闭式（FlowGRPO 论文）
两个等方差高斯的 KL 退化为均值速度差的平方：
`D_KL = (Δt/2)·(σ_t(1−t)/2t + 1/σ_t)²·‖v_θ − v_ref‖²`。Flow-DPPO 把 ref 换成
**rollout 策略**（不是 frozen ref），度量 current-vs-rollout drift——这正是
`old_prev_sample_mean` 这个第三个均值存在的理由。

## 2. 实际运行的实验

当前维护的入口是
`vrl/config/presets/experiment/flux/online_flow_dppo_pickscore_validation.yaml`。
它使用 flux/dev、PickScore + `pickscore_sfw`、`ppo_epochs=4`、group size 16 和
`return_prev_sample_mean=true`。仓库没有维护中的
`online_flow_dppo_geneval_validation` 配置。GenEval 只是原始论文锚点；其 scorer 在该环境中
不可运行，也不是本关闭记录实际使用的 reward。

threshold sweep 使用
`algorithm.kl_mask_threshold={0.0003,0.002,0.005,0.015}`。在该短探针中，配置默认值
`1.0` 是没有触发 mask 的 baseline，不是有效工作的 trust-region 设置。

## 3. 机制观测证据

- `ppo_epochs=1` 时，mask 与 drift 都为零。
- `ppo_epochs=4` 且 threshold=`1.0` 时，mask 仍为零，drift KL 升至 `0.0147`，reward
  从 `0.784` 降至 `0.681`。
- threshold 为 `0.0003`、`0.002`、`0.005`、`0.015` 时，mask fraction 均值分别是
  `8.8%`、`1.8%`、`0.8%`、`0.03%`；drift KL 均值保持约 `0.0002-0.0004`，最终
  reward 分别是 `0.783`、`0.782`、`0.776`、`0.780`。

因此，mask 变为非零并随 threshold 增大而单调下降，也阻止了未 mask baseline 的
drift/reward 退化。原始静态 `5%-40%` 目标不是可靠的通用 finishing band：mask 生效后，
负反馈会减少继续超过 threshold 的样本比例。

## 4. 关闭判据与结论

### 4.1 机制闭环——已完成

- 单元测试证明不对称四象限 mask：只移除扩大 policy gap 的 high-KL update，保留拉回旧策略的
  update。
- zero drift 保留所有样本；两条 KL variance 分支都有覆盖。
- 缺少 `old_prev_sample_mean` 或 `dt` 会 fail fast，typed algorithm dispatch 选择
  `FlowDPPO`。
- trust-region 算法拒绝 strict on-policy `ppo_epochs=1`，但允许 multi-epoch reuse。
- 短 threshold sweep 证明 mask 非零、随 threshold 单调变化，并能控制 policy drift。

这些 true/false 路径由
`tests/algorithms/test_flow_dppo_grpo_guard.py` 和
`tests/trainers/online/test_trust_region_engages.py` 固定。

### 4.2 原始完整曲线判据——未完成；已移交

原计划要求固定 held-out reward 在约 200-300 次 update 后上升 >2σ，并以 Flow-GRPO 的
GenEval `0.63 -> 0.95` 作为论文锚点。本 sprint 实际运行 PickScore 而非 GenEval；短探针
不能建立该学习结论。长曲线证据归可信 reference-curve 计划所有。

## 5. 非目标 / Non-Goals

- **不与 OP-GRPO 的 off-policy buffer 对比**——OP-GRPO 是另一条（序列级 IS 校正 + 截断）路线，
  本仓库未实现，超出范围（仅作信赖域参照）。
- **不调 `compute_kl_divergence` 数学**——闭式已与 FlowGRPO 母体一致。
- **不复现论文绝对数值**——验证信赖域**形状/可控性**，非数字。
- **不实现或运行 GenEval 检测器**——该 scorer 不属于本 sprint；机制探针使用的是 PickScore。

## References
- 实现：`vrl/algorithms/grpo/continuous.py`（`_require_trust_region_signals` +
  `FlowDPPO`）、`vrl/config/presets/base/algorithm/flow_dppo.yaml`、
  `vrl/config/presets/recipe/online/flow_matching_dppo.yaml`
- 母体/参照论文：`docs/papers/diffusion-flow-rl/flow-grpo-online-flow-matching-rl.pdf`
  （高斯 KL 闭式）、`docs/papers/diffusion-flow-rl/op-grpo-off-policy-flow.pdf`（信赖域/clip
  fraction 视角）
- KL 闭式：`vrl/math/denoise/flow_matching.py`
- proposal-mean 存储：`vrl/generation/bindings/joint_denoise/executor.py`
- 当前维护的实验：
  `vrl/config/presets/experiment/flux/online_flow_dppo_pickscore_validation.yaml`
- 奖励/数据：`vrl/rewards/functions/pickscore.py`、
  `vrl/config/presets/reward/pickscore.yaml`、
  `vrl/config/presets/dataset/pickscore_sfw.yaml`
- 回归测试：`tests/algorithms/test_flow_dppo_grpo_guard.py`、
  `tests/trainers/online/test_trust_region_engages.py`
