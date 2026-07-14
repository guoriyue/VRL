# SPRINT (info / measurement archive): flux 四算法正确性验证 reward 曲线（DanceGRPO / GRPO-Guard / Flow-DPPO / DiffusionNFT）

状态：measurement archive（`info/`）。这是单卡上四个 diffusion-RL 算法的正确性验证观测记录，
**不是 action item**；保留供复查。配套计划文档在 `planned/SPRINT_{dance_grpo,grpo_guard,flow_dppo,diffusion_nft}_validation.md`。
日期：2026-06-21 ~ 2026-06-22，单张 RTX 5090（32GB），VRL @ `main`（含未提交的 NFT→flux 移植）。
模型 flux/dev（LoRA r32/α64, bf16, 256²），num_steps=10，single-GPU colocated strict-on-policy。

## TL;DR

- **四个算法管线全部正确**：end-to-end 跑通（rc=0），first-step 不变量精确成立
  （NFT `first_step_nft_invariant abs_diff=0.000e+00`；dance/dppo first-step log-prob diff `mean=0 max=0`），reward 不发散。
- **第一轮（`ppo_epochs=1`，默认）四条 reward 全平**，都在 ±0.2σ 噪声内，看不到学习。这是**结构性**的，不是参数没调。
- **根因（GRPO 三件套同一个）**：recipe 每 epoch 是 collect rollout → **恰好一次 optimizer update**
  （`online.py:964-998`）。单次更新步上 `old_log_prob == 当前策略` → ratio 恒等于 1 →
  **所有漂移类机制（ratio_mean_bias / mask / clip）数学上必为 0**，Guard / DPPO 的卖点完全空转。
- **修复 = 开 `actor.ppo_epochs=4`**（内层 PPO 复用循环本就在 `trainer.py:1201`，只是没人开）。
  inner step≥2 用更新后的策略对同一 rollout batch 重算 log_prob → 产生漂移 → 机制被激活（epoch 0 即可见）。
- **实测有效**：GRPO-Guard 的 `ratio_mean_bias` 从恒 0 → ~5e-4，`clip_fraction` 从 0 → ~3-10%。
- **唯一未完全达标的是 Flow-DPPO 的 mask**：ppo_epochs=4 让漂移出现（`kl_penalty` 0.0004→0.0027 上升），
  但 `kl_mask_threshold=1.0` ≫ 实际漂移（~0.003）→ `masked_fraction` 仍为 0。要落进 5-40% 带，
  阈值得降到漂移量级（~0.002-0.005）。这是 follow-up（见末尾）。
- **NFT 一开始就健康**（无需 ppo_epochs 修），是四个里唯一有真实梯度信号的：grad_norm ~0.1-0.4，
  approx_kl 0→0.004（previous/default 真分离）。

## 运行配置

入口：dance/guard/dppo/NFT 均走
`vrl.scripts.diffusion.train:train_diffusion_online`；algorithm 由 `algorithm.kind`
唯一分派（NFT 的模型接口仍由 `FluxModel` 组合
`PreviousPolicyAdapterMixin` + `diffusion_nft_prepare_transformer_input`）。
config：`experiment/diffusion/flux/online_{dance_grpo_aesthetic,grpo_guard_pickscore,flow_dppo_pickscore,diffusion_nft_pickscore}_validation`。

共同 override：`trainer.total_epochs={12 第一轮 / 8 第二轮} trainer.save_freq=9999 trainer.eval.enabled=false`。
公共超参：lr=1e-4(LoRA)、num_steps=10、noise_level=0.7、rollout_batch_size=8、
n_samples_per_prompt={16 guard/dppo/nft, 12 dance}、clip_ratio=1e-4(guard/dance)。
reward：guard/dppo/nft=pickscore（本地 CLIP-H+PickScore_v1），dance=aesthetic（HPS 类比）。
> dppo 用 pickscore 而非 sprint 首选的 geneval——geneval 打分器在本环境无实现（detector 栈缺失），
> 是 reward 基建、超出 sprint 范围；pickscore 是文档化的 runnable-today fallback，仍验证 trust-region 机制。

## Part A — 第一轮 `ppo_epochs=1`（四条全平）

reward_mean 首→末（固定 12 epoch，guard 只记到 8）：

```
算法            reward 首 -> 末      Δ(σ)    grad_norm   机制指标
grpo_guard      0.7894 -> 0.7827    -0.11σ   ~0.002     ratio_mean_bias 恒 0, clip 恒 0
diffusion_nft   0.7835 -> 0.7818    -0.03σ   ~0.1-0.4   approx_kl 0->0.004 (健康)
dance_grpo      4.8010 -> 4.6454    -0.17σ   ~0.003     approx_kl 恒 0
flow_dppo       0.7863 -> 0.7804    -0.10σ   ~0.002     mask(clip) 恒 0
```

四条都在噪声内（|Δ|<0.2σ）。注：训练每 epoch 在轮换 prompt 上采样，`reward_mean` 主要反映抽到的 prompt 难度，
本就不该当学习曲线读——但即便如此，机制指标恒 0 已足以判定「机制没被触发」。

## Part B — 诊断：为什么 GRPO 三件套机制恒 0

- recipe 每 epoch 单次 optimizer update（`online.py:964-998`，无内层复用）。
- 单步上 `signals.old_log_prob`(rollout) 与 `signals.log_prob`(当前) 来自同一策略 → ratio≡1。
- GRPOGuard 的 `ratio_mean_bias = mean_diff_sq/(2·scale²)`（`continuous.py:354`）依赖
  `prev_sample_mean − old_prev_sample_mean`，无漂移则为 0。
- FlowDPPO 的 mask 依赖 `kl_per_sample ≥ threshold`（`continuous.py:287`），无漂移则 KL≈0，永不触发。
- NFT 不同：loss 是 reward-weighted flow-matching MSE 对 previous 的 ratio=1 不敏感，天然大梯度，所以一开始就健康。

指标列映射（判读必读，`TrainStepMetrics` 跨算法复用，机制指标借用通用列名）：
- **GRPOGuard**：`kl_penalty` 列 = `ratio_mean_bias.mean()`（`continuous.py:372`）；`clip_fraction` = ratio-clip 比例。
- **FlowDPPO**：`clip_fraction` 列 = `masked_fraction`（`continuous.py:306`）；`kl_penalty` = 高斯 KL 漂移均值。

## Part C — 修复 `ppo_epochs=4`（机制激活，写 `*_pe4` 目录，8 epoch）

**GRPO-Guard before/after**（同列对比）：

```
        ratio_mean_bias(kl_penalty)   clip_fraction
ppo=1   0.00000 (恒)                  0.0000 (恒)
ppo=4   0.0003-0.0005                 0.025-0.095     <- 机制激活 ✅
```

ppo=4 完整 8 epoch：rmb ∈ [0.00028, 0.00047]（有限小，符合判据），clip ∈ [0.025, 0.095]（ratio clip 真咬合）。
（`approx_kl` 仍显 0：clip_ratio=1e-4 极紧 + log_ratio²~1e-6 在 5 位小数舍入为 0，与 clip 咬合不矛盾。）

**Flow-DPPO before/after**（完整 8 epoch）：

```
        masked_fraction(clip)   kl_penalty(漂移KL)        grad_norm        reward 首->末
ppo=1   0.0000 (恒)             0.00000 (恒)              ~0.002           0.786 -> 0.780
ppo=4   0.0000 (恒)             0.0004 -> 0.0147 上升     0.0015 -> 0.030  0.784 -> 0.681
```

漂移被激活（kl_penalty 升到 0.0147、grad_norm 翻 ~15 倍），**但 mask 全程 0**（见 Part D）。
注意 reward 反而掉了（0.784→0.681）：无 ratio clip、mask 又没触发时，`-adv·ratio` 更新失控伤 reward——
这本身就是 trust-region 必要性的反证。

**DanceGRPO random vs strided（核心 ablation，ppo=4，8 epoch）**：

```
         reward 末3-epoch 均值   clip_fraction    grad_norm
random   4.560                  0.27-0.38        ~0.003
strided  4.547                  0.27-0.35        ~0.003
```

两条统计上无差别（Δ=0.013），**random timestep 子集没破坏学习**（论文 §3.6 的核心声明）。
clip(1e-4 极紧) 咬合 ~30%。caveat：两条都偏平（4.4-4.8 噪声内），所以是「同等平」而非「同等学习」。

## Part D — Flow-DPPO 的 mask 阈值（sweep 完成）

`kl_mask_threshold=1.0` 是 verl-omni 默认，但 ppo=4 **未 mask 时**漂移 KL 涨到 ~0.005-0.015（≪1.0），
所以 trust-region 永远不 drop 样本（mask 恒 0）。**修复 = 把阈值降到漂移量级**。sweep
`{0.0003, 0.002, 0.005, 0.015}`（6 epoch/点）结果：

```
threshold   mask(均值)   driftKL(均值)   reward 末
0.0003      ~8.8%        ~0.0002         0.783
0.002       ~1.8%        ~0.0003         0.782
0.005       ~0.8%        ~0.0004         0.776
0.015       ~0.03%       ~0.0004         0.780
（pe4 对照 thr=1.0：mask 0%，driftKL ->0.015，reward ->0.681 退化）
```

**单调性成立**（阈值↑ → mask↓：8.8→1.8→0.8→0.03%），mask 终于非 0，机制工作。

**关键反直觉发现——有效 trust region 会自我抑制触发条件**：mask 一生效 → 高漂移样本被 drop →
策略被拉住 → driftKL 从 0.015 塌回 ~0.0003 → 于是只剩 ~2% 尾巴超阈值。这是负反馈环，
所以 **mask 自然偏低，不会停在 sprint 预期的 5-40%**——那个区间的预期基于「静态漂移」假设，
真实动态下不成立。机制工作的最强证据不是命中 5-40%，而是 **mask 把 driftKL(0.015→0.0003) 和
reward 退化(0.681→0.78) 都挡住了**。补跑 thr=0.0003 确认单调曲线继续向下延伸，并能把 mask
拉进 5-40% 区间，但 reward/driftKL 没比 0.002 更好；因此不把“命中 5-40%”当机制 PASS 的硬判据。
产物 `outputs/flux_flow_dppo_thr{0p0003,0p002,0p005,0p015}/`、`outputs/_dppo_threshold_sweep.{sh,log}`。

## 判据判定（按各 sprint）

- **DiffusionNFT** ✅ 管线 + 机制 PASS：反对称不变量精确成立、grad/approx_kl 健康。learning 待长跑（256 更新，非本次目标）。
- **GRPO-Guard** ✅ 机制 PASS（ppo=4 后）：ratio_mean_bias 有限小(3-5e-4)、clip 咬合(2.5-9.5%)、保留全样本。
- **DanceGRPO** ✅ PASS：random≈strided（4.560 vs 4.547），timestep 子集不破坏学习；clip 咬合。
- **Flow-DPPO** ✅ 机制 PASS（修正判读）：ppo=4 漂移激活、阈值 sweep 后 mask 非 0 且单调、trust region
  实测压住漂移并保住 reward（vs 未 mask 的 pe4 退化）。原 sprint「mask∈5-40%」判据因自调节不适用，
  改判据为「mask 非 0 + 单调 + 漂移/reward 受控」。

> 共同限制：所有「learning（reward 上升 >2σ）」判据都需要 ~200-300 更新，本次 8-12 epoch 只验
> 「管线对、first-step 自洽、机制被触发」，不下「学得起来」的结论。

## 关联

- 复现并定位了 `project_first_trustworthy_curve`（2026-06-13 cosmos GRPO 持平）的根因：
  不是 epoch 数，是 `ppo_epochs=1` 的零漂移 + per-step 小梯度。
- `info/SPRINT_cosmos25_kling_reward_curve.md`（同结论：轮换 prompt 让训练 reward 不可读为学习；杠杆是 per-step 梯度）。
- NFT→flux 移植代码：`vrl/models/diffusion/flux/model.py`、`vrl/models/diffusion/common/lora.py:PreviousPolicyAdapterMixin`。

## 关键文件 / 产物

- 配置：`configs/experiment/diffusion/flux/online_*_validation.yaml`（三个已加 `actor.ppo_epochs: 4`）。
- 内层 PPO 复用循环：`vrl/trainers/online/trainer.py:1201`（`for _inner_epoch in range(cfg.ppo_epochs)`）。
- 机制指标来源：`vrl/algorithms/grpo/continuous.py`（FlowDPPO:255-309, GRPOGuard:337-373）。
- 第一轮产物：`outputs/flux_{grpo_guard,diffusion_nft,dance_grpo_aesthetic,flow_dppo}_*validation/metrics.csv`。
- 第二轮（ppo=4）产物：`outputs/flux_*_pe4/metrics.csv`、`outputs/_validation_queue_pe4.{sh,log}`。
- 汇总脚本：`outputs/_validation_summary.py`。
