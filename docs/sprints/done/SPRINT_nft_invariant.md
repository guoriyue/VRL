# SPRINT: NFT 的 lr=0 不变式（非 ratio 算法的 parity gate 等价物）

状态：**T1 / T2 / T3 全部 implemented（2026-06-11）**。T3 已按"可选协议方法"
要求落地（commit 7756de4）：DiffusionNFT.first_step_invariant_check（advantage
翻号不变式，fork_rng 钉噪声）+ trainer debug.first_step 的 uses_evaluator=False
分支，超阈值 warning 与 GRPO parity 警报对称，trainer 零算法特定逻辑。

落地摘要：

```text
T1  域守卫：diffusion_nft.py 的 /1000 归一化后越界即 RuntimeError
    （EDM-scale 网格 fail loud，含 incident 注释）
T2  两个不变式 pin 测试（tests/algorithms/test_diffusion_nft.py，
    复用真 LoRA 双 adapter 夹具）：
    - EDM 网格（timestep=80000）必须大声失败
    - lr=0 / previous 刚同步时，advantage 翻号 loss 必须逐位不变
      （NFT 版的 ratio==1）
验证  tests/algorithms 8 passed；广域回归 333 passed。
```

## 0. 为什么需要

lr=0 logprob parity gate 只覆盖 ratio 类算法（GRPO/PPO）。NFT 不算 logprob
ratio，对它失明——predict2.5 当时跑 NFT，所以即使有 sigma 域级别的 bug 也
不会被 parity 抓到。这是已定性的检查空窗；原一次性 parity sprint 已删除，取证长期
归档在 `docs/sprints/info/SPRINT_cross_model_smoke.md` §2 与 commit `c66bf116`。

## 1. NFT 在 lr=0 时必须成立什么（推导自 vrl/algorithms/diffusion_nft.py）

NFT 的三路前向（同一 transformer 的三个视角）：

```text
forward_prediction   当前策略（default adapter）
previous_prediction  上一策略（previous adapter，after_optimizer_step 同步）
ref_prediction       参考（adapter 关闭）
```

权重不动（lr=0）且 previous adapter 已同步时：

```text
I1（核心，reward 通道惰性）:
  forward == previous
  ⇒ positive_prediction = β·f + (1−β)·f = f
  ⇒ negative_prediction = (1+β)·f − β·f = f
  ⇒ positive_loss ≡ negative_loss（逐元素）
  ⇒ policy_loss 与 advantages/reward_mix 完全无关
  可测形式：lr=0 下把 advantages 随机打乱重算 loss，必须逐位相等。
  这是 NFT 版的 "ratio==1"：策略没动时，奖励通道必须是惰性的。

I2（三路一致性）:
  首步（同步后）forward 与 previous 的 max abs diff ≈ 0（bf16 确定性内）。
  fresh adapter（零初始化）时 ref 也应与 forward 一致 ⇒ kl_loss == 0。

I3（域守卫，必须做——潜伏炸弹）:
  diffusion_nft.py:143-144 的 timestep 归一化是启发式：
      if (t > 1.0).any(): t = t / 1000.0
  它假设 timestep 网格 ≤ 1000。predict2 的 FlowMatch timesteps 高达
  80000（EDM 域，见 `docs/sprints/info/SPRINT_cross_model_smoke.md` §2）——若未来有人把
  NFT 接到 predict2 类家族上，t/1000 = 80，xt = (1−t)x0 + t·noise 直接
  出域，无声产出垃圾——与 sigma 域事故同构。
  修法：归一化后 assert t ∈ [0,1]，越界大声报错。
```

## 2. 落地形态

```text
T1  I3 域守卫：diffusion_nft.py 归一化后加越界 raise（无条件，零开销）。
T2  I1/I2 单元测试：fake transformer + 同步 previous adapter，断言
    advantage-shuffle 不变性与三路一致性（CPU，确定性）。
T3  trainer debug.first_step 的 NFT 记录：uses_evaluator=False 分支目前
    直接跳过 parity——为 NFT 增加 I1 检查记录（打乱 advantages 重算一次
    policy_loss，diff > 阈值则 logger.warning），与 GRPO parity 警报对称。
```

## 3. Non-Goals

- 不改 NFT 损失本身的数学。
- 不做 predict2 × NFT 的接入（I3 只是守住边界）。

## 4. References

- `vrl/algorithms/diffusion_nft.py:142-203`（三路前向与损失结构）
- `vrl/trainers/online/trainer.py:555`（uses_evaluator gate，T3 挂点）
- `docs/sprints/info/SPRINT_cross_model_smoke.md` §2（方法论与域事故长期归档）
