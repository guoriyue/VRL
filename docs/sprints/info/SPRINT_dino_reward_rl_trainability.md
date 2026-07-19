# SPRINT: dino reward RL-trainability —— 用因果探针证明 reward 能被正确方向推动

Status: **INFO ARCHIVE (2026-07-18)**. The gradient-honest probe answered the
bounded causal question with a positive but not >2σ result. Its one-shot
continuation scripts and scratch outputs were removed after the answer was
recorded, so this document no longer owns executable follow-up work. Formal
video learning evidence belongs to the Cosmos trustworthy-curve sprint. Ray
cluster ownership remains recorded in
[`SPRINT_ray_cluster_ownership_and_shared_host_isolation.md`](../done/SPRINT_ray_cluster_ownership_and_shared_host_isolation.md).

> 来由：`online_grpo_droid_full_target_480p` 的 24h 长跑判"reward 往下掉 -4.5σ"（[[project_droid_overfit_validation]]），
> 追问"这是 reward 坏，还是操作点/梯度坏"。逐层拆解后发现前两次下跌各有 confound（采样脆 + 梯度 bug），
> 用修好的单-prompt 因果探针拿到第一条**干净**曲线。

---

## 0. 一句话

**RL 能把 dino reward 往对的方向推——方向稳、可复现、梯度逐位验证干净；但效果小（~+2–4%），
统计上正好卡在显著边缘（t≈1.6–1.96 随端点抖动），还没决定性越过 t>2。** 之前两次"往下掉"都是
artifact（采样 confound + 探针梯度 bug），不是 reward 本身坏。

## 1. 结果：干净因果曲线（单 prompt overfit，每点 16 样本，paired 同种子）

```
R0=0.4664(基线) R1=0.4720 R2=0.4692 R3=0.4706 R4=0.4698
R5=0.4772 R6=0.4831 R7=0.4836 R8=0.4878 R9=0.4907 R10=0.4856  （探针止于 R10；没有保留 R11/R12 结果）
slope/update ≈ +0.0024 (>+0.002 阈值 → 自动 VERDICT POSITIVE)
```

两个视角，都要看：
| 视角 | 读数 | 结论 |
|---|---|---|
| **趋势（over rounds）** | 斜率一直为正、R5–R9 连续 5 个新高、所有近期点远高于基线与早期 ~0.470 平台 | 方向不容置疑 |
| **配对显著性（R0 vs 端点）** | R6 t=1.20 → R8 t=1.41 → R9 t=1.96 → R10 t=1.62（回撤）；抗噪估计"末 3 点均值 vs R0" = +2.2%, t≈1.74 | near-significant，在阈值附近**抖动**，未决定性越过 |

**诚实措辞**：不是"已显著、板上钉钉",是"能推、稳定为正、效果小、刚到显著边缘"。不要拿 R9 的
t=1.96 当"crossed"——R10 回撤就掉回 1.62,单端点会两边跳。

- 每轮验证 `parity |Δlogp|_mean = 0.0000`、`clip_hits` 从 0 随策略移动上升（0→6→12→8→9→15）=
  梯度诚实 + 策略确实在动,不是随机。
- Historical one-shot artifacts were produced under `outputs/_level0_curve/`
  by `scratchpad/level0_curve.py`. They were removed after this record captured
  the result and are not maintained repository assets.

## 2. 关键 debug：为什么"往下掉"是假的

**第一次下跌（run10, 真 vrl-train, -4.5σ）= 采样 confound**：10 步生成太脆 + `same_latent` 共享噪声,
组内方差 = "哪条噪声抽签坏得少"（最差的是彩色 blob 怪）,GRPO 爬的是运气不是内容。修复 = 15 步（SDE
window [0,10) + 5 步确定性尾）+ 去共享种子（真内容多样性）。

**第二次下跌（level0 探针初版）= 梯度 bug（在探针,不在代码库）**：探针手写 replay 用了 `forward(es, 0)`,
但 cosmos 的 `forward_step` 按 `scheduler.sigmas[step_idx]` 取 sigma → step 2/4/6/8 全取到 step-0 的大
sigma → replay log-prob 差 0.6–1.5 → ratio 全错 → 梯度污染。**征兆 = ppo-epoch-1 的 `clip_hits` 恒
64/80（80=16样本×5步,只有 step0 对得上）。** 修复 = `forward_step(es, rec["step_idx"])`,parity 立刻归零：
```
step   forward(es,0) BUG   forward_step(es,k) FIX
 0        0.00000              0.00000
 2/4/6/8  0.6–1.5        →     0.00000
```
The one-shot proof script was removed with the scratch artifacts.

## 3. 你的代码库没有这个 bug（16 家族全审）

见 [[project_replay_parity_audit]]。两个正确 pattern,全家族一致：
- **sigma-indexer**（cosmos predict2/2.5/anima/cosmos3,forward_step 读 `sigmas[step_idx]`）→ 都用
  `CosmosReplayForward` mixin 的 `forward_step(state, timestep_idx)`（真索引）。`vrl/models/families/cosmos/__init__.py`。
- **timestep-only**（sd3/flux/qwen/wan/sana/cogvideox/hunyuan×2/pixart/mochi/lumina2）→ 都用
  `pack_eval_timestep` 把第 k 步 timestep 打包到位置 0,base `forward(state,0)`。`vrl/models/steps/denoise/base.py`。
- 非标准 sigma 域都**显式处理**：cosmos EDM（`sde_step_with_logprob` 自动检测 `sigmas.max()>1` 转域）、
  mochi/lumina2 反转域（重建 descending scheduler）、pixart epsilon-DDPM（走独立 `sde_type=ddim` log-prob 路径）、
  echo 从 timestep 值直接推 sigma（免疫）。
- **生产双保险**：evaluator 走 `model.replay_forward`（按家族正确分派）+ trainer `debug.first_step` parity
  检查。探针绕过了这两道,才中招。

## 4. 基建（抗环境）

- `~/.local/bin/run-until-success`：通用进程守护,任何命令套一下即得"跑到成功为止"——SIGTERM/HUP 后自动重启、
  `setsid` 独立会话、等 GPU 空、断点续、连续同类报错就停不空烧 GPU。见 [[feedback_unattended_run_survival]]。
- 93f 单卡显存地板：frozen offload + CPU-paged replay tensors + grad-ckpt + samples_per_chunk=1。
  见 [[project_single_gpu_93f_probe_oom]]。

## 5. 探针 ≠ 正式管线（诚实边界）

**现在跑的是探针,不是 `vrl-train`。** 探针复刻了 trainer 的 GRPO 数学(同 `group_relative_advantages`、
同 `sde_step_with_logprob`、同常数)且 parity=0.0000（对同一 rollout,梯度和 trainer 逐位一致）,所以因果
结论可迁移。但**正式管线在正确性上更 solid**（久经考验、用正确 mixin、自带 parity 守卫、run9/11 实测
parity=0.0）。我用探针还因为它是 Ray-free：现场旧日志证明 Ray worker 曾收到外部 `SIGTERM`，而探针没有
这类 Ray 进程暴露面。**现有证据不能识别发送方或命令，也不能确认是 `ray stop`。** 这项环境稳定性问题与
reward 数学正确性正交，归属独立 Ray sprint。

## 6. Closure and handoff

- The proposed warm-start continuation was not retained as a reproducible
  repository workflow. Its scratch scripts, adapter conversion, launcher, and
  output directory were deleted together as one-shot artifacts.
- First-step replay parity now hard-fails in the production trainer, preserving
  the durable correctness lesson from the probe.
- The bounded result remains positive but below the registered >2σ bar. This
  archive does not reopen sampling until significance appears.
- A formal multi-prompt video learning verdict belongs to the Cosmos
  trustworthy-curve sprint rather than a continuation of this scratch probe.

## 7. 非目标

- 不证明 dino 是"好"reward(能训 ≠ 训出好模型;泛化是多-prompt 长跑的事)。
- 不改 §3 已审过的家族 replay(它们都对)。
- 不在探针上追求生产级；正式学习证据由独立 reference curve 提供。

## 8. 引用

- Historical one-shot evidence (removed after recording):
  `scratchpad/level0_curve.py`, `scratchpad/parity_test.py`, and
  `outputs/_level0_curve/`.
- Correct pattern: `vrl/models/families/cosmos/__init__.py`,
  `vrl/models/families/cosmos/predict2/model.py`,
  `vrl/models/steps/denoise/base.py`, and
  `vrl/math/denoise/flow_matching.py`.
- Guard: `vrl/trainers/online/trainer.py` (first-step parity hard failure).
- 配方：`vrl/config/presets/experiment/cosmos_predict2/online_grpo_droid_overfit_validation.yaml`
- Ray 稳定性独立范围：`docs/sprints/done/SPRINT_ray_cluster_ownership_and_shared_host_isolation.md`
- 记忆：[[project_droid_overfit_validation]]、[[project_replay_parity_audit]]、
  [[project_single_gpu_93f_probe_oom]]、[[feedback_unattended_run_survival]]、[[project_first_trustworthy_curve]]
