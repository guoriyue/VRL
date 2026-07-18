# SPRINT: Low-precision rollout correction (TIS / MIS)（done）

状态：**DONE（2026-06-20 核实落地，commit `ba94732`）**。触发条件（当前可达为 FP8 rollout split；FP4 保留但不可用）已出现，核心交付物全部落地：
TIS `truncate`/`clip`/`mask`（`vrl/algorithms/logprob_mismatch.py:176-199`，bound 的是 mismatch ratio
`exp(replay-rollout)` 而非 behavior ratio——避开了本 doc §2 点名的头号 bug）+ 正交的 seq-level RS
（`seq_mean_k1`/`seq_max_k1`）+ mismatch 指标（`compute_logprob_mismatch_stats`，fp32 归约）+ FP8 rollout
精度轴（`vrl/config/precision.py`）。默认 off（`tis_mode="off"`/`rs_mode="off"`），bf16 下 no-op；precision split
时由 `vrl/config/builders.py:34-51` 自动启用。算法接线 `vrl/trainers/online/trainer.py:333-334`，测试
`tests/config/test_precision.py` + `tests/algorithms/test_grpo*.py` **87 passed**。
**刻意未做**（非 omission）：slime 的 token-level MIS 变体（SDE 窗口太短，`logprob_mismatch.py:145-152` 显式
报错拒绝）、decoupled full-precision old-recompute（`recompute_old_logprob="on"` 保留接口、`:167-173` raise
`NotImplementedError`，不做静默 no-op 旋钮）。

原始 parked 说明（保留备查）：从已删除的 `SPRINT_low_precision_rollout_production.md`
抽出，保留 slime 源码地图与正确的 TIS 语义，等触发条件出现再激活。

## 0. Core Decision

给 GRPO 训练补一条 **truncated importance sampling (TIS)** 修正路径，纠正 rollout 前向与
training replay 前向之间的数值不匹配（train/infer mismatch）。**默认关闭**，只有当我们刻意引入
rollout/replay 前向差异时才打开。

**为什么现在 parked**：fp16-rollout parity sprint 已完成——SD3.5 `fp16 rollout / fp16 replay /
fp32 math` 通过精度检查、零漂移（2026-06-07 run），并由 guard 强制
（`vrl/trainers/online/precision_guard.py` +
`tests/trainers/online/test_precision_drift_guard.py:90`
`test_precision_drift_guard_checks_fp16_same_role_precision`）。项目又默认 bf16
（rollout == replay），所以当前**根本没有 behavior/proximal mismatch 需要 TIS 去修**——现在建它
等于为一个已被工程消除的 mismatch 写修正。

## 1. 触发条件（激活本 sprint 的前提）

TIS 只有在**刻意引入 rollout/replay 前向不匹配**时才承重：

- **fp8 rollout**（比 bf16 训练前向更快但更低精度），或
- **分离式推理后端**（SGLang/vLLM 式 rollout serving，其前向 kernel 与 HF 训练前向不同——
  正是 slime 针对的经典 train/infer mismatch）。

若 rollout-perf 路线推进到任一项，这是自然的后续。在此之前不动。

## 2. 正确的 TIS 语义（承重公式）

存在**两个**独立的 importance 修正；**绝不能让一个 ratio 同时扮两个角色**：

```text
proximal_ratio  = exp(current_compute_log_prob - old_compute_log_prob)   # PPO clip
mismatch_ratio  = exp(old_compute_log_prob - behavior_rollout_log_prob)  # train/infer gap

policy_loss = bounded(mismatch_ratio).detach() * PPO(proximal_ratio, advantage)
```

mismatch 权重是**有界、detach、乘在 PPO 项上的**——它是一个 reweighting，不是第二条 policy
gradient。把它和 `exp(current - behavior_rollout)` 混为一谈是**头号 bug**。

**Diffusion 专属警告**：slime 的 TIS 是 **LLM token-level**；diffusion 路径是
**per-denoise-step log-prob**。token→timestep 的映射（sequence-level vs per-step 加权）是真正的
设计工作——**不要**照搬 slime 的 token-level `tis_clip=2.0`。

## 3. 本地管线（两个 log-prob 源已就位）

TIS 需要的两个 log-prob 源已存在：

- Rollout log-probs 由 `sde_step_with_logprob` 写进轨迹
  （`vrl/generation/diffusion/executor.py:470,486` → `buffers.log_probs`，`:507`）。
- Replay 由 `vrl/trainers/online/trainer.py` 的 evaluator 重算 log-probs。

## 4. 实施计划（触发后）

### T1: `correction_mode` flag（默认 off）
- 仿照 `denoise_compile` 的 off-by-default 模式加一个 `correction_mode` 开关，关闭时 bf16 路径
  逐位不变。

### T2: 把 rollout log-prob 接成 mismatch baseline
- 将 `buffers.log_probs`（behavior/rollout 侧）透传到 loss 层，作为 `behavior_rollout_log_prob`；
  `old_compute_log_prob` 仍由 replay evaluator 提供。

### T3: 在 GRPO loss 里加 bounded/detached reweight
- 按 §2 公式：`mismatch_ratio = exp(old - rollout)` → `clamp(tis_clip_low, tis_clip)` →
  `.detach()` → 乘上 PPO 项。**先确定 diffusion 的 per-step vs sequence-level 加权**（§2 警告）。

### T4: mismatch 指标 + 验收
- 报告 `train_rollout_logprob_abs_diff` 类指标；在一个**刻意 fp8/分离式 mismatch**的场景上验证
  TIS 能把 reward 曲线拉回接近 bf16 baseline。

## 5. slime 源码阅读地图（`~/Desktop/slime`）

实现前读真实源码，不要凭名字/README 工作：

- `slime/utils/arguments.py` — `--use-rollout-logprobs`、`--get-mismatch-metrics`、`--use-tis`、
  `--tis-clip` / `--tis-clip-low`、`custom_tis_function_path`。不变式：`use_rollout_logprobs` 与
  `use_tis` 互斥（`assert not args.use_tis`）；`get_mismatch_metrics` 强制
  `custom_tis_function_path`。
- `slime/ray/rollout.py` — `rollout_log_probs` 如何从 rollout samples 流入 `train_data` 并按
  data-parallel 分区切分。
- `slime/backends/megatron_utils/actor.py` — 训练引擎何时重算 `compute_log_prob`；为何
  `get_mismatch_metrics` 即使在 `use_rollout_logprobs` 下也强制额外一次前向：
  `if not use_rollout_logprobs or get_mismatch_metrics: rollout_data.update(compute_log_prob(...))`。
- `slime/backends/megatron_utils/loss.py` — `old_log_probs = rollout_log_probs if
  use_rollout_logprobs else log_probs`；`vanilla_tis_function` 权重方向；TIS 权重 detach；
  `pg_loss` 在哪相乘；`train_rollout_logprob_abs_diff` 报告：
  `tis = exp(old - rollout); w = clamp(tis, tis_clip_low, tis_clip); pg_loss *= w`。
- `examples/train_infer_mismatch_helper/mis.py` — token/sequence/geometric levels；
  truncate/clip/mask 模式；`SAFETY_BOUND = 20.0` 防 exp 溢出；batch 归一化到 mean=1.0；
  rejection sampling + veto 阈值；指标在 pre-RS mask 上聚合。

## 6. Non-goals

- 不在没有 fp8 / 分离式后端 mismatch 时实施（当前无 mismatch 可修）。
- 不照搬 slime 的 token-level TIS 到 per-step diffusion log-prob。
- 不动默认 bf16 路径（rollout == replay，TIS 对它是 no-op）。

## 7. 外部参考

- slime train/infer mismatch helper: <https://github.com/THUDM/slime/blob/main/examples/train_infer_mismatch_helper/README.md>
- slime loss: <https://github.com/THUDM/slime/blob/main/slime/backends/megatron_utils/loss.py>
- verl FP8 RL: <https://verl.readthedocs.io/en/latest/low_precision/fp8.html>
- lmsys FP8 RL: <https://www.lmsys.org/blog/2025-11-25-fp8-rl/>
- SGLang for RL: <https://sgl-project.github.io/advanced_features/sglang_for_rl.html>
