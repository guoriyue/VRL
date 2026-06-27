# SPRINT: RL-safe diffusion feature cache probe

状态：**planned / high-risk proof-gated（2026-06-26）**。这是近似计算复用，不是 exact 系统优化。目标是把 DeepCache / TeaCache / FasterCache / ToCa / FORA 这类 diffusion cache 方法放进 RL rollout 时，明确哪些 step 可以 cache、哪些 step 绝对不能 cache，并用 rollout-vs-replay drift 和 reward curve 做硬门禁。

## 0. 一句话

diffusion cache 方法能省 forward，因为相邻 denoise step 的部分 feature/output 相似。但 RL 不只关心视频质量，还关心：

```text
old_log_prob 是否可信
replay log_prob 是否能对齐
group reward variance 是否还活着
```

所以第一原则是：

```text
trainable_sde step 必须 fresh forward
cached step 不能参与 policy-gradient log_prob
```

## 0.5 实测判决（2026-06-26，SD3.5-medium 1024² / RTX 5090）—— 减少执行工作量这条轴

`vrl/scripts/perf/{rollout_bottleneck,dit_mfu}_probe.py` 给的是两条不同轴：

- rollout DiT **batch-scaling 平**（eager ms/sample batch 1→16 全程 ~155ms）→ 加 batch / step-wise batching 不降低单样本成本。
- DiT MFU：eager 69% → torch.compile **94%**（1.37x fusion）→ 换 kernel 仍有收益，compile 后才接近当前硬件上限。

**关键推论：在 compile/fp8 这类 kernel 轴吃完后，继续砍 rollout wall-clock 的主要杠杆就是减少实际执行的 denoise forward / token / layer 工作量。** feature cache 正是这条轴：它不靠更大 batch，也不靠 page store 省非瓶颈显存，而是直接减少要跑的计算。

但要诚实标注代价:feature cache **减少执行工作量,不保证提高 MFU**，且拿 RL 正确性换(cached ≠ trainable,见 §4 硬不变量)。所以它不是免费午餐,而是"用正确性风险换 compute"。本 sprint 的 step_kind contract + drift/reward 硬门**正是兑现这个 trade 的前提**。证据见记忆 `project_rollout_bound_class_probe`。

## 1. Related work

- DeepCache 利用 denoise step 之间的 temporal redundancy，复用高层 features 来减少重复计算：https://arxiv.org/abs/2312.00858
- TeaCache 用 timestep embedding aware 的输入差异估计 output 差异，尤其面向 video diffusion 的训练-free cache：https://arxiv.org/abs/2411.19108
- FasterCache 指出直接复用相邻 step feature 会损伤 video 细微变化，并提出 dynamic feature reuse 与 CFG-Cache：https://arxiv.org/abs/2410.19355
- ToCa 把 feature cache 做到 token-wise，按 token/layer 的 cache sensitivity 选择复用对象：https://arxiv.org/html/2410.05317v2
- FORA 在 DiT 里缓存 attention/MLP intermediate outputs，属于训练-free DiT acceleration：https://arxiv.org/abs/2407.01425

## 2. 当前代码证据

仓库已有 rollout-only TeaCache 入口：

```text
vrl/generation/diffusion/executor.py
  TeaCacheState
  teacache.should_run(...)
  teacache.cached_noise_pred
```

并且注释已经承认 TeaCache drift 和 fp8 drift 一样，需要 drift guard/TIS/RS 兜住：

```text
vrl/generation/diffusion/teacache.py
vrl/algorithms/logprob_mismatch.py
vrl/algorithms/grpo/continuous.py
```

问题是：现在 cache 只是 rollout-side optimization，没有一个 RL step-kind contract 明确禁止 cached step 被当成 trainable log-prob step 消费。

## 3. 设计

把 denoise step 明确分成三态：

```text
trainable_sde: fresh forward, has old_log_prob, participates in PG
ode_fresh: fresh forward, no PG log_prob
ode_cached: cached/skip/reuse, no PG log_prob
```

第一版只允许：

```text
TeaCache / feature cache -> ode_cached
SDE training window      -> trainable_sde
```

如果一个 step 同时是 cache 和 trainable，直接 config validation fail。

## 4. 正确性契约

- `cached => not trainable` 是硬不变量。
- `trainable_sde => fresh forward` 是硬不变量。
- `old_log_prob` 只对 trainable step 写入/消费；cached step 的 old_log_prob 不能混进 GRPO ratio。
- reward variance 是 proof gate：cache 开启后 `adv_zero_rate` 不能显著上升。
- rollout-vs-replay mismatch 是 proof gate：`ratio_abs_dev_max` / `logprob_abs_diff_max` 超阈值则关闭 cache。

## 5. 执行顺序

1. 加 measurement mode：不启用 cache，只记录 adjacent-step noise_pred/feature drift，找低变化窗口。
2. 把现有 TeaCache 接到 explicit step_kind，保证 `cached` step 不进入 train_indices。
3. 做 same-seed generation quality probe：cache off vs cache on 的 media/reward/latency 对比。
4. 做 RL dry-run：不开 optimizer，只跑 rollout->replay，检查 drift guard/TIS/RS 指标。
5. 只有 dry-run 过关，才跑短 RL curve；短曲线过关才允许长跑。

## 6. 验收

- config 层能拒绝 `trainable_sde + cached` 的非法组合。
- cache-on wall-clock 明显下降。
- `adv_zero_rate`、reward variance、eval reward curve 不退化。
- precision drift guard 不因 cache 大面积失败；如果失败，必须默认关闭 cache。
- TIS/RS 触发率可解释，不能靠把大量样本 mask 掉来假装稳定。

## 7. 非目标

- 不把 feature cache 称为 PagedAttention 类 exact optimization。
- 不让 cached step 产生 policy-gradient。
- 不在同一次实验里叠加 fp8、shared-prefix、step-wise batching；一次只测一个变量。
- 不追求 novelty claim；这些 cache 方法已有大量 prior art，本 sprint 的价值是 RL correctness contract。

## 8. 关键文件

- `vrl/generation/diffusion/executor.py`
- `vrl/generation/diffusion/teacache.py`
- `vrl/algorithms/logprob_mismatch.py`
- `vrl/algorithms/grpo/continuous.py`
- `vrl/trainers/online/trainer.py`
