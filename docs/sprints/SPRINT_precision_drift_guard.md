# SPRINT: Precision Drift Guard

状态：proposed（基于本地 `/home/mingfeiguo/Desktop/slime` 代码阅读，2026-06-06）。

## 0. 一句话

`rollout != compute` 不能只停留在“能配置、能记录”。如果正式支持这条路径，必须把采集点 `ratio == 1` 的不变量变成单次 run 内的 guard，并把 rollout-vs-replay logprob mismatch 变成常规 metrics；否则只能把这条配置降级为实验开关。

## 1. slime 证据

slime 没有假设训练后端和 rollout 后端天然一致。它分三层处理 training-inference mismatch。

### 1.1 first-step correctness gate

rollout 统计在 CI 下有硬断言：

```python
assert abs(reduced_log_dict["rollout/log_probs"] - reduced_log_dict["rollout/ref_log_probs"]) < 1e-8
```

训练第一步也有硬断言：

```python
assert log_dict["train/ppo_kl"] < 1e-8, f"{log_dict=}"
```

这说明 slime 把 first-step parity 当 correctness contract，不是纯 debug 输出。

### 1.2 mismatch metrics

slime 在 loss 里保留 rollout engine logprob，并在有 `rollout_log_probs` 时计算训练后端和 rollout 后端的差异：

```python
old_log_probs = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch["log_probs"]
...
train_rollout_logprob_abs_diff = sum_of_sample_mean((old_log_probs - rollout_log_probs).abs())
```

`examples/train_infer_mismatch_helper/README.md` 还列出：

```text
mismatch_kl
mismatch_k3_kl
train_rollout_logprob_abs_diff
```

这些是持续监控指标，不是只在 first step 看一次。

### 1.3 algorithm-level correction

slime 提供：

```text
--use-rollout-logprobs
--use-tis
--get-mismatch-metrics
```

对应代码里 `--use_rollout_logprobs` 和 `--use_tis` 互斥；`--get_mismatch_metrics` 会明确提示即使用 rollout logprobs，为了算 metrics 仍要多做一次 training engine forward。

TIS 的核心权重是：

```python
tis = torch.exp(old_log_probs - rollout_log_probs)
pg_loss = pg_loss * tis_weights
```

所以 slime 的策略不是“低精 rollout 直接相信”，而是检查、度量、或用重要性权重修正。

## 2. wm-infra 现状

### 2.1 已经有正确的精度轴

`PrecisionPolicy` 明确区分：

```python
compute: str
rollout: str
math: str
frozen: str
```

`math` 轴只控制 transformer 之后的 SDE/logprob 算术。`rollout` dtype 已经在 online runner 里传给 generation runtime：

```python
rollout_weight_dtype = resolve_torch_dtype(resolve_precision_policy(cfg).rollout)
```

### 2.2 已经有 first-step parity record，但不是 gate

当前 debug probe 会写：

```python
diff = (log_prob - old_log_prob).abs()
ratio = torch.exp(log_prob - old_log_prob)
```

并输出：

```text
event = first_step_logprob_parity
abs_diff
ratio
```

但它只写 `training_debug.jsonl`，没有阈值、没有 fail。现有测试也只断言记录被写出；测试里 `abs_diff.mean == 1.0` 和 `ratio.mean == exp(1.0)` 仍然通过，这正说明它不是 correctness guard。

### 2.3 `--use-rollout-logprobs` 不能机械照搬

slime 的 token PPO 有“training engine recomputed old logprob”和“rollout engine logprob”两套来源。wm-infra 的 diffusion evaluator 现在已经从 trajectory 读 rollout old logprob：

```python
resolved_old = self._old_log_prob_from_trajectory(...)
```

也就是说，我们的问题不是“denominator 要不要换成 rollout logprob”；它本来就是 rollout behavior logprob。真正问题是：

```text
fresh replay logprob (compute dtype) != old behavior logprob (rollout dtype)
```

所以 P0/P1 应先做 parity guard 和 mismatch metrics；P2 再研究 decoupled correction，而不是先加一个同名开关。

## 3. Sprint 目标

把 `precision.rollout != precision.compute` 从“expressible + observed”升级到“guarded + measurable”。

成功标准：

```text
1. 当 rollout dtype 和 compute dtype 不一致时，默认会跑 bounded first-step parity guard。
2. guard 在 optimizer step 之前执行，超阈值直接 fail，不污染训练状态。
3. 每次训练能记录 rollout-vs-replay logprob mismatch metrics。
4. 同 dtype 默认路径不增加额外 replay 成本。
5. 所有新增行为有单元测试覆盖，并保留现有 debug JSONL。
```

## 4. P0：single-run hard gate

新增一个长期配置，不挂在 `DebugConfig` 下，因为这是 correctness guard，不是临时 probe。

建议结构：

```python
@dataclass(slots=True)
class PrecisionDriftGuardConfig:
    mode: str = "auto"        # "auto" | "off" | "warn" | "fail"
    max_batches: int = 1
    max_timestep_checks: int = 3
    max_abs_log_ratio: float = 1e-3
    max_ratio_abs_dev: float = 1e-3
    fail_on_nonfinite: bool = True
```

`auto` 语义：

```text
precision.rollout == precision.compute -> off，除非 debug.first_step=true
precision.rollout != precision.compute -> fail
```

需要补一条 trainer 可见的 policy 信息。当前 `OnlineTrainer` 只拿到 `TrainerConfig`，不知道 rollout dtype；因此 `_apply_precision_policy` 需要把 `policy.rollout` 也桥接进 trainer config，例如：

```python
trainer_config.rollout_precision = policy.rollout
```

P0 检查范围：

```text
第一批 filtered batch
最多 3 个 timestep：first / middle / last
每个 timestep 复用 evaluator.evaluate(...)
检查 log_ratio = fresh_log_prob - old_log_prob
检查 ratio = exp(log_ratio)
```

失败信息必须带上：

```text
compute_precision
rollout_precision
math_precision
timestep_index
max_abs_log_ratio
max_ratio_abs_dev
old_log_prob stats
fresh_log_prob stats
```

测试：

```text
test_precision_drift_guard_passes_when_within_threshold
test_precision_drift_guard_fails_before_optimizer_when_ratio_drifts
test_precision_drift_guard_auto_enables_for_rollout_compute_mismatch
test_precision_drift_guard_auto_is_off_for_same_dtype_without_debug
test_first_step_debug_still_writes_jsonl_without_enforcing_when_guard_off
```

## 5. P1：常规 mismatch metrics

把 slime 的 metrics 思路映射到 diffusion GRPO：

```text
logprob_abs_diff_mean = mean(abs(fresh - old))
logprob_abs_diff_max
ratio_abs_dev_mean = mean(abs(exp(fresh - old) - 1))
ratio_abs_dev_max
mismatch_kl = mean(old - fresh)
mismatch_k3_kl = mean(exp(fresh - old) - (fresh - old) - 1)
```

落点优先选 `vrl/algorithms/grpo/continuous.py`，因为那里已经计算：

```python
ratio = torch.exp(signals.log_prob - old_log_probs)
approx_kl = 0.5 * torch.mean((signals.log_prob - old_log_probs) ** 2).item()
```

`TrainStepMetrics` 目前是固定字段 dataclass。这里不要用临时 JSONL 代替长期 metrics。两个可选实现：

```text
A. 给 TrainStepMetrics 增加固定字段：更可 grep，更符合现有 metrics 风格。
B. 增加 extra_metrics: dict[str, float]：扩展性强，但类型边界更松。
```

推荐 A。原因是这些不是任意 debug 值，而是 precision/on-policy correctness 的核心指标。

测试：

```text
test_grpo_reports_logprob_mismatch_metrics
test_online_trainer_aggregates_logprob_mismatch_metrics
test_metrics_are_zero_when_fresh_equals_old
```

## 6. P2：algorithm correction spike

P2 是研究项，不作为 P0/P1 的前置条件。

候选方向：

```text
1. decoupled behavior/proximal correction：显式区分 rollout behavior logprob 和 replay proximal logprob。
2. bounded IS / rejection：借鉴 slime TIS/MIS，但需要适配 diffusion per-timestep continuous logprob。
3. 禁止高风险 split：如果 drift gate 在真实模型上经常失败，配置层直接拒绝 rollout=bf16, compute=fp32。
```

非目标：

```text
不直接实现 slime 的 --use-rollout-logprobs 同名开关。
不把 token-level TIS/MIS 硬搬到 diffusion latent SDE。
不为了通过 guard 把 math 轴改成 bf16；math 轴默认 fp32 是正确保护。
```

P2 成功标准：

```text
写出一页 decision note：继续 guard-only、加入 IS correction、还是拒绝 split。
至少用一组 SD3.5 OCR rollout 数据报告 ratio drift 分布。
```

## 7. Architecture hygiene

### ALL_CAPS 常量

`vrl/config/precision.py::_CANONICAL` 应保留。它是配置协议 token 边界，不是业务大表，也没有重复 typed structure 的字段列表。

不要新增 `_ALLOWED_PRECISION_GUARD_MODES = {...}` 这种手写重复常量；如果新增 `PrecisionDriftGuardConfig.mode`，校验可以直接在 `__post_init__` 的局部 tuple/set 中完成，避免再引入模块级 ALL_CAPS。

### thin files/functions

`vrl/trainers/online/debug_probes.py` 这种薄文件应保留。它提供的是诊断边界，避免把 one-shot probe 塞回 `_step_impl` 主训练循环。

但是 hard gate 不应该只是 debug probe 的副作用。建议把共享的 parity 计算抽成一个纯 helper，例如：

```text
compute_logprob_parity_stats(...)
```

然后：

```text
debug JSONL 写入复用 stats
precision drift guard 复用 stats
```

这样 debug 和 gate 共用事实来源，但职责分离。

### 不动什么

```text
不重写 PrecisionPolicy 四轴模型。
不把 rollout/compute/math 合回一个 dtype。
不动现有同 dtype 训练默认行为。
不把 grad_split probe 和 precision drift guard 混在一起。
```

## 8. 验证命令

最小验证：

```bash
pytest tests/config/test_precision.py tests/scripts/test_online_precision_bridge.py tests/trainers/online/test_diagnostics.py tests/algorithms/test_grpo.py
```

新增 metrics 后补：

```bash
pytest tests/trainers/online/test_advantage_and_metrics.py
```

全量相关验证：

```bash
pytest tests/config/test_precision.py tests/scripts/test_online_precision_bridge.py tests/trainers/online tests/algorithms/test_grpo.py tests/rollouts/replay
```

## 9. 参考路径

slime：

```text
/home/mingfeiguo/Desktop/slime/slime/backends/megatron_utils/data.py
/home/mingfeiguo/Desktop/slime/slime/backends/megatron_utils/model.py
/home/mingfeiguo/Desktop/slime/slime/backends/megatron_utils/loss.py
/home/mingfeiguo/Desktop/slime/slime/utils/arguments.py
/home/mingfeiguo/Desktop/slime/slime/ray/rollout.py
/home/mingfeiguo/Desktop/slime/examples/train_infer_mismatch_helper/README.md
/home/mingfeiguo/Desktop/slime/examples/train_infer_mismatch_helper/mis.py
/home/mingfeiguo/Desktop/slime/docs/en/developer_guide/debug.md
/home/mingfeiguo/Desktop/slime/docs/en/blogs/release_v0.1.0.md
```

wm-infra：

```text
/home/mingfeiguo/Desktop/wm-infra/vrl/config/precision.py
/home/mingfeiguo/Desktop/wm-infra/vrl/scripts/common/online.py
/home/mingfeiguo/Desktop/wm-infra/vrl/trainers/core/types.py
/home/mingfeiguo/Desktop/wm-infra/vrl/trainers/online/trainer.py
/home/mingfeiguo/Desktop/wm-infra/vrl/trainers/online/debug_probes.py
/home/mingfeiguo/Desktop/wm-infra/vrl/rollouts/evaluators/trajectory.py
/home/mingfeiguo/Desktop/wm-infra/vrl/rollouts/evaluators/diffusion/sde_logprob.py
/home/mingfeiguo/Desktop/wm-infra/vrl/algorithms/grpo/continuous.py
/home/mingfeiguo/Desktop/wm-infra/tests/trainers/online/test_diagnostics.py
```
