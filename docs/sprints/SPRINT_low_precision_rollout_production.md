# SPRINT: Production 16-bit Rollout Parity

状态：proposed。`SPRINT_precision_drift_guard.md` 已删除；P0/P1 的 guard 和 mismatch metrics 已落地。本 sprint 接管剩余目标：让 16-bit rollout 成为安全生产路径。

## 0. 一句话

目标不是“低精 rollout 出现 drift 后用 TIS 勉强修正”。目标是 **16-bit rollout 和 16-bit replay 本身不 drift**。TIS / decoupled correction 只作为 fallback，用于不可避免的 backend mismatch；第一优先级是 precision-aligned rollout/replay parity。

## 1. 已知事实

当前已经确认：

```text
fp32 rollout / fp32 replay / TF32 off:
  exact parity

bf16 rollout / fp32 replay:
  ratio_abs_dev_mean ~= 1.4e-3
  ratio_abs_dev_max  ~= 1e-2
  clip_fraction      ~= 0.5~0.61
```

这说明：

```text
1. replay restore / trajectory slice / SDE logprob path 是对的。
2. drift 主要来自 rollout forward 和 replay forward 的 precision/backend policy 不一致。
3. 如果生产目标是 16-bit rollout，就不应该默认拿 fp32 replay 去做 behavior parity。
```

生产目标应改成：

```text
precision.rollout = bf16
precision.compute = bf16
precision.math    = fp32
```

也就是：

```text
transformer forward: rollout/replay 都走 16-bit
SDE/logprob math:    仍然 fp32
optimizer state:     不由本 sprint 强行降精
```

## 2. 为什么不是先上 TIS

TIS 的作用是 correction，不是 parity。它适合下面这种问题：

```text
behavior policy != replay/proximal policy
```

但这里的第一目标是：

```text
behavior policy == replay/proximal policy under same 16-bit precision policy
```

所以实施顺序必须是：

```text
1. 先证明 bf16 rollout / bf16 replay / fp32 math 能达到 near-zero parity。
2. 如果不能，再定位是 dtype、TF32、autocast、kernel、batch shape、CFG packing、Ray worker policy 哪一项不一致。
3. 只有确认 backend mismatch 无法消除，才进入 TIS / decoupled correction。
```

当前 precision drift guard 仍然有用，但它是安全闸：

```text
correction/alignment 未完成时:
  low-precision split fail fast

alignment 完成后:
  guard 用来证明 same precision policy 下 parity 通过
```

## 3. 需要跟 slime 的真实源码

`/home/mingfeiguo/Desktop/slime` 是真实参考代码。后续实现前必须读这些路径，而不是只看 README 或凭算法名搬：

```text
/home/mingfeiguo/Desktop/slime/slime/utils/arguments.py
```

重点 follow：

```text
--use-rollout-logprobs
--get-mismatch-metrics
--use-tis
--tis-clip / --tis-clip-low
custom_tis_function_path
use_rollout_logprobs 和 use_tis 互斥
get_mismatch_metrics 会强制要求 custom_tis_function_path
```

关键代码：

```python
if args.use_rollout_logprobs:
    assert not args.use_tis
```

```text
/home/mingfeiguo/Desktop/slime/slime/ray/rollout.py
```

重点 follow：

```text
rollout_log_probs 如何从 rollout samples 进入 train_data
rollout_log_probs 如何按 data parallel partition split
```

关键代码：

```python
if samples[0].rollout_log_probs is not None:
    train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]
```

```text
/home/mingfeiguo/Desktop/slime/slime/backends/megatron_utils/actor.py
```

重点 follow：

```text
什么时候 training engine 重新 compute_log_prob
use_rollout_logprobs 时如何跳过 recompute
get_mismatch_metrics 时为何即使 use_rollout_logprobs 也要多做一次 forward
```

关键代码：

```python
if not self.args.use_rollout_logprobs or self.args.get_mismatch_metrics:
    rollout_data.update(self.compute_log_prob(...))
```

```text
/home/mingfeiguo/Desktop/slime/slime/backends/megatron_utils/loss.py
```

重点 follow：

```text
old_log_probs = rollout_log_probs if use_rollout_logprobs else log_probs
vanilla_tis_function 的权重方向
TIS weight 是否 detach
pg_loss 在哪里乘 TIS
train_rollout_logprob_abs_diff 如何汇报
get_mismatch_metrics/use_tis 如何共享 metrics path
```

关键代码：

```python
tis = torch.exp(old_log_probs - rollout_log_probs)
tis_weights = torch.clamp(tis, min=args.tis_clip_low, max=args.tis_clip)
pg_loss = pg_loss * tis_weights
```

```text
/home/mingfeiguo/Desktop/slime/examples/train_infer_mismatch_helper/mis.py
```

重点 follow：

```text
token / sequence / geometric 三种 level
truncate / clip / mask 三种 mode
SAFETY_BOUND = 20.0 防 exp overflow
batch normalize to mean=1.0
rejection sampling 和 veto threshold
metrics 聚合时用 pre-RS mask
```

这些源码只作为 correction fallback 的参考。第一阶段不要照搬 slime TIS；先做 16-bit parity。

## 4. wm-infra 当前关键路径

必须 follow 这些本地 source：

```text
vrl/scripts/common/online.py
```

重点：

```text
trainer_config.mixed_precision = policy.compute
trainer_config.rollout_precision = policy.rollout
rollout_weight_dtype = resolve_torch_dtype(resolve_precision_policy(cfg).rollout)
```

这里已经有 rollout dtype 和 compute dtype 两条轴，但还没有证明 bf16/bf16 replay 与 rollout 完全同策略。

```text
vrl/trainers/online/trainer.py
```

重点：

```text
_get_autocast() 根据 trainer mixed_precision 控制 replay forward autocast。
debug first_step 和 precision_drift_guard 都复用 evaluator.evaluate。
训练 loss forward 也在同一个 autocast_ctx 下。
```

```text
vrl/models/diffusion/sd3_5/model.py
```

重点：

```text
forward_step() 用 self._transformer_dtype() 决定 latents / timestep / embeds dtype。
如果 compute=bf16，replay transformer dtype 应该是 bf16。
```

```text
vrl/generation/diffusion/executor.py
```

重点：

```text
rollout side 根据 prompt_embeds dtype 开 autocast。
noise_pred.float() 后再进 sde_step_with_logprob。
observations/actions/log_probs 存入 trajectory。
```

这条路径正好支持目标策略：

```text
forward precision aligned at bf16
SDE/logprob math remains fp32
```

## 5. Sprint 目标

成功状态：

```text
1. bf16 rollout / bf16 replay / fp32 math 在 SD3.5 OCR 上 first-step parity 接近 fp32/fp32 no-TF32 baseline。
2. guard 支持 precision-aligned low-precision path，不再把 bf16/bf16 误判成 unsafe split。
3. bf16 rollout / fp32 replay 仍然 fail，除非显式打开 correction fallback。
4. metrics 同时报告:
   - raw rollout-vs-replay mismatch
   - same-policy aligned parity
   - dtype/backend policy used by rollout and replay
5. 真实 run 能证明 bf16 rollout 有 rollout memory/throughput benefit，且不会引入 first-step ratio drift。
```

非目标：

```text
不把 SDE/logprob math 降成 bf16。
不默认开启 TIS。
不直接照搬 slime token-level tis_clip=2.0。
不为了通过 guard 隐藏 raw mismatch metrics。
```

## 6. T0：补齐 precision policy 可观测性

先让 debug 和 metrics 能清楚说明两边到底用了什么：

```text
rollout_precision
compute_precision
math_precision
trainer_autocast_enabled
trainer_transformer_dtype
rollout_transformer_dtype
allow_tf32
```

需要写入：

```text
training_debug.jsonl
precision_drift_guard record
runtime_debug / rollout chunk metadata
```

这样之后看到 drift，不会再猜是 dtype、TF32 还是 Ray worker policy。

## 7. T1：跑 bf16/bf16 parity gate

真实 run：

```bash
python -m vrl.scripts.train \
  --config experiment/diffusion/sd3_5/online_grpo_ocr \
  trainer.total_epochs=1 \
  actor.optim.lr=0.0 \
  eval.enable=false \
  precision.rollout=bf16 \
  precision.compute=bf16 \
  precision.math=fp32 \
  trainer.precision_drift_guard.mode=warn
```

需要对比：

```text
A. fp32/fp32/TF32 off baseline:
   ratio_abs_dev_mean = 0
   ratio_abs_dev_max  = 0

B. bf16/fp32:
   ratio_abs_dev_mean ~= 1.4e-3
   ratio_abs_dev_max  ~= 1e-2

C. bf16/bf16:
   目标是接近 A，而不是接近 B。
```

如果 C 仍然明显 drift，先定位一致性问题，不进入 TIS。

## 8. T2：对齐 rollout/replay backend policy

需要检查和修正：

```text
1. Trainer replay autocast 是否真的打开 bf16。
2. Replay model transformer dtype 是否真的是 bf16。
3. Ray rollout worker transformer dtype 是否真的是 bf16。
4. TF32 policy 是否两边一致；即使 bf16 下 TF32 不应是主项，也要记录。
5. CFG batching 是否同形状。
6. timestep packing 是否最终给 transformer 相同 dtype/shape。
7. trajectory tensors 是否保留了足够精度：latents/action 可以 fp32，forward input 再 cast 到 transformer dtype。
```

修复优先级：

```text
P0: dtype / autocast / model dtype mismatch
P1: Ray worker metadata 和 debug visibility
P2: deterministic kernel / batch shape alignment
```

## 9. T3：guard 语义更新

当前 guard 是：

```text
rollout_precision != compute_precision -> auto fail
```

这对 bf16/fp32 是对的，但生产目标要求：

```text
rollout=bf16, compute=bf16, math=fp32 -> allowed and checked
```

建议 guard 语义：

```text
same forward precision:
  run optional parity check when debug/guard enabled;
  pass/fail based on measured drift.

different forward precision:
  fail unless correction_mode != off.

math precision:
  not used to decide rollout/compute mismatch; math=fp32 is expected.
```

也就是说，guard 应比较 forward precision policy，而不是简单比较所有 precision axes。

## 10. T4：如果 bf16/bf16 仍 drift，再进入 fallback

只有当对齐后仍不能让 16-bit rollout/replay parity 通过，才打开 correction fallback。

候选：

```text
1. deterministic rollout/replay kernel alignment
2. replay 使用 rollout-equivalent forward path
3. decoupled PPO + TIS
4. rollout-aligned quantization-aware replay
```

TIS fallback 的正确语义：

```text
proximal_ratio =
  exp(current_compute_log_prob - old_compute_log_prob)

mismatch_ratio =
  exp(old_compute_log_prob - behavior_rollout_log_prob)

policy_loss =
  bounded(mismatch_ratio).detach()
  * PPO(proximal_ratio, advantage)
```

不要把 `exp(current_compute_log_prob - behavior_rollout_log_prob)` 继续当一个 ratio 同时承担两个角色。

## 11. T5：测试

单元测试：

```text
test_precision_guard_allows_same_forward_precision_bf16_math_fp32
test_precision_guard_fails_bf16_rollout_fp32_compute_without_correction
test_precision_debug_records_rollout_and_replay_transformer_dtype
test_sd3_replay_forward_uses_transformer_dtype_under_bf16_compute
test_rollout_executor_records_transformer_dtype
```

集成测试：

```text
test_online_precision_bridge_sets_bf16_compute_and_bf16_rollout
test_same_precision_low_precision_guard_runs_without_extra_policy_split
test_bf16_bf16_metrics_are_reported_as_regular_csv_columns
```

真实 run 验证：

```text
SD3.5 OCR:
  fp32/fp32/TF32 off
  bf16/fp32
  bf16/bf16

通过标准:
  bf16/bf16 first-step drift 接近 fp32/fp32 baseline；
  bf16/fp32 继续复现 drift；
  metrics 能解释差异来源。
```

## 12. T6：生产推荐

完成 T1/T2 后的推荐应是：

```text
production low-precision rollout:
  precision.rollout=bf16
  precision.compute=bf16
  precision.math=fp32

unsupported by default:
  precision.rollout=bf16
  precision.compute=fp32
```

如果业务必须用 bf16 rollout + fp32 compute，另开 correction sprint，参考 slime TIS/MIS 源码，但不要把它和 16-bit parity 目标混在一起。

## 13. 参考

本地源码：

```text
/home/mingfeiguo/Desktop/slime/slime/utils/arguments.py
/home/mingfeiguo/Desktop/slime/slime/ray/rollout.py
/home/mingfeiguo/Desktop/slime/slime/backends/megatron_utils/actor.py
/home/mingfeiguo/Desktop/slime/slime/backends/megatron_utils/loss.py
/home/mingfeiguo/Desktop/slime/examples/train_infer_mismatch_helper/README.md
/home/mingfeiguo/Desktop/slime/examples/train_infer_mismatch_helper/mis.py
/home/mingfeiguo/Desktop/slime/examples/train_infer_mismatch_helper/mis.yaml
```

外部资料：

```text
https://github.com/THUDM/slime/blob/main/examples/train_infer_mismatch_helper/README.md
https://github.com/THUDM/slime/blob/main/slime/backends/megatron_utils/loss.py
https://openreview.net/pdf?id=8MHqvb4lK9
https://verl.readthedocs.io/en/latest/low_precision/fp8.html
https://www.lmsys.org/blog/2025-11-25-fp8-rl/
https://sgl-project.github.io/advanced_features/sglang_for_rl.html
https://arxiv.org/abs/2601.14243
https://arxiv.org/abs/2604.07853
https://arxiv.org/abs/2605.13907
```
