# SPRINT: RL-safe diffusion feature-cache probe（parked）

状态：**parked / waiting for a new redundancy event**。

## 为什么不是 planned

TeaCache 已经落地，真实 Cosmos probe 也已完成：

- current owner：`vrl/generation/steps/denoise/teacache.py`
- real drift runner：`vrl/scripts/perf/teacache_drift_probe.py`
- 10–35 step 的现有 family 只有约 0–5% 可跳过，实际收益边际
- rollout/replay drift 在已测阈值上很小，但没有足够 compute saving 支撑继续投资

因此当前仓库没有一个值得马上实现的新 feature-cache sprint。等待事件是：出现 50–100 step 的新 family，或 measurement 证明某个现有 family 有显著 adjacent-step redundancy。

## 事件触发后的正确性合同

```text
trainable_sde : fresh forward, writes/consumes old_log_prob, participates in PG
ode_fresh     : fresh forward, no PG log_prob
ode_cached    : cached/skip/reuse, no PG log_prob
```

硬不变量：

- `cached => not trainable`
- `trainable_sde => fresh forward`
- cached step 不写入或消费 policy-gradient log-prob
- rollout/replay mismatch 与 reward variance 是 proof gates
- 不允许靠大量 TIS/RS mask 掉样本来制造“稳定”

## 重新启动条件

只有满足至少一项才回到 `planned/`：

1. 新 family 默认 schedule 达到约 50 steps 以上；
2. exact measurement 显示低变化窗口足以带来有意义的 forward skip；
3. 新 cache 方法能在不污染 trainable step 的前提下提供独立证据。

启动后的顺序：

1. measurement-only：不启用 cache，记录相邻 step drift；
2. same-seed cache off/on：比较 media、reward、latency；
3. optimizer-off rollout→replay dry-run：检查 log-prob mismatch；
4. 只有前三门通过才跑短 RL curve；本文件不授权长跑。

## 保持不变

- 保留现有 default-off TeaCache 与 drift probe；它们是长期验证资产。
- 不把 feature cache 描述为 PagedAttention 类 exact optimization。
- 不同时叠加 fp8、shared-prefix 或 step-wise batching。
- 不新增 `step_kind` enum/ALL_CAPS taxonomy，直到触发事件证明有第二个真实消费者；当前抽象会早于需求。
- 不删除现有 TeaCache thin state/helper：它们共同构成已测试的 skip decision machine 与 rollout adapter。

## References

- `docs/sprints/done/SPRINT_rollout_vllm_migration.md`
- `docs/sprints/parked/SPRINT_efficient_rollout_program.md`
- `vrl/generation/steps/denoise/teacache.py`
- `vrl/generation/steps/denoise/loop.py`
- `vrl/scripts/perf/teacache_drift_probe.py`
- `tests/generation/steps/denoise/test_teacache.py`
