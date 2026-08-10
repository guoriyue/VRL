# SPRINT：Continuous stage capacity calibration

状态：**parked（2026-07-21）**。等待
[UnifiedReward adaptive batching](SPRINT_unified_reward_adaptive_batching.md) 完成或走完 negative exit。

父 program：[Continuous three-stage pipeline](../planned/SPRINT_continuous_three_stage_pipeline_program.md)

## 0. 结论先行

在写 adaptive controller 前，先量出每个 stage 的安全 envelope。controller 只能在已验证的
generation inflight、unscored bytes、reward microbatch、ready bytes 范围内调整，不能把生产
长跑当 OOM 搜索器。

本 sprint 交付一个可重复运行的 capacity calibrator 和一份四 L4 profile。它不是 eval：输入
只用于测量内存、service time 和吞吐，不判断 checkpoint 质量，也不把分数反馈给训练。

## 1. Root cause / current behavior

当前 continuous config 的主要上限是人工填写：

```text
max_inflight_groups
max_ready_bytes_mb
reward service max_pending_requests
```

这些值没有统一表达三个不同风险：

```text
GPU execution capacity
host memory / artifact capacity
policy staleness / useful-work capacity
```

仅看显存占用也不够。unscored item 可能主要占 host tensor 和落盘视频；ready item 又携带 replay
trajectory。继续增加 inflight 可能不 OOM，却让 host RAM、artifact disk 或 staleness 先失控。

## 2. Goal and ownership boundary

定义 typed `ContinuousCapacityProfile`，字段只保留实际 runtime consumer：

```text
generation_inflight_max
unscored_groups_max
unscored_bytes_max
reward_request_groups_max
reward_microbatch_max
reward_pending_requests_max
ready_groups_max
ready_bytes_max
artifact_bytes_max
```

- calibrator 负责测量并输出 profile。
- config resolver 负责验证 profile 与当前 topology/model/workload signature 匹配。
- runtime settings 消费已 resolve 的硬上限。
- adaptive controller 只能输出这些上限内的当前 target。

profile provenance 必须包括会改变容量的 workload facts：model/reward version、resolution、frames、
denoise steps、samples per group、trajectory storage、dtype 和 GPU type/count。只用于显示的完整 GPU
pool 必须标注 provenance-only。

## 3. Correctness and resource invariants

1. calibration 使用与生产相同的 model build/runtime path，不用 synthetic tiny model 冒充容量。
2. candidate 失败必须完成 CUDA/Ray/reward/artifact cleanup 后才能测试下一个。
3. 安全值低于最大成功值，保留明确 headroom；不能把刚好一次不 OOM 当生产上限。
4. host RAM、GPU memory 和 artifact disk 都有独立 hard cap。
5. profile signature 不匹配时 fail closed 或退回保守 static defaults，不能静默套用旧值。
6. calibration 不执行 optimizer update，不写 training checkpoint，不污染正式 metrics.csv。
7. scratch artifacts 使用明确 `_probe`/`_scratch` 命名，结论归档后删除。

## 4. Implementation stages

### T0 — Workload signature

- 从实际 resolved config 派生 capacity-relevant signature。
- validation set 从 typed profile fields 派生，不维护手写允许键表。
- profile schema/version 有单一 source of truth。

### T1 — Measurement runner

在 `vrl/scripts/perf/` 建长期可复用 calibrator，分别测：

```text
generation inflight / chunk dispatch
unscored queue host+artifact footprint
reward request and model microbatch
ready queue trajectory footprint
end-to-end two-batch overlap
```

每项从保守值开始，失败后停止该轴；不同时扫所有维度造成不可解释组合。

### T2 — Headroom policy

- GPU memory 使用 peak allocated/reserved 和系统可见显存双证据。
- host RAM 使用 process + node pressure，避免只看 Python tensor estimator。
- artifact disk 使用 live bytes/high-watermark 和 cleanup 后残留。
- service latency 使用稳定段 percentile，不用单次最快结果。
- 计算 hard max 与 recommended target；runtime 只信 hard max，controller 从 recommended 开始。

### T3 — Profile consumption

- 四 L4 recipe 显式引用 profile 或嵌入其 resolved values/provenance。
- runtime startup 验证 topology/workload signature。
- 用户显式配置超过 hard max 时 fail fast，并显示具体轴和 profile provenance。

### T4 — Archive and cleanup

- 结果写入 `docs/sprints/info/`，包括命令、环境、候选表、失败点、headroom 和最终 profile。
- 删除同源 scratch media/log/CSV；保留 calibrator、profile schema 和代表性 tests。

## 5. Failure, cancellation and recovery semantics

- OOM candidate 触发 runtime teardown/rebuild；不能假设 `empty_cache()` 足以恢复未知 worker state。
- reward parse error 不是 capacity failure，必须终止该候选并修正确性问题。
- artifact cleanup failure 阻止继续 sweep，避免后续候选在污染环境中得出错误容量。
- calibrator 被中断时执行 terminal cleanup，并把未完成 profile 标为 invalid。

## 6. Telemetry

每个 candidate 记录：

```text
candidate settings
logical groups / artifacts completed
generation and reward throughput
GPU peak memory / duty / power
host RSS / available memory / pressure
queue peak items / bytes / age
artifact live and residual bytes
retry / OOM / cancellation counts
cleanup settled
```

## 7. Verification and acceptance gates

- signature 相同能加载，任一关键 workload field 改变都会拒绝旧 profile。
- OOM、reward failure、interrupt 三条路径均完成 cleanup。
- recommended target 始终不超过 hard max，并保留配置要求的 headroom。
- repeated candidate 的 service time/peak memory 在声明容差内稳定。
- profile fields 全部有 non-logging consumer；无 dead/no-op knob。
- 四 L4 profile 能被 continuous recipe resolve 并通过 config tests。

## 8. What changes / what stays

### 改变

- stage 上限从散落的人工猜测变为带 workload provenance 的 profile。
- config startup 验证 capacity signature。
- adaptive controller 获得不可越过的 envelope。

### 保持

- GPU role placement 固定。
- trainer batch size、reward semantics 和 staleness window。
- 用户仍可选择更保守值。

## 9. Non-goals

- 不自动增加 samples/prompts/PPO epochs。
- 不在线搜索 OOM 边界。
- 不动态迁移 trainer/rollout/reward 模型。
- 不执行 eval 或 checkpoint 排名。
- 不保留一次性 scratch dataset/media。

## 10. Definition of Done

- [ ] calibrator 可重复运行并安全清理失败 candidate。
- [ ] `ContinuousCapacityProfile` 有 schema、signature、validation 和 runtime consumers。
- [ ] 四 L4 profile 与完整测量档案已落地。
- [ ] recipe 超界/签名不匹配 fail closed。
- [ ] 所有 one-shot artifacts 已清理。
- [ ] adaptive sprint 可以只在 profile envelope 内工作。

## 11. References

- `vrl/config/schema.py`
- `vrl/trainers/core/types.py`
- `vrl/rollouts/orchestration/continuous/types.py`
- `vrl/rollouts/orchestration/continuous/scheduler.py`
- `vrl/scripts/perf/reward_overlap_benchmark.py`
- `vrl/config/presets/experiment/wan_2_1/online_grpo_robotics_physics_4x_l4_continuous.yaml`
- `vrl/config/reward_service/unified_reward_robotics.yaml`
