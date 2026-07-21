# SPRINT：Continuous three-stage 12-hour GA

状态：**parked（2026-07-21）**。触发条件：

- [Stage contracts](../planned/SPRINT_continuous_stage_contracts_and_baseline.md)、
  [generation/reward pump](SPRINT_continuous_generation_reward_pump.md)、
  [versioned lookahead](SPRINT_continuous_versioned_lookahead.md)、
  [reward batching](SPRINT_unified_reward_adaptive_batching.md)、
  [capacity calibration](SPRINT_continuous_stage_capacity_calibration.md)、
  [adaptive backpressure](SPRINT_continuous_adaptive_backpressure.md) 和
  [recovery](SPRINT_continuous_pipeline_recovery.md) 均完成。

父 program：[Continuous three-stage pipeline](../planned/SPRINT_continuous_three_stage_pipeline_program.md)

## 0. 结论先行

用同一 checkpoint、prompt sampler 和四 L4 拓扑做三臂验证：

```text
A: current finite-batch continuous baseline
B: three-stage pipeline + measured static capacity profile
C: three-stage pipeline + adaptive backpressure
```

先跑 correctness/fault smoke，再跑 12 小时 uninterrupted C arm。只有吞吐提高来自有用轨迹、
所有队列/内存有界、版本和 reward/gradient health 正常、cleanup/restart 通过，才把新路径转默认。

reward 均值不要求每个 epoch 单调上升；GRPO 小 prompt set 本来会有噪声。本 sprint 不运行 inline
eval，也不把 eval 结果设为 promotion gate。模型质量仍可另走离线 checkpoint evaluation，但与
流水线正确性/吞吐验收解耦。

## 1. Baseline and comparison contract

三臂必须固定：

```text
starting checkpoint
prompt sampler seed and dataset
prompts per update / samples per prompt
denoise schedule and replay timesteps
PPO epochs and optimizer config
reward model/version/rubric
precision and weight-sync mode
GPU role placement
supervisor health thresholds
```

允许改变的只有 orchestration、queue caps、reward batch 和 controller mode。A/B/C 的输出目录、
artifact namespace 和 metrics 分开，不能续写同一个 metrics.csv。

## 2. Primary success metric

首要指标：

```text
useful optimizer updates per hour
```

“useful”要求该 update：

- group/sample 完整；
- reward 成功且无默认/partial score；
- policy staleness 在允许窗口内；
- optimizer 实际 step，没有 NaN/Inf skip；
- 没有重复 logical group。

辅助指标：useful trajectories/hour、trainer wait、stage duty、queue wait、artifact throughput。
GPU utilization 只用于解释，不单独决定通过。

## 3. Correctness and resource gates

### Algorithm/data

- mixed policy version：0。
- future/too-stale batch：0；observed staleness `<= 1`。
- duplicate/missing prompt group or sample：0。
- reward result count/order mismatch：0。
- non-finite reward/advantage/loss/grad：0。
- supervisor parity、reward std、grad norm health gates持续通过。

### Throughput

- B 相对 A 的 median useful updates/hour 至少提高 10%，否则 stage split 不转默认。
- C 不低于 B；若 adaptive 没有额外收益但 correctness 正常，默认使用 B 的 static profile。
- trainer wait-for-ready 相对 A 明显下降，且收益不能来自更高 staleness 或 discarded work。
- eligible overlap window 内的 avoidable rollout idle 至少减少一半；weight sync/checkpoint 等不可避免
  barrier 单独归因，不强求全程 100%。

### Memory/queue/artifact

- GPU memory 在 capacity hard limits 内，无 OOM fallback storm。
- warmup 后 host RSS、queue bytes 和 artifact live bytes 无持续单调爬升。
- unscored/ready queue 不越 item/byte cap，不依赖 eviction 维持稳态。
- graceful shutdown/restart 后 live artifact lease、active reward request 和 owned Ray task 归零。

### Reward/training health

- reward components 范围/parse contract 正常。
- reward std 不长期塌到 health threshold 以下。
- grad norm 不长期为零，optimizer step 未被 scaler 连续跳过。
- reward 均值曲线用于诊断 reward hacking/崩溃，不要求 12 小时内单调增长。

## 4. Execution stages

### T0 — Deterministic correctness smoke

- A/B 使用相同固定 input/control，验证 batch/group/reward/trajectory contract。
- fake/gated tests 已通过后才占用真四卡。
- batch=1 reward control 与实际 batching 结果 parity。

### T1 — Fault smoke

在短 run 中注入 generation failure、reward retryable error、reward parse failure、SIGTERM 和 weight
sync failure，验证 supervisor、checkpoint resume 和 cleanup。任何 leak/protocol error 都阻止长跑。

### T2 — A/B/C throughput runs

- 每臂包含 warmup 和稳定采样段，记录同一 metric contract。
- 轮换执行顺序或记录环境温度/clock，避免单次热状态偏差。
- 不用一条 `nvidia-smi` snapshot 代表整个 run；使用固定频率 timeline 和 stage intervals。

### T3 — 12-hour uninterrupted C run

- 从已验证 checkpoint 启动 supervisor。
- 持续写 metrics/health verdict/checkpoints。
- telemetry 异常时 adaptive 自动退 static；correctness 异常时 fail closed，不继续烧卡。
- 结束后 graceful shutdown 并检查所有 resource/artifact owner。

### T4 — Promotion

- 将 A/B/C 表、timeline、queue high-watermark、health、negative observations 写入
  `docs/sprints/info/`。
- 所有 gate 通过才更新四 L4 recipe 默认。
- adaptive 不胜 static 时保留 static 默认；three-stage 不胜 baseline 时保留 current path，并删除
  无生产消费者的 migration flags。

## 5. Failure and rollback

| 结果 | 决定 |
|---|---|
| B 不胜 A | 不转 three-stage 默认；保留 telemetry/独立 reward batching 收益 |
| C 不胜 B | 默认 B static profile；adaptive 保持非默认或删除 |
| 吞吐提高但 stale/discard 增加 | 判失败，修 scheduler，不接受“更多无效 work” |
| 吞吐提高但 memory/queue 爬升 | 判失败，修 ownership/backpressure |
| reward batching parity 失败 | 退 batch=1，不接受近似 reward 漂移 |
| fault cleanup/restart 失败 | 不开始 12 小时 run |
| health gate 连续失败 | supervisor 停止；从最后完整 checkpoint 诊断，不自动放宽阈值 |

## 6. Telemetry/report

最终档案至少包含：

```text
config/checkpoint/workload provenance
A/B/C useful updates and trajectories per hour
generation/reward/training timeline
trainer wait and stage queue waits
GPU duty/power/memory timeline
queue item/byte/age high-watermarks
staleness and discard/retry counts
reward component/std and grad/optimizer health
artifact lifecycle and shutdown residue
controller decisions/fallbacks
all failures and negative results
```

## 7. Verification and acceptance gates

- 所有 child sprint test suites 全绿。
- config resolve/load-all experiments 通过。
- fault smoke 资源归零、checkpoint resume 正确。
- A/B/C 使用同一 comparison contract。
- 12 小时 C arm 无 correctness/health terminal failure。
- throughput、memory、queue、artifact、staleness gates 全部通过。
- 最终 diff/recipe/docs 与实际选择一致，没有保留 no-op knobs。

## 8. What changes / what stays

### 通过后改变

- 四 L4 continuous recipe 默认使用 three-stage pipeline。
- adaptive 只有在胜过 static 且无风险时转默认。
- capacity profile 和 supervisor command 记录在 canonical recipe/docs。

### 保持

- trainer GPU 0、rollout GPU 1/2、reward GPU 3。
- 16 trajectories/update、训练目标和 reward rubric。
- max stale 1 与现有 health gates。
- eval 独立于 hot path。

## 9. Non-goals

- 不声称 309 prompts 足以证明泛化或 robotics benchmark 质量。
- 不以 reward curve 单调增长作为调度 GA gate。
- 不在长跑中动态改模型、rubric、dataset 或 optimizer。
- 不要求每张 GPU 每个采样点都 100%。
- 不运行 inline eval。

## 10. Definition of Done

- [ ] A/B/C correctness、fault 和 throughput 结果完整。
- [ ] 12 小时 uninterrupted run 通过全部 hard gates。
- [ ] useful updates/hour 达到 promotion threshold。
- [ ] memory/queue/artifact 无增长或泄漏。
- [ ] final default/rollback 决策已写入 info archive 和 recipe。
- [ ] program 所有 child 文档按状态移动到 `done/`。

## 11. References

- `vrl/config/presets/experiment/wan_2_1/online_grpo_robotics_physics_4x_l4_continuous.yaml`
- `vrl/config/reward_service/unified_reward_robotics.yaml`
- `vrl/scripts/supervise.py`
- `vrl/scripts/common/online.py`
- `vrl/trainers/online/trainer.py`
- `vrl/rollouts/orchestration/continuous/`
- `tests/rollouts/orchestration/continuous/`
- `tests/rewards/service/test_service.py`
- `tests/scripts/test_supervise.py`
