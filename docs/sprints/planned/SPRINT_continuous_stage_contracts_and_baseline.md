# SPRINT：Continuous stage contracts and baseline

状态：**in progress（2026-08-22）**：T0/T1/T2 代码已落地并通过测试；T3 四卡基线待 4×L4
硬件（当前开发机为 1×RTX 5090，无法真实测量该拓扑）。

> 修订 2026-08-22：对照当前代码复核后调整——`collect_prompt_batches` 已重命名为
> `collect_prompt_groups`（83148c9e）；§6 改为逐指标定归约语义并删去三个已有指标的
> 别名；§2/T0 钉住 `batch_id` 的 Sprint-0 消费者；§3.3/§8 显式承认 `item_age_s` 的
> monotonic 口径变化；T0 的 `attempt` 复用 producer 现有 `failure_counts`；T1 增加
> 第 0 步（metrics_io 映射派生）。

父 program：[Continuous three-stage pipeline](SPRINT_continuous_three_stage_pipeline_program.md)

前置：[Online metrics IO contract](../done/SPRINT_online_metrics_io_contract.md)。这是 program 的首个
continuous 实施 sprint，但新增 telemetry columns 前先建立单一 CSV row schema，避免继续同步
header、row dict 与 format order 三份定义。

## 0. 结论先行

先把 generation、reward、trainer-ready 和 training 四个边界的 identity、ownership 与时间区间
钉住，再改并发。当前 metrics 能看到 collect 总阶段和 ready queue，但不足以回答“哪一段在等
谁、队列里是哪一个 batch、GPU 空闲是背压还是饥饿”。本 sprint 只补 contract、telemetry 和
baseline，不改变生产调度。

eval 不属于该 contract。真实 reward、group 完整性、policy parity 和 optimizer health 足以
验收；离线 checkpoint comparison 可以作为补充证据，但不进入事件循环。

## 1. Root cause / current behavior

`ContinuousRolloutItem` 只有 `group_key` 和 `rollout_policy_version`。单 finite batch 时足够，
但未来同时存在 current 和 lookahead batch 后，两个 batch 都会出现 `group_key=0..N-1`；没有
稳定 `batch_id` 就无法正确选择或诊断。

当前 producer state 记录全局 `submitted_count` / `completed_count` / `error_count`，ready queue
记录 items/bytes/age；缺少：

```text
batch_id
stage item id
generation queue wait / service interval
unscored ownership interval
reward queue wait / service interval
trainer-ready wait
per-stage retry / cancellation / discard reason
backpressure reason duration
```

`collect.generation_reward_overlap` 只统计一次 `collect_prompt_groups()` 调用内部的 interval。
continuous producer 每个 slot 只喂一个 prompt，且显式传 `RewardCollectionMode.BATCHED_SERIAL`
（`producer.py` `_collect_group`），单组调用内部没有第二个组可重叠——所以该指标在 continuous
路径上恒为 0，并发 slot 之间真实发生的 overlap 完全不被表达，不能用它证明四卡流水线。

另一个测量根因：consumer 把每个 item 的 `RolloutStats` `merge()` 进迭代 stats，而 merge 对
phase 是求和。串行 schedule 一次迭代只有一次 collect，求和等于 wall-clock；continuous 下多个
slot 并发时求和会系统性高估（4 个并发各 10s 的 `collect.engine_generate` 报 40s，实际 wall
是 10s）。新增 stage interval 若沿用该语义，T3 基线数字是错的——每个指标的归约语义必须按
§6 的表逐条定死。

## 2. Goal and ownership boundary

定义一个最小、可贯穿 stage 的 identity contract：

```text
batch_id
group_slot
policy_version
attempt
generation_request_id
reward_request_id
```

- `batch_id + group_slot` 是逻辑 work identity。
- `attempt` 只表达同一逻辑 work 的重试，不改变 sample seeds 或 group mapping。
- generation/reward request id 是外部调用 provenance，不替代逻辑 identity。
- policy version 在 batch admission 时冻结。

优先把 identity 放进现有 typed item/state；不建立松散 metadata 字典作为控制面。若某字段只有
日志消费者，必须在定义处标注 display/provenance-only；否则必须被 selection、validation、
runtime request 或错误分支消费。

Sprint 0 只保留一个 active batch（§8），`batch_id` 在本 sprint 内不会有 selection 分支消费者。
它的 Sprint-0 合法消费者是 T0 的架构测试（validation：禁止 trainer/reward 重新派生 identity）
与 metric row；字段定义处须注明「行为消费者自 Sprint 2 lookahead 起」，否则按 dead-field 规则
无法合入。

## 3. Correctness and resource invariants

1. 新 telemetry 为 observation-only；static schedule 的提交顺序、batch size 和结果逐位不变。
2. 指标不能从 mutable global reward cache 读取；每个 item 自己携带对应 timing。
3. interval 使用 monotonic clock；跨进程只传 duration 或各进程自己的时间线，不能直接比较
   未同步 wall-clock。现有 `ContinuousRolloutItem.completed_at`/`age_s` 是 wall clock
   （`types.py`），满足本条需换 monotonic；已在产的 `continuous.item_age_s` CSV 列口径随之
   变化，该变化列入 §8「改变」，不是悄悄的副作用。
4. stage counters 必须区分 logical item 和 attempt，避免 retry 被误报为吞吐。
5. GPU duty probe 是 one-shot validation artifact；结论写入 `docs/sprints/info/` 后删除 scratch
   CSV，不让 scratch output 进入 import graph。
6. 不读取或记录 prompt 文本、secret、完整 model output。

## 4. Implementation stages

### T0 — Identity contract

- 为 finite batch 建立 owner-assigned monotonic `batch_id`。
- 将 batch identity 贯穿 producer item、ready item、consumer selection 和 metric row。
- 为 request retry 建立 `attempt`：不新建机制，把 producer 现有 `failure_counts[slot]`
  （重试即 `pending_slots.append(slot)` 重新入队）提升为 item 上的字段，并锁定 seed/sample
  identity 不变。这同时满足 §5「retry 产生新 attempt 但沿用原 identity」。
- 添加 architecture test，禁止 trainer/reward 重新派生 batch identity；该测试就是
  `batch_id` 在 Sprint 0 的 validation consumer（见 §2）。

### T1 — Stage interval telemetry

- 第 0 步：先消除 `from_step_metrics` 里 11 行手写 `phases.get("continuous.*")` 映射——给
  `_csv_field` 增加 phase-key metadata，让映射从 field 定义派生（header/format 已由前置
  sprint 派生，这是剩下的另一半）。否则 §6 新列会把手工同步面翻倍。
- 之后通过 `OnlineMetricRow` 的单一 field/mapping 扩展稳定 CSV，不手改 header/format list。
- 记录 generation admission wait、service、receipt。
- 记录 reward admission wait、service、receipt。
- ready queue residence 与 trainer demand wait 不新增列：前者就是现有 `continuous.item_age_s`
  （§3.3 改 monotonic 后即 residence 的 max），后者就是现有 `continuous.queue_wait_s`
  （consumer 从发出需求到凑齐 iteration 的等待）。
- 记录 training replay、optimizer step、weight sync 的现有 phase，并在 update 级合并。
- 每种 backpressure reason 记录 count 和 duration，而不是只打印最近一次字符串。reason 集合
  从 `_admit()` 现有返回值派生（`no_active_prompt_batch` / `paused_for_weight_sync` /
  `inflight_full`），新增原因先加在 `_admit`，不另建第二套字符串。

### T2 — Queue/resource snapshot

- 输出 item count、estimated bytes、oldest age、distinct batches、distinct versions。
- 通过固定采样 probe 记录 GPU duty/power/memory，不把 `nvidia-smi` polling 塞进生产 hot loop。
- 记录 useful logical completions、attempts、retries、discarded/stale groups。

### T3 — Four-L4 baseline

用当前 robotics continuous recipe 跑可重复 baseline，至少覆盖：

```text
rollout generation active
reward backlog
trainer backward active
weight sync
current batch consumed and next batch installed
```

将 timeline、测量口径和结论写入一个 `docs/sprints/info/` 长期档案；scratch probe 文件按 one-shot
生命周期清理。

**T3 runbook**（在 4×L4 机器上执行）：

1. 启动采样探针（独立进程，不进生产 hot loop）：
   `python -m vrl.scripts.perf.gpu_duty_probe --output outputs/gpu_duty.csv --interval-s 0.5`
2. 用 `configs/.../online_grpo_robotics_physics_4x_l4_continuous.yaml` 跑当前 recipe 若干个
   optimizer update（≥30，覆盖至少 3 次 weight sync）。
3. 结束探针（Ctrl-C 或 --duration-s），从 metrics CSV 取
   `continuous_backpressure_*_s/_count`（累计快照，相邻行差分得 per-update 值）、
   `continuous_generation_service_s`、`continuous_reward_service_s`、`continuous_queue_wait_s`
   与 GPU duty 序列对齐（探针为独立进程 monotonic 时间线，只能对 duration 对齐，见 §3.3）。
4. 用 backpressure 差分回答 §7 的问题：GPU 1/2 idle 归因于 `inflight_full` /
   `no_pending_slots`（batch 耗尽等 trainer）/ `paused_for_weight_sync` 中哪一段。
5. 结论 + timeline 写入 `docs/sprints/info/`，删除 gpu_duty.csv。

## 4.1 实施记录（2026-08-22）

T0/T1/T2 已落地；实现中做了如下与草案不同或草案未定的决策：

- **`group_key` 更名 `group_slot`**：与 §2 的 contract 词汇一致，生产 + 测试全量统一。
- **request id 缓做**：`generation_request_id` 已存在于 `TrajectoryBatch.request_id`
  （`vrl/trajectory/types.py`），随 batch 走——item 上再放一份是重复构造点（data twin），
  不加；`reward_request_id` 停在 reward service client 内部无出口，等 Sprint 1 reward pump
  拥有 request 边界时再引入。
- **`attempt` 复用 `failure_counts`**：`failure_counts[slot] + 1` 在 enqueue 时定格为 item
  字段；重试后 batch_id/group_slot 不变（测试钉住）。本 sprint 内标注 display/provenance-only
  并以 `continuous.max_attempt` gauge 导出（item 是唯一载体，batch 切换后 failure_counts 即
  重置，事后不可推导）；行为消费者是 Sprint 1 reward pump 的 retry identity。
- **backpressure 第四个 reason**：`no_pending_slots`——batch 全部生成完、无 in-flight、等
  trainer 消费并安装下一批的饥饿态，这正是 GPU 1/2 idle 的主要归因。duration 按 tick 累计
  （长阻塞在发生中即可见），以 **cumulative gauge** 导出（对齐 producer_submitted 惯例），
  per-update 值离线差分；owner reset 时随 producer 归零。
- **service gauge 的取数来源**：`continuous.generation_service_s` / `reward_service_s` 直接
  取每 item 已有的 `collect.generation_wall` / `collect.reward_wall`（gauge=max 视角），不新增
  测量；busy 总量继续由这两个 phase（求和视角）承担，即 §6 表中 `_total_s` 的角色。
- **`groups_discarded` 缓做**：当前 finite batch 的所有 discard 路径都是 raise（terminal），
  该 counter 没有非零生产者，按 dead-field 规则等 Sprint 1 的 discard 语义出现时再加。
- **边界校验收敛（2026-08-22 追加）**：范围校验从 producer/consumer 构造函数的散点收敛为
  vLLM/SGLang 形状——`ContinuousRolloutConfig.__post_init__` 是唯一的用户边界校验点
  （config-key 错误域，补齐 bytes/wait/poll 三条），`_ActivePromptBatch.__post_init__` 拥有
  per-call 输入形状（prompts 非空、group_size≥1），机制层改为接收整个
  `ContinuousRolloutSettings` 载体并信任它（producer/consumer 签名从散装 int 改为
  `settings=`）。Settings 仅保留 max_stale 的调度路由检查（"零陈旧度用 strict_on_policy"
  是路由建议，不是范围）。queue 的容器自卫检查保留（独立公共容器边界）。
- **CSV 新增 12 列**：batch_id、active_batches、generation/reward 的 wait/service 4 列、
  3 组 backpressure `_s`/`_count`。schema 变化由 `prepare_metrics_csv` 的 header 校验兜底
  （旧文件续写会 fail loud）。

## 5. Failure, cancellation and recovery semantics

- telemetry 写入失败不能悄悄改变调度；必要字段无法构造时 fail fast 于 admission 前。
- producer retry 必须产生新 attempt，但沿用原 identity。
- cancellation 的最后状态只写一次；不能同时计为 success 和 cancelled。
- metrics CSV partial append、process restart 和 counter reset 的语义要有测试。

## 6. Telemetry contract

每个指标同时定名字和归约语义（`RolloutStats`：phase 求和、counter 求和、gauge merge 取
max）。并发 slot 的 interval 求和不等于 wall-clock（§1），所以 wait/service 一律走 gauge
（per-update 最坏观测 = 关键路径视角）；T3 duty ratio 需要总忙时的地方允许配一个
`*_total_s` counter，但总忙时永远不得当 wall-clock 呈现。

| 指标 | 类型 | 说明 |
|---|---|---|
| `continuous.generation_queue_wait_s` | gauge | admission 等待，最坏观测 |
| `continuous.generation_service_s` | gauge | 可配 `_total_s` counter 供 duty |
| `continuous.reward_queue_wait_s` | gauge | |
| `continuous.reward_service_s` | gauge | 可配 `_total_s` counter 供 duty |
| `continuous.backpressure_inflight_s` | counter | duration 累计 |
| `continuous.backpressure_unscored_bytes_s` | counter | |
| `continuous.backpressure_ready_bytes_s` | counter | |
| `continuous.logical_groups_completed` | counter | 只计 logical item，不计 attempt |
| `continuous.group_attempts` | counter | |
| `continuous.group_retries` | counter | |
| `continuous.groups_discarded` | counter | |
| `continuous.active_batches` | gauge | 队列内 distinct batch 数快照 |

从最初草案删去的三个名字，均为已有指标的别名（不同时保留一组别名）：

- `trainer_wait_for_ready_s` → 复用现有 `continuous.queue_wait_s`（consumer 需求等待）。
- `ready_residence_s` → 复用现有 `continuous.item_age_s`（monotonic 化后即 residence max）。
- `active_policy_versions` → 复用 `queue.stats()` 现有 `ready_versions`；T2 的 distinct
  versions 同理。

## 7. Verification and acceptance gates

- [x] fake-clock 单测证明 interval 无负数（`test_item_ages_never_go_negative_under_a_skewed_clock`）、
  retry 不重复 logical completion（retry 测试断言 completed=2/submitted=3/attempt 定格）。
- [x] producer/queue/consumer contract tests 证明 batch identity 和 policy version 不丢失
  （mixed-batch 拒绝、identity gauges、queue `ready_batches`；`test_identity.py` 架构测试
  钉住单一构造点与 trainer/reward 不得改写 identity）。
- [x] current static path 的 batches、group ids、reward tensors 和 selection 顺序与修改前相同
  （strict 路径零改动；`tests/rollouts/orchestration/` 203 项全绿）。
- [x] 两个并发 collect 的 timing 不互相覆盖（per-item stats 所有权 + 聚合测试，原有）。
- [ ] baseline 能解释 GPU 1/2 idle 时是 `inflight_full`、`no_pending_slots`、reward backlog、
  `paused_for_weight_sync` 或 failure 中的哪一种（机制已落地并有测试；数值验收待 T3 硬件）。
- `git diff --check`、相关 continuous/reward stats tests 全绿。

## 8. What changes / what stays

### 改变

- typed identity 和 stage timing。
- owner/producer/queue metrics 的 batch-aware 聚合。
- `continuous.item_age_s` 改为 monotonic 口径（列名不变，数值语义变化；见 §3.3）。
- `from_step_metrics` 的 continuous 列映射改为从 field metadata 派生（T1 第 0 步）。
- 长期保留一份四卡 baseline 信息文档。

### 保持

- 一个 active finite batch。
- `collect_prompt_groups()` 仍是 producer 的复合 task。
- max inflight、ready queue 和 weight-sync 行为。
- trainer batch、reward 和 optimizer 数值。

## 9. Non-goals

- 不新增 unscored queue。
- 不改变 prompt lookahead 数量。
- 不实现 reward batching 或 adaptive controller。
- 不做 eval worker、eval queue 或 inline checkpoint evaluation。
- 不以 GPU utilization 为唯一成功标准。

## 10. Definition of Done

- [x] Identity contract 有 production consumer 和测试（consumer 同批校验 + 架构测试）。
- [x] 所有 stage wait/service 和 backpressure reason 可在一次 update 中重建
  （wait/service 为 per-update gauge；backpressure 为 cumulative gauge，相邻行差分）。
- [ ] 四卡 baseline 已记录到 `docs/sprints/info/`（待 4×L4 硬件，runbook 见 T3）。
- [x] static path 数值与顺序无变化。
- [x] scratch artifacts 已按生命周期清理（探针 smoke CSV 已删；gpu_duty_probe.py 为长期
  perf 工具）。
- [ ] Sprint 1 的 unscored queue 可以直接复用这些 identity 和 metrics（待 Sprint 1 验证）。

## 11. References

- `vrl/rollouts/orchestration/continuous/types.py`
- `vrl/rollouts/orchestration/continuous/producer.py`
- `vrl/rollouts/orchestration/continuous/queue.py`
- `vrl/rollouts/orchestration/continuous/consumer.py`
- `vrl/rollouts/orchestration/continuous/owner.py`
- `vrl/rollouts/orchestration/prompt_collection.py`
- `vrl/trainers/online/trainer.py`
- `vrl/rollouts/stats.py`
- `vrl/trainers/metrics_io.py`
- `vrl/scripts/perf/gpu_duty_probe.py`
- `tests/rollouts/orchestration/continuous/test_identity.py`
- `tests/rollouts/orchestration/continuous/test_contracts.py`
- `tests/rollouts/orchestration/continuous/test_schedule.py`
- `tests/rollouts/orchestration/continuous/test_owner.py`
