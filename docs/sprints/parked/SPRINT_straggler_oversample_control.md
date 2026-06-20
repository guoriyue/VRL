# SPRINT: continuous rollout straggler control（over-sample / first-completed-wins / abort-tail）（parked）

状态：**parked（2026-06-20，profiling-gate 实测未通过）**。profiling-gate 现已用真机 run 数据评估，结论是
**现在做属于投机、gate 不通过**，故 park。证据（`docs/runs/`）：
- **没有任何 run 跑在 T6 适用的 regime**：被 profile 的 run 都是 `mode: strict_on_policy`（`sd3_5_ocr_grpo_crossnode_profile` / `cosmos_predict25_nft_kling_480p33f_rbs16_20260620` 的 resolved_config）；后者 continuous_* 列全 0（`ready_groups=0`、`weight_sync_barrier_mode=0` draining）。T6 是 **continuous + staleness-tolerant** 的特性，当前无此 run。
- **旗舰 NFT 视频 run 物理上不适用 T6**：DiffusionNFT `tolerates_off_policy_staleness=False`，被 soundness 闸强制 `max_stale=0`（`build_rollout_schedule` raise），只能 on-policy 串行；README 实测 ~13.5 min/group × 16 ≈ 3.6 h/epoch，逐组串行。
- **慢尾尚未 instrument**：per-group generate/reward 时长在 `consumer.py:174` 被 merge 成 iteration 总和，`phase_events.jsonl` 只记训练侧阶段——「同一步内最快 vs 最慢 group 的差」这个 gate 核心信号当前**量不出来**。
- **更大杠杆在别处**：GPU 占用 rollout 27.9% / trainer 42.6%、0↔100 反相位 = 零重叠，指向 [[SPRINT_async_rollout_train_overlap]]（已 parked 于 ≥2 卡）的真 rollout/train overlap，比慢尾尾巴大得多。

**解封条件**：出现一个真正的 GRPO 系（staleness-tolerant）continuous run，且其 per-group 尾巴（先建 T6.3 telemetry 量出占比）确实拖低吞吐。在那之前不动 T6.1/T6.2。

原始计划（slime T6，2026-06-18 从 [[SPRINT_slime_overlap_strategy]] 拆出，T1–T4 已 `done/`；profiling-gated：只有当慢尾确实拖低连续 rollout 吞吐时才做）：

## 0. 来历

[[SPRINT_slime_overlap_strategy]]（`done/`）做完后，T5（独立 reward stage）按 profiling gate 暂不做、T6 完全未动。本 sprint 只装 T6。当前消费端固定等 `min_groups` 个同策略组、无 surplus 准入；producer 的 `cancel()` 只是协作式关停，不是尾部中止。

## 1. 现状（surface）

- `vrl/rollouts/orchestration/continuous/consumer.py:130-149`：消费端固定等 `min_groups` 个同 policy-version 组，没有"多采 + 取先完成"的余量准入。
- producer 的 `cancel()`（`producer.py`）是 shutdown 协作取消，不是"够数即中止慢尾"。
- 全仓 grep `wasted_group | first_completed | over_sample | abort_tail | surplus` 无相关命中——T6 零基建（`vrl/ray/actor_pool.py:132` 的 `FIRST_COMPLETED` 是无关的 actor-pool 等待）。

## 2. Work items

- **T6.1 over-sample + first-completed-wins**：每轮多发 N 个 group，凑齐 `min_groups` 个**先完成**的即推进，不等慢尾。
- **T6.2 abort-tail**：够数后主动取消仍在跑的剩余 group（区别于 shutdown 取消）。
- **T6.3 telemetry**：`wasted_groups` / `wasted_samples`（被中止/丢弃的量）写进 metrics.csv，作为 over-sample 余量调参的依据。

## 3. 风险 / gate

- **正确性**：被中止的 partial group 绝不能进训练；over-sample 的多余样本要么用满 group、要么干净丢弃（与 GRPO group 语义对齐）。
- **与 staleness 交互**：多采的 group 可能跨 policy-version，需沿用现有 `policy_version captured at submission` 机制（`continuous/types.py`）。
- **gate**：profiling-gated。先用现有 per-item stats 量出慢尾占比，确认值得做再动手。

## 相关
- [[SPRINT_slime_overlap_strategy]]（`done/`，父 sprint，T1–T4 已落地）
- `docs/sprints/reading/slime.md`（slime async rollout 经验来源）
