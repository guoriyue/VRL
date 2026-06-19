# SPRINT: continuous rollout straggler control（over-sample / first-completed-wins / abort-tail）(planned)

状态：**planned（2026-06-18 从 [[SPRINT_slime_overlap_strategy]] 拆出）**。slime 对齐的 T1–T4 已落地、那个 sprint 已归档 `done/`；这里收它唯一完全未建的 T6——straggler 控制。profiling-gated：只有当慢尾确实拖低连续 rollout 吞吐时才做。

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
