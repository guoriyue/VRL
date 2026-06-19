# SPRINT: microbatch pipeline overlap —— retired scope guard

状态：**retired / do not implement（2026-06-17；2026-06-18 迁至 parked/——废弃决策记录、非待办）**。

本文件保留为防污染说明：**mini/microbatch 只表达同步的内存切片与梯度累积，不再承担 async 调度语义**。真正的 async 只指 rollout 生产侧和 trainer 消费侧的 wall-clock overlap，归 `SPRINT_continuous_scheduler_redesign.md`、`SPRINT_shadow_model_weight_sync.md` 和 parked 的 `SPRINT_async_rollout_train_overlap.md` 管。

---

## 0. 核心决定

不要把 `microbatch_size` 升级成 async pipeline item。

当前 sync streaming path 已经是正确基线：

```python
for microbatch in microbatches:
    batch = await trainer.collect_training_batch(microbatch)
    try:
        await trainer.backward_on_training_batch(batch, total_groups=rollout_batch_size)
    finally:
        del batch

await trainer.finish_optimizer_update(...)
```

语义是：

```text
collect microbatch 0 -> backward microbatch 0 -> release
collect microbatch 1 -> backward microbatch 1 -> release
...
optimizer.step once after the full target batch
```

这已经覆盖 paper-shaped target batch 在有限内存上的需求。它解决的是：

- host RAM 不再一次持有整个 target batch；
- GPU replay/backward 继续由 `sample_batch_size × timestep × rollout_microbatch` 控制；
- optimizer step / EMA / weight sync 仍然只在完整 target batch 后发生一次。

这不需要 async。

---

## 1. 为什么退掉 microbatch async

microbatch async 会把两个问题混在一起：

1. **mini/microbatch correctness**：target batch 怎么切、loss scale 怎么算、什么时候 optimizer step。
2. **rollout/train scheduling**：rollout actor 和 trainer 能不能在不同 GPU/进程上重叠工作。

这两个问题的风险完全不同。把 microbatch 作为 async item 会制造额外复杂度：

- `collect_training_batch(prompts=...)` 当前会写 trainer 状态；并发 prefetch 前必须重构采集 API。
- prefetch 会至少多持有一个 pending microbatch，直接增加 host RAM 峰值。
- `global_std=true` 时，streaming 已经是 per-microbatch std；把 item 做得更异步会让这个差异更难读。
- 单 GPU/colocated 场景下通常没有真 wall-clock overlap，容易得到复杂代码但收益很小。
- DiffusionNFT 的真正 async 风险在 stale rollout，不在 microbatch 内部。

所以这里不再推进“microbatch pipeline overlap”实现。

---

## 2. 保留的同步不变量

这些仍然属于 sync mini/microbatch 主线，继续由已完成 sprint 维护：

- `rollout.rollout_batch_size` 是 optimizer target prompt conditions。
- `rollout.microbatch_size` 是每次同步 collect/backward/release 的 prompt 条件数。
- `actor.gradient_accumulation_steps` 是派生/兼容视图，不是另一个手填 batch target。
- `rollout.sample_batch_size` 控制同 prompt 内 sample chunk，生成和 replay/backward 共用。
- 每个 prompt group 必须完整包含 `n_samples_per_prompt` 个样本，不能拆散 GRPO/NFT group。
- `finish_optimizer_update()` 只在完整 target batch 后调用一次。
- reward / advantage / phase metrics 聚合后仍然一行代表一个 optimizer update，不是一行一个 microbatch。

同步 mini/microbatch 的 source of truth：

- `docs/sprints/done/SPRINT_streaming_rollout_accumulation.md`
- `docs/sprints/done/SPRINT_memory_budgeted_microbatch.md`

---

## 3. 明确不做

以下内容从 async 计划中移除：

- 不在 `_run_streaming_optimizer_update` 里做 bounded microbatch prefetch。
- 不新增 `_PendingMicrobatch` / `RolloutMicrobatch` 作为跨模块 async 协议。
- 不新增 `drain_for_microbatch()` 或把 continuous consumer 边界降到 microbatch。
- 不把 `microbatch_size` 当成 rollout/train async scheduler 的 item size。
- 不为 microbatch prefetch 增加 `prefetch_ready_depth` / `collect_task_inflight` 等长期指标。
- 不为了 prefetch 去重构 `collect_training_batch()`；如果以后需要重构，必须由真正 rollout/train async sprint 证明收益后再做。

这些是非目标，不是 TODO。

---

## 4. 真正 async 的归属

只保留 rollout/train async 这条线：

```text
rollout producer / reward / ready queue   ||   trainer consumes previous ready batch
optimizer step
weight sync / version bump
next rollout generation continues under explicit staleness rules
```

对应 sprint：

- `docs/sprints/planned/SPRINT_continuous_scheduler_redesign.md`
  - 统一 rollout producer / in-flight / ready queue / staleness / admission。
- `docs/sprints/planned/SPRINT_shadow_model_weight_sync.md`
  - 去掉全 drain barrier 的安全前提：shadow-model / request-boundary weight swap。
- `docs/sprints/parked/SPRINT_async_rollout_train_overlap.md`
  - DiffusionNFT 专门裁决：真 overlap 需要多 GPU，stale rollout 没有理论安全版本，只能实测。
- `docs/sprints/planned/SPRINT_slime_overlap_strategy.md`
  - 参考 slime 的完整 rollout future 边界：ready queue 只能放已经 reward-scored、trainer-ready 的 batch。

---

## 5. 验收口径

以后看到“async”时，只按下面口径验收：

- 是否有独立 rollout owner / Ray actor / GPU 在 trainer 训练时继续生成？
- 是否有 ready queue 能跨 trainer step 保留已经打分的 rollout batch？
- 是否有明确 policy version / staleness / weight sync 边界？
- 是否证明 wall-clock 上 rollout 与 train 有重叠？

不要用“microbatch prefetch 成功”作为 async 验收。microbatch 是同步内存切片，不是 async 调度边界。

---

## 关键引用

- 同步 microbatch 基线：`vrl/scripts/common/online.py:_run_streaming_optimizer_update`
- 同步 backward/step 边界：`vrl/trainers/online/trainer.py:begin_optimizer_update`、`backward_on_training_batch`、`finish_optimizer_update`
- 已完成同步 sprint：`docs/sprints/done/SPRINT_streaming_rollout_accumulation.md`、`docs/sprints/done/SPRINT_memory_budgeted_microbatch.md`
- 真 async sprint：`docs/sprints/planned/SPRINT_continuous_scheduler_redesign.md`、`docs/sprints/planned/SPRINT_shadow_model_weight_sync.md`、`docs/sprints/parked/SPRINT_async_rollout_train_overlap.md`
