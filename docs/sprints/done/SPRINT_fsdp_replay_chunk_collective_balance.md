# SPRINT: FSDP/DDP replay chunk collective balance

状态：**done（2026-06-21）**。P1 replay execution-slot planner 已落地：
`vrl/trainers/online/trainer.py` 现在先按现有 prompt-group 语义生成真实 replay
chunks，再用 `all_reduce(MAX)` 对齐各 rank 的 execution slot 数；短 rank 用
`loss_weight=0` 的 dummy slot 补齐，所以 evaluator/backward collective 次数一致，
但 dummy 不进入梯度或训练指标分母。legacy `train_on_rollout_batch` 与 streaming
`backward_on_training_batch` 都走同一 planner。P2 跨 group compact/per-sample
weighted loss 未做，仍是非目标。

验证：

- `tests/trainers/online/test_skip_backward_agreement_distributed.py`：planner 单测、
  gloo 2-rank slot 对齐、trainer replay loop evaluate/backward 计数对齐。
- `tests/trainers/online/test_reward_update_flow.py::test_sample_batch_size_splits_training_replay_and_preserves_gradient`：
  单进程 sample chunk 梯度等价不回退。

原始问题来自 `feat/fsdp-online-orchestration` review：FSDP 多卡路径已经补了
“整 microbatch 是否一起跳过”的保护，但 trainer replay 内层仍会因为
`drop_zero_advantage + sample_batch_size` 让各 rank backward 次数不同，形成 latent
NCCL 死锁。

## 0. 先把三个“chunk / microbatch”分开

```text
rollout.microbatch_size
  切 prompt group 轴：一个 optimizer update 分几批 collect/train/release。

rollout.sample_batch_size
  切 sample 轴：一个 prompt group 或 replay slice 里，一次 evaluator
  forward/backward 最多跑多少 samples。

generation SampleChunkSchedule
  rollout 生成时的 sample-axis slicing；Ray 路径还会做 worker placement /
  OOM split。这不是本 sprint 的死锁根因。
```

当前 trainer replay chunk 是手动显存旋钮，不是自动调度器：

```python
chunk_size = int(sample_batch_size)
if chunk_size <= 0 or chunk_size >= batch_size:
    return [_TrainingSampleChunk(batch=batch, advantages=advantages, loss_weight=1.0)]

for start in range(0, batch_size, chunk_size):
    stop = min(start + chunk_size, batch_size)
```

含义：

```text
n_samples_per_prompt=8, sample_batch_size=1 -> 8 个 replay chunks
n_samples_per_prompt=8, sample_batch_size=2 -> 4 个 replay chunks
n_samples_per_prompt=8, sample_batch_size=8 -> 1 个 replay chunk
n_samples_per_prompt=8, sample_batch_size=0 -> 1 个 replay chunk（legacy full group）
```

现在没有根据显存自动判断“能切多小 / 能跑多大”。下限是 1 sample，上限是当前
filtered batch size。generation Ray executor 有 OOM 后二分重试；trainer replay/backward
没有 OOM 自适应，OOM 后只能用户手动调小 `rollout.sample_batch_size`。

## 1. 问题

FSDP/DDP backward 的 forward/backward 会触发 collective。多卡安全前提不是 batch
shape 完全一样，而是每个 rank 进入 collective 的次数和顺序完全一样。

当前 FSDP 分支只保护了“整个 microbatch 是否一起跳过”：

```python
if not _all_ranks_have_work(bool(batch.batches), self.device):
    return
```

但内层还有本地数据依赖循环：

```python
if cfg.drop_zero_advantage:
    mask = nonzero_advantage_mask(adv_b)
    if not bool(mask.any()):
        continue
    if not bool(mask.all()):
        b = select_batch(b, mask)
        adv_b = adv_b[mask.to(adv_b.device)]

for b, adv_b in zip(batch.batches, batch.advantages, strict=True):
    for sample_chunk in _training_sample_chunks(b, adv_b, sample_batch_size):
        for j in train_indices:
            signals = self.evaluator.evaluate(self.model, chunk_batch, j, ...)
            self._backward(loss)
```

死锁场景：

```text
world_size=2
microbatch_size=1
n_samples_per_prompt=8
sample_batch_size=1
drop_zero_advantage=true

rank0 当前 prompt group 存活 8 samples -> 8 * T 次 FSDP forward/backward collective
rank1 当前 prompt group 存活 3 samples -> 3 * T 次 FSDP forward/backward collective

rank0 第 4 个 replay chunk 进入 all-gather/reduce-scatter 时，
rank1 已经离开这段循环 -> NCCL 永久等待。
```

这不是 generation chunk schedule 的问题；它是 trainer replay chunk 和分布式 collective
之间的契约问题。

## 2. 为什么“直接 concat 非零样本”不是无脑正确

直觉方案是把多个非零 samples concat 成固定大小 chunk，再 backward。这个方向可以做，
但不能只 concat tensor，因为现有 loss 语义是“每个 prompt group 先按存活 samples 求均值，
再按 prompt group 累积到 optimizer update”。

当前 `_training_sample_chunks` 用 `loss_weight = chunk_len / filtered_group_size`
保持单组内切 chunk 后的梯度等价：

```text
group S 个存活样本，sample_batch_size=2
chunk loss 是 chunk 内 mean
乘 chunk_len / S 后，相当于还原 group mean
```

如果跨 prompt group compact/concat，就会丢掉每个 group 自己的 `1 / S_group`
归一化，除非新增 per-sample loss weight，并让 algorithm loss 支持带权 reduction。
否则存活样本多的 group 会被过度加权，行为不等价。

所以：

```text
可以 concat：同一个 group 内 compact 非零样本。
谨慎 concat：跨 group concat 必须先引入 per-sample loss weights。
不能做：all_reduce(MIN) 后只跑共同 chunk 数；这会直接丢弃某些 rank 的有效样本。
```

## 3. Padding 的真实代价

Padding 会多跑没训练信号的 replay forward/backward。它的价值不是贡献梯度，而是让所有
rank 发出相同数量的 collective。可以把它理解为“同步占位 backward”。

但 padding 必须满足两个条件：

1. dummy / zero-advantage rows 不进入有效 loss 分母。
2. dummy / zero-advantage rows 不贡献 KL / regularization 这类不乘 advantage 的项。

普通 GRPO 且 `kl_coef=0` 时，零 advantage 样本的 policy gradient 为 0。但代码里还有
`kl_coef > 0`、Flow-DPPO、GRPO-Guard、NFT 等路径；不能默认所有 padding row 自动梯度等价。
长期正确做法是显式携带 sample weight / valid mask，让 loss reduction 只看真实有效样本。

## 4. 推荐实现顺序

### P0. 立即 fail-fast，先防静默挂死

保守做法：在分布式训练路径上拒绝 `drop_zero_advantage=true` 的 replay sample
chunking，除非明确落在已分析过的出货安全子集。

```text
world_size > 1
strategy in {ddp, fsdp}
drop_zero_advantage = true
0 < sample_batch_size < n_samples_per_prompt
```

在当前 `microbatch_size=1` 的 FSDP 2x1 出货配置下，以下组合不会产生“同一非空
prompt group 内 replay chunk 次数不同”的内层死锁：

```text
drop_zero_advantage=false
sample_batch_size=0
sample_batch_size>=n_samples_per_prompt
```

理由：`sample_batch_size=0` 或 `>= n_samples_per_prompt` 时，每个 nonempty group 都是
1 个 replay chunk；当前 `_all_ranks_have_work` 已经把“某 rank 整组过滤为空、另一 rank
非空”的情况变成所有 rank 一起跳过，避免 collective 次数错配。

这不是完整解。如果 `microbatch_size>1`，`drop_zero_advantage` 还可能让各 rank 在同一个
microbatch 里保留不同数量的 nonempty prompt groups；即使每个 group 只有 1 个 chunk，
外层 group loop 次数仍可能不同。完整修复必须用 P1 的 execution-slot planner。

### P1. 分布式 replay chunk planner，保证执行槽数量一致

新增一个 trainer 侧 replay chunk planner，输出 execution slots，而不是直接让
`_training_sample_chunks` 决定本地 loop 次数。

最小正确形状：

```text
输入：当前 microbatch 的 per-group RolloutBatch + advantages
步骤：
  1. 保留原始 prompt-group 结构和 group normalization。
  2. 每个 group 内 compact 非零 advantage samples。
  3. 按 sample_batch_size 切真实 replay chunks。
  4. 每个 rank 统计本地真实 execution slot 数。
  5. all_reduce(MAX) 得到全局 max_slot_count。
  6. slot 少的 rank 补 dummy zero-weight slot 到 max_slot_count。
  7. 所有 rank 按相同 slot_count * train_indices 次数进入 evaluator/backward。
```

这个方案只补到全局最大 chunk 数，比“每个 group 永远 pad 回
`n_samples_per_prompt`”更省。但如果当前 `microbatch_size=1, sample_batch_size=1`，rank0
存活 8、rank1 存活 3，rank1 仍必须补 5 个 dummy slot；这是 collective balance 的最低成本。

### P2. 可选优化：跨 group compact，需要 per-sample weight

如果 P1 后仍觉得 padding 计算浪费太大，再做跨 group compact：

```text
真实样本按 per-sample weight 打平到 replay pool
按 sample_batch_size 重新装箱
dummy slot 只补到全局 max_slot_count
algorithm loss 使用 per-sample weights 做 weighted reduction
```

这一步改动算法契约，不应该混在 P0/P1 里。没有 per-sample weight 前，跨 group concat
会改变 group-level loss weighting。

## 5. 测试计划

1. 纯 planner 单测：
   - rank-local alive counts `[8]` vs `[3]`，`sample_batch_size=1`。
   - 两边 planner 都产生 8 个 execution slots。
   - rank1 后 5 个 slot 是 dummy / zero-weight。

2. 梯度等价单测：
   - 单进程 baseline：物理 drop zero advantage。
   - balanced planner：真实 slots + dummy slots。
   - 比较参数梯度一致；dummy 不改变 loss 分母。

3. 分布式 gloo/NCCL smoke：
   - 两 rank 都非空，但存活样本数不同。
   - fake evaluator 记录每 rank `evaluate/backward` 次数。
   - 两边次数一致，测试不会 hang。

4. Guard 测试：
   - `ddp/fsdp + drop_zero_advantage=true + sample_batch_size=1` 在未启用 P1 前 fail-fast。
   - `drop_zero_advantage=false`、`sample_batch_size=0`、`sample_batch_size>=n_samples_per_prompt`
     不报错。

## 6. 非目标

- 不改 generation `SampleChunkSchedule`。它只是 rollout/generation 的 sample-axis slicing，
  Ray placement / OOM split 也不解决 trainer replay collective 次数。
- 不把 `rollout.sample_batch_size` 伪装成自动显存调度器。当前它是用户给的手动上限；
  自动按字节准入属于 `SPRINT_memory_plan_full` 的 L2，不在本 sprint。
- 不用 `all_reduce(MIN)` 截断本地有效样本。这个方案不会死锁，但会静默丢训练信号。
- 不先做跨 group concat。没有 per-sample weighted loss 前，它会改变现有 group mean 语义。

## 7. 参考代码

- `vrl/trainers/online/trainer.py`
  - `_training_sample_chunks`
  - `collect_training_batch`
  - `backward_on_training_batch`
- `vrl/rollouts/batch/ops.py`
  - `nonzero_advantage_mask`
  - `pad_zero_advantage_mask`
- `vrl/trainers/core/types.py`
  - `TrainerConfig.sample_batch_size`
  - `TrainerConfig.microbatch_size`
- `vrl/generation/ray/executor.py`
  - generation-side OOM split only; trainer replay has no equivalent fallback.
- `docs/sprints/parked/SPRINT_multi_gpu_training.md`
  - FSDP2 orchestration context and existing `_all_ranks_have_work` lesson.
- `docs/sprints/planned/SPRINT_memory_plan_full.md`
  - future byte-admission work; related but separate from collective balance.
