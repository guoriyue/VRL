# SPRINT: Distributed fixed eval（Cosmos-RL style）

状态：DONE。fixed eval 现在是一个全局 rank-sharded phase：每个 rank 跑 disjoint eval prompt
shard（按 global index 切 + 定 seed），stats 经一次 all-reduce SUM 聚合，rank0 单独写
`eval_metrics.csv`，所有 rank 在 `strategy.barrier()` 汇合后再进下一个 training step。单卡输出与旧实现
一致（P3）。

落地内容（`vrl/scripts/common/online.py`）：
- 纯 helper：`_iter_fixed_eval_shard`（global-index 分片 + max_prompts 截断）、
  `_fixed_eval_group_seed`（seed 只由 global index 决定，与 world_size 无关）。
- stats 类型：`_FixedEvalLocalStats`（reward sum/sumsq/count + component sums/counts），
  `_FixedEvalResult` 保留为写 CSV 的全局结果。
- `_run_local_fixed_eval`（跑本 rank shard，返回 local stats）→ `_merge_fixed_eval_stats`
  （float64 打包 + all-reduce SUM；NCCL 上 GPU tensor、gloo 直接 CPU；component 用 sum/count，
  不做 mean-of-means）→ `_run_distributed_fixed_eval`（编排，全 rank 调用）。
- 调用点：baseline + 周期 eval 去掉 `is_primary` gate，改为全 rank 跑 eval、rank0 写、尾部 barrier。

测试：`tests/trainers/online/test_fixed_eval_distributed.py`（分片/聚合单测 + 真 2-rank gloo
all-reduce smoke）、`tests/trainers/test_rank_ownership.py`（重写为 train/eval/checkpoint 三段
ownership：全 rank 干活、rank0 单写）、`tests/trainers/online/test_reward_update_flow.py` 的
单卡 fixed-eval 回归改用 `_run_distributed_fixed_eval`（world_size=1，输出不变）。

> 历史背景（已不再成立的现状描述保留在下文）：当前 FSDP/DDP 多卡训练路径里，训练 rollout 已经按 rank
> 切 prompt shard，但 fixed eval 曾是 rank0-only driver。这个 sprint 目标是把 fixed eval 的生成/打分
> workload 分摊到多 rank / 多 rollout worker，同时保持 fixed seed grid、单 writer、全局指标语义。

## 0. 核心结论

不要把 `is_primary` 从 eval 上简单删掉，让所有 rank 各自写 eval。正确形状是：

```text
all ranks:
  enter eval point
  run disjoint eval prompt shard through local collector/runtime
  produce local reward sufficient statistics
  all-reduce local stats into global stats

rank0 only:
  append eval_metrics.csv
  log fixed eval summary
```

这和 Cosmos-RL 的 validation 思路一致：**rollout side 分摊生成/奖励计算，controller/primary 只聚合和
写日志**。区别是 VRL 当前没有 Cosmos 的 dispatcher/controller 进程，所以第一版不要为了 eval 新增
controller；先复用现有 torch distributed ranks 和每 rank 已有的 colocated Ray runtime。

## 1. 当前 VRL 现状

fixed eval 现在只在 primary rank 上跑：

```python
if eval_enabled and (epoch + 1) % int(eval_cfg.freq) == 0 and is_primary:
    await _fixed_eval_and_log(epoch)
```

`_run_fixed_eval()` 不碰 trainer，不 backward，不 optimizer step，只复用 collector 路径：

```python
for prompt_index, item in enumerate(examples):
    seed = int(base_seed) + prompt_index * int(samples_per_prompt)
    unscored.append(
        await collector.collect_unscored(
            [prompt],
            **_fixed_eval_collect_kwargs(item, group_size=samples_per_prompt, seed=seed),
        ),
    )
batches = await collector.score_rollouts(unscored)
```

这个实现是稳定的，但多卡下有两个问题：

1. eval prompt loop 是 rank0-only，训练 ranks 不分摊 eval prompt。
2. 即使底层 Ray executor 能把一个 request 的 sample chunks 分给多个 rollout workers，prompt 级别仍是
   rank0 顺序 `await`，eval set 大时会成为全局暂停点。

当前训练 rollout 已经是数据并行的：

```python
idx = sample_prompt_indices(..., rollout_batch_size=rank_batch * training_context.world_size)
shard = idx[training_context.rank * rank_batch : (training_context.rank + 1) * rank_batch]
example_batch = [examples[i] for i in shard]
metrics = await trainer.step(example_batch)
```

fixed eval 应该采用同一类数据并行思路，而不是继续 rank0-only。

## 2. Cosmos-RL 对照

Cosmos-RL 的 validation 不是 policy rank 自己各写一份 eval。它通过 controller/rollout worker
分工：

```python
payloads, is_end = self.api_client.get_next_prompt(
    batch_size * self.parallel_dims.mesh["dp"].size(), **kwargs
)
```

rollout replica 的 rank0 拉一批 validation prompts 后，在 rollout DP ranks 内 scatter：

```python
rank_prompts = prompts[rank::ranks_to_scatter]
dist.scatter_object_list(..., group=self.parallel_dims.mesh["dp"].get_group())
```

每个 rollout worker 生成并计算 validation reward 后，post 回 controller：

```python
response = ValidationReportRequest(
    src_replica_name=self.replica_name,
    validation_step=current_step,
    payloads=payloads,
    is_end=True,
)
self.api_client.post_validation_report(response)
```

controller 聚合所有 validation rollouts，等到完整 eval set * n_generation 都回来才 log：

```python
validation_finished = (
    n_items_of_this_step
    == (self.data_fetcher.val_datasize or len(self.data_fetcher.val_dataloader))
    * self.config.validation.n_generation
)
```

VRL 应该借鉴的是这个架构形状：

```text
distributed workers do eval generation/reward
one owner aggregates and writes metrics
validation dataset is consumed exactly once per eval step
```

不应该照搬的是：

```text
新增 HTTP controller / dispatcher
新增 rollout-side validation report protocol
把 fixed eval 混进训练 rollout queue
```

这些是 Cosmos 架构已经存在的基础设施；VRL 现在没有这个边界，硬搬会让 eval 改动过大。

## 3. 目标设计

### 3.1 Rank-sharded fixed eval

新增分片 helper，输入完整 eval examples 和 `DistributedTrainingContext`：

```text
all_examples = eval_examples[:max_prompts] if max_prompts > 0 else eval_examples
local_examples = [
  (global_eval_index, item)
  for global_eval_index, item in enumerate(all_examples)
  if global_eval_index % world_size == rank
]
```

必须用 `global_eval_index` 算 seed：

```text
group_seed = base_seed + global_eval_index * samples_per_prompt
```

不能用 shard-local index。否则 world_size 改变后，同一个 prompt 的 seed 会变，fixed eval 不再 fixed。

### 3.2 本地 eval runner

保留 `_fixed_eval_collect_kwargs()` 和 collector 路径，但让 runner 接收 `(global_index, item)`：

```text
for global_index, item in local_examples:
  seed = base_seed + global_index * samples_per_prompt
  collect_unscored([prompt], group_size=samples_per_prompt, seed=seed, ...)
score_rollouts(local_unscored)
return local sums/counts/component sums
```

这个函数不应该：

- 写 CSV；
- 修改 `trainer.prompts`；
- 调 `collect_training_batch()`；
- backward；
- optimizer step；
- weight sync；
- 读写 checkpoint。

### 3.3 全局聚合

每个 rank 输出 reward sufficient statistics：

```text
reward_sum
reward_sumsq
reward_count
```

多卡时 all-reduce SUM：

```text
global_mean = reward_sum / reward_count
global_var = reward_sumsq / reward_count - global_mean**2
global_std = sqrt(max(global_var, 0))
global_stderr = global_std / sqrt(reward_count)
```

这和现有 `_global_reward_stats()` 的思路一致，但 fixed eval 需要返回完整 `_FixedEvalResult`，并聚合
component means。

component 聚合不要平均 rank-local mean；要聚合 sum/count：

```text
local component stats:
  r_kling_sum
  r_kling_count

global component mean:
  global_sum / global_count
```

rank 本地 shard 为空时也必须参与 all-reduce，count=0。

### 3.4 Single-writer

所有 rank 都跑 eval 和聚合，但只有 rank0 写：

```python
result = await _run_distributed_fixed_eval(...)
if is_primary:
    run.write_eval_metric_row(...)
```

这个 gate 只应该包住写文件和 logger，不应该包住 eval runner 本身。

### 3.5 同步边界

fixed eval 前后要保证所有 ranks 在同一阶段：

```text
before baseline eval:
  all ranks enter eval together

after eval aggregation:
  strategy.barrier()
  all ranks continue into next training step
```

如果 eval helper 里的 all-reduce 已经覆盖了成功路径，尾部 barrier 仍建议保留，语义更清楚：
fixed eval 是一个 global phase，不是 rank0 的旁路任务。

## 4. 实现计划

### P0. 提取纯 helper

新增纯函数，不触碰 runtime：

```text
_iter_fixed_eval_shard(examples, max_prompts, rank, world_size)
_fixed_eval_group_seed(base_seed, global_index, samples_per_prompt)
_merge_fixed_eval_stats(local_stats, device)
```

验收：

- world_size=1 行为和旧实现 seed grid 一致。
- world_size=2/3 时每个 eval prompt 被恰好分给一个 rank。
- 同一个 prompt 的 seed 与 world_size 无关。

### P1. 分布式 eval stats 类型

新增内部 stats dataclass：

```text
_FixedEvalLocalStats
  reward_sum
  reward_sumsq
  reward_count
  component_sums
  component_counts
```

保留 `_FixedEvalResult` 作为写 CSV 的最终结果类型。

实现 all-reduce 时注意 backend：

```text
nccl: stats tensor 放到 training_context.device
gloo: CPU tensor 可以直接 reduce
```

这和 `_global_reward_stats()` 里处理 NCCL CPU tensor 的方式一致。

### P2. 替换 eval 调用点

把当前 rank0-only 调用：

```python
if eval_enabled and resume_checkpoint is None and is_primary:
    await _fixed_eval_and_log(-1)
```

改成：

```text
if eval_enabled and resume_checkpoint is None:
  result = await _run_distributed_fixed_eval(...)
  if is_primary:
    write/log
  strategy.barrier()
```

周期 eval 同理。checkpoint 逻辑不变；checkpoint 本来就所有 rank 调用、rank0 写文件。

### P3. 保持 single-process 行为

world_size=1 时必须保持旧输出：

- 相同 `eval_metrics.csv` schema；
- 相同 prompt order；
- 相同 seed grid；
- 相同 reward mean/std/stderr；
- 相同 component columns。

单卡不应该因为 distributed helper 引入额外分支复杂度或日志变化。

### P4. 后续优化：prompt-level Ray batching

当前 `_run_fixed_eval()` 是逐 prompt `await collect_unscored([prompt])`。rank-sharded eval 会减少 wall time，
但每个 rank 内仍是 prompt 顺序循环。

后续可选优化：

```text
把多个 plain-string eval prompts 合成一个 collect_unscored(prompts, group_size=...)
PromptExample / reference image/video 仍保持单 prompt request
```

这个优化要小心 seed：一个 request 内多 prompt 时，generation `build_sample_rows()` 的 flat index 会让
sample seed 变成 `base_seed + prompt_index * samples_per_prompt + sample_index`。如果多个 eval prompts
合进同一 request，就需要让 request base seed 对齐该 request 内第一条 prompt 的 global seed，且 request
内部 prompt order 不能改变。第一版先不做，避免 fixed eval 语义漂移。

## 5. 测试计划

### 5.1 纯 helper 单测

覆盖：

```text
examples=10, world_size=3
rank0 -> [0, 3, 6, 9]
rank1 -> [1, 4, 7]
rank2 -> [2, 5, 8]
```

断言：

- shards disjoint；
- union 覆盖全部 eval examples；
- `max_prompts` 生效；
- seed 只由 global index 决定。

### 5.2 聚合单测

构造两个 rank 的本地 stats：

```text
rank0 rewards: [1, 3]
rank1 rewards: [5]
```

期望：

```text
mean = 3
std(population) = sqrt(((1-3)^2 + (3-3)^2 + (5-3)^2) / 3)
stderr = std / sqrt(3)
```

component stats 使用 sum/count，不允许 mean-of-means。

### 5.3 distributed gloo smoke

用 `torch.multiprocessing` 启两 rank，fake collector 返回固定 rewards：

```text
rank0 shard: eval indices 0, 2
rank1 shard: eval indices 1, 3
```

断言：

- 两 rank 都进入 eval；
- all-reduce 后两 rank 得到相同 `_FixedEvalResult`；
- 只有 rank0 写 eval row；
- 没有训练 backward / optimizer 调用。

### 5.4 online loop ownership 回归

新增或更新 rank ownership 测试：

```text
training rollout:
  every rank owns local prompt shard

fixed eval:
  every rank runs local eval shard
  rank0 writes metrics only

checkpoint:
  every rank gathers
  rank0 writes only
```

这个测试要替换旧的“non-primary never touches Ray”假设；当前主训练路径已经不是 rank0-only Ray owner。

## 6. 风险和边界

### 6.1 Reward component 聚合

当前 `reward_fn.last_components` 是本地 reward function 状态。分布式 eval 后，rank0 不能只读自己的
`last_components`。必须显式把 component sums/counts 并入 all-reduce。

### 6.2 空 shard

当 `max_prompts < world_size` 时，部分 rank 没有 eval prompt。它们仍必须参与 stats all-reduce 和
barrier，不能提前 return。

### 6.3 Eval failure

如果某个 rank eval 抛异常，其他 rank 可能卡在 all-reduce/barrier。第一版不需要实现复杂 fault
tolerance，但要让异常在本 rank 明确打印，避免 silent hang。后续可以加 all-rank error flag。

### 6.4 与 FSDP collective 的关系

fixed eval 不跑 FSDP model forward/backward；它走 rollout collector/runtime。它的 distributed
collectives 只有 stats all-reduce/barrier，不能混进 trainer replay 的 FSDP collectives。

### 6.5 不做 Cosmos controller

本 sprint 不新增：

- validation HTTP API；
- rollout worker validation report protocol；
- central validation dataloader actor；
- async validation queue。

这些只有在 VRL 后续引入 Cosmos-style controller/dispatcher 时才值得。当前目标是消除 rank0 fixed eval
瓶颈，不重写运行时架构。

## 7. 验收标准

1. 单卡 fixed eval 输出与现有实现一致。
2. 多卡 fixed eval 中，每个 rank 都处理 disjoint eval prompt shard。
3. `eval_metrics.csv` 仍只由 rank0 写。
4. reward mean/std/stderr/component means 是全局 eval set 指标，不是 rank0 本地 shard。
5. eval seed grid 与 world_size 无关。
6. eval phase 结束后所有 ranks 一起进入下一次 training step，不产生 collective mismatch。

## 8. 参考路径

- `vrl/scripts/common/online.py`：当前 `_run_fixed_eval()`、`_fixed_eval_and_log()`、eval 调用点。
- `vrl/trainers/online/trainer.py`：`_global_reward_stats()` 的 distributed stats 聚合参考。
- `vrl/trainers/strategy.py`：`Strategy.barrier()`。
- `/home/mingfeiguo/Desktop/cosmos-rl/cosmos_rl/rollout/worker/rollout_control.py`：Cosmos validation prompt fetch / scatter / report。
- `/home/mingfeiguo/Desktop/cosmos-rl/cosmos_rl/dispatcher/data/data_fetcher.py`：Cosmos validation dataloader。
- `/home/mingfeiguo/Desktop/cosmos-rl/cosmos_rl/dispatcher/status.py`：Cosmos validation aggregation and logging。
