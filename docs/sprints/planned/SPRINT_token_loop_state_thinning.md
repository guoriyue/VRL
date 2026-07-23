# SPRINT：Token loop state thinning

状态：**planned（2026-07-22）**。

父 program：[Argument and state ownership](SPRINT_argument_and_state_ownership_program.md)

前置：无行为依赖；建议在
[Trajectory and rollout single source](SPRINT_trajectory_rollout_single_source.md) 的 token reader迁移
稳定后执行，减少同一测试 fixture冲突。

## 0. 结论先行

当前 token loop每次只为一个 `GenerationRequest` 创建 scheduler。所有 active sequence 的
family、task、tokenizer、dtype、max token都相同，却仍为尚未实现的 cross-request scheduler保存
per-sequence key和IDs。

本 Sprint 保留 request内真实 microbatch调度：

```text
position queueing
row gather/scatter
KV/cache lane ownership
scheduler_batch_size
family runner hook
```

删除没有 producer能使其变化的 grouping state。未来 cross-request scheduler必须在 parked Sprint
启动时重新引入它真正需要的 typed state，不能让当前 production对象为未来测试预付复杂度。

## 1. T0 — Pin current production envelope

加 architecture tests证明：

- `TokenAutoregressiveLoop` 一次只接收一个 request；
- family/task/tokenizer/dtype/max token来自 loop/request级配置；
- scheduler只在该 request的 rows之间按 position形成 microbatch；
- 没有 registry、Ray mailbox或continuous producer向同一个 scheduler追加第二个 request。

若以上任一条件已因后续更新失效，停止本 Sprint并转到
`docs/sprints/parked/SPRINT_cross_request_step_scheduler.md`；不能在错误前提下删除真实异质状态。

## 2. T1 — `ActiveSequence` 只保留调度状态

### 删除

- `request_id`、`sample_id`：只有测试/property reader；
- `family`、`task`、`tokenizer_key`、`dtype`、`max_new_tokens` per-sequence副本；
- `remaining_tokens` test-only property；
- free-form `metadata`。

### 派生/显式化

- `finished` 从 `position >= loop.max_new_tokens` 派生；
- metadata唯一活键 `"row_index"` 变成 typed `row_index`；
- family/task/tokenizer/dtype/max tokens放在 loop级 immutable config。

### 保留

- `position`：真实控制同 position batching；
- row index：真实控制 cache row与gather/scatter；
- sequence持有的生成 token/cache引用；
- explicit cancellation/error state（若有行为 reader）。

constructor不再能给同一 loop里的 sequence传矛盾 family/dtype/max token。

## 3. T2 — 删除没有异质 producer 的 key/batch字段

删除：

- `TokenAutoregressiveSequenceKey`；
- scheduler按 family/task/tokenizer/dtype/max tokens分组的 branch；
- `TokenBatch.key`；
- `TokenBatch.request_ids/sample_ids`。

`TokenBatch` 可以保留为最薄的 `TokenBatch(sequences)`，因为 runner boundary接收具名 batch提高
cross-family grepability；若所有 runner都直接接收 list且没有验证行为，再删除 wrapper。不能仅为了
少一行预先决定内联。

`TokenScheduler` **KEEP**：即使只有一个 request，它仍真实执行 position-aware bounded
microbatch和cache row routing，不是零价值薄层。

测试：

- `scheduler_batch_size < rows`；
- `scheduler_batch_size >= rows`；
- mixed position只选择当前可运行 rows；
- gather/scatter row顺序；
- invalid/duplicate row index失败。

删除“同一池不同 family/max token分组”测试；它只证明当前无 production producer的 future feature。

## 4. T3 — Step/result payload收缩

### `TokenAutoregressiveResult`

当前 wrapper只有 `finalized`，所有 production caller立即 `.finalized`。loop直接返回 finalized
sequence/output list，删除 one-field result和解包。

### `TokenStepBatch.positions`

envelope已经证明一个 step batch共享 scalar `position`，runner随后又检查 positions set。删除 list，
只保留 scalar：

- envelope负责 mixed-position不进入同 batch；
- runner接收一个 position；
- false test在 envelope边界构造 mixed rows并失败。

### 保留

- `TokenLoopInit`；
- `TokenAutoregressiveEnvelope.cache_lanes/row_lanes`；
- runner input/output protocol；
- `call_with_supported_kwargs` family hook compatibility adapter。

这些字段表达真实 KV/cache ownership或framework适配，不能跟 future-only key一起删。

## 5. T4 — Dotted/indirect reference cleanup

删除前后搜索：

- class/function symbol；
- string class name；
- registry dotted path；
- `__all__`；
- test parametrization；
- `getattr(sequence, ...)`；
- serialization/asdict；
- trace/debug format。

如果 request/sample ID只在日志中需要，不把它放回每个 sequence；从 enclosing request和row
临时格式化。若出现无法派生的跨进程 provenance consumer，再在 protocol层显式保留并标注，而不是
塞进 scheduler key。

## 6. What changes / what stays

### 改变

- loop级不变量不再复制到每个 sequence；
- row index从 metadata变 typed；
- finished改派生；
- future-only grouping key、IDs和one-field wrapper删除。

### 保持

- ActiveSequence具名 state；
- TokenScheduler；
- position queueing；
- bounded microbatch；
- cache/KV lane ownership；
- runner/family hook统一形状；
- request/sample identity在 generation protocol层。

## 7. ALL_CAPS / thin functions

没有大型业务 ALL_CAPS 表需要迁移。保留：

- token/model architecture constants；
- special token IDs；
- runner protocol names；
- family hook adapters。

不要把 scheduler、envelope或runner薄文件合并到 loop；它们分别提供调度、cache ownership和
family framework边界。

## 8. Non-goals

- 不实现 cross-request batching。
- 不更改 sampling算法、seed、top-k/top-p、temperature。
- 不更改 cache布局或CUDA kernel。
- 不把 token与denoise scheduler合并。
- 不为未来 feature保留 dead state。
- 不运行 GPU/Ray。

## 9. Acceptance gates

- token composition/loop/scheduler/runner全部 CPU fake tests；
- scheduler batch小于、等于、大于 rows；
- mixed position、duplicate row、finished rows false cases；
- output token/sample order与修改前逐项相同；
- registry/generation request integration；
- `ruff` touched files、`git diff --check`。

## 10. Definition of Done

- [ ] per-sequence只剩能在同 request内变化的状态。
- [ ] row index有 typed field，无 metadata string key。
- [ ] scheduler无 dead heterogeneous grouping branch。
- [ ] step/result payload无重复 position或one-field wrapper。
- [ ] request内 output顺序、token和cache routing不变。
- [ ] parked cross-request Sprint的触发条件与本 Sprint无冲突。

## 11. References

- `vrl/generation/composition/token_autoregressive/token_loop.py`
- `vrl/generation/steps/token/protocol.py`
- `vrl/generation/bindings/token_autoregressive/executor.py`
- `vrl/models/families/emu3/runner.py`
- `vrl/models/families/glm_image/runner.py`
- `vrl/models/families/janus_pro/runner.py`
- `vrl/models/families/llamagen/runner.py`
- `vrl/models/families/nextstep_1/runner.py`
- `tests/generation/composition/token_autoregressive/`
- `docs/sprints/parked/SPRINT_cross_request_step_scheduler.md`
- `docs/sprints/done/SPRINT_ar_step_result_dead_fields_cleanup.md`
