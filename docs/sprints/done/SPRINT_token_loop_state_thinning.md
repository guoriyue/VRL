# SPRINT：Token loop state thinning

状态：**done（2026-07-22）**。

父 program：[Argument and state ownership](SPRINT_argument_and_state_ownership_program.md)

前置：
[Trajectory and rollout single source](SPRINT_trajectory_rollout_single_source.md) 的 token reader迁移
已完成。

## 0. 结论先行

当前 token loop每次只服务一个 request。审计确认没有异质 producer后，production已经删除
`TokenScheduler`、`ActiveSequence`、`ARSequenceKey`、`TokenBatch` 与
`TokenAutoregressiveResult`，将单调用者概念拆分合回 `TokenAutoregressiveLoop.run()`：

```python
for position in range(init.step_count):
    for start in range(0, init.row_count, batch_size):
        step_batch = envelope.build_step_batch(...)
```

这不是删除真实调度能力。request内仍保留 position-major执行、bounded row microbatch、
`TokenAutoregressiveEnvelope.row_lanes` 的 gather/scatter cache ownership，以及统一 family hook
协议。未来 cross-request scheduler必须在出现真实多请求 producer时重新引入它实际消费的 typed
admission/key/cache state，不能复活当前已证明无 producer的字段。

## 1. T0 — Pin current production envelope

加 architecture tests证明：

- `TokenAutoregressiveLoop` 一次只接收一个 request；
- family/task/tokenizer/dtype/max token来自 loop/request级配置；
- loop只在该 request的 rows之间按 position形成 bounded microbatch；
- 没有 registry、Ray mailbox或continuous producer把第二个 request追加到同一次 loop执行。

若以上任一条件已因后续更新失效，停止本 Sprint并转到
`docs/sprints/parked/SPRINT_cross_request_step_scheduler.md`；不能在错误前提下删除真实异质状态。

## 2. T1 — 删除 per-sequence future-only state

`ActiveSequence` 的 request/sample IDs、family/task/tokenizer/dtype/max-token镜像、
`remaining_tokens`、`finished`与 metadata都没有异质 production producer。随着 scheduler合回
唯一 caller，该 wrapper整体删除，而不是保留一个只剩 row index的空壳。

位置由外层 `range(init.step_count)`拥有；row identity由当前 batch的显式
`TokenStepBatch.row_indices`拥有；生成 token/cache由具名 `row_lanes`拥有。constructor不再能构造
同一 request内互相矛盾的 sequence key或 max-token状态。

## 3. T2 — 删除没有异质 producer 的 key/batch字段

删除：

- `TokenAutoregressiveSequenceKey` / `ARSequenceKey`；
- scheduler按 family/task/tokenizer/dtype/max tokens分组的 dead branch；
- `TokenBatch`及其 key/IDs；
- `TokenScheduler`本身。

读取函数体后确认，scheduler只有 loop一个 caller；异质 grouping输入没有 producer，剩余逻辑只是
固定 `position × row chunk` 的两层循环。它属于 dead-code form 3
“single-caller concept split”，合并后决定可以在一个函数中从上到下阅读。真实 cache routing没有
内联到 family：它仍由 `TokenAutoregressiveEnvelope`与 `TokenStepBatch`协议承担。

测试：

- `scheduler_batch_size < rows`；
- `scheduler_batch_size >= rows`；
- 每个 position覆盖全部 rows且顺序稳定；
- gather/scatter row顺序；
- invalid/duplicate row index失败。

删除“同一池不同 family/max token分组”测试；它只证明当前无 production producer的 future feature。

## 4. T3 — Step/result payload收缩

### `TokenAutoregressiveResult`

迁移前 wrapper只有 `finalized`，所有 production caller立即 `.finalized`。loop直接返回 finalized
sequence/output list，删除 one-field result和解包。

### `TokenStepBatch.positions`

迁移前 envelope已经证明一个 step batch共享 scalar `position`，runner随后又检查 positions set。删除 list，
只保留 scalar：

- envelope负责 mixed-position不进入同 batch；
- runner接收一个 position；
- `TokenStepBatch`在协议边界拒绝负 position；scalar shape使 mixed-position状态不可构造。

### 保留

- `TokenLoopInit`；
- `TokenAutoregressiveEnvelope.row_lanes`；
- runner input/output protocol；
- `call_with_supported_kwargs` family hook compatibility adapter。

仓库没有单独的 `cache_lanes`镜像；cache/KV状态通过具名 `row_lanes`表达。这些字段表达真实
ownership或framework适配，不能跟 future-only key一起删。

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

## 6. 实施结果与审计判定

| Suspect | 判定 | 落地结果 |
|---|---|---|
| per-sequence family/task/tokenizer/dtype/max-token/IDs | **REMOVE** | request级不变量不再复制 |
| `finished/remaining_tokens` | **REMOVE/DERIVE** | position由 loop控制；没有 stored完成状态 |
| `ARSequenceKey`与 heterogeneous grouping branch | **REMOVE** | 无 production producer |
| `TokenScheduler` | **REMOVE** | 与唯一 caller合并，保留同样的 position-major bounded row行为 |
| `TokenBatch`、one-field result、positions list | **REMOVE** | runner协议直接使用 `TokenStepBatch` scalar position与 row indices |
| `TokenAutoregressiveEnvelope.row_lanes` | **KEEP** | 真实 cache/KV row ownership与 gather/scatter边界 |
| `call_with_supported_kwargs` | **KEEP** | family hook兼容 adapter，不是装饰性 helper |

CPU closure gate：

```text
55 passed
```

覆盖 batch bound小于/等于/大于 rows、逐 position顺序、空/重复/负数/越界 row、错误 hook result与
row-lane gather/scatter；未启动 Ray 或 GPU。

### 改变

- loop级不变量不再复制到每个 sequence；
- row index从 metadata变 typed；
- finished stored state由外层 position loop消除；
- future-only grouping key、IDs和one-field wrapper删除。

### 保持

- `TokenAutoregressiveLoop` composition边界；
- 直接 position-major bounded microbatch；
- `TokenAutoregressiveEnvelope.row_lanes` cache/KV ownership；
- `TokenLoopInit`、`TokenStepBatch`、`TokenStepOutput` runner协议；
- runner/family hook统一形状；
- request/sample identity在 generation protocol层。

## 7. ALL_CAPS / thin functions

没有大型业务 ALL_CAPS 表需要迁移。保留：

- token/model architecture constants；
- special token IDs；
- runner protocol names；
- family hook adapters。

保留 `token_loop.py`、token step protocol与 family binding的分层：它们分别提供 composition、
model-facing协议和 framework adapter。已删除的 scheduler文件没有独立 protocol、public API或异质
producer，不为跨 family外形而保留。

## 8. Non-goals

- 不实现 cross-request batching。
- 不更改 sampling算法、seed、top-k/top-p、temperature。
- 不更改 cache布局或CUDA kernel。
- 不把 token与denoise scheduler合并。
- 不为未来 feature保留 dead state。
- 不运行 GPU/Ray。

## 9. Acceptance gates

- token composition/loop/binding/runner全部 CPU fake tests；
- scheduler batch小于、等于、大于 rows；
- empty/negative/duplicate/out-of-range row与错误 hook result false cases；
- output token/sample order与修改前逐项相同；
- registry/generation request integration；
- `ruff` touched files、`git diff --check`。

## 10. Definition of Done

- [x] per-sequence future-only wrapper与不变量镜像已删除。
- [x] row index由 `TokenStepBatch.row_indices`显式拥有，无 metadata string key。
- [x] dead heterogeneous grouping branch与单调用者 scheduler已删除。
- [x] step/result payload无重复 position或one-field wrapper。
- [x] request内 output顺序、token和cache routing不变。
- [x] parked cross-request Sprint要求从真实多请求 producer重新设计 admission/cache ownership。

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
