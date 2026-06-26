# SPRINT: 消除 parse→runtime 结构的 pass-through 字段重复

状态：**planned（2026-06-25）**。exemplar 已修（见 §3），其余按本文逐条处理。范围是一个**架构卫生**问题，不是 correctness bug——但它会孕育 correctness bug（silent no-op knob），所以值得收口。

## 0. 一句话

仓里存在一类重复:**A 结构(parse/config 层)的字段被逐个手抄进 B 结构(runtime/resolved 层)**。每加一个旋钮要同时改 A、B、和它们之间的拷贝点;漏抄拷贝点 → 旋钮在运行时变成 no-op(用户设了却没效果),而且**不报错**。修法不是合并两层(两层往往是正当边界),而是**让 B 引用/内嵌 A**,把纯透传字段从"手抄"变成"只读一处"。

## 1. 问题定义与判据

一个 (A → B) 拷贝是**该修的 pass-through**,当且仅当:

```
B 的多数字段是 A 字段的逐字拷贝(B.x = a.x),没有 rename / 解析 / 计算;
A、B 是不同类型(不是子类、不是 1:1 wrapper);
加一个 A 的字段必须同时碰 B 的定义 + 拷贝点,否则静默失效。
```

它**不该修**(是正当的 curated 边界),当:

```
B 的多数字段是 A 的转换/解析/新增(如 per-chunk 解析窗口、resolved length);
拷贝只是少数字段、且 B 还有大量独立来源的字段;
或这是跨 family 的 provenance/identity 戳(刻意一致,grep 锚点)。
```

判据来自 AGENTS.md 架构卫生原则:**"不要手抄一个复制了 typed 结构的常量/字段——源类型加字段时它会静默 rot"**。pass-through 拷贝就是这条的"字段版"。

## 2. 修复模式(B 做的)

把 B 里那批纯透传字段删掉,改成**持有一个 `a: A` 引用**,运行时读 `b.a.<knob>`。B 只保留:它**独有**的字段(身份、resolved 值、跨组拉来的旁支配置)。加旋钮 → 只改 A(+wire schema)。

代价:消费端从 `b.x` 变成 `b.a.x`(多一层),且直接构造 B 的测试要嵌套构造 A。**只有当透传字段够多、转换字段够少时才值得**——否则保留显式拷贝(可审计)更好。

## 3. exemplar(已修,模板)

`DiffusionSDEParams`(`generation/diffusion/layout.py`)→ `DiffusionDenoiseConfig`(`generation/diffusion/executor.py`),`build_denoise_config` 里曾逐字抄 6 个字段(`same_latent / return_kl / noise_level / sde_type / return_prev_sample_mean / cache_ref_noise_pred`)。

**已改为** `DiffusionDenoiseConfig` 持有 `sde: DiffusionSDEParams`,运行时读 `config.sde.<knob>`;`DiffusionDenoiseConfig` 只保留 chunk 身份(`prompt_index/sample_start/sample_count/seed`)、**per-chunk 解析后**的 `sde_window`(不能放进请求级 sde)、和旁支 `denoise_mode/teacache`。`build_denoise_config` 不再有 6 行拷贝。测试改为构造 `sde=DiffusionSDEParams(...)`。验证:`tests/generation/diffusion/` 32 passed。

这是本 sprint 其余条目的模板。

## 4. 待处理清单(worst-first,经联网外的全仓审计 + 逐条 file:line 核对)

| # | A(源) | B(目标) | 拷贝点 | 纯抄 / 转换 | 处置 |
|---|---|---|---|---|---|
| **1** | `LogprobMismatchStats` `algorithms/logprob_mismatch.py:28` | `TrainStepMetrics` `algorithms/types.py:8` | `GRPO.compute_loss` `grpo/continuous.py:190-195` | **6 纯 / 0 转换** | **FIX** |
| **2** | `AxisCapability` `generation/capabilities.py:25` | `ResolvedAxis` `generation/execution/planner.py:26` | `ResolvedAxis.from_capability:36-49` | **4 纯 / 1 新增**(`length`) | **FIX(轻量)** |
| 3 | `ExecutionStageCapability` `generation/capabilities.py:60` | `ExecutionStage` `generation/execution/planner.py:52` | `EnginePlanner._execution_stages:291-303` | 5 纯 / 7 新增+转换 | **多半保留**(见下) |

### #1 — `LogprobMismatchStats` → `TrainStepMetrics`(最该修,与 exemplar 同型)

`continuous.py:190-195` 把 6 个字段(`logprob_abs_diff_mean/max, ratio_abs_dev_mean/max, mismatch_kl, mismatch_k3_kl`)逐字抄进 `TrainStepMetrics`,零转换。且 `LogprobMismatchStats.finite` 被**静默丢弃**——已核实。加第 7 个 mismatch 指标要改两个结构 + 拷贝行,正是 exemplar 的失败模式。

**注意 B 是 22 字段的聚合器**,不能整体内嵌 `LogprobMismatchStats`(语义不对)。修法选其一:
- 给 `LogprobMismatchStats` 加 `to_metrics_kwargs()`(返回 dict),`compute_loss` 用 `TrainStepMetrics(..., **mismatch.to_metrics_kwargs())`;或
- `TrainStepMetrics` 加 `from_mismatch(mismatch, **rest)` 工厂,把 6 字段映射收口到一处。
两种都把"6 行散抄"变成"一处映射",新增指标只碰映射点。顺带决定 `finite` 是该进 metrics 还是确认丢弃。

### #2 — `AxisCapability` → `ResolvedAxis`(轻量修)

`ResolvedAxis.from_capability` 抄 `name/kind/batchable/chunkable`,只有 `length` 是 per-request resolved。5 个字段里 4 个纯抄。

**修法**:`ResolvedAxis` 持有 `capability: AxisCapability` + `length`,消费端读 `resolved.capability.kind` / `resolved.length`。或者更保守:保留 `ResolvedAxis` 但加一行断言 `set(fields(ResolvedAxis)) - {"length"} == 字段来自 capability`,防 rot。前者彻底,后者改动更小。实现时按 `ResolvedAxis` 的消费面大小定(消费点少就内嵌,多就先加断言)。

### #3 — `ExecutionStageCapability` → `ExecutionStage`(建议保留,记录理由)

`_execution_stages` 只在 5 个 `ExecutionStage(...)` 构造点里的**1 个**从 capability 抄 5 字段(`name/segment/axis/cache_read/cache_write`);另 4 个构造点(`plan/cache_read/cache_write/forward_chunk`)是**凭空合成**的。B 还加了 7 个 planner-runtime 字段(`stage_id/axis_index/prompt_index/sample_start/sample_count/metadata`)并转换 `profiler_name`。

**判断:保留。** B 是名副其实的"运行时计划单元",拷贝是少数派、转换/新增是多数派——这是正当 curated 边界,不是 exemplar 那种纯透传。只把这 5 字段重叠记为"已知、可接受"。强行内嵌会逼那 4 个合成构造点也塞一个假 capability,反而更脏。

## 5. 明确不动的(防 over-cleanup)

AGENTS.md:cleanup 不要外溢到正当 convention。以下**已核查、确认不是本 pattern,保留**:

- **`(request_id, family, task)` 三元 provenance 戳**:`GenerationRequest` → `TrajectoryBatch` 及 AR runtime payload(`trajectory/builders.py:97,158,360,514`;`models/ar/*/runtime.py` ~10 处;`trajectory/ops.py:36,72,104`)。纯抄 3 字段,但这是**刻意一致的跨 family 身份戳**(grep 锚点、不期望转换),不是 config-vs-resolved 边界。收掉它会破坏代码库看重的跨 family 一致性。
- **`RuntimeBundle` 构造**(8 个 diffusion family runtime.py):从 model 对象**收集** 3 个独立属性进 bundle,不是 A 结构→B 结构拷贝。handle-gathering wrapper,保留。
- **`ray/resources.py`** 的 `(num_gpus, num_workers, gpus_per_worker)` 三元(`:665,687,831,851,875`):小 GPU 资源三元,周围字段是计算/归一化的,非高重叠透传。保留。
- **`rollouts/`** 的 `RolloutBatchBuildContext / RolloutIteration / ContinuousRolloutItem`:各 ≤1 纯透传,其余是 `cfg_get` 提取 / `int()` 包装 / 计算(时间戳、nbytes)/ 条件默认。真正的 resolve/materialize 边界,保留。
- kling/videoscore2 reward 里的 5-6 字段连写:是 `kv(...)`/`logger.info` 日志构造,不是 dataclass。保留。

## 6. Phase plan

```
P0  exemplar(DiffusionSDEParams→DiffusionDenoiseConfig)            # 已完成,作模板
P1  #1 LogprobMismatchStats→TrainStepMetrics:to_metrics_kwargs/from_mismatch + 决定 finite
P2  #2 AxisCapability→ResolvedAxis:内嵌 capability 或加 rot 断言(按消费面定)
--  #3 不立项,只在 ExecutionStage 旁加一行注释记"5 字段重叠已知、可接受"
```

每条统一验收:

```
加一个源字段只碰源结构(+ wire schema),不碰目标结构;
目标行为不变(现有测试全绿 + 该结构的构造/消费测试覆盖新形状);
若引入"读一处"的间接,消费点改动一次到位,不留半套。
```

## 7. 非目标

- **不合并 parse 层与 runtime 层**——两层多数是正当边界(per-chunk 解析、resolved length)。只去掉"逐字抄",不抹掉边界。
- **不动 §5 列出的 provenance 戳 / handle-gathering / 真 resolve 边界**——那是 over-cleanup,会破坏跨 family 一致性。
- 不为了省几行把有转换/合成的结构(#3)硬塞进源结构。
- 不在没有消费面证据时盲目内嵌(#2 给了"内嵌 vs 断言"两条路按消费面定)。

## 8. 参考代码

- exemplar(已修):`vrl/generation/diffusion/executor.py`(`DiffusionDenoiseConfig` / `build_denoise_config`)、`vrl/generation/diffusion/layout.py`(`DiffusionSDEParams`)。
- #1:`vrl/algorithms/logprob_mismatch.py:28`(`LogprobMismatchStats`,注意 `finite`)、`vrl/algorithms/types.py:8`(`TrainStepMetrics`)、`vrl/algorithms/grpo/continuous.py:190-195`(拷贝点)。
- #2:`vrl/generation/capabilities.py:25`(`AxisCapability`)、`vrl/generation/execution/planner.py:26,36-49`(`ResolvedAxis` / `from_capability`)。
- #3:`vrl/generation/capabilities.py:60`、`vrl/generation/execution/planner.py:52,291-303`。

> 实现任一条前,按 evidence-first 重新 grep 一遍 file:line(本文行号取自 2026-06-25 审计,代码移动后以实际为准)。
