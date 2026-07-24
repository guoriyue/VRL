# SPRINT：Online metrics IO contract

状态：**done（2026-07-22）**。

父 program：[Argument and state ownership](SPRINT_argument_and_state_ownership_program.md)

前置：无。现有
[Continuous stage contracts and baseline](../planned/SPRINT_continuous_stage_contracts_and_baseline.md)
在新增 telemetry columns前依赖本 Sprint。

## 0. 结论先行

`OnlineRecipeRun` 当前把同一约 40 列 CSV schema手写三次：

```text
prepare_metrics_csv(): header order
write_metric_row():    row dict mapping
write_metric_row():    format/order list
```

这不是应删除的 IO层，而是一个缺少具名 payload的真实跨进程协议。新增 `OnlineMetricRow`，让
column name、order、format和value mapping只有一个 owner；`TrainStepMetrics` 的 nested runtime
结构保持不变。

## 1. 正确的 dataclass 位置

新增：

```text
vrl/trainers/metrics_io.py
    OnlineMetricRow
    build_online_metric_row(...)
    online_metric_columns(...)
    format_online_metric_row(...)
```

这个薄文件 **KEEP**，因为 metrics CSV同时被：

- online trainer recipe写入；
- supervisor health gate读取；
- resume/truncate逻辑解析；
- continuous telemetry扩展；
-离线分析工具消费。

它是稳定 IO protocol，不是为了减少 `online.py` LOC而拆的 helper。

不要把它放进：

- `vrl/algorithms/types.py`：CSV不是 algorithm result；
- `vrl/scripts/common/online.py`：schema不应由一个 workflow method私有拥有；
- `vrl/config/schema.py`：它不是用户 config；
- `vrl/utils/`：它是 trainer domain protocol，不是泛用 utility。

## 2. T0 — 定义固定 row 与动态 extension

`OnlineMetricRow` 保存固定列的已物化 scalar。每个 dataclass field metadata声明：

```text
csv_name
format
```

若 field name就是稳定 CSV name，`csv_name` 从 field name派生，不重复写。format只在需要非默认
格式时声明。

dynamic reward component columns不伪装成 dataclass fields：

```text
r_<component_name>
```

它们作为明确尾部 mapping，列顺序由 run开始时冻结的 `component_names` 决定。

构造时验证：

- component name非空且不会产生重复列；
- fixed与dynamic列不冲突；
-所有 value finite policy保持现有语义；
- epoch/integer-like fields不能被 float format悄悄改变。

不建立手写 `_ONLINE_METRIC_FIELDS` ALL_CAPS list；column list从 dataclass fields派生。

## 3. T1 — 从 `TrainStepMetrics` 物化一次

`build_online_metric_row(epoch, metrics, component_names)` 是唯一 mapping：

- top-level loss/reward/advantage fields；
- nested `update`；
- nested `initial_replay`；
- nested `logprob_mismatch`；
- `phase_times` 中 continuous字段；
- dynamic reward component means。

strict mode没有 continuous phases时继续写稳定的 0，保持现有 external schema。missing reward
component继续写 NaN，除非现有 reader contract要求其他值。

保留 nested `TrainStepMetrics`：

- `PolicyUpdateStats` 是 objective update；
- `InitialReplayStats` 是 pre-optimizer parity snapshot；
- `LogprobMismatchStats` 是 replay mismatch；
- flatten只发生在 IO boundary。

display-only metric fields继续在 runtime type定义处注明；“不控制训练”不是删除 public telemetry的
理由。

## 4. T2 — Header、format、resume 共用 schema

- header从 `OnlineMetricRow` field order + frozen component columns生成；
- format逐 field metadata执行；
- row serialization按同一 order；
- `prepare_metrics_csv` 接收 column contract，不再接收手拼整行 header；
- resume定位的 `epoch` 从同一 schema确认存在；
- existing CSV header不匹配时 fail fast，不能按位置写入错误列。

`OnlineRecipeRun.prepare_metrics_csv/write_metric_row` **KEEP** 作为 run controller facade，但只委托
protocol module，不再维护 schema。

## 5. T3 — Supervisor 与 continuous 接入

`_REQUIRED_HEALTH_METRICS` / `_CONTINUOUS_HEALTH_METRICS` **KEEP**：它们是 health policy选择的
小型 required subset，不是完整 CSV schema副本。

增加启动期断言/测试：

- required subset全部存在于 `OnlineMetricRow` columns；
- typo或已删除列在测试中失败；
- supervisor仍只依赖它需要的列。

Continuous stage后续新增 column时：

1. 在 `OnlineMetricRow` 定义字段；
2. 在唯一 builder映射 phase/source；
3. health需要时才加入 required subset；
4. 不再编辑 header string和format list。

## 6. Tests

### True paths

- fixed row header与serialized value逐列一一对应；
- strict mode continuous列为0；
- continuous phase mapping；
- reward components有/无；
- resume/truncate正常；
- supervisor读取完整 row；
-新 field fixture自动进入 header与serialization。

### False paths

- duplicate/empty component name；
- fixed/dynamic name collision；
- header reorder/missing列；
- required health subset typo；
- row value数与header不一致；
- malformed epoch/resume cursor。

用 temporary directory和pure file IO；不启动 trainer、Ray或GPU。

## 7. ALL_CAPS / thin functions

### 改变

- 不新增完整 metrics column ALL_CAPS表；
- 删除 workflow中的大型 header literal和format order list。

### 保持

- health required subset常量：真实 consumer policy；
- checkpoint/metrics filename；
- `prepare_metrics_csv`：共享 resume/truncate IO boundary；
- `OnlineRecipeRun` methods：controller facade；
- nested training metric dataclasses。

## 8. Non-goals

- 不改任何 metric定义或数值。
- 不改 CSV column name/order，除非现有三份定义已不一致且测试证明 bug。
- 不改为数据库/Parquet/TensorBoard。
- 不把 dynamic reward columns做成运行中可变 schema。
- 不与 continuous scheduling实现合并。

## 9. Acceptance gates

- byte-for-byte header与代表性 row snapshot；
- strict/continuous/reward component/resume/supervisor CPU tests；
-现有 metrics consumer fixtures；
- `ruff` touched files；
- `git diff --check`；
-无 Ray/GPU。

### 实施记录

`9c2344c2` 新增 `OnlineMetricRow` 与唯一 row builder；fixed columns、顺序和格式从 dataclass fields
及 metadata 派生，dynamic reward component columns 在 run 开始时冻结。header 创建、row 写入、
resume/truncate 与 supervisor required-subset validation 共用同一 column contract。

保留了 `TrainStepMetrics` nested runtime 结构、health required subset、metrics filename、
`prepare_metrics_csv` IO 边界和 `OnlineRecipeRun` controller facade。没有新增完整 columns
ALL_CAPS 副本，也没有改变 metric 数值、列名或列顺序。验证只使用 temporary file 与 CPU tests；
没有启动 trainer、Ray 或 GPU。包含本 Sprint 改动面的 program 累计 CPU gate 为
`1703 passed, 23 deselected`；deselect 仅来自显式非 CPU lane 与缺失 vendored source 的两个
digest 用例。

## 10. Definition of Done

- [x] fixed CSV schema只存在于 `OnlineMetricRow`。
- [x] header、order、format从同一 field source派生。
- [x] dynamic component columns在 run开始时冻结。
- [x] supervisor required subset被 schema验证。
- [x] continuous Sprint新增 metric只需改一个 row定义和一个 mapping。
- [x]现有 CSV names/order/value保持。

## 11. References

- `vrl/scripts/common/online.py`
- `vrl/algorithms/types.py`
- `vrl/trainers/checkpointing.py`
- `vrl/scripts/supervise.py`
- `tests/scripts/test_supervise.py`
- `docs/sprints/planned/SPRINT_continuous_stage_contracts_and_baseline.md`
