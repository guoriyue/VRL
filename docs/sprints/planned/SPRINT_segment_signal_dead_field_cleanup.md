# SPRINT: SegmentSignal 死字段清理 + 派生结构体字段审计（planned）

状态：未开始（2026-06-20）。
范围：清理 `SegmentSignal` / `SignalRequest` 上**有人构造、无人按行为读取**的死字段（字段级，承接 [[SPRINT_trainer_rollout_dead_alias_cleanup]] #5 已落地的标识符级清理 —— 那次删的是三份重复的 `segment` 标识符，这次删的是字段本身）。附带把一轮派生/已解析结构体（`Resolved*` / `*Bundle` / `*Capability`）的字段扫描结果一并收口。**不**重构 evaluator→algorithm 契约，**不**删 `SegmentSignal` 结构体本身（经核实是活契约）。

> 本 sprint 由一次 dynamic workflow 审计产出（21 agents：5 路并行 map → 15 条死/冗余 claim 对抗验证 → 综合）。所有「死」判定都过了对抗复核，下文逐条标注是 confirmed-dead 还是 needs-design-intent。

## 0. Core Decision（先看这一段）

裁决一切「字段到底活没活」的，是 `AlgorithmAdapter.validate_inputs` 里这段反射：

```python
# vrl/algorithms/trajectory.py:54-56
available = sorted(
    f.name for f in fields(signal) if getattr(signal, f.name) is not None
)
```

`fields(signal)` 会枚举 `SegmentSignal` 的**所有**字段，但 `getattr(...) is not None` 过滤器把永远为 `None` 的字段直接滤掉。结合「只有 `GRPO` 在 `vrl/algorithms/grpo/continuous.py:47` 声明了 `required_signal_keys = ("log_prob", "old_log_prob")`，其余算法继承 `base.py:33` 的空元组」这一事实，得出三档判定：

1. **永远 `None` 的字段**（`entropy`）→ 被 `is not None` 滤掉，永远进不了 `available`，永远不影响 `missing`，永远不改任何报错。审计中「通过 `fields()` 反射存活」的翻案对它是**机械性错误**。→ **删**。
2. **永远非 `None` 但只进 `available` 诊断串、自身从不被任何 `required_signal_keys` 要求、也无按值读取**（`axes`、`aux`）→ 唯一读者是诊断/日志构造，按 AGENTS.md 派生结构体规则属 display-only 死字段。→ **默认删，需 owner 确认是否留作未来日志**。
3. **有真实控制流消费**（校验能 raise、或喂给 compute、或被某 `required_signal_keys` 要求）→ **留**。`axis` 因 `types.py:54` 的非空校验守卫落在这档。

`SegmentSignal` 结构体**整体保留**：`distribution` 字段在 `continuous.py:142` 驱动 `flow_matching` 分支选择（latent-space KL vs 普通 logprob KL），是真行为开关；`TrajectorySignalBatch` 是 evaluator→algorithm 的一等契约（`__post_init__` 校验 + `primary` 属性 + 多段 dict 被 `multisegment.py` 消费）。把它塌进 trajectory 层是契约重设计，不是清理 —— 列为非目标。

## 1. 现状实锤

结构体定义：`vrl/rollouts/evaluators/types.py:9-26`（`SegmentSignal`）、`:82-88`（`SignalRequest`）。

### 1.1 `SegmentSignal.entropy` + `SignalRequest.need_entropy` —— confirmed dead（成对）

`vrl/rollouts/evaluators/types.py:21`：

```python
entropy: Any | None = None
```

所有 evaluator 构造点**硬编码 `entropy=None`**：`vrl/rollouts/evaluators/ar/token_logprob.py:75`、`ar/continuous_token_logprob.py:64`、`ar/multi_segment_token_logprob.py:91`；builder 形参一路默认 `None` 穿透（`vrl/rollouts/evaluators/trajectory.py:38,57,83,141`）。

生产者侧开关 `vrl/rollouts/evaluators/types.py:87`：

```python
need_entropy: bool = False
```

`grep -rn "need_entropy"` 全仓库**仅命中这一行定义**：从不被读、从不被设为 `True`。git 历史显示它在 commit `4ee8095`（"Unify replay model contracts"）引入，但配套实现路径从未接上。

合证：`entropy` 永远 `None` → 被 `trajectory.py:55` 的 `is not None` 滤除 → 永不进 `available`；任何算法的 `required_signal_keys` 都不含它；零 `.entropy` 按值读取。它是一条**半成品信号链**：占位字段 + 占位开关，两端都没接。

### 1.2 `SegmentSignal.aux` —— 死，但写入了真元数据（需 owner 拍板删/标注）

`vrl/rollouts/evaluators/types.py:26`：

```python
aux: dict[str, Any] = field(default_factory=dict)
```

`vrl/rollouts/evaluators/ar/multi_segment_token_logprob.py:93-96` 填入了真实内容（`segment_name` / `segment_modality`）；`trajectory.py:146` 填空 dict。但 `grep` 证实 `.aux` / `["aux"]` / `getattr(...,"aux")` **零读取**：无算法在 `required_signal_keys` 列它，无测试、无 config、无序列化读它。

与 `entropy` 的区别：`aux` 被填了**有意义的元数据**，像是给未来 per-segment 日志预留的钩子，只是日志构造从未写出。按 AGENTS.md「只被尚不存在的日志构造读取 = 死」默认删；若团队确认要留作未来逐段日志，就地标注 `# display/provenance-only`（对齐 `ResolvedDistributedResources.visible_devices` 的处理，见 §3.2）。

### 1.3 `SegmentSignal.axes` —— display-only（构造时算了真值，下游只进诊断串）

`vrl/rollouts/evaluators/types.py:15`：

```python
axes: tuple[str, ...]
```

构造侧 `vrl/rollouts/evaluators/trajectory.py:135` 用 `_axes_from_value`（`trajectory.py:299-303` 真算：按 ndim + axis 推 tuple）填值。但下游：

- **无非空校验**（`types.py:54` 只校验 `axis`，不校验 `axes`）。
- **无按值读取**：`grep` 证实 `vrl/algorithms/` 内零 `signal.axes` / `segment.axes`。
- **唯一「读者」** 是 `trajectory.py:55` 的 `fields()` 反射 —— `axes` 永远非 `None`，故永远落进 `available` 列表，但 `available` 只在**别的** required key 缺失时拼进报错串 `Available signal keys: [...]`。`axes` 自身从不被任何 `required_signal_keys` 要求，进 `available` 不改任何控制流。

即：构造侧花了真实计算，消费侧只把它当诊断串的填充。属 display-only。

### 1.4 `SegmentSignal.axis` —— 留（有能 raise 的校验守卫）

`vrl/rollouts/evaluators/types.py:14` `axis: str`，构造侧 `trajectory.py:134` 经 `_axis_from_segment_or_signal`（`trajectory.py:289-296` 扫 `segment.tensors` 找 `old_log_prob` role 取轴名）填值。

与 `axes` 的关键差别：`axis` 在 batch 的 `__post_init__` **有真实校验守卫**，能 raise：

```python
# vrl/rollouts/evaluators/types.py:54
if not segment.axis:
    raise ValueError(f"trajectory signal {name!r} must have a non-empty axis")
```

该校验在所有 `TrajectorySignalBatch` 实例化路径执行（`trajectory.py:66`、`multi_segment_token_logprob.py:101`、`multisegment.py:96`）。按派生结构体规则「能 raise 的校验 = 真消费者」，`axis` 是活的 —— 但它是**契约不变量守卫**，其值从不参与 loss/grad 计算。保留，不动校验。

### 1.5 活字段清单（留，附消费点）

| 字段 | 定义 | 消费点 |
|---|---|---|
| `name` | `types.py:13` | dict key + 校验 `types.py:48-52`；`trajectory.py:67,70` |
| `distribution` | `types.py:16` | `continuous.py:142` `== "flow_matching"`；`batch_builder.py:97,99` |
| `log_prob` | `types.py:17` | `continuous.py:101,154,166`；`token.py:46` |
| `old_log_prob` | `types.py:18` | `continuous.py:99,101,154,170`；`token.py:47` |
| `mask` | `types.py:19` | `token.py:54`；shape 校验 `types.py:65-70` |
| `ref_log_prob` | `types.py:20` | `continuous.py:134,154`；`token.py:80,89` |
| `prev_sample_mean` | `types.py:22` | `continuous.py:143,147`（flow-matching KL） |
| `ref_prev_sample_mean` | `types.py:23` | `continuous.py:144,148` |
| `std_dev_t` | `types.py:24` | `continuous.py:149` |
| `dt` | `types.py:25` | `continuous.py:150`（`cfg.flow_kl_use_dt` 门控） |
| `SignalRequest.need_ref` | `types.py:86` | `sde_logprob.py:103`；`multi_segment_token_logprob.py:65` |
| `SignalRequest.need_kl_intermediates` | `types.py:88` | `sde_logprob.py:92,123` |

## 2. 落地方案

### A. 删 `entropy` + `need_entropy`（confirmed dead，机械删）
- 删 `vrl/rollouts/evaluators/types.py:21` 的 `entropy` 字段、`:87` 的 `need_entropy` 字段。
- 删 builder 形参与穿透：`vrl/rollouts/evaluators/trajectory.py:38,57,83,141` 的 `entropy` 形参/传参。
- 删 evaluator 构造点的 `entropy=None`：`ar/token_logprob.py:75`、`ar/continuous_token_logprob.py:64`、`ar/multi_segment_token_logprob.py:91`。
- 同步任何在测试里构造 `SegmentSignal(..., entropy=...)` 的点（`tests/algorithms/test_input_contract.py`、`vrl/scripts/perf/fp8_rollout_drift_probe.py` 如有传参）。

### B. `aux` —— 默认删，owner 二选一
- **默认**：删 `vrl/rollouts/evaluators/types.py:26` 的 `aux` 字段 + `trajectory.py:146`、`multi_segment_token_logprob.py:93-96` 的填充。
- **若要保留**（未来逐段日志）：保留字段，就地加 `# display/provenance-only：写入但当前无读者，留作 per-segment 日志钩子`，并把它从「死字段」改记为「显式标注的 provenance 字段」。
- 决策依据：`aux` 是否进未来日志 —— 这是 owner 的产品意图，非机械可判。

### C. `axes` —— 默认删，与 `aux` 同档
- **默认**：删 `vrl/rollouts/evaluators/types.py:15` 的 `axes` 字段、`trajectory.py:135` 的 `_axes_from_value(...)` 填充（若 `_axes_from_value` 仅服务此字段，连带删 `trajectory.py:299-303`）。
- **若保留**：标注 `# display/provenance-only：仅进 validate_inputs 的 available 诊断串`。
- 注意：删 `axes` **不影响** `axis` —— 两者校验/读取路径不同（§1.3 vs §1.4），不要连坐。

### D. `axis` —— 不动
- 保留字段与 `types.py:54` 校验守卫。仅在字段定义处补一行注释，说明「值不参与计算，仅作契约非空不变量」，避免未来审计重复怀疑。

## 3. 附录：派生结构体字段扫描（不另开 sprint，就近收口）

workflow 的 broader sweep 扫了 `ResolvedArtifact` / `FamilyCapability` / `EnginePlan`（均判 justified，无死字段），只捞出两条值得动的，量不够单独成 sprint，并入本 sprint 附录处理。

### 3.1 删 `RuntimeBundle.ref_modules`（confirmed dead）
`vrl/models/interfaces/runtime.py:170`：

```python
ref_modules: dict[str, Any] | None = None
```

`grep -rn "ref_modules"` 全仓库**仅命中这一行定义**：14 处 `RuntimeBundle` 实例化全部省略它，引入后从无引用，class docstring 也未提及。零生产者零消费者 → 直接删。

### 3.2 `ResolvedDistributedResources.visible_devices`（已合规，仅记录）
`vrl/ray/resources.py:135`，唯一读者 `resources.py:440` 在 `format_distributed_resource_plan`（日志串构造，`online.py:652` / `train_dpo.py:143` 的 `logger.info` 下调用）。**无需改动** —— 它已按 AGENTS.md 规则在 `resources.py:132-134` 标注 `display/provenance-only`。在此登记为「已审计、合规、不动」，避免下一轮 sweep 重复 flag。

## 4. 验证（finishing criteria）

- `grep -rn "entropy" vrl/rollouts/evaluators/ vrl/algorithms/` 不再命中 `SegmentSignal.entropy` / `need_entropy` / `entropy=None`（仅可能剩 RNG 注释类无关命中）。
- `grep -rn "ref_modules" vrl/` 零命中。
- `grep -rn "\.aux\b\|\.axes\b" vrl/rollouts/evaluators/ vrl/algorithms/`：按 owner 决策，删则零命中、标注则仅命中定义行。
- `pytest tests/algorithms/ -q` 全绿（重点 `test_input_contract.py`：删字段后 `available` 列表相应缩短，断言若硬编码字段名需同步更新）。
- `pytest tests/rollouts/ -q` 全绿（evaluator 构造点改动）。
- 全量 `pytest -q` 与现有 config 解析零回归。

## 5. 非目标 / Non-Goals

- **不删 `SegmentSignal` / `TrajectorySignalBatch` 结构体本身** —— 经核实是活契约（`distribution` 驱动 `continuous.py:142` 分支；batch 被多段算法消费）。
- **不做 evaluator→algorithm 契约重设计** —— workflow 提出的两个 `design_question`（把 `name/axis/axes/distribution` 元数据移回 trajectory 层、把 `segments` dict 改成 lazy-load）是架构重构，不是死代码清理，留待独立讨论。
- **不动 `axis` 的 `types.py:54` 校验守卫** —— 删它会静默放宽接受的输入 shape，属契约收窄决策，需 owner 单独签字。
- **不扩展到其他 sprint 历史产物**；本 sprint 的派生结构体扫描仅收口 §3 两条，不递归审计全仓库每个 dataclass。

## References

- `vrl/rollouts/evaluators/types.py:9-26,48-70,82-88`
- `vrl/rollouts/evaluators/trajectory.py:38,57,83,128-147,289-303`
- `vrl/rollouts/evaluators/ar/token_logprob.py:75`、`ar/continuous_token_logprob.py:64`、`ar/multi_segment_token_logprob.py:65,91,93-96,101`
- `vrl/rollouts/evaluators/diffusion/sde_logprob.py:92,103,123`
- `vrl/algorithms/trajectory.py:43-63`
- `vrl/algorithms/base.py:28-33`
- `vrl/algorithms/grpo/continuous.py:47,142-154`、`grpo/token.py:46-89`、`grpo/multisegment.py:96`
- `vrl/models/interfaces/runtime.py:170`
- `vrl/ray/resources.py:132-135,440`
- `tests/algorithms/test_input_contract.py`、`vrl/scripts/perf/fp8_rollout_drift_probe.py:80-82`
- 关联：[[SPRINT_trainer_rollout_dead_alias_cleanup]]（已删 `SegmentSignal.segment` 标识符，本 sprint 承接字段级清理）
