# SPRINT(auto): vrl/generation/capabilities.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/generation/capabilities.py` (343 LOC)
角色判定: core
结论: improve

## 0. 一句话
文件整体是 planner 的核心 capability 契约（真边界，重用广），但 `ExecutionStageCapability.to_dict()` 与 `from_value()` 对 `profiler_name` 字段不对称，存在 round-trip 数据腐烂的真 bug，需修。

## 1. 现状（读代码得出）
`ExecutionStageCapability` 是个 frozen dataclass，`profiler_name` 是可选原始字段，`profiler_label` 是个派生 property：

```python
profiler_name: str | None = None              # line 69

@property
def profiler_label(self) -> str:              # line 91-93
    return self.profiler_name or f"engine.{self.name}"
```

序列化时 `to_dict` 把 `profiler_name` 这个 key 填成了派生的 `profiler_label`：

```python
def to_dict(self) -> dict[str, Any]:
    return {
        ...
        "profiler_name": self.profiler_label,  # line 87  <-- 写的是 label 不是 name
        ...
    }
```

但反序列化时 `from_value` 把 `profiler_name` 当原始字段读回：

```python
profiler_name=None if value.get("profiler_name") is None
              else str(value.get("profiler_name")),   # line 108
```

该 `to_dict` 是热路径，不是装饰：`NEXTSTEP_1_FAMILY_CAPABILITY.to_dict()` / `JANUS_PRO_FAMILY_CAPABILITY.to_dict()` 会被塞进 `runtime_caps["family_capability"]`，再经 `family_capability_from_value` 重建（见 `vrl/models/ar/nextstep_1/runtime.py:58,85`、`janus_pro/runtime.py:67,100`、`capabilities.py:219-230` 的 `with_runtime_caps`）。

## 2. 质疑点 / 改进机会
- **序列化非对称（真 bug，证据 capabilities.py:87 vs :108）**：对一个 `profiler_name=None` 的 stage 做 `to_dict()`，key 会被写成 `"engine.<name>"`；再 `from_value()` 读回后，`profiler_name` 从 `None` 变成了 `"engine.<name>"`。表面行为（`profiler_label`）没变，但 raw 字段被悄悄改写，违反了 frozen dataclass 的 round-trip 不变性，也让 `__post_init__`（line 75-78 校验 profiler_name 非空）面对的语义和构造时不一致。这是 AGENTS.md "源类型加字段/数据会悄悄腐烂" 类问题的同构变体——序列化把派生值写回了原始 slot。
- 这是文件里唯一的 hygiene 缺陷。其余结构（`AxisCapability` / `ExecutionStageCapability` / `FamilyCapability` 三层 + `from_value`/`to_dict`/`with_runtime_caps`）都是 planner 真边界，重用广（`execution/planner.py`、`execution/worker.py`、`execution/scheduler.py`、`rollouts/families/registry.py`、各 model family），不属于薄 wrapper。

## 3. 建议动作
把 `to_dict` 第 87 行改为序列化原始字段，保持 round-trip 对称：

```python
"profiler_name": self.profiler_name,   # 原始 None/字符串，而非 profiler_label
```

派生的 `engine.<name>` 仍由 `profiler_label` property 在读取侧按需计算（`profiler_labels` line 174-175 已经走的是 property，不受影响）。若确实希望对外暴露已解析的 label，应另起一个独立 key（如 `"profiler_label"`）而不是覆盖 `profiler_name` 这个反序列化要用的 slot。

非删除项，无需 grep 无引用确认。

## 4. 不动什么 / 为什么不是过度清理
- 三个 dataclass 的 `from_value`/`to_dict`/`from_request*` 形状跨家族统一（diffusion / ar 都消费），属于 AGENTS.md "跨家族一致性 > LOC" 的保留项，不要拍平。
- `TrajectoryKind` Literal + `_trajectory_kind` 校验是刻意隔离的 taxonomy 校验边界，保留。
- `with_runtime_caps` 的 `bool_fields` 元组（line 194-208）是手列的字段名集合，理论上可由 dataclass fields derive；但它是"只合并 backend 动态 bool 子集"的刻意白名单（并非全字段镜像，static trajectory facts 故意排除在外），属于真边界，保留——不要为省行 derive，会把不该被 runtime 覆盖的字段也放进来。

## 5. 验证
- 新增/补一个 round-trip 单测：`ExecutionStageCapability(name="x")` → `from_value(s.to_dict())`，断言 `result.profiler_name is None` 且 `result.profiler_label == "engine.x"`。
- 跑 `python -m pytest -k "capability"`（确认现有 capability 测试不回归）。
- `ruff check vrl/generation/capabilities.py`。
- `grep -rn "profiler_name" vrl --include=*.py` 确认无别处依赖 to_dict 输出的是 label 语义。
