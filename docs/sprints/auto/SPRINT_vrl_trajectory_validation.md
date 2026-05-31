# SPRINT(auto): vrl/trajectory/validation.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/trajectory/validation.py` (417 LOC)
角色判定: interface-boundary
结论: improve

## 0. 一句话
这是 trajectory 契约校验器，是真边界、必须保留；但 "trainable segment 必须各有且仅有一个 action/old_log_prob/mask" 这条规则的角色三元组在同一文件里写了两遍，应统一成一个命名常量，另有一处恒为真的死分支可清理。

## 1. 现状（读代码得出）
`TrajectoryValidator` 校验 `TrajectoryBatch` 及其派生的 training/reward view，被 `builders.py`、`resolver.py`、`ops.py` 广泛调用（如 `vrl/trajectory/ops.py:211` `TrajectoryValidator(out).validate_batch()`），是 trajectory 契约的核心入口，不是薄 wrapper。

两个 module-level ALL_CAPS：

```python
FORBIDDEN_TRAJECTORY_METRICS = frozenset(
    {"queue_wait_s", "execution_s", "peak_memory_mb", "chunks"}
)
SINGLETON_TENSOR_ROLES = frozenset({"action", "old_log_prob", "mask"})
```

`SINGLETON_TENSOR_ROLES` 用于段内角色唯一性检查（`validation.py:225`），而同一方法内紧接着又用一个**字面三元组**做"必需角色齐全"检查：

```python
# validation.py:233
missing = [role for role in ("action", "old_log_prob", "mask") if role not in roles]
```

## 2. 质疑点 / 改进机会
- **同一概念写两遍（重复，非派生问题）**：`("action", "old_log_prob", "mask")` 表达的是同一个领域事实——"可训练段必须各有且仅有一个的核心角色三元组"。它在 `validation.py:26`（frozenset，做唯一性）和 `validation.py:233`（tuple，做齐全性）各出现一次。改一个忘改另一个会让两条校验规则悄悄分叉。这不是 AGENTS.md 说的"手抄 typed 结构"反模式（它确实是 `TensorRole` Literal 的语义子集，不是镜像），但属于"同一常量散落多处"的 DRY 问题。证据：`validation.py:26`、`validation.py:233`。
- **degenerate 死分支**：`_looks_runtime_only` 末尾

  ```python
  # validation.py:386
  if type_module.startswith("ray.") or "actor" in type_name:
      return True
  return True
  ```

  最后无条件 `return True`，使前面那个 `ray.`/`actor` 判断永远走不到独立分支——它和兜底返回值相同，是 dead 条件。语义上该函数对"任何非 str/bytes/数值/dict/list/tuple/None、且非 tensor-like" 的对象一律判为 runtime-only，那条 ray 检查纯属装饰，删掉不改变行为。证据：`validation.py:386-388`。

## 3. 建议动作
- 把必需角色三元组提为单一有序常量并在两处复用，例如：

  ```python
  REQUIRED_TRAINABLE_ROLES = ("action", "old_log_prob", "mask")
  SINGLETON_TENSOR_ROLES = frozenset(REQUIRED_TRAINABLE_ROLES)
  ```

  然后 `validation.py:233` 改为 `for role in REQUIRED_TRAINABLE_ROLES`。保持有序 tuple 用于"缺哪个角色"的稳定报错顺序，frozenset 仍用于唯一性查找。
- 删除 `_looks_runtime_only` 中 `if type_module.startswith("ray.") or "actor" in type_name: return True` 这一行（连同其上方取 `type_module`/`type_name` 的两行，如确认别处不再用），直接保留兜底 `return True`。若想保留 ray-actor 的语义文档价值，改成注释而非 dead 分支。

## 4. 不动什么 / 为什么不是过度清理
- `FORBIDDEN_TRAJECTORY_METRICS` 保留：这是"刻意隔离的 taxonomy/denylist"——本模块存在意义之一就是阻止引擎运行时指标泄进可序列化契约（`validation.py:94` 用它做交集检查）。符合 AGENTS.md ALL_CAPS 例外。
- 整个 `TrajectoryValidator` 及其 `_validate_segment`/`_validate_tensor_axes`/`_reject_runtime_state` 等私有方法保留：它们是真实的契约校验复杂度，不是可拍平的薄转发。
- `tensor_ref` / `replay_input_ref` 两个 module 级函数保留：它们是 segment.tensor 引用串的 canonical 构造点，被 `builders.py`、`resolver.py`、`views.py` 多处复用，是跨家族一致的引用约定（grepability），不要内联。
- 不要把 `SINGLETON_TENSOR_ROLES` 改成从 `TensorRole` Literal 派生——它是该 Literal 的**语义子集**（不含 observation/replay_input），派生反而会把无关角色拉进来，属于误伤。

## 5. 验证
- `rg -n '"action", "old_log_prob", "mask"' vrl/trajectory/` 确认改后只剩常量定义一处。
- 跑 trajectory 相关测试：`pytest tests/trajectory tests/models/test_diffusion_model_base.py tests/models/test_janus_replay.py -q`，重点覆盖 trainable 段缺角色 / 多角色的报错路径。
- `ruff check vrl/trajectory/validation.py`。
- 针对死分支：临时构造一个普通对象传给 `_looks_runtime_only`，确认删前删后返回值一致（恒为 `True`）。
