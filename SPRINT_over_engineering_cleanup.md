# Sprint: Over-Engineering Cleanup

## 恢复说明

这个文件之前被误删。它应该被视为仍有价值的 cleanup backlog，而不是完成后可以清掉的历史 sprint。

原文件没有被 git 跟踪，无法通过 git 完整恢复；下面内容来自当前会话和 Codex 日志中能恢复到的 cleanup 计划。以后如果清理 sprint 文件，这类“待执行架构清理 backlog”只能移动或合并，不能直接删除。

---

## 已恢复的原文件上下文

目标：删除薄包装、仪式化层级和多余间接层。不改变行为，只减少代码和概念负担。

### Task 1 - Flatten reward registry to a plain dict

**File:** `vrl/rewards/functions/registry.py`

**Problem:** `register_reward()` 和 `get_reward()` 是 dict `__setitem__` / `__getitem__` 的薄包装。`_register_builtins()` 是 lazy-init 仪式层，存在的原因只是 dict 初始为空、运行时再填充。这个模式为约 10 个 registry entry 增加了约 30 行脚手架。

**Cut:**

- 删除 `register_reward()`、`get_reward()` 和 `_register_builtins()`
- 改成 module-level dict，在 import 时直接定义
- `MultiReward.from_dict()` 直接从 dict 读取

**Result:**

```python
# Before
def get_reward(name): ...
def register_reward(name, cls): ...
def _register_builtins(): ...  # called inside from_dict

# After
_REWARD_REGISTRY: dict[str, type[RewardFunction]] = {
    "aesthetic": lambda: _import("vrl.rewards.functions.aesthetic", "AestheticReward"),
    ...
}
```

**Tests to run:** `tests/rewards/`

---

### Task 2 - Remove `collect_policy_version()` from `RolloutLifecycle`

**File:** `vrl/rollouts/orchestration/lifecycle.py`

**Problem:** `collect_policy_version()` 是一层很薄的代理，隐藏了真实调用点，让 lifecycle 看起来有额外职责，但它只是把值转发给 runtime 或 worker。

**Cut:**

- 删除 `RolloutLifecycle.collect_policy_version()`
- 调用方直接访问真实来源
- 如果需要保留边界，使用现有 runtime/context 对象，不新增 wrapper

**Tests to run:** rollout orchestration 相关测试。

---

### Task 3 - Remove `RolloutScheduleMode.from_value()`

**File:** rollout schedule mode 定义处

**Problem:** `from_value()` 只是 enum 构造的重复包装，增加了一个“看起来有自定义解析逻辑”的入口，但实际没有额外行为。

**Cut:**

- 删除 `RolloutScheduleMode.from_value()`
- 调用方直接使用 `RolloutScheduleMode(value)`
- 如果需要错误信息，放在调用边界，而不是 enum helper

**Tests to run:** rollout scheduling 相关测试。

---

## 额外恢复的 cleanup backlog

下面是从 Codex 日志中恢复到的相关 over-engineering cleanup 计划。它与本文件目标一致：删除不承载真实复杂度的抽象，同时保留真正有边界价值的接口。

### Phase 1 - 删除 `CompositeReward`

**目标：** 删除只包了一层 list loop 的 reward 抽象。

**需要改动：**

- 删除 `vrl/rewards/composite.py`
- 删除 `tests/rewards/test_composite.py`
- 更新 `vrl/rewards/__init__.py`，不再导出 `CompositeReward`
- 搜索 `CompositeReward` 全仓库引用并替换为直接调用多个 reward function 或 `MultiReward`

**判断标准：**

- 如果调用方只是把多个 reward function 串起来，直接用 list + loop。
- 如果调用方需要配置驱动的多 reward 聚合，用现有 `MultiReward`，不要保留第二套组合抽象。

**Tests to run:**

```bash
python -m pytest -q tests/rewards
```

---

### Phase 2 - 把 `GenerationIdFactory` 改成普通函数

**目标：** 删除无状态 class，只保留生成 sample row 的函数。

**需要改动：**

- 找到 `GenerationIdFactory` 定义和所有调用点
- 改成函数，例如：

```python
def build_sample_rows(
    *,
    prompts: Sequence[str],
    request_ids: Sequence[str],
    generation_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    ...
```

- 删除 class wrapper 和不必要的实例化
- 测试继续验证 output schema，而不是验证 class 存在

**判断标准：**

- 如果对象没有长期状态、没有替换实现、没有生命周期，就不要 class。
- 如果未来需要不同 id 策略，再引入显式参数或小函数，而不是提前做 factory class。

**Tests to run:**

```bash
python -m pytest -q tests/engine/generation tests/generation
```

---

### Phase 3 - 合并 executor protocol 层级

**目标：** 删除 `FamilyPipelineExecutor` / `ChunkedFamilyPipelineExecutor` 的层级仪式，只保留一个清晰的 executor protocol。

**需要改动：**

- 删除或合并 `FamilyPipelineExecutor`
- 将 `ChunkedFamilyPipelineExecutor` 重命名为 `PipelineExecutor`，或者直接使用现有最贴近调用方的名字
- 调用方只依赖一个 protocol
- 类型测试和 architecture boundary 测试同步更新

**判断标准：**

- 只要当前只有一种 executor 调用形态，就不要用两个 protocol 描述同一个边界。
- protocol 应该对应真实可替换边界，而不是为未来分层。

**Tests to run:**

```bash
python -m pytest -q \
  tests/engine/generation \
  tests/generation/ray \
  tests/architecture/test_generation_rollout_boundaries.py
```

---

### Phase 4 - 删除空 marker protocol `PipelineChunkResult`

**目标：** 删除没有方法、没有字段、没有约束力的 marker protocol。

**需要改动：**

- 删除 `PipelineChunkResult`
- 调用方改用真实类型；如果当前结果形态仍然由不同 executor 决定，可以先用：

```python
ChunkResult = Any
```

- 在真正需要字段访问的位置定义具体 dataclass 或 TypedDict

**判断标准：**

- 空 protocol 不提供类型安全，只制造“这里有抽象”的假象。
- 如果还不知道结果 shape，就承认是 `Any`；等访问字段稳定后再定义结构。

**Tests to run:**

```bash
python -m pytest -q tests/engine/generation tests/generation/ray
```

---

## 明确不清理的抽象

这些不是当前 sprint 的删除目标，因为它们承载了真实边界或复杂度：

| 抽象 | 保留原因 |
| --- | --- |
| `ARPipelineExecutorBase` | 自回归 generation 的共享状态机和 hook 边界，有真实复用价值 |
| Family capability system | 模型 family 的能力差异是真实复杂度，不应压平成 if/else |
| `ChunkGatherer` protocol | Ray streaming gather 和本地测试替身之间有真实替换边界 |
| `RolloutConfig` wrapper | 训练配置边界需要稳定入口，不能把 raw dict 扩散到 runtime |
| `RayGenerationExecutor` | 隔离 Ray actor / distributed execution 细节 |
| `GenerationRuntime` protocol | rollout 和 generation runtime 的跨模块边界 |
| Reward Ray launcher | Ray 绑定应留在 `vrl/rewards/ray/launcher.py`，不要塞进 reward domain API |

---

## Acceptance Criteria

- 删除的 abstraction 不再出现在 public imports 中。
- 没有新增 compatibility alias 来假装没删。
- 行为测试仍然通过。
- 类型边界更少、更直接。
- 相关 sprint 或 docs 中不再引用已删除 abstraction。

**Full verification:**

```bash
python -m pytest -q \
  tests/rewards \
  tests/engine/generation \
  tests/generation/ray \
  tests/rollouts/test_runtime_inputs.py \
  tests/architecture/test_generation_rollout_boundaries.py
```
