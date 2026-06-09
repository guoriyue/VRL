# SPRINT: rollouts 编排层 DIP 收口（getattr → protocol）

状态：planned。父：`SPRINT_solid_architecture_audit.md`（子 sprint A，优先级最高）。

## 0. Core Decision

把 rollouts 编排层「用 `getattr` 走对象内部嵌套结构来探测能力」改成**向 protocol 提问**。

- **为什么**：现在编排层知道 `runtime` 内部长什么样（`runtime.config.resources.reward_shared_with_rollout`）、知道 `weight_syncer` 内部有个 `runtime.current_policy_version`。这是 DIP 倒置——高层模块依赖低层模块的**具体内部结构**。不影响当前正确性，但 runtime / weight_syncer 内部任何重构都会让这些 `getattr(..., None)` **静默返回 None**，编排层默默走错分支而不报错。
- **范围**：只动两处探测点 + 给两个协议各加一个方法。最低 LOC、最高确定性，**先做这个**。

## 1. 现状（两处证据，已实测）

### 1a. 内存释放决策走三级 getattr —— `vrl/rollouts/collector/core.py:160-166`
```python
def _should_release_runtime_before_reward_model(self) -> bool:
    runtime = self._runtime
    config = getattr(runtime, "config", None)
    if not bool(getattr(config, "release_before_reward_model", False)):
        return False
    resources = getattr(config, "resources", None)
    return bool(getattr(resources, "reward_shared_with_rollout", False))
```
collector 穿透 `runtime → config → resources` 三层私有结构。

### 1b. policy 版本探测走 collector + weight_syncer 内部 —— `vrl/rollouts/orchestration/lifecycle.py:121-135`
```python
def _runtime_policy_version(self, *, default: int | None) -> int | None:
    runtime = self._collector_runtime()
    value = getattr(runtime, "current_policy_version", None)
    if value is None:
        sync_runtime = getattr(self.weight_syncer, "runtime", None)
        value = getattr(sync_runtime, "current_policy_version", None)
    ...
```
lifecycle 知道「policy version 可能在 collector.runtime 上，也可能在 weight_syncer.runtime 上」——这是把两个低层对象的内部布局硬编码进编排层。

## 2. 目标架构

```
GenerationRuntime (协议, vrl/.../interfaces or runtime base)
   + should_release_memory_before_reward() -> bool     ← 新增,把决策内聚到 runtime 自己
   + current_policy_version: int | None                ← 已隐式存在,提升为协议契约

PolicyVersionProvider (新协议)                          ← runtime / weight_syncer 都实现它
   + current_policy_version: int | None

collector / lifecycle 只调协议方法,不再 getattr 走内部结构
```

## 3. 分步实施

### T1 [核心] `should_release_memory_before_reward()` 上移到 runtime
- 在 `GenerationRuntime` 协议（先确认协议定义位置：grep `class GenerationRuntime`）加方法 `should_release_memory_before_reward(self) -> bool`。
- 把 `core.py:160-166` 的三级 getattr 逻辑搬进 runtime 的**具体实现**——runtime 自己持有 config/resources，访问是合法的内部访问，不是跨层穿透。
- `_should_release_runtime_before_reward_model` 退化为 `return self._runtime.should_release_memory_before_reward()`（或直接内联删掉这个 wrapper）。
- runtime 是 `None` 时的兜底（`self._runtime` 可空）：协议方法不存在的分支保留 `False` 默认。

### T2 [核心] `PolicyVersionProvider` 协议替掉 policy 版本 getattr
- 定义协议 `PolicyVersionProvider`（property `current_policy_version: int | None`），放在 rollouts 协议层（与 `GenerationRuntime` 同处）。
- 让 collector.runtime 和 weight_syncer 都声明实现它（它们已经有这个属性，只是没进协议）。
- `lifecycle.py:_runtime_policy_version` 改成：依次问两个 provider 的 `current_policy_version`，删掉 `getattr(..., None)` ——版本不存在时仍走 `default` 兜底，但「对象没有这个属性」从静默 None 变成类型可检。

### T3 [收尾] 扫剩余同类 getattr-探测
- grep `getattr(.*runtime`、`getattr(.*config`、`getattr(.*syncer` 在 `vrl/rollouts/` 下的其它出现，判断哪些是「探测低层内部结构」（要改）vs「读可选配置字段」（可留）。
- 只改前者，后者（如读 `getattr(cfg, "optional_flag", default)` 这种纯配置默认值）保留。

## 4. 测试策略

- 现成：grep `tests/rollouts/` 下覆盖 collector lifecycle / policy version 的用例，重构后必须仍过。
- 新增：给 `should_release_memory_before_reward()` 和 `PolicyVersionProvider` 各加一个 protocol 契约测试（参照 `tests/nn/layers/test_paged_attention_contract.py` 的契约测试风格）。
- 等价性：T1 前后，给定同一 runtime config，`_should_release...` 返回值不变（先抓一组 golden）。

## 5. Non-Goals

- 不重构 `runtime` / `weight_syncer` 的内部数据结构——只把「问法」从 getattr 改成协议。
- 不动 `release_reward_artifact_if_needed` 等已经走显式 policy 对象的路径（那些已经是对的）。
- 不为「纯可选配置字段」的 `getattr(cfg, k, default)` 强加协议（YAGNI）。

## 6. 关键参考文件

- 改动点：`vrl/rollouts/collector/core.py:160-166`、`vrl/rollouts/orchestration/lifecycle.py:121-135`
- 协议定义（实施前 grep 确认）：`class GenerationRuntime`、rollouts 协议层
- 契约测试范式：`tests/nn/layers/test_paged_attention_contract.py`
