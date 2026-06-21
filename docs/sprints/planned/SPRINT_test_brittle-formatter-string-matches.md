# SPRINT: 测试断言诊断 formatter 的渲染子串（去脆化）（planned）

状态：未开始（2026-06-21）。
范围：把三处「冻结渲染串」的 formatter/log 测试改成「从已解析对象派生」的断言。涉及两个测试文件：`tests/ray/test_resources.py`（`format_distributed_resource_plan` 的 2 个断言块）、`tests/generation/test_capabilities.py`（`profiler_label` / `profiler_labels` 的 fallback 断言）。优先级 low —— 这些测试当前全绿，只是耦合了人类可读输出的标点/键名/前缀模板，会在任何 cosmetic 重排版时假性失败。**不**删除这些 formatter 单测本身（它们覆盖了真实的渲染路径），只把「冻结串匹配」换成「从 `Resolved*` 字段 / `f"engine.{name}"` 派生 expected」。

> 本 sprint 承接同批 test-brittleness 审计中「brittle_string_match」一类（与 frozen_snapshot、duplicated_constant 分列不同 sprint）。判定标准：被断言的渲染值，其底层结构化值是否已在别处（或同对象上）可派生 —— 若可派生，冻结串就是 bug。

## 0. Core Decision（先看这一段）

裁决一切「这个串断言到底脆不脆」的，是一句话：**被断言的人类可读子串，其底层值在被断言的同一个对象上已经是结构化字段了吗？**

三处全部命中：

- `"trainer=[0]"` / `"rollout_gpu_memory_fraction=None"` / `"trainer_reservation=True"` / `"lifecycle=rollout:on_demand/reward:on_demand"` / `"before_reward:True"` —— 它们渲染的全是 `ResolvedDistributedResources` 上的解析字段（`trainer_devices`、`rollout_gpu_memory_fraction`、`requires_trainer_reservation`、`lifecycle.rollout.mode`、`lifecycle.handoff.release_rollout_before_reward`），而 `format_distributed_resource_plan` 的整个函数体就是把这些字段拼成 `key=value` 串（`vrl/ray/resources.py:435-456`）。测试把 formatter 的**布局**（分隔符 `=`、`:`、`/`、键拼写 `trainer_reservation`）钉死，等于在测「这一行长什么样」，而不是「这些解析值对不对」—— 后者才是行为契约。
- `"engine.denoise_step"` / `"engine.prepare"` —— 它们渲染的是 `profiler_label` 属性的 fallback 模板 `f"engine.{self.name}"`（`vrl/generation/capabilities.py:94-95`）。测试把 `engine.` 前缀模板再抄一遍当 expected，前缀一旦改名（如 `engine.` → `gen.`）测试就因纯模板改动而红，而 fallback 行为本身没坏。

这与 `wan_2_2` 漏注册类 bug 同源：那里的反模式是「手抄一份 family key 列表」而不是 `registered_rollout_families() == tuple(FAMILY_REGISTRY)` 让 source 自证；这里的反模式是「手抄一份渲染串」而不是从 `Resolved*` 字段 / `f"engine.{name}"` 现算 expected。**source（解析对象 / 派生属性）才是 single source of truth；冻结串应当消失进派生表达式。**

裁决产出三档：

1. **底层值已在同对象上结构化、且 formatter 是纯派生**（全部三处）→ expected 改成从该结构化值现算。formatter 单测保留（它确实覆盖渲染路径），只是不再钉死标点。
2. **断言的是 formatter 唯一能验证的、无结构化来源的字面量**（本 sprint 无此项）→ 维持字面量，标注为「即契约」。
3. **断言的值是测试自己喂进去的输入回显**（如 capabilities 测试里显式设的 `"generation.decode_latents"`）→ 本来就不脆，保留。

## 1. 现状实锤

### 1.1 `tests/ray/test_resources.py:406-424` —— formatter 渲染串钉死解析字段

`tests/ray/test_resources.py:418-424`：

```python
    text = format_distributed_resource_plan(resolved)

    assert "trainer=[0]" in text
    assert "rollout=[1]" in text
    assert "reward=[]" in text
    assert "rollout_gpu_memory_fraction=None" in text
    assert "trainer_reservation=True" in text
```

formatter 源（`vrl/ray/resources.py:435-447`，全是 `key=value` 拼接）：

```python
    parts = [
        f"visible={list(resolved.visible_devices)}",
        f"trainer={list(resolved.trainer_devices)}",
        f"rollout={list(resolved.rollout_devices)}",
        f"reward={list(resolved.reward_devices)}",
        ...
        f"rollout_gpu_memory_fraction={resolved.rollout_gpu_memory_fraction}",
        ...
        f"trainer_reservation={resolved.requires_trainer_reservation}",
```

底层解析字段在 `ResolvedDistributedResources` 上全是一等字段：`trainer_devices`/`rollout_devices`/`reward_devices`（`vrl/ray/resources.py:131-133`）、`rollout_gpu_memory_fraction`（`:139`）、`requires_trainer_reservation`（`:143`）。测试拼死的 `trainer=[0]`、`reward=[]`、`trainer_reservation=True`、键名 `rollout_gpu_memory_fraction=None`，全部是这些字段经 `f"key={value}"` 渲染的结果 —— 键拼写、`=` 分隔、`[...]` list 格式一旦改（例如把 `trainer_reservation` 改写成 `needs_trainer_reservation`，或 list 改成逗号串），测试因 cosmetic 改动而红，解析逻辑没动。

### 1.2 `tests/ray/test_resources.py:734-754` —— lifecycle 渲染串钉死分隔符

`tests/ray/test_resources.py:752-754`：

```python
    text = format_distributed_resource_plan(resolved)
    assert "lifecycle=rollout:on_demand/reward:on_demand" in text
    assert "before_reward:True" in text
```

formatter 源（`vrl/ray/resources.py:450-455`）：

```python
        f"lifecycle=rollout:{resolved.lifecycle.rollout.mode}"
        f"/reward:{resolved.lifecycle.reward.mode}",
        "handoff="
        f"before_train:{resolved.lifecycle.handoff.release_rollout_before_train}"
        f",before_reward:{resolved.lifecycle.handoff.release_rollout_before_reward}"
        ...
```

底层值是 `RayLifecyclePlan` 上的结构化字段：`lifecycle.rollout.mode` / `lifecycle.reward.mode`（`mode: Literal["resident","on_demand"]`，`vrl/ray/resources.py:89`）、`lifecycle.handoff.release_rollout_before_reward`（`bool`，`:102`）。同文件 `test_lifecycle_*` 系列已对这些 `mode` / 布尔做了结构化断言（如 `:730-731` 的 `plan.rollout.mode == "resident"` / `plan.handoff.release_rollout_before_train is False`）。本断言把它们重复成一段冻结渲染串，连 `:` 和 `/` 分隔符都钉死 —— 分隔符或键名一改即红。

### 1.3 `tests/generation/test_capabilities.py:27,61` —— `profiler_label` fallback 钉死 `engine.` 前缀模板

`tests/generation/test_capabilities.py:14-27`（fallback 路径：`profiler_name` 不设）：

```python
    stage = ExecutionStageCapability(
        name="denoise_step",
        ...
    )
    ...
    assert restored.profiler_name is None
    assert restored.profiler_label == "engine.denoise_step"
```

`tests/generation/test_capabilities.py:61`：

```python
    assert restored.profiler_labels == ("engine.prepare", "generation.decode_latents")
```

派生属性源（`vrl/generation/capabilities.py:94-95`）：

```python
    def profiler_label(self) -> str:
        return self.profiler_name or f"engine.{self.name}"
```

`profiler_labels` 即逐 stage 取 `profiler_label`（`vrl/generation/capabilities.py:168-169`）。测试验证的是真实的 fallback 行为（`profiler_name is None` → `engine.<name>`）与 override 透传，这是真契约；但 expected 里的 `"engine.denoise_step"` / `"engine.prepare"` 把 `engine.` 前缀模板手抄了一份，前缀模板改名即红。注意第二个 tuple 里的 `"generation.decode_latents"` 是测试自己在 `:48` 显式设进 `profiler_name=` 的输入回显，**不脆**，保留。

## 2. 落地方案

### 派生范式（canonical pattern）

不引入任何新字面量。expected 一律从「已解析对象的结构化字段」或「派生属性的同一模板」现算：

- ray formatter：`text = format_distributed_resource_plan(resolved)` 之后，用 `resolved.<field>` 现拼 expected 子串再 `in text`。这样既保留「这些值确实出现在日志行里」的覆盖，又让 formatter 重排版（改键名/分隔符）时——若拼装方式同步——自然跟随；若只想验证值在串里，连键名都不拼，直接断言 `str(value)` 子串。本 sprint 取**保留键名但从字段拼**的折中：键名/值都来自 `resolved`，formatter 改值的渲染方式才会触发更新，纯 list-格式微调不再误伤。
- capabilities：fallback 断言 expected 用 `f"engine.{stage.name}"` 现算（与源同模板）；override 断言用测试自己设进去的输入值。

### A. `tests/ray/test_resources.py:406-424`

BEFORE（冻结串）：

```python
    text = format_distributed_resource_plan(resolved)

    assert "trainer=[0]" in text
    assert "rollout=[1]" in text
    assert "reward=[]" in text
    assert "rollout_gpu_memory_fraction=None" in text
    assert "trainer_reservation=True" in text
```

AFTER（从 `resolved` 派生）：

```python
    text = format_distributed_resource_plan(resolved)

    # The formatter renders resolved fields as `key=value`; assert the resolved
    # values reach the log line, not a frozen layout. A reword of the plan line
    # (key spelling / separators) must not break behavioral coverage.
    assert f"trainer={list(resolved.trainer_devices)}" in text
    assert f"rollout={list(resolved.rollout_devices)}" in text
    assert f"reward={list(resolved.reward_devices)}" in text
    assert (
        f"rollout_gpu_memory_fraction={resolved.rollout_gpu_memory_fraction}" in text
    )
    assert f"trainer_reservation={resolved.requires_trainer_reservation}" in text
```

### B. `tests/ray/test_resources.py:734-754`

BEFORE：

```python
    text = format_distributed_resource_plan(resolved)
    assert "lifecycle=rollout:on_demand/reward:on_demand" in text
    assert "before_reward:True" in text
```

AFTER（从 `resolved.lifecycle.*` 派生）：

```python
    text = format_distributed_resource_plan(resolved)
    plan = resolved.lifecycle
    # Lifecycle modes/flags are structured fields; build the expected substring
    # from them so a separator/keyword reword in the formatter does not break
    # this acceptance check.
    assert (
        f"lifecycle=rollout:{plan.rollout.mode}/reward:{plan.reward.mode}" in text
    )
    assert f"before_reward:{plan.handoff.release_rollout_before_reward}" in text
```

### C. `tests/generation/test_capabilities.py:27,61`

BEFORE（fallback 钉死 `engine.` 前缀）：

```python
    assert restored.profiler_label == "engine.denoise_step"
```

AFTER（从 `f"engine.{stage.name}"` 派生，与源同模板）：

```python
    # profiler_label falls back to f"engine.{name}" when profiler_name is unset;
    # derive the expected from stage.name so a prefix-template rename does not
    # break the fallback-behavior contract.
    assert restored.profiler_label == f"engine.{stage.name}"
```

BEFORE（labels tuple 钉死 fallback 前缀 + 输入回显）：

```python
    assert restored.profiler_labels == ("engine.prepare", "generation.decode_latents")
```

AFTER（fallback 项派生，override 项用输入值；显式构造两个 stage 引用以现算）：

```python
    prepare_stage, decode_stage = capability.execution_stages
    # First stage has no profiler_name -> engine.<name> fallback (derived);
    # second stage's label is the explicitly-set profiler_name input (echoed).
    assert restored.profiler_labels == (
        f"engine.{prepare_stage.name}",
        decode_stage.profiler_name,
    )
```

> 注：`capability.execution_stages` 在 `:44-50` 已构造为 `(ExecutionStageCapability(name="prepare"), ExecutionStageCapability(name="decode", profiler_name="generation.decode_latents"))`，故 `prepare_stage.name == "prepare"`、`decode_stage.profiler_name == "generation.decode_latents"`，AFTER 与原 expected 等值，但来源是 source-of-truth 字段而非冻结串。

## 3. 验证（finishing criteria）

- `grep -rn '"trainer=\[0\]"\|"reward=\[\]"\|"trainer_reservation=True"\|"rollout_gpu_memory_fraction=None"' tests/ray/test_resources.py` 零命中（冻结串已全部派生化）。
- `grep -rn '"lifecycle=rollout:on_demand/reward:on_demand"\|"before_reward:True"' tests/ray/test_resources.py` 零命中。
- `grep -rn '"engine\.denoise_step"\|"engine\.prepare"' tests/generation/test_capabilities.py` 零命中；允许 `f"engine.{...}"` 派生表达式继续存在。
- `pytest tests/ray/test_resources.py -q` 全绿（重点 `test_resource_plan_formatter_includes_key_fields`、`test_resource_plan_formatter_includes_lifecycle`）。
- `pytest tests/generation/test_capabilities.py -q` 全绿（两个 round-trip 测试）。
- 反向证明去脆有效（可选手验）：在 `vrl/ray/resources.py` 临时把 `f"trainer_reservation=..."` 改写成 `f"needs_trainer_reservation=..."`，派生化后的断言会因「值仍在串里、键名由测试现拼」而跟随——确认测试不再因纯键名重排版假性失败；验证后回滚该临时改动。

## 4. 非目标 / Non-Goals

- **不删 formatter 单测本身**。`format_distributed_resource_plan` / `profiler_label` 是真实日志/诊断渲染路径，单测覆盖它们有价值；本 sprint 只换断言来源，不删测试。
- **不重写 formatter 输出格式**。键名/分隔符是日志可读性决策，不在本 sprint 改动；本 sprint 让测试不再钉死它，而非去优化它。
- **不动同文件已合规的结构化断言**（如 `:730-731` 的 `plan.rollout.mode` / `plan.handoff.*` 直接断言）—— 它们本就是从 `resolved.lifecycle` 读的，是范本而非问题。
- **不动 capabilities 测试里的输入回显断言**（显式 `profiler_name="generation.decode_latents"` 的回显）—— 那是测试自己喂的输入，不是冻结的 source 模板。
- **不扩展到 frozen_snapshot / duplicated_constant 类**（如 `_module_filenames(...) == {...}` 目录清单、`AlgorithmConfig.kind` Literal 抄写、protocol method tuple 抄写）—— 那些是另一档反模式，归属各自 sprint，本 sprint 仅收口 `brittle_string_match`（formatter 渲染串）三处。

## References

- `tests/ray/test_resources.py:406-424,734-754`（本 sprint 修改）
- `tests/generation/test_capabilities.py:14-27,30-61`（本 sprint 修改）
- `vrl/ray/resources.py:131-149`（`ResolvedDistributedResources` 字段：`trainer_devices`/`rollout_devices`/`reward_devices`/`rollout_gpu_memory_fraction`/`requires_trainer_reservation`/`lifecycle`）
- `vrl/ray/resources.py:89,102,107`（`RayLifecyclePlan.rollout.mode` / `PhaseHandoffPolicy.release_rollout_before_reward`）
- `vrl/ray/resources.py:435-459`（`format_distributed_resource_plan`，纯 `key=value` 派生）
- `vrl/generation/capabilities.py:71,94-95,168-169`（`profiler_name` / `profiler_label = profiler_name or f"engine.{name}"` / `profiler_labels`）
- 关联（同批 test-brittleness 审计的姊妹 sprint slug，若已建则互链）：[[SPRINT_test_frozen-directory-listing-snapshots]]（目录清单 == set 漂移，含 `hub.py` 实锤）、[[SPRINT_test_literal-allowlist-vs-typed-literal]]（`AlgorithmConfig.kind` / `TrainingSection.strategy` 抄写，含 `ddp` 漏覆盖）、[[SPRINT_test_protocol-method-tuple-duplication]]（`ReplayModel.__protocol_attrs__` 抄写）、[[SPRINT_test_hardcoded-family-entity-lists]]（`registered_rollout_families() == tuple(FAMILY_REGISTRY)` 范式，`wan_2_2` 漏注册同源）
