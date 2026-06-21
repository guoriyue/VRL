# SPRINT: Rollout-wiring 测试冻结 registry `entry.task` 字面量清理（planned）

状态：未开始（2026-06-21）。
范围：把 rollout runtime 接线测试里**手抄的 canonical task 字面量**（`t2i` / `ar_t2i` / `ar_t2i_r1` 等）改成从已绑定的 `RolloutFamilyEntry.task` 派生。这些测试已经把 entry 取在手里（`get_rollout_family_entry(family)`），且 `runtime_builder` / `executor_cls` 已经是对着 `entry.*` 比的 —— 唯独 `task` 还在断言一份冻结副本。registry 是 task 词表的唯一真源，registry 里改名一个家族的 canonical task，这些字面量就得手改、否则假性 fail。优先级：medium。

> 这是 `tests/` 反 frozen-snapshot 系列的一员：与 [[SPRINT_test_registry-family-list-snapshot]]（`registered_rollout_families()` 字面 key 列表 = `tuple(FAMILY_REGISTRY)`，wan_2_2-missing 那一类）、[[SPRINT_test_literal-config-assertions]] 同源。本 sprint 只收口 `task` 字段这一条主题，**两个**测试文件。

## 0. Core Decision（先看这一段）

`RolloutFamilyEntry.task`（`vrl/rollouts/families/registry.py:67`）是每个 family 的 canonical task 字符串的**唯一真源**。整条接线路径都从它派生：

```python
# vrl/rollouts/collector/core.py:250,253 —— collector.task 与 request.task 都来自 entry.task
task=entry.task,
...
    task=entry.task,
```

`build_ray_generation_inputs_for_family` 也是把同一个 entry 交给 launcher，`launch_contract.task` 同样从 entry 出。也就是说，**被测代码读的就是 `entry.task`**；测试若另写一份字面量，等于让两份 copy 比对——一旦 registry 改 task 名，被测代码自动跟上、字面量却不跟，测试为「非行为原因」red。

这跟 `registered_rollout_families() == tuple(FAMILY_REGISTRY)` 的 wan_2_2-missing bug 是同一个反模式：手抄一份 registry 已机械派生的东西。修法是统一——**删字面量，断言 `== entry.task`**，复用测试里已 fetch 的 entry，与同文件已有的 `runtime_builder` / `executor_cls` 断言保持同形。

非目标里有一条要先说清：本 sprint **不**碰 `expected_gatherer`（那是 type，registry 里是 dotted-path 字符串 `gatherer.import_path`，不是同一形状，需另案处理）；只动 `task`。

## 1. 现状实锤

### 1.1 `tests/rollouts/runtime/test_runtime_inputs.py` —— parametrize 手抄 10 行 `expected_task`

`tests/rollouts/runtime/test_runtime_inputs.py:22-23`，parametrize 把 `expected_task` 作为一列硬塞：

```python
@pytest.mark.parametrize(
    ("experiment", "family", "expected_task", "expected_gatherer"),
    [
        ("diffusion/sd3_5/online_grpo_ocr", "sd3_5", "t2i", DiffusionChunkGatherer),
        ...
        ("ar/janus_pro/online_r1_grpo_ocr", "janus_pro_r1", "ar_t2i_r1", JanusProR1ChunkGatherer),
        ("ar/nextstep_1/online_grpo_ocr", "nextstep_1", "ar_t2i", NextStep1ChunkGatherer),
    ],
)
```

测试体里 entry 已经取好了，且 `runtime_builder` / `executor_cls` 已经是对着 `entry.*` 断言的，唯独 `task` 还在比那份手抄字面量（`test_runtime_inputs.py:97,109,111-112`）：

```python
entry = get_rollout_family_entry(family)
...
assert inputs.launch_contract.task == expected_task          # ← 冻结副本
assert inputs.launch_contract.runtime_builder == entry.runtime_builder  # ← 已经派生
assert inputs.launch_contract.executor_cls == entry.executor_cls        # ← 已经派生
```

逐条核对参数表里的 `expected_task` 与 registry `entry.task`，全部一一对应、当前皆吻合（故现在是「碰巧没 drift」，不是「不会 drift」）：

| family | 参数表 `expected_task` | registry `task=`（registry.py） |
|---|---|---|
| `sd3_5` | `t2i` | `:135 t2i` |
| `wan_2_1` | `t2v` | `:178 t2v` |
| `wan_2_1_i2v` | `i2v` | `:191 i2v` |
| `cosmos-predict2` | `v2w` | `:205 v2w` |
| `cosmos-predict2-anima` | `t2i` | `:247 t2i` |
| `janus_pro` | `ar_t2i` | `:266 ar_t2i` |
| `janus_pro_r1` | `ar_t2i_r1` | `:288 ar_t2i_r1` |
| `nextstep_1` | `ar_t2i` | `:315 ar_t2i` |

派生可达性已核实：`launch_contract.task` 由 launcher 从交进去的同一个 entry 取，`get_rollout_family_entry(family)` 在测试体已绑定 `entry`。`entry.task` 直接可用。

> 同文件 `test_wan_i2v_runtime_inputs_include_reference_image_from_cfg`（`test_runtime_inputs.py:248`）也有一处 `assert inputs.launch_contract.task == "i2v"` 字面量，且该测试**没有** fetch entry，属同一主题、一并修（见 §2.B）。

### 1.2 `tests/rollouts/runtime/test_janus_pro_r1_wiring.py` —— collector/request 双重手抄 `ar_t2i_r1`

`tests/rollouts/runtime/test_janus_pro_r1_wiring.py:84-87`：

```python
assert collector.family == "janus_pro_r1"
assert collector.task == "ar_t2i_r1"          # ← 冻结副本
assert plan.request.family == "janus_pro_r1"
assert plan.request.task == "ar_t2i_r1"        # ← 冻结副本
```

被测路径 `vrl/rollouts/collector/core.py:250,253` 把 `collector.task` 与 `request.task` 都从 `entry.task` 派生（§0 已贴）。该测试**当前未** import `get_rollout_family_entry`（文件 import 段只到 `vrl.rollouts.collector` / `vrl.trajectory`），需补一个 import。`family` 字面量先留（family 本身就是入参 `"janus_pro_r1"`，是测试自己控制的输入，echo 回来合理；要派生也可 `== entry.family`，但非本主题重点）。

## 2. 落地方案

canonical 派生模式（与同文件 `runtime_builder` / `executor_cls` 断言同形）：**取已绑定的 `entry`，断言 `== entry.task`**，registry 永远是唯一真源。

### A. `test_runtime_inputs.py` 主参数化测试 —— 删 `expected_task` 列

BEFORE（`test_runtime_inputs.py:22-23,77-82,109`）：

```python
@pytest.mark.parametrize(
    ("experiment", "family", "expected_task", "expected_gatherer"),
    [
        ("diffusion/sd3_5/online_grpo_ocr", "sd3_5", "t2i", DiffusionChunkGatherer),
        ("diffusion/wan_2_1/online_grpo_ocr", "wan_2_1", "t2v", DiffusionChunkGatherer),
        ... (每行第三列都是手抄 task) ...
    ],
)
def test_rollout_runtime_inputs_are_serializable_and_registry_backed(
    experiment: str,
    family: str,
    expected_task: str,
    expected_gatherer: type,
) -> None:
    ...
    entry = get_rollout_family_entry(family)
    ...
    assert inputs.launch_contract.task == expected_task
```

AFTER —— 从 parametrize 删掉 `expected_task` 列、删掉同名形参，断言改对 `entry.task`：

```python
@pytest.mark.parametrize(
    ("experiment", "family", "expected_gatherer"),
    [
        ("diffusion/sd3_5/online_grpo_ocr", "sd3_5", DiffusionChunkGatherer),
        ("diffusion/wan_2_1/online_grpo_ocr", "wan_2_1", DiffusionChunkGatherer),
        ("diffusion/wan_2_1/online_grpo_kling_video_reward", "wan_2_1", DiffusionChunkGatherer),
        ("diffusion/wan_2_1/online_grpo_physics_i2v", "wan_2_1_i2v", DiffusionChunkGatherer),
        ("diffusion/cosmos_predict2/online_grpo_kling_video_reward", "cosmos-predict2", DiffusionChunkGatherer),
        ("diffusion/anima_preview3/online_grpo_aesthetic", "cosmos-predict2-anima", DiffusionChunkGatherer),
        ("diffusion/anima_preview3/online_grpo_aesthetic_nsfw_safety", "cosmos-predict2-anima", DiffusionChunkGatherer),
        ("ar/janus_pro/online_grpo_ocr", "janus_pro", JanusProChunkGatherer),
        ("ar/janus_pro/online_r1_grpo_ocr", "janus_pro_r1", JanusProR1ChunkGatherer),
        ("ar/nextstep_1/online_grpo_ocr", "nextstep_1", NextStep1ChunkGatherer),
    ],
)
def test_rollout_runtime_inputs_are_serializable_and_registry_backed(
    experiment: str,
    family: str,
    expected_gatherer: type,
) -> None:
    ...
    entry = get_rollout_family_entry(family)
    ...
    assert inputs.launch_contract.task == entry.task   # registry is the single source of truth
    assert inputs.launch_contract.runtime_builder == entry.runtime_builder
    assert inputs.launch_contract.executor_cls == entry.executor_cls
```

注意：`expected_gatherer` 一列**保留不动**（type 断言 `isinstance(inputs.gatherer, expected_gatherer)`，与 task 不同形状，非本 sprint 主题）。

### B. `test_runtime_inputs.py::test_wan_i2v_...` —— 单测里的字面 `"i2v"`

BEFORE（`test_runtime_inputs.py:246-248`）：

```python
assert inputs.launch_contract.family == "wan_2_1_i2v"
assert inputs.launch_contract.task == "i2v"
```

AFTER —— 取 entry 后对 `entry.task` 断言：

```python
entry = get_rollout_family_entry("wan_2_1_i2v")
assert inputs.launch_contract.family == entry.family
assert inputs.launch_contract.task == entry.task
```

（`get_rollout_family_entry` 已在该文件 import，无需新增 import。）

### C. `test_janus_pro_r1_wiring.py` —— 绑定 entry，collector/request 双断言改派生

补 import（文件顶部 import 段，紧随 `from vrl.rollouts.collector import build_rollout_collector`）：

```python
from vrl.rollouts.families import get_rollout_family_entry
```

BEFORE（`test_janus_pro_r1_wiring.py:84-87`）：

```python
assert collector.family == "janus_pro_r1"
assert collector.task == "ar_t2i_r1"
assert plan.request.family == "janus_pro_r1"
assert plan.request.task == "ar_t2i_r1"
```

AFTER —— bind entry 一次，task 两处都对 `entry.task`：

```python
entry = get_rollout_family_entry("janus_pro_r1")
assert collector.family == "janus_pro_r1"
assert collector.task == entry.task
assert plan.request.family == "janus_pro_r1"
assert plan.request.task == entry.task
```

`family == "janus_pro_r1"` 字面量保留：family 是测试自己传给 `build_rollout_collector("janus_pro_r1", ...)` 的输入，echo 回来是合法的输入-透传断言，不是冻结的 registry 副本。

## 3. 验证（finishing criteria）

- `grep -rn 'expected_task' tests/rollouts/runtime/test_runtime_inputs.py` 零命中（参数列与形参均已删）。
- `grep -rn '"ar_t2i_r1"\|"ar_t2i"\|"t2i"\|"t2v"\|"i2v"\|"v2w"\|"t2w"' tests/rollouts/runtime/test_runtime_inputs.py tests/rollouts/runtime/test_janus_pro_r1_wiring.py` 不再命中 **task 断言** 行（允许 `family`/experiment 路径里出现的无关子串，逐条 confirm 不是 `launch_contract.task ==` / `collector.task ==` / `request.task ==` 的字面右值）。
- `grep -rn 'entry.task' tests/rollouts/runtime/test_runtime_inputs.py tests/rollouts/runtime/test_janus_pro_r1_wiring.py` 命中改后的派生断言。
- `pytest tests/rollouts/runtime/test_runtime_inputs.py tests/rollouts/runtime/test_janus_pro_r1_wiring.py -q` 全绿。
- 反向证明派生有效：临时把 `vrl/rollouts/families/registry.py:288` 的 `task="ar_t2i_r1"` 改成别的串，跑上面两测试应**仍全绿**（被测代码与断言一起跟着 registry 走）；改前的字面量版本会 red。验证完 revert registry。
- `pytest tests/rollouts/ -q` 无回归。

## 4. 非目标 / Non-Goals

- **不动 `expected_gatherer`**：那是 type 断言（`isinstance`），registry 侧存的是 `gatherer.import_path` dotted-path 字符串，形状不同，需另案；本 sprint 只统一 `task`。
- **不动 `family == "..."` 字面量**：family 是测试自己传入 builder 的输入参数，echo 回来是输入-透传断言，不是 registry 冻结副本。要进一步派生（`== entry.family`）可顺手，但不是本主题的 root cause。
- **不动其它 frozen-snapshot 主题**：本仓库审计还有 `registered_rollout_families` 字面 key 列表、alias map 手抄、literal-config 断言、目录 listing 快照等多条，各归各的 sprint（见 References 的 sibling links），本 sprint 不越界。
- **不改 registry 本身**：registry 的 `task` 词表是真源，保持不动。

## References

- `tests/rollouts/runtime/test_runtime_inputs.py:22-23,77-82,109,111-112,246-248`
- `tests/rollouts/runtime/test_janus_pro_r1_wiring.py:84-87`（+ import 段）
- `vrl/rollouts/families/registry.py:62-77`（`RolloutFamilyEntry.task` 定义）、`:134-315`（各 family `task=`）、`:361-364`（`registered_rollout_families`）
- `vrl/rollouts/collector/core.py:250,253`（`collector.task` / `request.task` 从 `entry.task` 派生）
- 关联：[[SPRINT_test_registry-family-list-snapshot]]（`registered_rollout_families() == tuple(FAMILY_REGISTRY)`，alias map 手抄，同源 registry 冻结副本）、[[SPRINT_test_literal-config-assertions]]（configs 是声明，勿断言字面值）
