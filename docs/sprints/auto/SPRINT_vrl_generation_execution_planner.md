# SPRINT(auto): vrl/generation/execution/planner.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/generation/execution/planner.py` (480 LOC)
角色判定: core
结论: improve

## 0. 一句话
文件本体（`EnginePlan` / `EnginePlanner` / `build_engine_plan` / `attach_engine_plan`）是真核心，但 `resolve_executor_capability` 和它的私有助手 `_merge_runtime_caps` 是一对无人调用的死代码，应删除。

## 1. 现状（读代码得出）
这是 engine planning 的契约中心，被 ray/ar/diffusion executor 和 janus_pro runtime 大量复用：
- `build_engine_plan(...)` — planner.py:395，被 diffusion/executor、janus_pro/runtime、scheduler 调用。
- `attach_engine_plan(...)` — planner.py:437，被三个 executor + janus_pro 调用。
- `EnginePlan` / `EnginePlanner` / `ExecutionStage` / `ResolvedAxis` — 都被实际消费（`EnginePlan` 还是 `GenerationOutput.engine_plan` 的类型，见 generation/types.py:155）。

但文件尾部有一对函数无人使用：
```python
def resolve_executor_capability(executor, request) -> FamilyCapability:  # planner.py:418
    method = getattr(executor, "capability", None)
    if callable(method):
        return _merge_runtime_caps(method(), executor, request)
    ...

def _merge_runtime_caps(value, executor, request) -> FamilyCapability:  # planner.py:454
    ...
    runtime_caps = getattr(executor, "runtime_caps", None)
    if isinstance(runtime_caps, Mapping):
        return capability.with_runtime_caps(runtime_caps)
```

## 2. 质疑点 / 改进机会
- 死代码：`resolve_executor_capability` 仅出现在自身定义、`__init__.py` re-export 和 `__all__`，无任何 call site（`grep -rn resolve_executor_capability vrl/ tests/` 只命中 planner.py:418/478 + execution/__init__.py:17/51）。
- `_merge_runtime_caps`（planner.py:454）只被 `resolve_executor_capability` 调用，随之一起死亡（`grep -rn _merge_runtime_caps vrl/` 只命中 planner.py:426/430/454）。
- 它想做的「合并 executor.runtime_caps 进 capability」这件事，活的路径已经在别处实现：worker.py:299-317 直接用 `self.capability.with_runtime_caps(runtime_caps)`；executor 则直接 `self.capability()` + `build_engine_plan(capability=...)`（diffusion/executor.py:208/226）。所以这对函数是被取代后遗留的旧入口。

## 3. 建议动作
- 删除 `resolve_executor_capability`（planner.py:418-434）与 `_merge_runtime_caps`（planner.py:454-468）。
- 从 `__all__`（planner.py:478）和 `vrl/generation/execution/__init__.py`（line 17 import + line 51 `__all__`）移除 `resolve_executor_capability`。
- grep 证据：`resolve_executor_capability` 在 vrl/ 与 tests/ 均无调用方；`_merge_runtime_caps` 仅被被删函数调用；`with_runtime_caps` 的活路径在 worker.py，不受影响。

## 4. 不动什么 / 为什么不是过度清理
- `EnginePlan` / `EnginePlanner` / `build_engine_plan` / `attach_engine_plan` / `ExecutionStage` / `ResolvedAxis` 全部保留，它们是被广泛 import 的真核心契约。
- `EnginePlanner` 里 `_build/_chunk_size/_resolved_axes/_axis_length/_execution_stages/_chunk_stages` 都是该类内聚的构建逻辑，不是薄转发，保留。
- 不要去动 `with_runtime_caps`（在 capabilities.py），那是活的。本次只删除被取代的旧 resolve 入口，符合 "fix root cause not symptom"（删掉死入口，不是给它加调用方）。

## 5. 验证
- 删除后 `grep -rn "resolve_executor_capability\|_merge_runtime_caps" .` 应为空。
- 跑 `pytest tests/generation/execution -q` 与 `pytest tests/generation -q`。
- `ruff check vrl/generation/execution/planner.py vrl/generation/execution/__init__.py`（确认无未使用 import / `__all__` 漂移）。
