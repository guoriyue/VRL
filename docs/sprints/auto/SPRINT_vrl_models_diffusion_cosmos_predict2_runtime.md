# SPRINT(auto): vrl/models/diffusion/cosmos/predict2/runtime.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/models/diffusion/cosmos/predict2/runtime.py` (337 LOC)
角色判定: registry-config
结论: improve

## 0. 一句话
文件本身职责合理（family runtime builders + rollout executor），但中间有一行 line 222 的“游离 module-level 字符串”是合并残留的死语句，应删除；其余结构属于跨家族刻意一致，不要动。

## 1. 现状（读代码得出）
这个文件是 cosmos-predict2 家族的 runtime 装配层，做三件事并被 registry 引用：

- `extract_cosmos_predict2_runtime_spec` / `build_cosmos_predict2_runtime_bundle` / `..._from_cfg`：从整份 RL cfg 切出 spec、按 `spec.backend_preference` 装配 backend model、应用 LoRA/compile/num_steps，返回 `RuntimeBundle`。
- `build_cosmos_predict2_replay_runtime_bundle` / `..._from_cfg`：装配只含 transformer+scheduler 的 trainer replay bundle。
- `CosmosPipelineExecutor(DiffusionPipelineExecutorBase)`：rollout 时的 Video2World 执行器。

这些符号都被真实引用：`vrl/rollouts/families/registry.py:162-171` 通过 `executor_cls=...:CosmosPipelineExecutor` 与 `runtime_builder=...:build_cosmos_predict2_runtime_bundle`、`...:extract_cosmos_predict2_runtime_spec` 装配；`vrl/scripts/diffusion/cosmos/train.py:66,78` 调用两个 `*_from_cfg`。

问题点在 line 222：

```python
"""Cosmos Predict2 Video2World diffusion pipeline executor."""


class CosmosPipelineExecutor(DiffusionPipelineExecutorBase):
```

这是一个 module-level 裸字符串表达式语句，位于文件中段。Python 里只有“模块第一条语句的字符串”才是 docstring，这一行不是 docstring，只是一个被求值后丢弃的无副作用表达式——是把 executor 从独立文件并入 runtime 时留下的旧文件头。

## 2. 质疑点 / 改进机会
- **死语句 / 合并残留**：`runtime.py:222` 的游离字符串没有任何作用（既非 docstring，也非赋值/常量），属于 dead statement。证据：line 1 已经有真正的模块 docstring `"""Cosmos Predict2 family runtime."""`，line 222 不在模块开头。
- **跨家族同样存在**：sibling `vrl/models/diffusion/cosmos/predict2_5/runtime.py:182` 有完全相同的游离字符串 `"""Cosmos Predict2.5 diffusion executor."""`。所以这是 cosmos 家族通用残留，不是 predict2 独有。修复应两家一起改以保持一致。
- **god-file 疑虑（判定为不拆）**：runtime builders 与 `CosmosPipelineExecutor` 同居一文件，看似职责过载，但 predict2_5 与 predict2 采用完全相同的“builders + executor 同文件”形状（`predict2_5/runtime.py:185` 同样把 executor 放在 runtime 里），属于刻意的跨家族一致。按 AGENTS.md “consistency over cleanup”，不拆此文件。

## 3. 建议动作
- 删除 `runtime.py:222` 的游离字符串语句。如果想保留“executor 段”的视觉分隔，用真正的注释（如 `# --- rollout executor ---`）而不是裸字符串。
- 同步修复 `vrl/models/diffusion/cosmos/predict2_5/runtime.py:182` 的同款游离字符串，保持两家族一致。
- 不新增/不删除任何公共符号。

## 4. 不动什么 / 为什么不是过度清理
- 不拆分 builders 与 executor：两家族同形状，拆开会破坏 grepability/一致性，违背 “consistency over cleanup”。
- 不动 `_MODEL_BY_BACKEND`（runtime.py:40）：它是 `backend -> "module:ClassName"` 的延迟导入派发表，提供 lazy import 边界（避免 runtime 顶层 import diffusers），是真边界，保留。
- 不动 `runtime_caps` / `metadata` 字典里的字面 key：它们是 `RuntimeBundle` 契约 key，属 schema 边界。
- 不动 `COSMOS_PREDICT2_FAMILY_CAPABILITY`（runtime.py:34）：由 `diffusion_family_capability(...)` derive 而来，不是手抄常量。

## 5. 验证
- `ruff check vrl/models/diffusion/cosmos/predict2/runtime.py vrl/models/diffusion/cosmos/predict2_5/runtime.py`（B018 useless-expression 应能识别游离字符串）。
- `python -c "import vrl.models.diffusion.cosmos.predict2.runtime"` 确认导入无误。
- `pytest tests/rollouts/test_family_registry.py tests/models/test_minimal_replay_runtime_wiring.py -q` 确认 registry 装配与 replay bundle 仍正常。
