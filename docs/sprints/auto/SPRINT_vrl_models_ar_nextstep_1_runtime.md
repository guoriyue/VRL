# SPRINT(auto): vrl/models/ar/nextstep_1/runtime.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/models/ar/nextstep_1/runtime.py` (753 LOC)
角色判定: core
结论: improve

## 0. 一句话
核心 runtime（builders + executor + gatherer 都在此，且被 registry 引用），整体职责分布是刻意对齐 `janus_pro` 的家族约定、不该拆；唯一真实缺陷是文件中段有一个被求值后丢弃的裸三引号字符串（伪 docstring），应改成正式块注释。

## 1. 现状（读代码得出）
文件提供 NextStep-1 rollout/replay 的全部 runtime 入口，全部被外部引用：
- `vrl/rollouts/families/registry.py:278-286` 注册 `NextStep1PipelineExecutor` / `build_nextstep_1_runtime_bundle` / `extract_nextstep_1_runtime_spec` / `NextStep1ChunkGatherer`。
- `vrl/scripts/ar/nextstep_1/train.py:31-47` 调 `build_nextstep_1_runtime_bundle` / `build_nextstep_1_replay_runtime_bundle` / `extract_nextstep_1_runtime_spec`。

文件第 243-273 行是一个**裸字符串语句**（不是任何函数/类/模块的 docstring，因为它出现在 `_optional_int` 定义之后、`NextStep1ARChunkResult` dataclass 之前）：

```python
"""NextStep-1 pipeline executor.

Owns the continuous-token autoregressive sampling loop ...
Determinism contract: same prompts + same generator state ⇒ ...
"""
```

Python 会构造这个 str 然后立即丢弃 —— 它对 `help()`/`__doc__`/IDE 全部不可见，只是看起来像 section 文档。

## 2. 质疑点 / 改进机会
1. **伪 docstring 裸语句（清晰度缺陷）**：runtime.py:243-273 的三引号串是 no-op 表达式语句，既不是 docstring 也不是注释，容易让读者误以为是 `NextStep1ARChunkResult` 或模块的文档。内容本身（determinism contract、boundary "MUST NOT import rollouts / MUST NOT compute reward"）是有价值的设计说明，应保留为正式块注释或挂到 `NextStep1PipelineExecutor` 的类 docstring 上。

## 3. 建议动作
- 把 runtime.py:243-273 的裸字符串改为 `#` 块注释（或并入 `NextStep1PipelineExecutor` 类 docstring 的开头），删除其作为独立表达式语句的存在。注意：`janus_pro/runtime.py:275` 有完全相同的模式，若要修建议两家一起改以维持一致性，但本 sprint 只针对 nextstep_1。

## 4. 不动什么 / 为什么不是过度清理
- **不要拆这个文件**。它把「runtime bundle builders + config spec 抽取 + pipeline executor + chunk gatherer」放在一个 runtime.py，与 `janus_pro/runtime.py`（同样 18 个 top-level def/class、~45KB、同一段中段字符串模式）完全同构。这是 AGENTS.md「跨家族统一形状 > LOC 缩减 / consistency over cleanup」明确要保护的对象，拆开反而破坏 grepability。
- `_dtype_to_config_string` / `_optional_int` 等私有 helper 是 `ConfigField` cast 边界，保留。
- `NEXTSTEP_1_FAMILY_CAPABILITY = ar_continuous_family_capability(...)`（runtime.py:43）是从工厂派生、非手抄常量，保留。
- 两个 builder（full vs replay bundle）对应 generation worker 与 trainer replay 两条真实路径，不合并。

## 5. 验证
- `ruff check vrl/models/ar/nextstep_1/runtime.py`（ruff 的 B018 useless-expression 可佐证裸字符串问题）。
- 改完 `python -c "import vrl.models.ar.nextstep_1.runtime"` 无报错。
- `grep -nE '^"""' vrl/models/ar/nextstep_1/runtime.py` 应只剩文件顶部 1 处模块 docstring。
