# SPRINT(auto): vrl/models/ar/janus_pro/runtime.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/models/ar/janus_pro/runtime.py` (1246 LOC)
角色判定: core
结论: improve

## 0. 一句话
一个 1246 行的文件塞进了三条分别带独立模块级 docstring 的管线（runtime bundle 构建 / Janus-Pro executor / Janus-Pro-R1 executor），用裸三引号字符串当章节分隔，并复制了一份 `_call_with_supported_kwargs`；命名 "runtime" 与同包 `RuntimeBundle`/`RuntimeBuildSpec` 概念有轻微碰撞但属跨家族一致命名，保留。

## 1. 现状（读代码得出）
文件混了三段职责，且用 **module-level 三引号字符串** 当 section divider（这些字符串既不是任何对象的 docstring，也不赋值，纯粹是注释滥用）：

```python
# line 275 —— 既不是函数也不是类的 docstring，悬空字符串
"""Janus-Pro AR text-to-image pipeline executor.
...
"""

# line 830 —— 同样悬空
"""Janus-Pro-R1 AR text-to-image pipeline executor.
...
"""
```

内部又复制了一份与 `ARDecodeLoop` 完全相同的 helper：

```python
# runtime.py:245
def _call_with_supported_kwargs(fn, *args, **kwargs):
    try:
        signature = inspect.signature(fn)
    ...
```

对照 `vrl/generation/ar/decode_loop.py:520` 的 `ARDecodeLoop._call_with_supported_kwargs`，二者实现逐字相同。

## 2. 质疑点 / 改进机会
- **悬空模块级字符串当章节分隔 (runtime.py:275, 830)**：Python 只把模块/类/函数**第一条**语句的字符串当 docstring，这两段三引号既不在文件头也不在任何定义内，是死字符串字面量。它们透露的真问题是：这文件其实是三个文件被物理拼在一起（builder 段 + Pipeline executor 段 + R1 executor 段），靠注释假装分章。这是 god-file 信号。
- **`_call_with_supported_kwargs` 重复 (runtime.py:245 vs decode_loop.py:520)**：逐字复制，两处维护同一逻辑。runtime.py 仅在 R1 路径（line 889/985）调用它来兼容 `generate_with_refine` 的可选 kwargs。应复用单一来源。
  ```
  $ grep -rn "_call_with_supported_kwargs" vrl/ --include=*.py
  vrl/generation/ar/decode_loop.py:520:    def _call_with_supported_kwargs(...)
  vrl/models/ar/janus_pro/runtime.py:245:def _call_with_supported_kwargs(...)
  ```
- **职责过载 / 三管线同居**：`build_*_runtime_bundle` + `extract_*_runtime_spec`（spec/装配层）、`JanusProPipelineExecutor` + `JanusProChunkGatherer`（基础 T2I 管线）、`JanusProR1PipelineExecutor` + `JanusProR1ChunkGatherer` + 4 个 R1 module-level helper（R1 三段管线）三者互不依赖运行期状态，仅共享少量 import。1246 行单文件不利于定位与 review。

注意：命名 "runtime" 这个词本身在本仓是**跨家族约定**（`nextstep_1/runtime.py` 同名同构，见 §4），不按 §3 命名规则 flag。

## 3. 建议动作
- 删掉 runtime.py:275 与 830 的两段悬空三引号字符串；把其中真正有价值的"边界契约"说明（"MUST NOT import rollouts / MUST NOT compute reward" 等）移成对应 executor 类的 class docstring 或正常 `#` 注释。
- 删掉 runtime.py:245 的本地 `_call_with_supported_kwargs`，改为复用 decode_loop 的实现：要么把它提成 `vrl/generation/ar/` 下的 module-level 函数供两处 import，要么直接 `from vrl.generation.ar.decode_loop import ...`（若它被提为模块函数）。不要在两处各留一份。
- 拆 god-file（与 `nextstep_1/runtime.py` 一起做，保持跨家族对称）：建议按现有三段切成 `runtime.py`（仅 bundle builders + spec extractor）、`executor.py`（base T2I executor + gatherer）、`executor_r1.py`（R1 executor + gatherer + R1 helper）。注册表里的 `runtime_builder="...janus_pro.runtime:build_janus_pro_runtime_bundle"` 字符串路径只要 builder 留在 runtime.py 就不受影响（已 grep `vrl/rollouts/families/registry.py:229/251`）。**拆分须跨家族一致**，否则不如不拆——见 §4。

## 4. 不动什么 / 为什么不是过度清理
- 文件名 `runtime.py` 及 `build_*_runtime_bundle` / `extract_*_runtime_spec` 命名是**跨家族统一形状**：`vrl/models/ar/nextstep_1/runtime.py` 结构逐项对称（`build_nextstep_1_runtime_bundle` / `build_nextstep_1_replay_runtime_bundle` / `extract_nextstep_1_runtime_spec` / `_nextstep_1_config_from_runtime_spec` / `*PipelineExecutor` / `*ChunkGatherer`）。AGENTS.md "consistency over cleanup" 明确要求保留这种 grepability，不要单独给 janus 改名或单独拆它一个家族。任何拆分/重命名必须 nextstep_1 同步做，否则维持现状。
- `_dtype_to_config_string` / `_optional_int` 等小 helper 是 spec 装配的局部工具，保留。
- `JanusProR1PipelineExecutor` 继承 `JanusProPipelineExecutor` 复用 tokenize/embed/runner，是合理复用，不要拍平。
- `build_janus_pro_replay_runtime_bundle` 虽未进 `__init__.py` 的 `__all__`，但被 `vrl/scripts/ar/janus_pro/train.py:88` 直接 import 使用，非死代码。

## 5. 验证
- 删悬空字符串 + 去重 helper 后：`ruff check vrl/models/ar/janus_pro/runtime.py`（应消除任何 dead-code / unused 提示）。
- `python -c "import vrl.models.ar.janus_pro.runtime"` 正常 import。
- 注册表路径仍有效：`grep -rn "janus_pro.runtime:" vrl/rollouts/families/registry.py` 指向的符号仍存在于 runtime.py。
- 跑 `pytest -k "janus and r1"` 验证 R1 路径（依赖被去重的 `_call_with_supported_kwargs`）。
