# SPRINT: Runtime Protocol fail-closed contracts（planned）

状态：**planned / CPU-only**。

## 根因

Python 的 `@runtime_checkable Protocol` 只检查属性存在，不检查调用签名。它也不是
concrete implementation base。当前两种误用会把扩展错误推迟到深层执行：

```python
class ChunkExecutorBase(GenerationChunkExecutor):
    ...

class RayGenerationRuntime(GenerationRuntime):
    ...
```

`ChunkExecutorBase` 没有自己的 `forward_chunk_plan`，因此继承 Protocol 中的
`...` 空桩。当前 `_require_chunked_executor()` 只检查 `callable`，这个不完整类会
通过并在运行时返回 `None`。CPU 反例已复现：

```text
incomplete_structural_match=True
inherited_stub_result=None
```

model 边界有同类缺口：

```python
if isinstance(value, ReplayModel):
    return cast("ReplayModel", value)
```

缺少 `request=` 或 `state_dict` 参数的同名方法仍会通过，直到 evaluator/weight sync
真正调用时才失败。

## 改动

### 1. Generation implementation ownership

- `ChunkExecutorBase` 改为 `ABC`，声明 abstract `forward_chunk_plan`；
- `forward_plan(..., plan: Any)` 收紧为已有公共 `EnginePlan`；
- concrete `RayGenerationRuntime` 不再 nominally 继承 `GenerationRuntime`；
- 保留 structural `isinstance(..., GenerationRuntime/GenerationChunkExecutor)` contract；
- 更新只使用 base 局部方法的 test fake，显式实现最小 `forward_chunk_plan`。

### 2. Model call-shape validation

`require_replay_model` / `require_runtime_model` 不比较注解或精确文本签名，而用
`inspect.signature(...).bind(...)` 验证生产实际调用：

```text
replay_forward(batch, timestep_idx)
replay_forward(batch, timestep_idx=...)
replay_forward(batch, request=...)
disable_adapter()
load_trainable_state(state_dict)
```

这允许 `*args/**kwargs` 与额外可选参数，同时拒绝“名字存在但调用不了”的实现。
registry-derived class matrix 通过 `__new__` 建立不加载权重的 bound method，再复用同一
production guard；不维护第二份手写 method list。

## 审计裁决

| Suspect | 裁决 | 原因 |
|---|---|---|
| `GenerationRuntime` / `GenerationChunkExecutor` Protocol | **KEEP** | collector/worker 的真实 public capability |
| concrete class nominal Protocol inheritance | **REMOVE** | 错误 implementation ownership，Protocol stub 可进入 MRO |
| `ChunkExecutorBase.forward_chunk_plan` | **FIX** | shared implementation base 的必需 extension point，应由 ABC fail early |
| `plan: Any` | **FIX** | 已有唯一公共 `EnginePlan`，开放 bag 没有消费者需求 |
| Replay/Runtime callable-only guard | **FIX** | live caller 需要明确 positional/keyword ABI |
| `ChunkResult = Any` | **KEEP** | family-owned opaque payload transport，具体 gatherer 拥有验证 |
| thin generation binding bases | **KEEP** | 跨 family 一致公共形状与 lazy registry boundary |

## True / false regressions

- 缺 `forward_chunk_plan` 的 subclass 无法实例化；
- 完整 fake 可实例化并 structural-match `GenerationChunkExecutor`；
- `GenerationChunkExecutor` / `GenerationRuntime` 不在 concrete MRO；
- `RayGenerationRuntime` 仍 structural-match `GenerationRuntime`；
- 合法 minimal ReplayModel/RuntimeModel 通过；
- 缺 `request`、缺 keyword `timestep_idx`、缺 positional timestep、缺
  `state_dict` 的 callable fake 在 boundary 被拒绝；
- 所有 registry replay/runtime class 通过实际 call-shape bind。

## 保持不变

- 不让 family model nominally 继承 Protocol；
- 不合并 Protocol 与 ABC；
- 不新增 universal validator registry；
- 不改变 Ray async lifecycle、version slots、gatherer registry 或 transition table；
- 不改当前用户正在编辑的 model family capability/config 文件。

## 验收

```bash
.venv/bin/python -m pytest -q \
  tests/models/interfaces \
  tests/generation/execution \
  tests/generation/bindings \
  tests/generation/ray \
  tests/rollouts/test_runtime_protocol_contract.py
```

没有热路径算法变化，不需要 benchmark 或 GPU。

## References

- `vrl/generation/protocols.py`
- `vrl/generation/execution/executor_base.py`
- `vrl/generation/execution/worker.py`
- `vrl/generation/ray/runtime.py`
- `vrl/models/interfaces/replay.py`
- `tests/models/interfaces/`
- https://docs.python.org/3/library/typing.html#typing.runtime_checkable
