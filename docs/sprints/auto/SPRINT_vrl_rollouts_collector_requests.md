# SPRINT(auto): vrl/rollouts/collector/requests.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/rollouts/collector/requests.py` (124 LOC)
角色判定: core
结论: improve

## 0. 一句话
核心适配器没问题，但模块尾部的 `RolloutEngineRequestBuilder = GenerationRequestBuilder` 别名只剩测试在用，是历史改名遗留的薄别名，应删并改测试引用真名。

## 1. 现状（读代码得出）
`GenerationRequestBuilder` 把 resolved rollout config + per-call kwargs 组装成 `GenerationRequest`，是真核心适配器。文件尾部有一个纯别名：
```python
RolloutEngineRequestBuilder = GenerationRequestBuilder   # line 117
```
并把它一起导出到 `__all__`（line 123）。

## 2. 质疑点 / 改进机会
- `RolloutEngineRequestBuilder` 没有任何边界价值：不是 protocol、不是 facade、不是 lazy import 边界，只是给同一个类起了第二个名字。grep 确认生产代码（vrl/ 内）零引用，唯一引用全部在测试：`tests/rollouts/test_engine_requests.py`、`test_collector_runtime.py`、`test_video_world_reference_metadata.py`。这是一次类改名后为兼容旧名留下的别名，按 AGENTS.md "thin / 死代码" 规则应清掉。

## 3. 建议动作
- 删除 requests.py:117 的别名和 `__all__` 里的 `"RolloutEngineRequestBuilder"`（line 123）。
- 把上述 3 个测试文件里的 `RolloutEngineRequestBuilder` 改为 `GenerationRequestBuilder`（直接 import 真名）。grep 确认这是仅有的 4 处引用，改完无残留。

## 4. 不动什么 / 为什么不是过度清理
- `GenerationRequestBuilder`、`CollectorRequest`（NamedTuple，承载 request + 派生 metadata 的轻量返回值）、`_sampling` / `_metadata` 都是真核心，保留。
- `_sampling` 里对 tuple->list 的归一化（line 90-93）是为了让 sampling payload 可序列化进 engine request，有真实理由，不动。

## 5. 验证
- `grep -rn "RolloutEngineRequestBuilder" --include=*.py` 改完应为空。
- `pytest tests/rollouts/test_engine_requests.py tests/rollouts/test_collector_runtime.py tests/rollouts/test_video_world_reference_metadata.py` 通过。
