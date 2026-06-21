# SPRINT: `ARAttention*Output.metrics` display 字段清理（planned）

状态：未开始（2026-06-20）。**整体 needs owner sign-off**（唯一读者是 backend 回归测试断言）。
范围：`vrl/nn/layers/attention/paged.py` 两个 attention output struct 的 `metrics` 字段。
来源：dead-dataclass-hunt + 手动验证（非测试读者为零）。承接 [[SPRINT_segment_signal_dead_field_cleanup]]。

## 0. Core Decision

`ARAttentionPrefillOutput.metrics`（`paged.py:62-71`）和 `ARAttentionStepOutput.metrics`（`paged.py:97-106`）在构造时算了值（attention backend 名等），但所有 behavior 代码只读 `.last_hidden` / `.sequence_states`——`grep` 证实 `vrl/models/ar/`、`vrl/generation/ar/` 内**零** `.metrics` 读取。唯一读者是两个 vLLM backend 测试断言：

```python
# tests/.../test_janus_vllm_paged_attention_backend.py:60
assert paged_prefill.metrics["attention_backend"]
```

按 AGENTS.md 规则属 display-only 死字段。**但**这两个断言是有意的 backend-selection 回归守卫（确认走了 paged 而非 torch_native 后端）的可能性存在——删字段会同时删守卫。**故整体需 owner 签字**：确认这些断言不是有意守卫，或把守卫改成读别的信号，再删。

## 1. 现状实锤

- 定义：`paged.py:62-71`（`ARAttentionPrefillOutput.metrics`）、`paged.py:97-106`（`ARAttentionStepOutput.metrics`）。
- 构造：`vrl/nn/modules/ar_decoder.py:96-104`（prefill）、`:128-132`（step）、`vrl/nn/modules/torch_attention.py:79-83`。
- behavior 读取：runner（`janus_pro/runner.py:88-93`、`nextstep_1/runner.py:99-120`）只读 `.last_hidden` / `.sequence_states`，`.metrics` 零命中。
- 测试读取：`test_janus_vllm_paged_attention_backend.py:60`、`test_nextstep_vllm_paged_attention_backend.py:65` 断言 `metrics["attention_backend"]`。

两字段同文件同模式，合一个 sprint。

## 2. 落地方案

**前置：owner 判断两个 backend 断言是否有意守卫。**
- **若非守卫（默认）**：删 `paged.py:62-71`、`:97-106` 两 `metrics` 字段，删 `ar_decoder.py:96-104,128-132`、`torch_attention.py:79-83` 构造，删两测试断言。
- **若是有意守卫**：保留字段，但改标注为 `# test-only backend-selection guard`，并在字段注释写明唯一消费者是哪两个测试——避免下轮审计再判死。

## 3. 验证
- 按决策后 `grep -rEn "\.metrics\b" vrl/nn/layers/attention/ vrl/nn/modules/` 收敛。
- `pytest tests/nn/ tests/models/ar/ -q` 全绿（含两个 backend 测试，删或改后）。
- AR rollout 冒烟（janus_pro / nextstep_1）输出一致。

## 4. 非目标 / Non-Goals
- 不动 `.last_hidden` / `.sequence_states`（活）。
- 不在确认断言性质前机械删——这是本 sprint 与纯机械删 sprint 的区别。

## References
- `vrl/nn/layers/attention/paged.py:62-71,97-106`
- `vrl/nn/modules/ar_decoder.py:96-104,128-132`、`vrl/nn/modules/torch_attention.py:79-83`
- `tests/nn/.../test_janus_vllm_paged_attention_backend.py:60`、`test_nextstep_vllm_paged_attention_backend.py:65`
- 关联：[[SPRINT_segment_signal_dead_field_cleanup]]
