# SPRINT(auto): vrl/generation/diffusion/executor.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/generation/diffusion/executor.py` (711 LOC)
角色判定: core
结论: improve

## 0. 一句话
这是 diffusion 家族共享的真核心基类（4 个 model family 继承它），但里面挂着一个从未被调用的 `run_denoise_chunk` 死方法，应删除。

## 1. 现状（读代码得出）
`DiffusionPipelineExecutorBase` 是 `GenerationRequest -> GenerationOutput` 的共享执行路径，被 5 个 family runtime 继承：

```
vrl/models/diffusion/cosmos/anima/runtime.py:207   class AnimaPipelineExecutor(DiffusionPipelineExecutorBase)
vrl/models/diffusion/sd3_5/runtime.py:213          class SD3_5PipelineExecutor(...)
vrl/models/diffusion/wan_2_1/runtime.py:204        class Wan_2_1PipelineExecutor(...)
vrl/models/diffusion/cosmos/predict2/runtime.py:225
vrl/models/diffusion/cosmos/predict2_5/runtime.py:185
```

主路径 `forward_chunk_plan`（executor.py:316）已经把 prepare -> denoise -> decode 串起来：它分别调用 `prepare_denoise_state` / `run_denoise_steps` / `decode_denoise_result`，而不是经过 `run_denoise_chunk`。

## 2. 质疑点 / 改进机会
- 死代码：`run_denoise_chunk`（executor.py:383-408）声称 "Run one fused diffusion sample chunk: prepare -> denoise -> decode"，但 grep 全仓库（`vrl/` + `tests/`）只有定义处一行命中：

  ```
  vrl/generation/diffusion/executor.py:383:    def run_denoise_chunk(
  ```

  没有任何调用方，连测试都不用它。它和 `forward_chunk_plan` 的三步内联调用功能重复，属于历史残留的便捷封装。

- ALL_CAPS / engine_counters 字符串（`"diffusion_num_denoise_steps"` 等，executor.py:557-570）不是 ALL_CAPS 常量，是内联 metrics schema key，和 `gather.py` / `metrics.py` 的同名 key 配套。这部分是真边界（rollout counter schema），不 flag。

## 3. 建议动作
删除 `run_denoise_chunk`（executor.py:383-408）。grep 已确认零调用方（仅定义行命中，见上）。删除后主路径 `forward_chunk_plan` 不受影响。

不要拆分这个基类：它虽长，但全是同一条 diffusion 执行管线的连续阶段 + family hook，不是 god-file（没有塞无关管线）。

## 4. 不动什么 / 为什么不是过度清理
- 保留所有 `@dataclass` 结果类型（`DiffusionDenoiseConfig` / `DiffusionDenoiseResult` / `DiffusionChunkResult` / `DiffusionDenoiseBuffers`）：它们是跨阶段传递的 typed 边界，被 gather/tests 引用。
- 保留 family hook 方法（`encode_prompt_for_chunk` / `build_chunk_encoded` / `build_prepare_kwargs` / `build_video_request_extra`）即使部分是薄默认实现 —— 它们是跨家族一致的 override 点，符合 AGENTS.md "跨家族一致性优于 LOC 缩减"。
- 保留 `preallocate_denoise_buffers`：被独立测试 `tests/generation/diffusion/test_denoise_preallocation.py` 覆盖，是真实预分配逻辑。

## 5. 验证
- `grep -rn "run_denoise_chunk" --include="*.py" .` 应只剩 0 命中（删除后）。
- 跑 `pytest tests/generation/diffusion/ tests/generation/execution/test_chunk_gatherer.py` 全绿。
- `ruff check vrl/generation/diffusion/executor.py`。
