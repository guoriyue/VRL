# SPRINT: Step-wise diffusion batching probe

状态：**PARKED — 实测证伪（2026-06-26，见 §0.5）**：SD3.5 1024² 全程 compute-bound，ms/sample 随 batch 全平 → 加 batch 不降低单样本成本；唯一可能有余量的 launch-bound 小 batch 区，第一解是直接调大 `sample_batch_size`（planner 已支持），不是先建跨请求调度器。下为原提案，留作记录。
> （原状态：planned / proof-gated，2026-06-26）这是一个低风险探针：把多个 diffusion 请求在同一个 denoise step 边界合 batch 执行，类似 LLM continuous batching 的调度思想，但不声称能复用 KV 或减少单样本 FLOPs。目标是验证：在 under-utilized rollout 场景里，跨请求 step-wise batching 是否能提高 GPU utilization，同时保持 trajectory 逐位一致。

## 0. 一句话

diffusion denoise step 之间不能并行：

```text
x_t -> x_{t-1} -> x_{t-2}
```

但不同请求如果处于相同 family / resolution / dtype / timestep bucket，就可以在 **同一个 step** 拼成更大的 batch：

```text
request A step 7
request B step 7
request C step 7
  -> one transformer forward with batch=A+B+C
```

这不是 PagedAttention；更像 diffusion 版 iteration-level batching。

## 0.5 实测判决（2026-06-26，SD3.5-medium 1024² / RTX 5090）—— 证伪

`vrl/scripts/perf/rollout_bottleneck_probe.py` 的 DiT batch-scaling（eager，同一 kernel 栈）：

```
batch 1→16:  ms/sample = 161 → 155 → 158 → 155 → 155   (平)
verdict:     全部 compute-bound (saturated)
```

ms/sample 全程平 = **同一 kernel 栈下，加 batch 不降低单样本成本**。这足以证伪 step-wise batching 在 SD3.5 1024² 这档的收益；它不等价于“没有任何 kernel 优化 headroom”（torch.compile / fp8 是另一条轴）。

唯一可能有余量的是真正小 batch/低分辨率/小模型的 launch-bound 区间,但那里的第一解是直接调大 `sample_batch_size`(planner 已支持,见 `planner.py:_chunk_size`),不是先建跨请求调度器。**本 sprint 维持 parked**:除非在某个具体 under-util 配置上 `rollout_bottleneck_probe` 实测 ms/sample 随 batch 明显下降,否则不解 park。证据见记忆 `project_rollout_bound_class_probe`。

## 1. Related work

- PagedAttention 和 LLM continuous batching 共同说明：系统收益常来自 iteration 边界的调度和大状态管理，而不是改模型数学：https://arxiv.org/abs/2309.06180
- vLLM-Omni 已经把 diffusion step execution 和 continuous batching 放到 scheduler/runner 层，文档明确说 base step-execution contract 不变，主要工作在 compatibility gating 和 batch packing：https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/diffusion_continuous_batching/
- Streamlined Inference 说明 video diffusion 的 peak memory 和计算可通过 feature slicing/operator grouping/step-level 重组优化，但其中 Step Rehash 属于近似路径；本 sprint 只做 exact batching 探针：https://openreview.net/forum?id=iNvXYQrkpi

## 2. 和现有 parked sprint 的关系

`docs/sprints/parked/SPRINT_cross_request_step_scheduler.md` 已经判断终局是 family-neutral StepScheduler，但当时被 parked，因为当前瓶颈未证明、且单请求负载下收益不成立。

本 sprint 只做更窄的 probe：

```text
不建终局引擎
不统一 AR/diffusion
不改 trainer
只证明 diffusion step-wise batching 在真实 rollout 负载下是否值得解 park
```

## 3. 设计

先从 diffusion executor 中抽出一个可恢复的一步执行接口：

```text
advance_denoise_step(state, step_idx, generator) -> StepResult
```

再做 scheduler-side grouping：

```text
DiffusionStepKey
  family
  task_variant
  height
  width
  frame_count
  dtype
  cfg_mode
  timestep_index_or_value
  policy_version
```

不能手写一个和 runtime capability 重复的 ALL_CAPS key 列表；`DiffusionStepKey` 要从现有 request layout / runtime capability / policy version 派生。

## 4. 正确性契约

- 单请求路径和 batched 路径在同 seed、同 request 下必须生成相同 observations/actions/old_log_prob。
- 不跨 policy version batch。
- 不跨 scheduler config batch。
- 不跨 `sde_window` stochastic/deterministic 边界 batch，除非 RNG stream 有严格 per-sample generator contract。
- 任何 CFG batching 必须保持 cond/uncond branch 顺序一致。

## 5. 执行顺序

1. 先做 measurement-only harness：记录每个 rollout chunk 的 family/resolution/frame_count/step_idx/GPU utilization，估计可合批机会。
2. 抽 `advance_denoise_step`，单请求路径仍走旧 loop，只加 parity 测试。
3. 写 in-process scheduler probe，只在一个 worker 内合并两个 synthetic same-shape requests。
4. 扩到 Ray worker 内多个 pending requests，不跨 actor。
5. 只有当 SM utilization 和 wall-clock 实测过关，再讨论解 park `SPRINT_cross_request_step_scheduler.md`。

## 6. 验收

- two-request probe 与串行单请求路径逐位一致。
- under-utilized 小 batch 场景 SM utilization 上升，wall-clock/request 下降。
- video high-memory 场景不允许通过 padding 强行合 batch；如果 padding waste 吃掉收益，直接记录负结果。
- RL trajectory parity 和 precision drift guard 仍绿。

## 7. 非目标

- 不做跨 denoise step 并行。
- 不做 attention KV cache。
- 不处理 shape 不兼容请求的 padding batching，除非 measurement 证明 padding waste 可接受。
- 不默认打开；这是 proof-gated scheduler probe。

## 8. 关键文件

- `vrl/generation/diffusion/executor.py`
- `vrl/generation/execution/planner.py`
- `vrl/generation/ray/executor.py`
- `vrl/generation/execution/worker.py`
- `docs/sprints/parked/SPRINT_cross_request_step_scheduler.md`
