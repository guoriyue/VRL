# SPRINT：Diffusion Rollout Denoise 性能优化

状态：proposed。

依赖：

```text
SPRINT_trajectory_runtime_performance_optimization.md
```

## 核心结论

这份 sprint 只处理 diffusion-specific 优化：

```text
denoise loop preallocation
diffusion-specific counter population
diffusion adoption of generic storage / slice helpers
diffusion replay timestep slicing
SD3.5 OCR profile gate
```

通用的 trajectory storage policy、CPU/GPU 搬运、reward artifact 生命周期、profiler counters 不在这里重复设计，统一放到 trajectory runtime sprint。

diffusion 不能套 AR KV cache。每个 denoise step 的 latent、timestep、conditioning 都参与 transformer forward；跨 timestep 复用 attention KV 会改变模型语义。

## 当前 diffusion 路径

```text
vrl/engine/diffusion/executor.py
  forward_plan(...)
  forward_chunk_plan(...)
  gather_chunks(...)

vrl/engine/diffusion/layout.py
  DiffusionRequestLayout
  parse_spec(...)
  repeat_batch(...)
  ordered_chunks(...)

vrl/engine/diffusion/denoise.py
  run_diffusion_denoise_chunk(...)

vrl/engine/diffusion/gather.py
  gather_diffusion_chunks(...)
  build_diffusion_output_batch(...)

vrl/models/diffusion/model_base.py
  DiffusionModelBase.replay_forward(...)

vrl/engine/trajectory/builders.py
  build_diffusion_trajectory(...)

vrl/rollouts/evaluators/diffusion/flow_matching.py
  trainer replay signal extraction
```

当前 diffusion-specific 风险：

- `run_diffusion_denoise_chunk(...)` 每步 append 到 Python list，最后 `torch.stack(...)`。
- observations / actions / log_probs / timesteps / kl / decoded video 同时存在，diffusion peak memory 高。
- evaluator 每次 replay 只用一个 timestep，但 diffusion trajectory 保存整条 `[B, T, ...]`。
- diffusion counters 还不能回答 latent/action/video/replay tensors 分别占多少。

## 非目标

本 sprint 不做：

- 不实现通用 `TrajectoryStoragePolicy`；只消费它。
- 不设计通用 reward artifact lifetime；只验证 diffusion path 接入后不退化。
- 不给 diffusion 加 KV cache。
- 不改变 SDE / flow-matching logprob 数学。
- 不改变 reward、advantage、GRPO 语义。
- 不删除 SD3.5 OCR 当前可跑路径。
- 不默认 CPU offload。

## 设计目标

### 1. Denoise loop 预分配

当前主路径：

```text
obs_steps.append(...)
act_steps.append(...)
lp_steps.append(...)
kl_steps.append(...)
t_steps.append(...)
torch.stack(...)
```

目标：

```text
preallocate observations/actions/log_probs/timesteps/kl
step_idx 直接写入对应 slice
```

要求：

- shape / dtype / device 与当前行为一致。
- preallocated buffer 在 `run_diffusion_denoise_chunk(...)` 内创建，调用点在 `model.prepare_sampling(...)` 返回之后、denoise loop 开始之前。
- 新增 helper 放在 `vrl/engine/diffusion/denoise.py`：

```python
@dataclass(slots=True)
class DiffusionDenoiseBuffers:
    observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    timesteps: torch.Tensor
    kl: torch.Tensor


def preallocate_denoise_buffers(
    *,
    state: Any,
    config: DiffusionDenoiseConfig,
) -> DiffusionDenoiseBuffers:
    ...
```

call site：

```text
state = model.prepare_sampling(...)
buffers = preallocate_denoise_buffers(state=state, config=config)
for step_idx in range(len(state.timesteps)):
    ...
    buffers.observations[:, step_idx].copy_(latents_ori.detach())
```

- `observations` / `actions` shape 来自 `state.latents`。
- `timesteps` shape 是 `[chunk_batch, num_steps]`。
- `log_probs` shape 是 `[chunk_batch, num_steps]`，dtype 默认 `torch.float32`，device 跟 `state.latents.device`。
- `return_kl=False` 先保持当前 schema：`DiffusionChunkResult.kl` 仍是 `[chunk_batch, num_steps]` zero tensor，避免让 layout / gather / trajectory / batch builder 全部 optional 化。
- `return_kl=True` 时写入真实 `[chunk_batch, num_steps]` KL tensor。
- optional KL 不是本 sprint 的第一阶段目标。只有 byte counters 证明 KL tensor 是实质内存问题时，另开 follow-up 把 `kl` 改成 optional。
- profiler labels 使用 diffusion 语义，不能继续使用 AR KV cache 名字：

```text
engine.prepare_sampling
engine.prompt_encode
engine.denoise_step
engine.denoise_forward
engine.scheduler_step
engine.latent_write
engine.trajectory_buffer_write
engine.decode_latents
```

当前代码里 `engine.cache_read` / `engine.cache_write` 是 diffusion path 的错误命名；`engine.vq_decode` 对 SD3/Wan/Cosmos 这类 latent diffusion 也不准确。Phase 2 必须把它们替换掉，不能把 AR/VQ label 带进 diffusion profile 结果。

### 2. Diffusion byte counters

在通用 `GenerationMetrics.engine_counters` 中填 diffusion counters：

```text
diffusion_num_denoise_steps
diffusion_sample_batch_size
diffusion_observation_bytes
diffusion_action_bytes
diffusion_logprob_bytes
diffusion_timestep_bytes
diffusion_kl_bytes
diffusion_replay_tensor_bytes
diffusion_video_bytes
diffusion_storage_device
diffusion_storage_dtype
```

这些 counters 只用于 profile，不参与算法。

### 3. Diffusion storage policy first adoption

消费通用 trajectory runtime sprint 提供的 storage policy：

```yaml
rollout:
  trajectory_storage:
    device: preserve
    dtype: preserve
```

diffusion-specific 要求：

- 默认 preserve，不改变 SD3.5 OCR 行为。
- CPU offload 只在显式配置时启用。
- dtype 改变必须有 numerical tolerance test。
- old_log_prob / mask / distribution source of truth 不变。

### 4. Diffusion replay timestep slicing

`FlowMatchingEvaluator` 每次只需要：

```text
observations[:, timestep_idx]
actions[:, timestep_idx]
old_log_prob[:, timestep_idx]
mask[:, timestep_idx]
```

目标：

- 先确认当前路径是否真的在 copy；不能把普通 tensor view 当成性能 bug。
- replay forward 不触发整条 trajectory 的额外 device copy。
- resolver 能返回当前 step slice。
- `ReplayResult` 仍只带 current payload，例如 `noise_pred`。

当前代码事实：

```text
FlowMatchingEvaluator.evaluate(...)
  observations = batch.observations[:, timestep_idx]
  actions = batch.actions[:, timestep_idx]

DiffusionModelBase.replay_forward(...)
  trajectory_replay_tensor_dict(batch, "denoise")
  batch.observations[:, timestep_idx]
```

如果 `batch.observations[:, timestep_idx]` 已经在目标 device 上，这通常只是 view，不是 copy。真正要防的是：

```text
resolver / storage policy 先把整条 [B, T, ...] trajectory tensor .to(device)
然后 evaluator / model 再 slice 当前 timestep
```

因此 Phase 4 的实现标准不是“无条件重写 slicing”，而是：

- 新增或扩展 step-slice resolver，让带 `timestep` axis 的 replay tensor 先 slice 后按需 move。
- sample-only prompt embeds 等无 `timestep` axis 的 replay tensor 保持原样。
- tests 用 sentinel tensor / fake tensor 断言不会对 unsliced `[B, T, ...]` 调 `.to(device)`。
- 如果当前 direct `batch.observations[:, timestep_idx]` 已经是 view，则保留它，只加 regression guard。

## 需要编辑的文件

### Diffusion engine

```text
vrl/engine/diffusion/denoise.py
vrl/engine/diffusion/gather.py
vrl/engine/diffusion/layout.py
vrl/engine/diffusion/executor.py
vrl/engine/diffusion/spec.py
vrl/engine/diffusion/__init__.py
```

目标：

- `DiffusionDenoiseConfig` 继续只描述 denoise runtime knobs；storage policy 不进入 denoise config。
- `run_diffusion_denoise_chunk(...)` 主路径改为预分配。
- `preallocate_denoise_buffers(...)` 在 `run_diffusion_denoise_chunk(...)` 中 `prepare_sampling` 之后调用。
- `return_kl=False` 保持 zero KL tensor schema，只避免 Python list append + final stack 成为主路径。
- `build_diffusion_output_batch(...)` 在 `vrl/engine/diffusion/gather.py` 写 diffusion counters。
- `DiffusionRequestLayout.parse_spec(...)` 不读取 storage policy；storage policy 是 trajectory runtime policy，不是 sampling/model 语义。

### Diffusion model / evaluator

```text
vrl/models/diffusion/model_base.py
vrl/rollouts/evaluators/diffusion/flow_matching.py
```

目标：

- `DiffusionModelBase.replay_forward(...)` 只恢复当前 step 所需 state。
- evaluator 不回退 legacy extras。
- step-level old logprob / mask 继续从 `TrajectoryBatch` resolver 获取。
- 确认 direct `batch.observations[:, timestep_idx]` 是 view 时不做多余重构。
- 只有发现 resolver/storage path 先整条 move 再 slice 时，才改为 step-slice resolver。

### Diffusion trajectory builder

```text
vrl/engine/trajectory/builders.py
vrl/engine/trajectory/resolver.py
```

目标：

- `build_diffusion_trajectory(...)` 接入通用 storage policy。
- 添加或使用 step-slice resolver。
- 保持 `denoise.kl` tensor schema；optional KL 不进入本 sprint 第一阶段。
- 不改变 `TrajectoryBatch` schema。

### Configs / profile

```text
configs/base/rollout/diffusion.yaml
configs/experiment/sd3_5_ocr_grpo.yaml
configs/profile/*
configs/profiling/torch_profiler.yaml
README.md
```

目标：

- 默认 preserve。
- profile recipe 能输出 diffusion counters。
- README 只记录命令和结果位置。

## Tests

新增：

```text
tests/engine/diffusion/test_denoise_preallocation.py
tests/engine/diffusion/test_diffusion_counters.py
tests/engine/diffusion/test_diffusion_storage_policy_adoption.py
tests/rollouts/test_diffusion_replay_slice_access.py
```

编辑：

```text
tests/models/test_diffusion_model_base.py
tests/trainers/test_memory_guards.py
tests/rollouts/test_runtime_inputs.py
```

测试要求：

- preallocated output 与旧 list/stack 行为 shape、dtype、device 一致。
- `return_kl=False` 输出 zero KL tensor，shape 与旧 path 一致。
- `return_kl=True` 输出真实 KL tensor，shape 与旧 path 一致。
- diffusion counters 可 JSON 序列化。
- storage policy 默认 preserve 不改变当前 tests。
- CPU offload 只在显式配置时启用。
- evaluator 只读取当前 timestep slice；如果 tensor 已在目标 device 上，slice 必须保持 view 行为。
- step-slice resolver tests 覆盖“slice first, then optional move”，禁止 unsliced `[B, T, ...]` `.to(device)`。
- SD3.5 OCR baseline gate 通过。

## 实施阶段

### Phase 1：接入通用 metrics / storage contract

完成标准：

- diffusion runtime 能读取通用 storage policy。
- `GenerationMetrics.engine_counters` 有 diffusion counters。
- 默认行为不变。

### Phase 2：denoise loop 预分配

完成标准：

- `run_diffusion_denoise_chunk(...)` 不再以 list append + final stack 作为主路径。
- `preallocate_denoise_buffers(...)` 在 `prepare_sampling` 后、loop 前创建 buffer。
- `return_kl=False` 保持 zero KL tensor schema，但不再通过逐步 append + stack 生成。
- `engine.cache_read` / `engine.cache_write` / `engine.vq_decode` 不再出现在 diffusion profiler labels。
- fake denoise tests 覆盖 num_steps、batch、latent shape。
- profiler labels 不退化。

### Phase 3：storage policy first adoption

完成标准：

- `build_diffusion_trajectory(...)` 支持通用 storage policy。
- preserve path 与现有 SD3.5 OCR 行为一致。
- 显式 CPU path 有 tests。

### Phase 4：replay timestep slice

完成标准：

- 先用测试确认当前 `observations[:, timestep_idx]` 是 view 时不 copy。
- 如果 hidden copy 来自 resolver/storage policy，则修复为“先 slice 当前 timestep，再 move 当前 slice”。
- `FlowMatchingEvaluator` / `DiffusionModelBase.replay_forward(...)` 只读取当前 step 所需 tensor。
- old_log_prob / mask / distribution 仍来自 trajectory resolver。
- 不引入 extras fallback。

### Phase 5：profile 对比

完成标准：

- 跑 SD3.5 OCR baseline profile。
- 跑 preallocation + counters profile。
- 如启用 CPU offload，单独跑 offload profile。
- 对比：

```text
collect.engine_generate wall-clock
peak_memory_mb
host RSS
GPU memory
diffusion_*_bytes counters
cudaLaunchKernel count
cudaMemcpy / device synchronize events
```

## 验收命令

```bash
pytest tests/engine/diffusion/test_denoise_preallocation.py \
  tests/engine/diffusion/test_diffusion_counters.py \
  tests/engine/diffusion/test_diffusion_storage_policy_adoption.py \
  tests/rollouts/test_diffusion_replay_slice_access.py \
  tests/models/test_diffusion_model_base.py

pytest tests/models tests/engine tests/rollouts tests/trainers/test_memory_guards.py
python -m compileall vrl tests
git diff --check
```

Profile：

```bash
python -m vrl.scripts.train --config experiment/sd3_5_ocr_grpo \
  trainer.profile=true
```

## 风险与处理

- preallocation shape 容易错：fake denoise tests 覆盖 latent/image/video shape。
- `return_kl=False` 的 zero KL tensor 仍占少量内存：先用 counters 量化，不在本 sprint 第一阶段牵动 schema。
- CPU offload 可能降速：默认 preserve，offload 单独 profile。
- dtype 降级可能改变 replay logprob：默认 preserve，显式 dtype policy 必须有 tolerance test。
- reward artifact lifetime 在通用 sprint 中处理；本 sprint 只验证 SD3.5 OCR 不退化。

## 最终完成标准

完成后必须能回答：

- diffusion rollout peak memory 主要花在 observations/actions/video/replay tensors 哪一项。
- denoise preallocation 是否降低 allocation 和 peak memory。
- `return_kl=False` 的 KL tensor 是否只是可接受的小项，是否值得后续 optional 化。
- storage policy preserve / cpu 对 SD3.5 OCR 的影响。
- replay timestep slicing 是否存在真实整条 trajectory 搬运；如果存在，是否已经变成先 slice 后 move。
- SD3.5 OCR 当前工作路径是否仍然通过。
