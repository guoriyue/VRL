# SPRINT：Diffusion Generation 性能与内存优化

状态：deferred，但仍然有效。

依赖关系：

```text
Diffusion shared model-call support
  已落地到 vrl/models/diffusion/common 和各 family runner；不再保留独立 sprint 文件。

SPRINT_trajectory_runtime_performance_optimization.md
  提供通用 storage policy、reward artifact lifecycle、byte counter helper。
```

注意：denoise loop 预分配和 profile label 清理可以独立先做；storage policy / reward artifact 部分应消费通用 trajectory sprint，不在 diffusion sprint 里重复发明。

## 核心结论

这个 sprint 仍然有意义，但旧文件里的 `vrl/engine/diffusion/*` 路径已经过时。当前 diffusion rollout 路径在：

```text
vrl/generation/diffusion/executor.py
vrl/generation/diffusion/gather.py
vrl/generation/diffusion/layout.py
vrl/models/diffusion/base.py
vrl/trajectory/builders.py
vrl/trajectory/resolver.py
vrl/rollouts/evaluators/diffusion/sde_logprob.py
```

本 sprint 只处理 diffusion-specific 性能问题：

```text
denoise loop preallocation
diffusion-specific engine counters
diffusion profiler label cleanup
diffusion replay timestep slice regression guard
SD3.5 / Wan / Cosmos profile gate
```

通用 trajectory storage policy、reward artifact 生命周期、跨 family byte helper 不在这里重复设计。

## 当前代码事实

- `vrl/generation/diffusion/executor.py::run_denoise_steps(...)` 仍然每步 append Python list，最后 `torch.stack(...)`。
- diffusion path 仍使用 `engine.cache_read` / `engine.cache_write` profile label，但 diffusion denoise 不是 AR KV cache。
- decode label 仍叫 `engine.vq_decode`，对 latent diffusion 的 SD3 / Wan / Cosmos 不准确。
- `vrl/generation/diffusion/gather.py` 目前只写 `stage_durations_s`，没有 diffusion tensor byte counters。
- `vrl/models/diffusion/capabilities.py` 仍把 denoise stage 标成 `cache_read/cache_write`，并使用 `vq_decode` stage name。
- `vrl/trajectory/resolver.py` 已经支持 `LossUnit.axis_index`，所以 replay slice 的重点不是盲目改写所有 slicing，而是防止 storage/offload path 先整条 move 再 slice。

## 非目标

本 sprint 不做：

- 不给 diffusion 加 KV cache；跨 timestep 复用 attention KV 会改变 denoise 语义。
- 不改变 SDE / flow-matching logprob 数学。
- 不改变 reward、advantage、GRPO / DiffusionNFT 语义。
- 不默认 CPU offload。
- 不把 decoded video 当 trainer replay source of truth。
- 不实现 generic `TrajectoryStoragePolicy`；只消费它。
- 不替换 DiT attention backend；那属于后续 diffusion module / kernel sprint。

## Phase 1：profile label 和 capability cleanup

编辑：

```text
vrl/generation/diffusion/executor.py
vrl/models/diffusion/capabilities.py
vrl/generation/execution/planner.py
tests/engine/generation/test_chunk_gatherer.py
```

目标：

- diffusion profiler label 改成 diffusion 语义：

```text
generation.prepare_sampling
generation.prompt_encode
generation.denoise_step
generation.denoise_forward
generation.scheduler_step
generation.latent_write
generation.trajectory_buffer_write
generation.decode_latents
```

- diffusion capability 不再把 denoise stage 表达成 cache read/write。
- AR 可以继续使用 cache label；diffusion 不应复用 AR/KV 术语。

完成标准：

- diffusion profile 结果里不再出现 `engine.cache_read` / `engine.cache_write` / `engine.vq_decode`。
- existing AR profile label 不被误改。

## Phase 2：denoise loop 预分配

编辑：

```text
vrl/generation/diffusion/executor.py
tests/engine/diffusion/test_denoise_preallocation.py
```

新增内部 helper：

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
    state: object,
    config: DiffusionDenoiseConfig,
) -> DiffusionDenoiseBuffers:
    ...
```

目标：

- `run_denoise_steps(...)` 不再以 list append + final stack 作为主路径。
- `observations` / `actions` shape 来自 `state.latents` 和 `len(state.timesteps)`。
- `timesteps` shape 是 `[chunk_batch, num_steps]`。
- `log_probs` shape 是 `[chunk_batch, num_steps]`，dtype 默认 `torch.float32`，device 跟 `state.latents.device`。
- `return_kl=False` 继续保持当前 schema：`kl` 是 `[chunk_batch, num_steps]` zero tensor。
- `return_kl=True` 写入真实 `[chunk_batch, num_steps]` KL tensor。

完成标准：

- fake denoise tests 覆盖 num_steps、batch、latent shape、`return_kl=True/False`。
- 输出 shape / dtype / device 与旧行为一致。
- 不改变 SDE step 数学。

## Phase 3：diffusion engine counters

编辑：

```text
vrl/generation/diffusion/executor.py
vrl/generation/diffusion/gather.py
vrl/trajectory/storage.py
tests/engine/diffusion/test_diffusion_engine_counters.py
```

目标 counter：

```text
diffusion_num_denoise_steps
diffusion_sample_batch_size
diffusion_observation_bytes
diffusion_action_bytes
diffusion_old_logprob_bytes
diffusion_timestep_bytes
diffusion_kl_bytes
diffusion_replay_tensor_bytes
diffusion_video_bytes
diffusion_storage_device
diffusion_storage_dtype
```

要求：

- counters 只用于 profile，不参与算法。
- counter values 必须 JSON-serializable。
- family gatherer 聚合 chunk-level counters。
- 如果通用 `trajectory_tensor_bytes(...)` 尚未落地，可先在 diffusion sprint 内加私有 helper，但最终应迁到 `vrl/trajectory/storage.py`。

## Phase 4：storage policy adoption

依赖：

```text
SPRINT_trajectory_runtime_performance_optimization.md Phase 2
```

编辑：

```text
vrl/trajectory/builders.py
vrl/generation/diffusion/gather.py
configs/base/rollout/diffusion.yaml
tests/engine/diffusion/test_diffusion_storage_policy_adoption.py
```

目标：

- 默认 `preserve`，不改变当前 SD3.5 / Wan / Cosmos 行为。
- CPU offload 只在显式配置时启用。
- dtype 改变必须有 tolerance test。
- old_log_prob / mask / distribution source of truth 不变。

## Phase 5：replay timestep slice guard

编辑：

```text
vrl/models/diffusion/base.py
vrl/rollouts/evaluators/diffusion/sde_logprob.py
vrl/trajectory/resolver.py
tests/rollouts/test_diffusion_replay_slice_access.py
```

当前判断：

- 直接 `batch.observations[:, timestep_idx]` 如果 tensor 已在目标 device 上，通常只是 view，不是性能 bug。
- 真正要防的是 storage/offload path 先把整条 `[B, T, ...]` trajectory `.to(device)`，再 slice 当前 timestep。

完成标准：

- evaluator / replay forward 只读取当前 denoise step 所需 tensor。
- old_log_prob / mask / distribution 仍来自 `TrajectoryBatch` / `TrainingView`。
- tests 用 sentinel tensor 或 fake tensor 断言不会对 unsliced `[B, T, ...]` 调 `.to(device)`。
- 不引入 legacy extras fallback。

## Phase 6：profile gate

目标：

- 跑当前 diffusion OCR baseline profile。
- 跑 preallocation + counters profile。
- 如果启用 CPU offload，单独跑 offload profile。

对比：

```text
collect.generate wall-clock
peak_memory_mb
host RSS
GPU memory
diffusion_*_bytes counters
cudaLaunchKernel count
cudaMemcpy / device synchronize events
```

## Tests

新增或编辑：

```text
tests/engine/diffusion/test_denoise_preallocation.py
tests/engine/diffusion/test_diffusion_engine_counters.py
tests/engine/diffusion/test_diffusion_storage_policy_adoption.py
tests/rollouts/test_diffusion_replay_slice_access.py
tests/models/test_diffusion_model_base.py
tests/engine/generation/test_chunk_gatherer.py
tests/trainers/test_memory_guards.py
```

## 验收命令

```bash
pytest tests/engine/diffusion \
  tests/rollouts/test_diffusion_replay_slice_access.py \
  tests/models/test_diffusion_model_base.py \
  tests/engine/generation/test_chunk_gatherer.py

pytest tests/models tests/rollouts tests/trainers/test_memory_guards.py
python -m compileall vrl tests
git diff --check
```

Profile：

```bash
python -m vrl.scripts.train --config experiment/online/ocr/image_flow_grpo \
  trainer.profile=true
```

## 最终完成标准

完成后必须能回答：

- diffusion rollout peak memory 主要花在 observations/actions/video/replay tensors 哪一项。
- denoise preallocation 是否降低 allocation 和 peak memory。
- `return_kl=False` 的 zero KL tensor 是否只是可接受的小项，是否值得后续 optional 化。
- storage policy preserve / cpu 对 SD3.5 / Wan / Cosmos 的影响。
- replay timestep slicing 是否存在真实整条 trajectory 搬运；如果存在，是否已经变成先 slice 后 move。
