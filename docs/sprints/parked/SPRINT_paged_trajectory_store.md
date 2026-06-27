# SPRINT: Paged Trajectory Store for diffusion RL

状态：**PARKED — MFU 前提证伪（2026-06-26，见正文实测判决）**：SD3.5 1024² 实测 trajectory buffer 不是峰值/吞吐瓶颈，PagedTrajectoryStore 不是这档的 MFU 杠杆，最多是"防 OOM / 扩容量 / 减少搬运"工具；要成为性能 sprint，必须先在目标 video 配置上证明 trajectory buffer 是瓶颈。下为原提案，留作记录。
> （原状态：planned / proof-gated，2026-06-26）目标不是把 vLLM 的 KV page 硬搬到 diffusion attention；目标是把 diffusion RL 已经必须保存的巨大 rollout trajectory 变成可分页、可延迟物化、可溢出到 CPU/NVMe 的数据结构。这个 sprint 是 **exact 系统优化**：不改变采样、不改变 log-prob、不改变训练 loss，只改变 trajectory tensors 的存储和搬运。

## 0. 一句话

PagedAttention 的本质是：LLM serving 里 KV cache 生命周期长、大小动态、碎片浪费大，所以把它做成 page/block 管理。diffusion RL 里对应的大状态不是 attention KV，而是：

```text
observations / actions / old_log_prob / timesteps / replay_tensors
```

尤其 video rollout 的 latent 是 5D：

```text
[B, C, T_video, H, W]
```

训练 replay 每次只消费一个 microbatch 和一个 denoise step，但当前 dense trajectory 会把整条 `[batch, denoise_steps, ...]` 当作大块张量处理。PagedTrajectoryStore 的目标是把这条轨迹拆成 pages，按 replay 需要再 materialize。

## 0.5 实测判决（2026-06-26，SD3.5-medium 1024² / RTX 5090）

跑了 `vrl/scripts/perf/{rollout_bottleneck,dit_mfu,backward_mfu}_probe.py`，本 sprint 的 **SD3.5 1024² MFU/吞吐前提**被证伪；不要把这个结论外推到所有 video 配置：

- **rollout DiT batch-scaling（eager 同一 kernel 栈）**：ms/sample 在 batch 1→16 全程平（161→155→158→155→155 ms）。这只证明“加大 batch / 跨请求合 batch”在这档不会降低单样本 forward 成本；它不否认 torch.compile / fp8 这种换 kernel 的收益。
- **训练侧显存确实 binding**（batch 4 no-ckpt OOM；batch 2 = 26.8GB），但当前证据指向的是**激活值**而不是 trajectory buffer：grad-checkpointing 把峰值从 26.8GB → 9.8GB，replay 又本来按 step 取（`_replay_inputs_for_step`）。

结论：PagedTrajectoryStore 不是 SD3.5 这档的 MFU 杠杆。它最多是"防 OOM / 扩容量 / 减少搬运"工具。要让它成为性能 sprint，必须先在目标 video 配置上证明 trajectory buffer 是峰值或吞吐瓶颈；否则先不做。证据见记忆 `project_rollout_bound_class_probe`。

## 1. Related work

- PagedAttention 证明了 block/page 管理能把长生命周期 KV cache 的碎片和重复分配降下来，并提升 serving throughput，但它依赖 LLM decode 的 KV cache 结构，不直接适用于 diffusion latent self-attention：https://arxiv.org/abs/2309.06180
- vAttention 给了另一个系统启发：保留虚拟连续性、物理内存按需分配，核心仍是把动态大状态从普通 tensor allocation 里解耦出来：https://arxiv.org/abs/2405.04437
- vLLM-Omni 的 diffusion continuous batching 把 step execution 拆成 scheduler/runner 层，说明 diffusion 也可以在 step 边界做系统调度，但它处理的是请求调度，不是 RL trajectory storage：https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/diffusion_continuous_batching/

## 2. 当前代码证据

rollout 当前在 denoise loop 内每步写 dense buffers：

```text
vrl/generation/diffusion/executor.py
  buffers.observations[:, step_idx]
  buffers.actions[:, step_idx]
  buffers.log_probs[:, step_idx]
  buffers.timesteps[:, step_idx]
```

replay 每次只取一个 denoise step：

```text
vrl/models/diffusion/base.py
  _replay_inputs_for_step(batch, timestep_idx)
```

GRPO loss 只需要当前 step replay 出来的 `signals.log_prob` 和 trajectory 里的 `old_log_prob`：

```text
vrl/algorithms/grpo/continuous.py
  raw_ratio = exp(signals.log_prob - old_log_probs)
```

这就是分页的入口：producer 可以保存 page table；consumer/replay 只把当前 step 需要的 pages 物化成现有 batch 视图。

## 3. 设计

新增一个 trajectory storage boundary，不先改算法：

```text
PagedTrajectoryStore
  page_size_samples
  page_size_steps
  page_table: sample_id, step_range, tensor_kind -> page_handle
  materialize(segment, tensor_kind, sample_indices, timestep_idx, device) -> Tensor
```

第一版只支持 step-page：

```text
page = one tensor_kind for one denoise step over N samples
```

不要一开始做 temporal/spatial sub-page。video latent sub-page 会影响 VAE/decode/replay shape，风险太高；先把 step axis 分页打通。

## 4. 正确性契约

- page store 是 storage detail，不改变 `RolloutBatch` 对 evaluator/algorithm 的语义。
- `old_log_prob` 必须 bitwise 等于 dense buffer 路径。
- materialized `observations/actions/timesteps/replay_tensors` 必须和 dense trajectory 的同一切片逐位一致。
- 不新增 hardcoded ALL_CAPS tensor-kind 列表；合法字段从 trajectory schema/segment metadata 派生，避免手写字段集随 schema 漂移。

## 5. 执行顺序

1. 加一个只读 adapter：从现有 dense `RolloutBatch` 构造 `PagedTrajectoryStore`，再 materialize 回 dense slice，做 parity 测试。
2. 改 rollout writer：生成时同时写 page store 和 dense fallback，默认仍消费 dense fallback。
3. 改 `_replay_inputs_for_step` / `TrajectoryResolver`：如果 batch 带 page store，就只 materialize 当前 step。
4. 加 CPU offload policy：GPU 只保当前 replay window，冷 pages 留 CPU pinned memory。
5. 只有在 parity 和 throughput 都过后，才允许把 dense fallback 变成 debug mode。

## 6. 验收

- same-seed dense vs paged trajectory：observations/actions/old_log_prob/timesteps/replay_tensors 逐位一致。
- GRPO first-step ratio parity：same precision 下 `ratio_abs_dev_max == 0` 或保持现有测试阈值。
- 峰值 GPU memory 降低；replay wall-clock 不退化超过 materialization 成本预算。
- video batch 的最大可训练 sample count 增加，或者同 batch 下 OOM 频率下降。

## 7. 非目标

- 不做 diffusion attention KV cache。
- 不做 shared-prefix/tree rollout；那是 `SPRINT_signal_paged_rollout.md` 的范围。
- 不改变 `sde_step_with_logprob` 或 GRPO loss。
- 不把 page store 暴露成用户-facing config，先作为 executor/trainer 内部能力。

## 8. 关键文件

- `vrl/generation/diffusion/executor.py`
- `vrl/models/diffusion/base.py`
- `vrl/trajectory/`
- `vrl/rollouts/batch.py`
- `vrl/algorithms/grpo/continuous.py`
