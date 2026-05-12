# SPRINT：Torch Profiling 与性能优化

这份 sprint 的目标是先建立可视化 profiling 基线，再做 `torch.compile`、多 GPU rollout、reward 并发、kernel 级优化。不能先写自定义 kernel，也不能只看单个 wall-clock 数字；每个优化都必须有 trace 前后对比。

## 1. 背景结论

当前 Janus-Pro-R1 Codex-QA real run 已经证明训练链路能真实更新参数：

```text
outputs/janus_pro_1b_r1_codex_qa_real_dense_n4_1epoch/
  metrics.csv
  checkpoint-final/checkpoint_meta.json
```

关键观察：

- `n_samples_per_prompt=4` 时有真实 optimizer update。
- Codex reward 是外部 CLI judge，可能成为 wall-clock 主瓶颈。
- R1 自回归生成很可能比 replay/backward 更重。
- trainer 端 profiling 只能看到 driver replay/backward；Ray rollout worker 的 family inference GPU op 不会自动出现在 driver trace 里。
- `torch.compile` 已在部分 diffusion family 里有配置，但还没有统一 trace 对比流程。

## 2. 目标

需要完成一个 repo 级 profiling/optimization 工作流：

- 所有 `OnlineTrainer` recipe 可以打开 PyTorch profiler，并写 TensorBoard trace。
- 所有 family inference 路径可以单独 profile，包括 Ray rollout worker 内的 executor forward。
- trace 能区分这些阶段：
  - rollout engine generation
  - reward scoring
  - advantage/stat tracking
  - replay forward/evaluate
  - loss computation
  - backward
  - optimizer step
  - weight sync
- 每个优化实验有 before/after trace、metrics、命令和结论。
- 多 GPU 机器到手后，可以直接跑同一套 profiling 命令比较单卡与多卡。

## 3. 非目标

本 sprint 不直接做这些事：

- 不先写 Triton/CUDA 自定义 kernel。
- 不默认打开 `torch.compile`。
- 不把 `torch.compile` 的首次编译时间伪装成训练吞吐提升。
- 不把 Codex reward 速度问题误判成模型 kernel 问题。
- 不追 paper number。

如果 profiler 显示瓶颈在外部 reward 或自回归 sampling Python loop，kernel 优化不是第一优先级。

## 4. Profiling 架构

需要两层 profiler。

### 4.1 Trainer Profiler

位置：

```text
vrl/trainers/online.py
```

覆盖：

- driver 上的 replay/evaluator forward
- loss
- backward
- optimizer step
- weight sync call 的 driver-side cost

输出：

```text
outputs/<run>/torch_profiler/trainer/
```

配置建议：

```yaml
trainer:
  profile: true
  torch_profiler:
    enabled: true
    output_dir: ""
    activities: [cpu, cuda]
    record_shapes: true
    profile_memory: true
    with_stack: false
    with_flops: false
    skip_first: 0
    max_steps: 1
```

`trainer.profile=true` 继续负责轻量 phase timer；`trainer.torch_profiler.enabled=true` 负责 TensorBoard trace。

### 4.2 Family Inference Profiler

位置：

```text
vrl/distributed/ray/rollout/worker.py
vrl/models/families/*/executor*.py
vrl/engine/core/worker.py
```

覆盖：

- family executor `forward`
- distributed executor `forward_chunk`
- gather 前后的 chunk-level latency
- Janus-Pro / Janus-Pro-R1 autoregressive generation
- SD3.5 / Wan / Cosmos diffusion generation
- NextStep continuous-token AR generation

输出：

```text
outputs/<run>/torch_profiler/rollout/<worker_id>/
outputs/<run>/torch_profiler/family/<family>/
```

关键点：

- Ray rollout worker 内必须自己启动 profiler。driver profiler 看不到 worker 进程里的 CUDA kernels。
- profile 文件名必须带 `worker_id`、`family`、`task`、`policy_version`、`step`。
- rollout profiling 默认只开少量 steps，避免 trace 文件爆炸。
- release-after-collect 模式下，trace 必须在 actor shutdown 前 flush。

## 5. 配置设计

Profiling 必须通过可复用 YAML preset 管理，不能把 profiling 开关复制到每个 experiment 里，也不能只靠 README 里的长 CLI override。

默认训练配置只保留关闭状态：

```yaml
# configs/base/trainer.yaml
trainer:
  profile: false
  torch_profiler:
    enabled: false
    output_dir: ""
    activities: [cpu, cuda]
    record_shapes: true
    profile_memory: true
    with_stack: false
    with_flops: false
    skip_first: 0
    max_steps: 1
```

Profiling overlay 单独放在：

```text
configs/profiling/torch_profiler.yaml
```

内容负责同时打开 trainer trace 和 rollout trace：

```yaml
trainer:
  profile: true
  torch_profiler:
    enabled: true

rollout:
  torch_profiler:
    enabled: true
```

每个常用 profiling run 用 wrapper config 表达，而不是在命令行重复配置。例如：

```text
configs/profile/janus_pro_r1_codex_qa_1epoch.yaml
```

该文件继承真实训练 recipe，再叠 profiler overlay：

```yaml
defaults:
  - /experiment/janus_pro_1b_r1_codex_qa_grpo
  - /profiling/torch_profiler

trainer:
  total_epochs: 1
  save_freq: 1
  output_dir: outputs/profile_janus_r1_codex_qa_1epoch
```

Ray rollout profiling 配置必须进入 `GenerationRuntimeSpec.extra`，不能依赖全局环境变量。这样 Ray actor、release-after-collect、multi-GPU worker 都能拿到同一份 profiler 配置。

## 6. 实施阶段

### Phase 1：Trainer PyTorch profiler

完成标准：

- `TrainerConfig` 有 typed profiler config。
- `configs/base/trainer.yaml` 声明默认关闭的 profiler 字段。
- `configs/profiling/torch_profiler.yaml` 能作为 overlay 打开 profiler。
- `configs/profile/janus_pro_r1_codex_qa_1epoch.yaml` 能直接加载并继承真实 recipe。
- `OnlineTrainer.step()` 能在启用时写 TensorBoard trace。
- trace 默认输出到 `trainer.output_dir/torch_profiler/trainer`。
- `trainer.profile=true` 仍然能输出 phase timer。
- config load 测试覆盖 CLI override。
- trainer 单元测试验证 trace 文件生成。

### Phase 2：Family inference profiler

完成标准：

- Ray rollout worker 能按配置启动 profiler。
- executor forward / forward_chunk 能被 trace 包住。
- trace 输出路径带 worker/family/task 信息。
- release-after-collect 不丢 trace。
- 单元测试用 fake executor 验证 worker trace 写出。
- Janus-Pro-R1 real 1-epoch profile 能产出 rollout worker trace。

### Phase 3：Baseline profiling runs

至少跑这些 baseline：

```bash
python -m vrl.scripts.train --config profile/janus_pro_r1_codex_qa_1epoch
```

需要记录：

- total wall time
- `collect.engine_generate`
- `collect.reward_score`
- replay forward/evaluate time
- backward time
- optimizer time
- weight sync time
- GPU memory peak
- top CUDA kernels
- top CPU ops

### Phase 4：`torch.compile` 对照实验

按 family 分开做，不混在一起：

- SD3.5 transformer
- Wan transformer
- Cosmos transformer
- Janus-Pro language model / image-token head
- Janus-Pro-R1 generation path
- NextStep policy

每个实验必须比较：

- compile off baseline
- compile on `mode=default`
- compile on `mode=reduce-overhead`
- first-step compile cost
- second-step steady-state cost
- memory peak
- graph breaks / recompiles

如果 batch shape、sequence length、segment length 动态变化导致频繁 recompile，默认结论是“不打开 compile”，除非有明确稳定收益。

### Phase 5：多 GPU profiling

多 GPU 机器到手后先跑 profiling，不先跑长训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vrl.scripts.train \
  --config profile/janus_pro_r1_codex_qa_1epoch \
  /base/distributed=ray_rollout \
  distributed.resources.trainer.num_gpus=1 \
  distributed.resources.rollout.num_gpus=auto \
  trainer.output_dir=outputs/profile_janus_r1_codex_qa_4gpu_1epoch
```

比较点：

- rollout worker 数量是否等于 rollout GPU 数量。
- 每个 worker 是否有 trace。
- actor 是否重复冷启动。
- weight sync 是否成为瓶颈。
- reward 是否串行卡住。
- rollout GPU utilization 是否均衡。

### Phase 6：优化候选排序

按 trace 决定优先级：

1. 如果 `collect.reward_score` 最大：优化 reward 并发、缓存、batch judge、或替换成本更低的 local judge。
2. 如果 `collect.engine_generate` 最大：优化 family inference、microbatch、KV/cache、sampling loop、`torch.compile`。
3. 如果 replay forward/backward 最大：优化 evaluator、segment mask、gradient checkpointing、compile replay path。
4. 如果 weight sync 最大：优化 LoRA-only sync、sync frequency、multi-worker broadcast。
5. 如果 CPU Python overhead 最大：减少 per-token Python loop、合并 packer/evaluator 操作。
6. 只有 profiler 指向具体 op/kernel 时，才考虑 Triton/CUDA kernel。

## 7. 验收标准

本轮 Phase 1-3 完成需要同时满足：

- 能用 TensorBoard 打开 trainer trace。
- 能用 TensorBoard 打开至少一个 family inference / rollout worker trace。
- Janus-Pro-R1 Codex-QA 1-epoch profile run 成功。
- 至少一份 profile report 写明当前最大瓶颈。
- README 写清楚怎么打开 profiler 和怎么看 trace。
- profiler 相关单元测试和配置加载测试通过。

查看 trace：

```bash
tensorboard --logdir outputs/profile_janus_r1_codex_qa_1epoch/torch_profiler
```

后续 Phase 4-6 的验收另算：

- 至少一个 `torch.compile` 对照实验完成，并明确是否值得默认开启。
- 多 GPU 机器到手后，同一个 wrapper config 能跑出每个 rollout worker 的 trace。
- 优化前后都有 trace、metrics、命令和结论。
- 全量测试通过；如果失败，必须区分 profiler 改动失败和工作树里已有的无关失败。

## 8. 当前实现与实测结果

本轮已经落地：

- `configs/base/trainer.yaml` 默认关闭 `trainer.torch_profiler`。
- `configs/profiling/torch_profiler.yaml` 作为可复用 overlay，同时打开 trainer trace 和 rollout trace。
- `configs/profile/janus_pro_r1_codex_qa_1epoch.yaml` 继承真实 Janus-Pro-R1 Codex-QA recipe，并叠加 profiler overlay。
- `OnlineTrainer.step()` 在启用时写 trainer TensorBoard trace 和 top-op summary。
- Ray rollout worker 从 `GenerationRuntimeSpec.extra` 接收 profiler 配置，在 `executor.forward_chunk()` 外层写 rollout TensorBoard trace 和 top-op summary。
- `trainer.profile=true` 仍然写 phase timer，不依赖 torch profiler。

已执行真实 baseline：

```bash
python -m vrl.scripts.train --config profile/janus_pro_r1_codex_qa_1epoch
```

结果：

```text
output_dir: outputs/profile_janus_r1_codex_qa_1epoch
checkpoint: checkpoint-final
global_step: 1
trainer_step: 1
reward_mean: 0.1600
reward_std: 0.0163
loss: 0.0472
grad_norm: 0.3375
```

phase timer：

```text
total=500.918s
collect=474.328s
collect.engine_generate=435.502s
collect.reward_score=38.826s
release_rollout=24.507s
evaluate=0.446s
backward=0.404s
optim_step=0.045s
```

trace 文件：

```text
outputs/profile_janus_r1_codex_qa_1epoch/torch_profiler/trainer/
  *.pt.trace.json      ~68 MB
  *.summary.txt        top CPU/CUDA ops

outputs/profile_janus_r1_codex_qa_1epoch/torch_profiler/rollout/rollout-0/
  *.pt.trace.json      ~3.1 GB
  *.summary.txt        top CPU/CUDA ops
```

当前瓶颈结论：

- 最大 wall-clock 瓶颈是 rollout generation，不是 backward 或 optimizer。
- `collect.engine_generate` 占单 step 约 86.9%，`collect.reward_score` 占约 7.8%。
- trainer trace 里 replay/backward 很小，主要 CUDA time 是 tensor copy 和 matmul/attention backward。
- rollout trace 里 CUDA time 主要是 Janus 自回归生成的 cutlass matmul 和 attention kernel；CPU side 主要是 `cudaStreamSynchronize`、`Command Buffer Full` 和大量 kernel launch。
- 下一轮优化优先级应是 Janus generation path、sampling loop、KV/cache/microbatch；reward 并发是第二优先级；不应该先写自定义 kernel。

References:

```text
vrl/trainers/online.py
vrl/trainers/types.py
vrl/trainers/profiling.py
vrl/distributed/ray/rollout/worker.py
vrl/rollouts/runtime/launch_inputs.py
vrl/engine/core/runtime_spec.py
vrl/models/families/
vrl/engine/core/worker.py
configs/base/trainer.yaml
configs/profiling/torch_profiler.yaml
configs/profile/janus_pro_r1_codex_qa_1epoch.yaml
outputs/profile_janus_r1_codex_qa_1epoch/metrics.csv
outputs/profile_janus_r1_codex_qa_1epoch/phase_events.jsonl
```
