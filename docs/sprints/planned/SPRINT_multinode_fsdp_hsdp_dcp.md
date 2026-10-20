# SPRINT：多节点 FSDP——HSDP、分片 checkpoint、分布式调试 hook（不引入 Megatron）

状态：**planned（2026-10-20），触发式**。每一项都有一个"何时做"的触发事件。当前所有运行都是单机，触发前不动代码。

来源：PyTorch Conference NA 2026 Day 1：069 Pinterest、070 Dell、081 GCS fsspec、093 torch.distributed 新 API、103 AWS 五个瓶颈、105 Intel XCCL；海报 P19、P48、P62。

## 0. 结论先行：Megatron 的决定

**现在不引入 Megatron-Core、Megatron-Bridge 或 NeMo。** 理由：

- **规模不需要。** 最大的模型是 33B dense DiT（MiniMax-H3），而且走 LoRA。最长序列约 8.5 万 token（Cosmos 704p）。硬件是 PCIe 连接的 L4、L40S 或 5090，没有 NVLink。TP 每层两次 all-reduce，在这种拓扑上是负收益。
- **会上的证据：**
  - 103：多余的 PP stage 让吞吐损失 28%；讲者自己的 Megatron 最优配置是 TP=1、PP=1、纯 DP，本质上就是 ZeRO/FSDP。
  - 070：HSDP 比纯 FSDP 多 59%。
  - 105：纯 FSDP2 到 128 节点，扩展效率仍有约 80%。
- **业界做法：** 做 diffusion RL 和 diffusion 微调的框架，训练全部是 FSDP 系，包括 Miles-diffusion、VeRL-Omni、DanceGRPO、FastVideo、VeOmni 和 NeMo AutoModel。
- **最大的代价：** rollout 和 replay 将不再跑同一份 diffusers 模块，逐位一致被破坏。此外每个家族都要重写成 Megatron 模块，peft LoRA 的 key 和分片布局不兼容，`_cp_plan`、`_no_split_modules` 这些 diffusers 自带的并行支持也会失去。

**重新评估的触发条件**：(a)(b)(c) 至少满足一条，并且通过 (d)：
- (a) 全参训练约 100B 以上的 dense DiT，或需要跨节点 EP 的 token-routed MoE DiT。
- (b) 几百卡以上的预训练，且同硬件实测 MFU 比 Megatron 低 25% 以上。
- (c) 换到 NVLink 平台，且 HSDP + CP 的扩展实测停滞。
- (d) 否决条件：Megatron replay 对 diffusers rollout 的 logprob 差能过 parity gate。

即便满足，也先评估 TorchTitan 的 GraphPP、DualPipeV、Dist-MoE。

## 1. 当前代码现实（HEAD a914644c6 核对）

- **mesh**：`build_fsdp_mesh`（`vrl/trainers/fsdp.py:34-53`）只建 1D mesh `("dp_shard",)`，docstring 写明了 HSDP 还没做。CP 的 mesh 是 `(dp_shard, ring, ulysses)`（:56-82）。schema 只接受 `["dp_shard"]` 和 `["dp_shard","cp"]`（`vrl/config/schema.py:586-622`）。`num_nodes`/`gpus_per_node`（:651-653）只用来校验 `WORLD_SIZE`（`vrl/trainers/distributed.py:140-160`）。
- **多节点启动**：目前只有一个手写的逐节点 torchrun 示例，`docs/training_examples/online_nft_kling_video_reward_fsdp_2x1/fsdp_2x1_launch.sh`。Ray 的 `cross_node` 只覆盖 rollout 和 reward actor，trainer 不经 Ray 放置。
- **checkpoint**：
  - 保存（`vrl/trainers/checkpointing.py:963-1196`）：所有 rank 一起 gather 成完整张量，然后 primary 一次 `torch.save`（:911），再走原子 replace。是同步、单文件写入。
  - 加载：每个 rank 用 mmap 读整个文件。
  - DCP 只用到了 `state_dict` 辅助函数（`fsdp.py:251-331`、`:405-430`、`:495-610`），没有用 `dcp.save` 或 `async_save`。
  - 换 world size 后 resume 会被 RNG 校验挡住（`checkpointing.py:1366-1380`）。
- **调试 hook**：没有 NCCL flight recorder，没有任何 `TORCH_NCCL_*` 或 `TORCH_DISTRIBUTED_DEBUG` 设置，没有 collective 的 NaN 检查（只有全局 grad-norm 的有限性检查，`trainer.py:803-823`）。进程组超时用默认的 30 分钟，CP 开启时是 12 小时。

## 2. 阶段

### P0. 分布式调试 hook。触发：下一次多卡长跑，或首次出现卡死、NaN

1. 在启动环境里打开 NCCL flight recorder：`TORCH_NCCL_TRACE_BUFFER_SIZE`，加上 `TORCH_NCCL_DUMP_ON_TIMEOUT`，dump 目录放在 run 目录下。放在 launch 脚本或 `vrl/trainers/distributed.py` 的进程组初始化处，二选一，只能有一个 owner。
2. 如果 torch 版本里已有 `FlightRecorderHook`/`NanCheckHook`（093，见升级 sprint），就用 hook，不用环境变量。
3. **验收**：故意让一个 rank 卡住（chaos 测试，海报 P19），能拿到每个 rank 卡在哪个 collective 的 dump；`vrl/scripts/supervise.py` 能区分卡死和崩溃。

### P1. 2D HSDP。触发：第一次跨节点运行 FSDP

1. `build_fsdp_mesh` 改为接受 `num_nodes × gpus_per_node`，建 `("dp_replicate","dp_shard")` 的 2D mesh：节点内 shard，跨节点 replicate。schema 的 mesh 校验加一个合法值。`DistributedTrainingContext` 要带上 `num_nodes`，目前它只用于校验。
2. 和 CP 组合时，轴序保持 CP 在最内层（节点内），即 `(dp_replicate, dp_shard, ring, ulysses)`，同步更新 `build_context_parallel_mesh`。
3. **验收**：2 节点 × N 卡，同一 preset 下 HSDP 对比 1D FSDP 的步时与吞吐。070 的参考值是 +59%。如果测得的收益在噪声以内，就不合入。

### P2. 分片 checkpoint。触发：H3 或其他全参 ≥14B 的模型开始做 checkpoint，或换卡数 resume 的需求出现

1. 模型和优化器状态改用 `torch.distributed.checkpoint.save`/`load` 分片写入，每个 rank 只写自己的分片，不再在 rank0 拼完整张量。可选 `async_save`（093：异步 checkpoint 能完全藏在一个 step 里）。
2. RNG 状态按 rank 存放；放宽 world-size 校验，换卡数时 RNG 由新拓扑重新播种，并在 checkpoint 元数据里记一条"拓扑已改变"。这是语义变化，要在 `docs/` 里写明。
3. 导出给 rollout 和推理用的完整权重（diffusers 格式），走现有的 gather 路径，不受影响。
4. 写入远端存储时，用流式 put，避免 O(N²) 的内存拷贝（081：48GB checkpoint 从 47 分钟降到 2.6 分钟）。
5. **验收**：H3 或 Wan 14B 的 checkpoint 时间和 rank0 峰值内存，前后对比；resume 后 parity gate 和 loss 曲线连续；4 卡存、2 卡读能成功。

### P3. 通信重叠核对。触发：多卡训练中 backward 占步时 30% 以上

1. 用 profiler 时间线确认 FSDP 的 all-gather 和 reduce-scatter 是否真的并行。海报 P62 指出，共享同一个 communicator 会让两者串行执行。
2. 测到串行，再用 torch 2.13 的 `set_separate_reduce_scatter_group(True)`（需要先升级）。
3. 方法论（069/103）：跟着关键路径走，按 compute / memory / comm / IO 四类归因，结果写进本文件。

## 3. 不做

- 不做 DTensor TP 和 PP（理由见 §0）；PP 不进计划。
- 不做 EP、DeepEP、Dist-MoE：VRL 没有 token-routed MoE。
- 不换 Ray 编排层（024 Monarch 只作参考）。
