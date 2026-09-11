# SPRINT：Optimizer streaming：容量不足时才把全参 Adam 状态流式送入 GPU

> Retired on 2026-09-10: the custom disk-streaming optimizer, its configuration
> fields, and its probe were removed in favor of standard `torch.optim.AdamW`.
> The implementation and measurements below describe the historical revision;
> their commands do not apply to the current checkout.


Status: **implementation started under the explicit six-item user request.**
The opt-in online AdamW path below is implemented; real-model capacity and performance
acceptance remain open. The historical plan follows; its original capacity trigger
is not evidence of a measured VRL capacity failure.

## Implemented path (2026-09-09)

Set `actor.optim.disk_state_directory` to a local scratch directory to select
`vrl/trainers/disk_optimizer.py::DiskStreamingAdamW` through the existing online
optimizer factory. `actor.optim.disk_state_bucket_bytes` defaults to 268435456.
No preset enables it by default. This option cannot be combined with `optim_8bit`;
offline DPO rejects it instead of silently using resident AdamW.

- FP32 moments are loaded, updated with native non-fused AdamW, written, and
  released one parameter-aligned bucket at a time. The byte cap counts the two
  moments and step counters, not arithmetic temporaries, host copies, serialization,
  allocator cache, model parameters, gradients, or FP32 masters. An individual
  parameter exceeding the cap is rejected; intra-parameter splitting is not implemented.
- Existing FP32 master wrapping remains responsible for low-precision model weights.
  Existing phase parking remains responsible for phase transitions. No additional
  scheduler, background prefetcher, or algorithm rule table is introduced.
- Each optimizer owns an isolated scratch directory. Updates write a new generation;
  file checksums detect subsequent corruption, and publication occurs after all buckets
  are written. Any update I/O failure or cancellation makes the optimizer terminal:
  parameters may already be partially updated, so restore the complete training
  checkpoint into a new model/optimizer. This is not transactional model rollback.
- Portable `state_dict` includes moments, group settings, positional parameter/bucket
  layout, and initialized-state membership. It contains no scratch paths. Exports use
  file-backed CPU views that survive later generation replacement on local POSIX
  filesystems. Restores reject missing initialized state or incompatible layout.
  Existing trainer checkpoint loading can still materialize all CPU moments;
  bounded checkpoint peak memory and crash-resumable scratch are not implemented.
- Plain FP32 parameters/masters are supported. DTensor/FSDP shards, closures, dynamic
  parameter layout, AMSGrad, capturable/differentiable and fused/foreach modes are
  rejected. Cross-shard/rank restoration is not supported or claimed.
- Scratch disk requires approximately two full moment generations during a step,
  plus serialization overhead. Old exported mmap views can keep unlinked files live.
  Process crashes may leave scratch directories requiring later cleanup.

CPU and CUDA acceptance in `tests/trainers/test_disk_optimizer.py` compares multiple parameter
groups and missing-gradient schedules against native resident AdamW, including
moments, counters, exact parameters and the next step after portable restore.
It also exercises BF16 source/FP32 master residual restoration, bounded live moment
state, corrupted/missing bucket files, injected write failures/partial writes and
cancellation, failed restore, and checkpoint/layout rejection.
The CUDA case passed on the shared RTX 5090 with tiny parameters; it is a numerical
transport check, not a full-model capacity run. The online trainer export/restore
path is also exercised in `tests/trainers/online/test_state_restore.py`.
These tests do not establish CUDA peak HBM, host RSS, throughput, disk endurance,
full-parameter recipe convergence, or distributed checkpoint compatibility.

## Measured optimizer-only capacity probe

The [shared RTX 5090 probe](../../research/optimizer_streaming_5090_20260909.md)
measured 488 MiB lower peak CUDA allocation for 67,108,864 synthetic FP32 parameters,
with exact final parameter/moment agreement and approximately 134 times slower
optimizer steps. This is evidence of a residency tradeoff, not a real-recipe budget
or a reason to enable disk state by default.

## Remaining acceptance

Measure real recipe memory and I/O budgets before recommending this option. The
historical acceptance below remains open wherever CPU tests cannot establish it.


## 阅读基线与执行边界

日期：2026-09-09。VRL 基线 `76b4fa228`。The plan below predates the opt-in implementation described above.
论文、upstream pin、完整功能清单和已知限制见
[研究总表](../../research/miles_v01_2609_08368.md)。
Miles 论文 v1 与当前 main 的差异必须保留，不能混成同一份复现证据。

## 现状与适用性

论文 §3.2 区分 phase eviction 与 step 内 optimizer streaming。
VRL 的 `vrl/trainers/strategy.py` 已负责 phase parking；
`vrl/trainers/optimizer.py::FP32MasterWeightOptimizer` 拥有 FP32 masters，
`vrl/trainers/fsdp.py` 负责分布式 optimizer state，
`vrl/trainers/checkpointing.py` 负责保存/恢复。不能把 disk streaming 加进 rollout sleep。

Miles 的 Megatron disk path 不等于 VRL FSDP 可直接使用。LoRA optimizer 很小，
Default adoption still requires measured capacity/performance evidence.

## 启动后步骤

1. 量测每 rank 模型/梯度/master/moments HBM、host/pinned RAM、NVMe 容量、
   实际吞吐与寿命相关写入量；估算每步 I/O 下界和可容忍 step 时间。
2. 在 optimizer owner 内按真实参数顺序建立有界 bucket；最多一个更新 bucket
   加一个受限预取，参数更新完成才释放状态。先 host streaming，再决定 disk。
3. 保留 FP32 master；是否压低 moments 精度是独立数值实验，不随 I/O 实现默认改变。
4. 将 bucket 文件版本、参数布局、rank/shard 和完成状态纳入既有 checkpoint。
   原子发布；不能拿半写 bucket 恢复。没有状态时拒绝 resume，不能重置 Adam。
5. FSDP reshard/checkpoint layout 不一致先 fail closed；需要可移植恢复时另做转换工具。

## 验收和 kill gate

与 resident Adam 从相同权重/梯度运行多步，比较参数、moments、step 计数和恢复后下一步。
注入磁盘满、短写、损坏、取消、缺文件和布局变化；不得悄悄跳过参数更新。
真实运行满足预先写出的 HBM/host/step-time 预算；若 I/O 超预算，记录不采用。
checkpoint 保存时间单列，不能隐藏在异步保存的名称后面。

## 应改／应留／非目标

改变 optimizer 的状态 residency 与 checkpoint 完成规则，保留 phase lease/FSDP strategy 边界。
bucket 是有实际 I/O 的共享抽象；文件名/schema 常量必要；不新建后台调度框架。
不为 LoRA 默认启用、不承诺跨任意分片恢复、不导入 Megatron 或复制其 allocator。
