# SPRINT：Optimizer streaming：容量不足时才把全参 Adam 状态流式送入 GPU

状态：**parked。触发：真实 full-parameter recipe 因 optimizer/host 容量失败，且 profile 证明 LoRA、现有分片/parking 不满足该任务。**

## 阅读基线与执行边界

日期：2026-09-09。VRL 基线 `76b4fa228`。本文件是待执行计划，不代表功能已实现。
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
若没有容量事件，本 sprint 不动。

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
