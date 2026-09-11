# SPRINT：权重同步传输——先测瓶颈，再选择传输

> Configuration rename (2026-09-10): the current option is
> `distributed.rollout.update_weight_buffer_size`. Historical commands below
> retain the former `weight_sync_bucket_bytes` name used for those measurements.


状态：**implementing（用户已要求实现传输选择）；默认传输仍由真实测量决定。**
真实 full-parameter 多 GPU 作业的性能验收尚未完成，LoRA-only 测试不能替代。
2026-09-09 按当前代码和 [Miles v0.1 研究](../../research/miles_v01_2609_08368.md)
重写旧的“全参默认 NCCL”计划；新默认必须由真实测量决定。

## 现状与唯一 owner

- `vrl/trainers/weight_sync.py::RayRuntimeWeightSyncer` 获取 immutable CPU snapshot、
  分配单调版本并串行提交。成本分为 state_to_cpu 与 push。
- `vrl/generation/ray/weight_sync.py::GenerationWeightSync` 是已有传输接口；
  Ray 实现只做一次 ray.put，所有 worker 共享 ObjectRef，验证每 rank 安装版本。
- `vrl/generation/execution/worker.py::update_weights` 安装实际状态。
- `vrl/rollouts/orchestration/continuous/owner.py` 保留 pause/drain-or-slot、
  ACK、publish/purge/resume 与失败关闭 admission。
- `vrl/generation/ray/launcher.py` 构建唯一同步实例。

GPU→CPU 的事实来自 snapshot producer；不是 ray.put 自身无条件执行 GPU 拷贝。
如果增加 GPU transport，必须修改它上游的 snapshot 生命周期，不能只替换最末端。

## 来源与范围

Miles 论文 §4 的借鉴是准备与传输分开，以及内容验证。
[pinned bucket code](https://github.com/radixark/miles/blob/e5125a97e1fd383f005f4de258a5985026e09425/miles/backends/training_utils/weight_update/hf_weight_iterator/bucketing.py)
和 [transport implementations](https://github.com/radixark/miles/tree/e5125a97e1fd383f005f4de258a5985026e09425/miles/backends/training_utils/weight_update/protocols)
仅作参考；Megatron tensor/expert layout 不是 VRL 的 state schema。

## 实施阶段

1. 量测 adapter/full-param bytes、CPU snapshot、序列化、transport、安装/重量化、
   pause 和总 step 时间；同一真实拓扑做至少 warm-up 后多个稳态更新。
2. 当瓶颈成立，复用现有接口添加一个最小可用传输，比较 object-store 与
   NCCL（或经过原型验证的 RDT/IPC）。明确谁持有 GPU snapshot、何时可释放，
   不允许 optimizer 更新仍在发送的底层存储。
3. 准备路径按确定 key/schema 顺序生成有界 bucket；转换一次再发送。
   参数族需要一起 requantize 时，不可任意按字节切开。记录实际 bucket 峰值，
   不照搬 Miles 512MB/benchmark 1GB。
4. 每 worker 所有 bucket 就绪才安装/发布版本；部分失败不得让新版本进入 admission。
   与 [真实内容验证](../planned/SPRINT_miles_weight_delivery_verification.md) 共用验收。
5. 仅在双端都跨多节点、有可达 RDMA 且 broadcast 被证实为瓶颈后评估 P2P。
   Miles 报告单节点 P2P 可慢约 70%，不设置“节点越少也越快”的默认。
6. 若没有直连 fabric、已有共享存储且 delta 的压缩/补丁耗时值得，再单独实现
   disk-delta 阶段：base hash、names/shapes/dtypes/layout、版本 index-last 原子发布、
   结果 tensor digest；XOR 必须精确一次应用，重试无法证明时选 overwrite/full snapshot。
   该事件未出现前，不实现文件格式、后台同步或额外配置。

## 验收

- 同一 state 经基线与新传输，在全部 rank 上产生相同正确结果。
  非量化逐位相等；量化路径验证同一转换结果和误差预算。
- 单卡双进程 Gloo 只能验证一部分控制逻辑，不算 NCCL/GPU direct 验收。
- OOM、取消、部分 ACK、错误版本、漏 bucket、重复 delta、错 base、
  源 snapshot 提前释放：均不能发布坏策略。
- trace/NVML/host metrics 证明确实消除目标拷贝；CUDA tensor 名义传输不等于零拷贝。
- 测总 step 改善及新增资源成本；达不到预先声明的阈值则不改默认，记录负结果。
- 新传输不可用时可在启动时显式拒绝；禁止训练中静默切换破坏版本事务。

## 应改／应留／非目标

改变已测瓶颈所在的 snapshot/transport 生命周期。保留 GenerationWeightSync、
薄 Ray actor adapter、trainable-state owner 和现有版本事务，因为它们是真实边界。
协议名/schema key/file name 常量有用；不添加按算法名的传输名单。

不先添加 YAML 菜单再找消费者，不按 use_lora 布尔值自动断言最优传输。
不迁移 Ray，不导入 Megatron，不为 LoRA 复制全参优化栈。


## 2026-09-09: Optional staged bucket transport

`distributed.rollout.weight_sync_bucket_bytes=67108864` selects the new transport
for weight-bearing updates (64 MiB of tensor data per bucket). Omit it or use null
to keep the existing one-put path. The public schema and frozen worker projection
validate a positive integer and the launcher passes it to the existing sync owner;
there is no second transport registry or automatic mode selection.

The sender describes tensor shapes/dtypes, splits oversized tensors into independent
storage slices, and packs small slices into buckets. It puts one bucket and waits
for every engine rank before advancing. Metadata and serialization framing are
outside the tensor-byte ceiling. Each receiver copies slices into owned CPU staging
buffers so retained Ray views cannot pin all prior buckets. Missing/duplicate,
out-of-order, wrong-shape/dtype and foreign-transfer chunks fail. Only a complete
mapping is passed once to the existing model loader/slot installer, which retains
its full-key validation and policy ACK behavior. No partial mapping reaches the
model. None payloads keep their existing version-only behavior.

Abort uses bounded direct cleanup calls because a failed dispatcher may already
reject ordinary admissions. It clears matching staged state without touching a
different transfer. Worker release also drops staging. A commit failure can leave
another rank already updated; the driver still fails and the existing runtime
quarantine prevents publication/admission. This is not a cross-rank rollback
transaction. No new version is globally accepted merely because begin/chunk calls
returned their transfer version.

Memory/performance limits are explicit: the sender still owns a complete immutable
CPU snapshot; each receiver needs a full host-side staging copy, and its existing
model install may allocate more. Only live transport references are paced; Ray can
cache evictable objects after their references are released. This is not a hard
RSS/object-store-used ceiling, GPU direct transfer, RDMA, or a claim of faster
training. Additional RPCs and copies may make small snapshots slower. The default
remains unchanged pending real full-parameter measurements. Broadcast/GPU transport,
large-model benchmarks and delta/alternative transport evaluation remain open.

`StagedWeightTransfer` owns actual assembly state and validation, and the new Ray
methods are RPC boundaries. The existing loader, version slots, default one-put
sync, snapshot owner and runtime failure boundaries remain intact. No per-model
ALL_CAPS vocabulary or declarative contract class was added.

Validation: 262 Ray/execution/config/architecture tests passed before the final
small-tensor packing addition. Then 29 focused transfer/Ray tests passed, including
actual two-rank Ray transfer, lost-chunk rejection, abort cleanup, packed bucket
sizes, independent backing storage, noncontiguous/BF16/NaN/negative-zero payloads,
empty tensors and explicit CPU staging under a different default device. The
real-process models are tiny CPU fixtures; no full-model/multi-GPU performance
result is claimed. Source/checkpoint dtype policies are not changed by transport.

## 2026-09-09: Configured-transport acceptance CLI

The existing weight-delivery CLI now routes target snapshots through the production
sync owner and consumes the resolved bucket setting. Its v2 report labels the
transport and complete sync/verification wall times. Two-receiver real-Ray CPU
checks include missing-bucket rejection, not just direct actor installation.
See the [acceptance details and limits](../planned/SPRINT_miles_weight_delivery_verification.md).
This closes a CLI coverage gap; warm full-parameter measurements, GPU/fabric
comparisons and end-to-end training improvements remain unproven.

## 2026-09-09: Real 4.49 GB full-parameter CPU acceptance

The configured 64 MiB bucket path passed exact poisoned-state recovery and two
installs on two real SD3.5 CPU rollout models, each receiving all 908 trainable
parameters (4,486,343,040 bytes). The default snapshot comparison on the same
CPU topology was terminated by Ray's host-memory monitor and published no report.
[Commands, results and limits](../../research/weight_delivery_sd3_5_fullparam_cpu_20260909.md)
record both outcomes. This closes real full-parameter bucket byte acceptance for
two CPU replicas; it does not establish GPU/fabric performance or a controlled
peak-memory improvement. The default remains unchanged.
