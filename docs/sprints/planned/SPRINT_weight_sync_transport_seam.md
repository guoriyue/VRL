# SPRINT：权重同步传输——先测瓶颈，再选择传输

状态（2026-10-21 更新）：**blocked on GPU 测量。** 唯一的传输是 object-store
一次 put（`vrl/generation/ray/session.py::RayGenerationSession.update_weights`）。
2026-09-09 加入的分桶传输和 `update_weight_buffer_size` 已于 2026-09-19 删除
（`b744da3af`：单机上比一次 put 慢 5–15%），下文 2026-09-09 各节只是历史记录。
**解锁事件**：在真实拓扑上按文末“实施记录”的命令测出 `weight_sync.state_to_cpu`
/ `weight_sync.push`，且其中一段被证实是步时瓶颈；之后才选新传输。
2026-09-09 按当时代码和 Miles v0.1 研究重写旧的“全参默认 NCCL”计划；新默认必须由
真实测量决定。

## 现状与唯一 owner

- `vrl/trainers/weight_sync.py::RayRuntimeWeightSyncer` 获取 immutable CPU snapshot、
  分配单调版本并串行提交。成本分为 state_to_cpu 与 push。
- `vrl/generation/ray/weight_sync.py::RayGenerationWeightSync` 是已有传输 owner；
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
   与 [真实内容验证](parked/SPRINT_miles_weight_delivery_verification.md) 共用验收。
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

改变已测瓶颈所在的 snapshot/transport 生命周期。保留 RayGenerationSession.update_weights、
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
See the [acceptance details and limits](parked/SPRINT_miles_weight_delivery_verification.md).
This closes a CLI coverage gap; warm full-parameter measurements, GPU/fabric
comparisons and end-to-end training improvements remain unproven.

## 2026-09-09: Real 4.49 GB full-parameter CPU acceptance

The configured 64 MiB bucket path passed exact poisoned-state recovery and two
installs on two real SD3.5 CPU rollout models, each receiving all 908 trainable
parameters (4,486,343,040 bytes). The default snapshot comparison on the same
CPU topology was terminated by Ray's host-memory monitor and published no report.
[Commands, results and limits](../research/weight_delivery_sd3_5_fullparam_cpu_20260909.md)
record both outcomes. This closes real full-parameter bucket byte acceptance for
two CPU replicas; it does not establish GPU/fabric performance or a controlled
peak-memory improvement. The default remains unchanged.

## 2026-10-20: PyTorch Conference NA 2026 transport options (evidence only)

Read the existing `weight_sync.state_to_cpu` / `weight_sync.push` ranges before
choosing; LoRA payloads (tens of MB) do not justify a new transport. Options
reported on Day 1, in order of fit for VRL:

- Colocated: CUDA IPC handle on one reused packed buffer (SkyRL / vLLM native
  RL API, session 099/039). LoRA in-memory sync measured 12-14x faster than disk.
- Ray Direct Transport (NCCL/NIXL/CUDA_IPC/Gloo, Ray 2.56-2.59, session 007):
  object-store baseline 188 ms vs nixl 81 ms per step on B200. Check asyncio
  support before using it from the async syncer, and make the trainer wait for
  receive completion before `optimizer.step` (RDT objects are mutable).
- `torch.distributed._transport` one-sided RDMA, CUDA-graph capturable, built
  for RL weight sync (session 093; experimental).
- vLLM `sharded_rdt` / Monarch WeightBridge: discover each worker's shard layout
  by dry-running the loader on shape-only tensors, then pull only the local
  slice (Qwen3-235B 64.7 s -> 3.49 s; session 039/024).

Independent of transport: after install, recompute the trainable digest on the
rollout side and compare with the trainer's (zero-risk correctness check; SkyRL
also reports "first divergent parameter" and a logprob-gap-unchanged check).

## 实施记录（2026-10-21）

### 现状核对（替换上文“现状与唯一 owner”里的过期路径）

- 版本分配与串行提交：`vrl/trainers/weight_sync.py::RayRuntimeWeightSyncer`。
- 快照生产者：`RolloutRuntimeCoordinator.prepare_weight_sync_state`
  （`vrl/rollouts/orchestration/rollout_runtime.py`）在 trainer 线程调用
  strategy 的 `export_rollout_state`。DDP/单进程走 `to_cpu_snapshot`，FSDP 走
  `gather_trainable_state_dict` → `full_cpu_tensor`（`vrl/trainers/fsdp.py`）。
- 传输：`RayGenerationSession.update_weights`（`vrl/generation/ray/session.py`）。
  一次 `ray.put`，所有 engine 共享同一个 ObjectRef。原来的
  `vrl/generation/ray/weight_sync.py::RayGenerationWeightSync` 已不存在。
- 安装：`vrl/generation/execution/worker.py::GenerationWorkerCore.update_weights`。
- pause/drain 与提交：`vrl/rollouts/orchestration/continuous/thread.py`。上文的
  `continuous/owner.py` 已不存在。

### 本轮代码

1. `trainers(weight_sync): time the snapshot where it is taken and copy it once`
   - 问题：两个 strategy 的 `export_rollout_state` 都已经返回不与 live 参数共享
     存储的 CPU 快照，`prepare_weight_sync_state` 的 docstring 也写明快照由 getter
     负责。但 `RayRuntimeWeightSyncer.push` 在 `weight_sync.state_to_cpu` range 里
     又整份 `to_cpu_snapshot` 了一次。
   - 后果有两个：每次同步多一份完整的 host 拷贝（SD3.5 全参是 4.49 GB）；10-20
     补充让“选传输前先读”的 `state_to_cpu` range，量到的是这次 CPU→CPU 拷贝，
     而真正的 FSDP gather 和 D2H 拷贝不在任何 `weight_sync.*` range 里。
   - 改动：range 移到 `prepare_weight_sync_state` 调 getter 的地方；push 直接转交
     已准备好的快照。`docs/PROFILE_PHASES.md` 同步更新。
   - 测试：`test_snapshot_export_and_push_are_separate_profile_ranges`（真实
     collector + strategy + syncer）和
     `test_syncer_hands_on_the_prepared_snapshot_under_growing_versions`。两者在旧
     代码上都失败。
2. `trainers(fsdp): copy each gathered tensor to host once`
   - `full_cpu_tensor` 原来是 `.cpu().clone()`：对 CUDA tensor，`.cpu()` 已经拷到
     host，`clone()` 又拷一遍。改成 `to(device="cpu", copy=True)`，两种情况都只拷
     一次。CPU tensor 不 alias 的保证由 `tests/trainers/test_ema.py` 继续覆盖。
   - FSDP 每次权重同步少一次整模型的 host memcpy。checkpoint gather 和 EMA 导出
     同样受益。

### 各项状态

| 项 | 状态 |
| --- | --- |
| 阶段 1 量测 | 工具就绪（上面两个 commit），**待 GPU 实测**，命令见下。adapter/全参字节数用 `sum(t.numel() * t.element_size())` 从快照算出即可，不另加指标 |
| 阶段 2 新传输（object-store vs NCCL/RDT/IPC） | **不做，等阶段 1**。“瓶颈成立”这个门槛还没有证据。分桶传输实现后实测更慢，已撤回（`b744da3af`）。仓库里没有第二个传输，所以不引入 transport seam。10-20 列出的 CUDA IPC、RDT、`torch.distributed._transport`、`sharded_rdt` 都只是证据：LoRA 几十 MB 不值得换传输；全参要等测出 push 的占比后，再选一种直接实现（RDT 要先确认 asyncio 支持，并且 `optimizer.step` 前必须等接收完成） |
| 阶段 3 有界 bucket 准备 | **已撤回**（`b744da3af`，单机慢 5–15%） |
| 阶段 4 全部 bucket 就绪才安装 | 随分桶一起撤回。一次 put 路径本来就把完整 mapping 交给 loader 安装，版本事务不变 |
| 阶段 5 P2P | **不做**：前提是双端多节点、有 RDMA、broadcast 被证实是瓶颈；仓库只有单节点配置 |
| 阶段 6 disk-delta | **不做**：触发事件（无直连 fabric 但有共享存储，且 delta 值得做）没有出现 |
| 10-20：安装后在 rollout 侧重算 trainable digest，与 trainer 对比 | **需要用户决定**：这和 `b744da3af` 删掉的 `verify_content` 读回是同一类检查。每次同步都要在 worker 上读回并 hash 一遍，trainer 侧也要 hash 一份 4.49 GB 快照；至今没有测到过内容损坏。SkyRL 的“logprob gap 不变”检查本仓已经有：`replay_parity_gate`（`vrl/trainers/online/trainer.py`），首步和失败时还会记录 `trainable_state_digest`。要恢复读回，需要 owner 推翻 09-19 的删除决定 |
| 验收各条（逐位一致、部分 ACK、OOM 等） | 只针对新传输；没有新传输之前不适用 |

### 阶段 1 的 GPU 量测命令（未运行）

单卡 LoRA（payload 几十 MB，作参照）：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m vrl.scripts.train \
  --config experiment/sd3_5/online_grpo_ocr +profile=torch_profiler \
  trainer.torch_profiler.activities=[cpu,cuda] \
  trainer.torch_profiler.skip_first=2 trainer.torch_profiler.max_steps=3 \
  trainer.total_epochs=5 trainer.output_dir=outputs/wsync_measure_lora
```

全参 FSDP 2x1（两台机器都要运行；最后的 `0`/`1` 是节点号）：

```bash
OUT=outputs/wsync_measure_fullparam \
CONFIG=experiment/sd3_5/online_grpo_ocr_fsdp_2x1_fullparam \
EXTRA_OVERRIDES="+profile=torch_profiler trainer.torch_profiler.activities=[cpu,cuda] \
  trainer.torch_profiler.skip_first=2 trainer.torch_profiler.max_steps=3 trainer.total_epochs=5" \
  bash docs/training_examples/online_nft_kling_video_reward_fsdp_2x1/fsdp_2x1_launch.sh 0
```

两条命令的 override 已在 CPU 上用 `load_config` + `parse_config` 解析通过，训练本身没有运行。
读 trace：

```bash
.venv/bin/python - outputs/wsync_measure_lora <<'EOF'
import glob, gzip, json, sys
for path in sorted(glob.glob(f"{sys.argv[1]}/torch_profiler/trainer/*.pt.trace.json*")):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as handle:
        events = json.load(handle)["traceEvents"]
    for event in events:
        if event.get("ph") == "X" and event.get("name", "").startswith("weight_sync."):
            print(path.rsplit("/", 1)[-1], event.get("cat"), event["name"],
                  f"{event['dur'] / 1e3:.1f} ms")
EOF
```

这个读取脚本已在 CPU 上生成的 trace 上验证过。每步总时间看 `phase_events.jsonl`
（profile preset 会打开 `trainer.profile=true`）。判定：如果 `state_to_cpu + push`
占稳态步时的比例低于预先声明的阈值，就记为负结果，不换传输。
