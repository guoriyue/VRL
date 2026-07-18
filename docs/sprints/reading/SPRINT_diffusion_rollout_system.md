# SPRINT：Diffusion rollout system 方向讨论

状态：discussion（不是 proposed/approved；这份文档的目的是把方向讨论清楚，再决定要不要拆 phase 落地）。

> **2026-07-13 decision update：** 两层框架已被
> [Native generation engine program](../SPRINT_native_generation_engine_program.md)
> 采纳：wm-infra 的 rollout/control plane 继续自研并作为 source of truth；
> FlashDreams/SGLang 只作为可选 diffusion execution provider。本文继续保留为
> reading 证据；native transformer ownership 与跨请求 step scheduling 仍按各自
> profiling/event gate parked。

关联：[[SPRINT_diffusion_native_transformer_executor]]（模型执行层，正交关系，见下文「两层框架」）。

## 核心结论（先看这一段）

1. **「rollout 系统」和「native transformer executor」是两个正交的层**，不要混在一起讨论。sglang-omni 解决的是前者（调度/编排），那份 native executor sprint 解决的是后者（transformer forward 的 layer ownership）。
2. **diffusion 喂满 GPU 本来就比 AR LLM 简单得多**，所以 sglang/vLLM 那套最值钱的机制（paged KV、block table、prefix cache、chunked prefill、prefill/decode 拆分、tree cache）对 diffusion **几乎零收益**。这个判断要先达成共识，否则会照搬一堆用不上的复杂度。
3. **VRL 已经有一套相当完整的 RL rollout 系统**（约 1 万行，见「现状盘点」）。它做的是 **data-parallel 样本分块 + 分辨率分桶批处理 + OOM 重试 + 异步 producer/consumer 重叠训练 + staleness/权重同步**。照搬 sglang-omni 等于重写正在用的代码，不做。
4. 唯一**可能**值得从 sglang-omni 借的点，是**单请求内的 stage 流水线重叠**（text-encode → denoise → VAE 三段跨请求重叠）。但当前系统**没有**做这个，要不要做必须由 profiling 决定，不能拍脑袋。

本 sprint 不写代码，目标是把上面 4 点和文末「待决策问题」讨论定，再决定开 phase。

## 两层框架（消除困惑的关键）

把任何「diffusion 性能/架构」讨论先归到这两层之一，再往下谈：

```text
Layer A — Rollout / 编排层（本 sprint 的主题）
  谁来调度请求、怎么分批、怎么跨 GPU/节点分发、
  怎么和 training 重叠、怎么控 staleness、怎么同步权重。
  对标物：sglang-omni / vLLM 的 scheduler+engine。
  VRL 现状：vrl/generation/execution/* + vrl/generation/ray/* +
            vrl/rollouts/orchestration/continuous/*

Layer B — 模型执行层（另一份 sprint 的主题）
  谁拥有 transformer 的 forward：block / attention / MLP / norm / kernel。
  对标物：vLLM 的 model_executor、自研 Triton/Flash kernel。
  VRL 现状：大部分仍由 diffusers 拥有；
            SPRINT_diffusion_native_transformer_executor 计划接管 Wan/Cosmos。
```

两层独立：可以在不动 Layer A 的情况下做 Layer B 的 native executor，反之亦然。**用户最初的困惑来自把这两层叠在一起看。**

## 为什么 diffusion 喂满 GPU 本来就简单（必须先达成的共识）

AR LLM serving 难喂满，所以才需要 vLLM/SGLang 那套；diffusion 这些痛点一个都没有：

| AR LLM 的痛点 | 为什么 diffusion 没有这个痛点 |
|---|---|
| decode 每步只出 1 token，矩阵小、**显存带宽瓶颈** | 每个 denoise step 是对**整个 latent 的全量 dense forward**，本来就是算力瓶颈、本来就大 |
| 序列变长、batch ragged、padding 浪费 | step 数固定（约 28–50），同分辨率下 batch 完全 homogeneous |
| KV cache 随生成长度无限增长 → 需要 paging | self-attn 的 K/V 每步从当前 latent 重算，**没有跨 step 复用**，不需要 paged KV |
| prefill（算力瓶颈）vs decode（带宽瓶颈）形态不同 → 需解耦 | 每一步形态相同，没有 prefill/decode 之分 |
| batch=1 喂不饱卡 | CFG 天然给 cond+uncond → batch≥2；中高分辨率单样本常已接近喂满 |

**推论：** Layer A 对 diffusion 真正的喂满杠杆是这三件，**都不是 AR serving 那套**：

```text
1. 按分辨率分桶的静态批处理 + CFG 批     —— 最大头，VRL 已做（batch_group_key）
2. 单请求内 stage 流水线重叠               —— VRL 当前未做，本 sprint 的真正开放问题
3. 原始 kernel 速度（Flash/Triton/compile）—— 属于 Layer B，见 native executor sprint
```

而且别忘了 **这是 RL 训练的 rollout，不是对外 serving**：前面有一整批 prompt（rollout buffer），同模型、可统一分辨率、可堆超大静态批，**在线连续批处理的复杂度基本用不上**。RL 这边真正的难点是 staleness、权重同步、生成与训练重叠——这些 VRL 已经有实现。

## 现状盘点（带证据，避免边讨论边猜）

### Layer A 当前已有的东西

```text
vrl/generation/execution/
  planner.py        EnginePlan / ExecutionStage / build_engine_plan
  scheduler.py      DeviceAssignment：把一个 chunk 映射到一个 worker(gpu_ids)
  worker.py         GenerationWorker core（独立于 Ray actor 包装）
  chunks.py         SampleChunk / build_prompt_chunk_schedule / run_sample_chunks_with_oom_retry
  request_batch.py  RequestBatch
vrl/generation/diffusion/
  executor.py       diffusion 家族的执行 scaffolding（684 行）
  gather.py layout.py metrics.py
vrl/generation/ray/
  launcher.py placement.py weight_sync.py worker.py runtime.py
vrl/rollouts/orchestration/continuous/
  producer.py       后台 asyncio 任务，维持 N 个 collect job 在飞，标记 policy version
  consumer.py       同 policy-version 选批，重排 group_id 保证 advantage 归一化正确
  queue.py staleness.py schedule.py
```

### 关键事实：当前的 "stage" 是数据并行分块，不是 pipeline 阶段

`ExecutionStage`（`vrl/generation/execution/planner.py:54`）的字段是 `prompt_index / sample_start / sample_count / batch_group_key / cache_read / cache_write`，`chunk_key` 形如 `prompt:{p}:samples:{start}:{end}`。也就是说：

```text
当前 "stage" = 「为 prompt P 生成第 [start,end) 个样本」这一块数据并行任务，
              按 batch_group_key（分辨率）分桶批处理。
不是          text-encode / denoise / VAE 这种单请求内的处理阶段。
```

`vrl/generation/diffusion/executor.py` 内部把 text-encode → denoise loop → VAE decode **整条链在一个 chunk 里跑完**（配 `run_sample_chunks_with_oom_retry`），不同处理阶段之间**没有跨请求重叠**。

**这对 RL rollout 通常是对的设计**：吞吐来自 chunk 内的批大小和跨 worker 的数据并行，不来自 stage 流水线。是否需要再上 stage 重叠，是本 sprint 要讨论的开放问题，不是已知缺陷。

## sglang-omni 评估（要不要照着建）

子 agent 通读 `/home/mingfeiguo/Desktop/sglang-omni` 的结论：

- 它本质是 **AR-LLM 多模态 serving 框架**（Coordinator / Stage / Worker / Relay / ZMQ control plane / OmniEngine schedule→execute→update / SGLang AR backend）。**不含任何 diffusion/图像生成代码**（只借用了 profiler 的注释）。
- 可复用的约 60% 全在「pipeline 编排 + relay + control plane」这一无关计算语义的层；AR engine/scheduler 那 40%（paged KV、prefill/decode、tree cache、logits sampling）对 diffusion 不适用。

| sglang-omni 组件 | 对 diffusion | VRL 是否已有对位 |
|---|---|---|
| 多 stage pipeline 编排（Coordinator/Stage/Worker） | 概念可借 | 部分：execution + ray + continuous 已覆盖数据并行与异步重叠 |
| Relay（SHM/NCCL/RDMA，payload 无关搬张量） | 可借（搬 latent） | 部分：Ray object store + weight_sync 已承担跨节点搬运 |
| ZMQ control plane | 可借 | 用 Ray 替代 |
| 连续批处理（AR 版） | 弱相关 | continuous/ 已有 RL 版（按 policy-version 选批） |
| paged KV / block table / prefix cache / chunked prefill / tree cache / logits sampling | **不适用** | 不做（native executor sprint 的 Phase 0.5 已明确拒绝） |

**结论：不照搬。** 可复用的那 60%，VRL 这边已有功能对位的实现；照搬等于重写能跑的架构，违背「先修正确性、别重写工作架构」原则。真正值得认真考虑借鉴的只有一个点 → 下一节。

## 真正的开放问题：要不要做单请求内的 stage 流水线重叠

这是本 sprint 唯一一个「sglang-omni 有、VRL 没有、且对 diffusion 可能真有用」的点。

```text
现状：每个 chunk 内 text-encode → denoise×N → VAE decode 串行跑完。
设想：把三段拆成可重叠的 stage，让
      请求 A 在 denoise 时，请求 B 在 text-encode，请求 C 在 VAE decode，
      从而填满三段资源画像差异很大的空隙（文本编码器小 / transformer 巨大 / VAE 中等）。
```

**支持做的理由：** transformer 段是绝对大头，但 text-encode 和 VAE decode 期间 transformer 那块算力可能闲置；流水线能把这些空隙填上。

**反对/暂缓的理由：**

```text
- RL rollout 是离线吞吐场景，靠大静态批 + 数据并行通常已经把卡喂得很满，
  stage 重叠的边际收益可能很小。
- 三段如果跑在同一张卡上，它们抢的是同一个 SM/显存，"重叠" 不等于 "免费"。
  真正受益通常要把 text-encoder/VAE 放到不同设备（stage 解耦 + 跨设备 relay），
  这会显著增加系统复杂度。
- 增加的复杂度会直接落在已经能跑的 execution/continuous 代码上。
```

**结论：必须先 profiling，再决定。** 不允许在没有「transformer 段之外的空闲占比」实测数据之前就开工。

## 待决策问题（讨论目标）

```text
Q1. 是否认同「diffusion 喂满 GPU 简单、AR serving 机制不适用」这个前提？
    —— 如果认同，Layer A 的工作就锁定在「批处理 + 数据并行 + 异步重叠」，不碰 AR 那套。

Q2. 当前 RL rollout 的实际瓶颈是什么？（需要先量）
    候选：generation 吞吐 / 权重同步开销 / staleness 导致的等待 /
          单 chunk 内 transformer 段之外的 GPU 空闲。
    —— 没有这个测量，下面的 Q3/Q4 都不该开工。

Q3. 要不要做单请求内 stage 流水线重叠（text/denoise/VAE）？
    取决于 Q2 里「transformer 段之外空闲占比」是否显著，以及是否愿意做跨设备 stage 解耦。

Q4. Layer A 和 Layer B 的优先级？
    Layer B（native executor）带来 kernel 加速的地基但短期 0 收益；
    Layer A（rollout 重叠）可能直接提吞吐。优先做哪个取决于 Q2。
```

## 建议的下一步（非承诺，待讨论后定）

```text
Step 1（必做、低成本）：rollout profiling pass
  在现有 vrl/generation 路径上加一次 profiling：
    - 单 chunk 内 text-encode / denoise / VAE 三段各自占用 wall-clock 与 GPU busy 比例
    - 数据并行下 worker 间的负载均衡 / 尾延迟
    - 权重同步与 staleness 等待在一个 training step 里的占比
  产出：一份 measured breakdown，直接回答 Q2。
  （这属于 one-shot validation 工件，profiling 脚本用完即弃，结论写回本文档。）

Step 2（条件触发）：仅当 Step 1 显示 transformer 段之外空闲显著，
  才设计 stage 解耦 + 跨设备 relay 的 phase 方案，届时再开独立 sprint。

Step 3（与 Layer B 解耦）：native transformer executor 按
  SPRINT_diffusion_native_transformer_executor 自己的节奏走，
  其 Phase 7 benchmark gate 的数据也能反哺 Q4 的优先级判断。
```

## Non-goals

```text
不照搬 sglang-omni 的 Coordinator/Stage/Relay/ZMQ 重写现有 rollout
不把 AR paged KV / block table / prefix cache / chunked prefill 引入 diffusion
不在没有 profiling 数据前就动 execution/continuous 的工作架构
不把本 sprint（Layer A）和 native executor sprint（Layer B）合并成一个大改造
不预建任何没有被实测瓶颈支撑的 stage-pipeline skeleton
```

## 参考

- `vrl/generation/execution/planner.py`（`ExecutionStage:54`、`EnginePlan:100`）
- `vrl/generation/execution/scheduler.py`（`DeviceAssignment`）
- `vrl/generation/execution/chunks.py`（`SampleChunk`、`build_prompt_chunk_schedule`、`run_sample_chunks_with_oom_retry`）
- `vrl/generation/diffusion/executor.py`
- `vrl/generation/ray/{launcher,placement,weight_sync,worker}.py`
- `vrl/rollouts/orchestration/continuous/{producer,consumer,queue,staleness}.py`
- `docs/sprints/planned/SPRINT_diffusion_native_transformer_executor.md`（Layer B）
- 外部对标：`/home/mingfeiguo/Desktop/sglang-omni`（AR-LLM 多模态 serving 框架，不含 diffusion）
