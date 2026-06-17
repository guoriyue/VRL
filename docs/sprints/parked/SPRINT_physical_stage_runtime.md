# SPRINT: Physical stage runtime（future）

状态：foundation contract layer 已落地（b224383，PipelineTopology / PipelineStagePayload / SerialPipelineRunner / RayPipelineStageWorker / RayPipelineRunner + 12 passing contract tests），但仅是未接线的 Level-1 actor-safe seam（无任何 rollout/trainer/config 引用）；physical Ray stage runtime（StageScheduler、bounded queues、tensor relay，§6 P0-P4）尚未开工，仍 parked 等待 multi-GPU + cross-family profiling gate（§3）。

## 0. Core Decision

不要把 SGLang-Omni 式 physical stage runtime 塞进当前的 planner label 或
cross-model performance sprint。要做 staging，但分两层：

```text
foundation staging:
  现在做。typed payloads、serial_staged、per-stage metrics、placement schema。

physical staging:
  多 GPU / cross-family profile gate 打开后做。Ray stage actors、queues、relay。
```

现有 `ExecutionStage` / chunk planner 仍然只是逻辑执行标签和 profiler label：

```text
prompt P 的 sample chunk
batch_group_key / cache label / profiler stage
```

未来要做的 physical stage runtime 是另一层边界：

```text
prompt_encode -> denoise -> decode_latents -> reward_artifact
```

每个 physical stage 会拥有自己的模型组件、Ray actor 生命周期、GPU placement、batch
策略、队列、backpressure、payload contract 和 failure/cancel 语义。它不是给当前
planner label 加字段可以自然演进出来的东西。

### 0.1 Chunk vs pipeline

`SampleChunk` 和 physical pipeline 是两层，不要互相替代。

```text
GenerationRequest
  -> split into SampleChunk(s)
      -> each SampleChunk is executed through a stage pipeline
          prompt_encode -> prepare_latent -> denoise -> decode_latents
```

`SampleChunk` 回答的是：

```text
这次 rollout 要生成哪一小批样本？
prompt index 是多少？
sample range 是多少？
OOM retry 要 split 哪个 batch?
Ray chunk worker 要接收哪个 work unit?
```

Physical pipeline 回答的是：

```text
这个 work unit 要经过哪些 runtime stage?
哪个 stage owns 哪些 model components?
哪个 stage 放在哪张 GPU?
每个 stage 用什么 batch / queue / memory policy?
stage 之间用什么 tensor relay?
```

因此边界应该是：

```text
GenerationChunkExecutor:
  logical family executor boundary.
  Receives one SampleChunk and returns one ChunkResult.

PipelineTopology / PipelineStage:
  physical runtime topology.
  Decides how that SampleChunk moves through actors/stages.
```

没有 `SampleChunk`，rollout 会丢失 group/sample identity、batch split、seed、OOM retry
和 gather 语义。没有 physical pipeline，`prompt_encode / denoise / decode_latents` 只能在
同一个 worker 里串行执行，不能给 denoise 和 decode 独立 placement、batch policy、
memory policy 或 backpressure。

两轴在 foundation 阶段严格正交（chunk = 数据轴按 sample 切行，stage = 计算轴按
denoise/decode 切列，stage 方法的工作单位就是 chunk）。但 physical 阶段有两个
**有意的交叉点**，不能误读成永远正交：

```text
1. per-stage 重组 batch（§2.4）：denoise batch 16 / decode batch 2 意味着 chunk
   不再是跨 stage 不变量 —— 物理流通单位是 payload，stage 可重新分组；payload
   携带 sample_identity 正是为此准备的接缝。
2. OOM/失败语义（§2.7）：今天 OOM 切 chunk（数据轴），但 decode OOM 会把该
   chunk 的所有 stage（含 denoise）重跑 —— 失败边界把两轴耦合，拆分前必须显式定义。
```

当前已落地的 foundation contract：

```text
vrl/generation/pipeline/topology.py
vrl/generation/pipeline/payload.py
vrl/generation/pipeline/runner.py
vrl/generation/ray/stage_worker.py
vrl/generation/ray/pipeline_runner.py
```

这些文件只建立 actor-safe contract 和 serial/Ray routing seam；它们不是完整 scheduler。

## 1. 为什么不属于当前 cross-model sprint

当前 cross-model performance sprint 的目标是：

```text
GPM scale bump
VAE tiling / decode memory policy
rollout.n / sample_batch_size live gate
OOM retry 保持可用
```

这些都是在现有 chunk worker 边界内做容量与吞吐验证。physical stage runtime 会改变
runtime topology、tensor transport、weight sync ownership 和 RL replay payload 语义，
属于分布式 runtime 重构，不能混入当前性能 sprint。

## 2. 必须新增的 runtime 边界

### 2.1 Stage graph / topology

目标拓扑不是现有的“一个 chunk 内串行跑完”，而是显式图：

```text
prompt_encode -> denoise -> decode_latents -> maybe reward_artifact
```

拓扑必须声明：

```text
stage 名字
上游 / 下游
terminal stage
payload schema
ordering / cancellation rule
retry boundary
```

### 2.2 Per-stage Ray actors

每个 stage 单独 owns：

```text
model components
GPU placement
memory lifecycle
compile / warmup policy
health check
restart behavior
```

典型 ownership：

| stage | owns | trainable |
| --- | --- | --- |
| `prompt_encode` | text encoder / tokenizer side payload | usually no |
| `denoise` | trainable diffusion transformer / scheduler step | yes |
| `decode_latents` | VAE decode | no |
| `reward_artifact` | artifact serialization / reward input preparation | no |

### 2.3 Tensor relay

stage 之间不能默认靠 Ray object store 反复 CPU copy 大 tensor，否则 latent / replay tensor
传输会吞掉收益。

候选 relay 机制：

```text
CUDA IPC for same-node GPU handoff
NCCL or CUDA-aware transfer for cross-GPU tensor movement
shared memory for CPU artifact handoff
Mooncake / NIXL-style relay only after profiling proves it is needed
```

首版可以用 bounded Ray queues 验证调度收益，但必须把它标成 validation path，不得把
CPU tensor relay 当作长期 runtime 设计。

### 2.4 Per-stage scheduler / queue / backpressure

每个 stage 的最佳 batch policy 不一样：

```text
denoise: batch 8/16, cost ~= samples * timesteps * latent tokens
decode_latents: batch 1/2/4, cost ~= samples * frames * height * width
reward_artifact: batch N, cost 由 reward backend / artifact format 决定
```

因此需要 per-stage：

```text
max_batch_size
max_batch_wait_ms
max_inflight
cost function
queue depth
cancel / drain rule
backpressure credits
```

核心规则：denoise 不能生产超过 decode 能 drain 的 latent payload；decode 不能生产超过
reward 能 drain 的 artifact payload。

### 2.5 RL replay semantics

diffusion RL 的 denoise stage 不只产 final latent。训练还需要：

```text
observations
actions
log_probs
timesteps
kl / reference policy fields
policy_version
sample / group identity
```

这些 replay tensors 是 trainer contract，不是普通 inference pipeline 的临时中间结果。
physical stage runtime 必须先定义 replay payload schema，再谈 stage 拆分。

### 2.6 Weight sync ownership

denoise stage owns trainable transformer，所以它必须参与每次 policy update 的 weight sync。
frozen stages 不应该跟着同步：

```text
prompt_encode: frozen, no policy sync unless text encoder is trainable
denoise: trainable, policy_version barrier + weight sync
decode_latents: frozen, no policy sync
reward_artifact: no policy sync
```

同步协议必须覆盖：

```text
pause admission
drain in-flight denoise / replay payloads
sync denoise weights
stamp new policy_version
resume admission
```

不能让一个 prompt 的 replay tensors 混入两个 policy versions。

### 2.7 Profiler / determinism / OOM retry

现有 chunk worker 绑定了这些语义：

```text
chunk OOM retry / split
seed and generator state
runtime_debug
policy_version
stage_durations
trajectory assembly
```

physical stage runtime 后必须重新定义这些跨 stage 行为：

```text
OOM 发生在 decode 时，是否回退 denoise batch？
denoise 重试是否复用同一 seed / latent init？
cancel 后哪些 replay tensors 必须丢弃？
stage duration 如何汇总回一个 rollout sample？
debug artifact 如何跨 stage 关联？
```

## 3. Gate：什么时候重新打开

Foundation staging 已经可以开始，因为 video-heavy families 显示 decode 不是小尾巴。
Physical Ray stage actors 只有同时满足下面条件才进入 prototype：

```text
1. profiling 显示现有 chunk boundary 是主要瓶颈，而不是 denoise forward / reward CPU / weight sync。
2. encode/decode/reward bubble 足够大，超过 physical pipeline 的 relay + queue 开销。
3. 单纯提升 sample_batch_size、VAE tiling、OOM split、reward batching 已不能继续解决。
4. 有明确的 GPU placement 方案，例如 denoise 和 decode/reward 分卡后收益大于传输成本。
```

不满足 gate 时，继续优化现有 chunk worker 和 continuous rollout，不预建 pipeline skeleton。

### 3.1 Cross-family evidence

不要再用 SD3.5 OCR 单独决定 staging。现有 evidence 已经显示 video-heavy families
和 SD3.5 OCR 不同：

| profile | timing | batch | steps | encode | denoise | decode | encode+decode share | decode/denoise |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SD3.5 OCR old profile | wall | ? | ? | 0.499s | 78.664s | 0.571s | 1.3% | 0.7% |
| Anima rbs4 smoke | wall | 4 | 10 | 0.128s | 2.771s | 0.489s | 18.2% | 17.6% |
| Wan sm_profile | wall | 4 | 20 | 0.184s | 19.002s | 3.744s | 17.1% | 19.7% |
| Cosmos2.5 bs6 profiler | CUDA total | ? | 10 | 0.169s | 1.832s | 0.920s | 37.3% | 50.2% |

Read:

```text
SD3.5 OCR:
  not enough encode/decode bubble to justify physical actors.

Wan / Anima:
  decode is around 18-20% of denoise, so staging foundation is justified.

Cosmos2.5:
  decode CUDA total is about half of denoise CUDA total, so this is the first
  physical-stage candidate once multi-GPU relay can be measured.
```

These numbers are not a single strict benchmark because the timing source differs:
Wan / Anima use runtime wall-time counters, while Cosmos2.5 uses torch profiler
CUDA total. The next measurement pass must record both for all video families.

## 4. Proof / throughput inequality

Physical pipeline is justified only when the staged steady-state bound beats the
current fused chunk bound after relay and queue overhead are included.

For one logical rollout chunk, define measured serial stage costs:

```text
T_prompt
T_denoise
T_decode
T_reward
```

The current fused runtime processes a chunk as:

```text
T_fused = T_prompt + T_denoise + T_decode + T_reward
```

A physical pipeline with independent workers has a steady-state lower bound:

```text
T_pipeline_step = max(
  (T_prompt + R_prompt_out + Q_prompt) / N_prompt_workers,
  (T_denoise + R_denoise_in + R_denoise_out + Q_denoise) / N_denoise_workers,
  (T_decode + R_decode_in + R_decode_out + Q_decode) / N_decode_workers,
  (T_reward + R_reward_in + Q_reward) / N_reward_workers,
)
```

where:

```text
R_* = measured relay / tensor movement cost on that stage boundary
Q_* = measured queueing / scheduler / backpressure cost
N_* = number of physical workers for that stage
```

The pipeline has a throughput reason to exist only if:

```text
T_pipeline_step < T_fused
```

The stricter practical gate is:

```text
T_pipeline_step + correctness/ops overhead margin < T_fused
```

Equivalently, for a first split around denoise:

```text
relay + queue overhead < work hidden behind denoise
```

If denoise is the dominant stage and encode/decode/reward together are tiny,
then the maximum possible overlap win is tiny before relay is even counted.
The existing SD3.5 OCR profile had:

```text
encode:   0.499s
decode:   0.571s
denoise: 78.664s
```

So hiding encode+decode can recover at most:

```text
(0.499 + 0.571) / (78.664 + 0.499 + 0.571) ~= 1.34%
```

before queueing, relay, failure handling, and extra actors. That does not prove
that physical pipeline is bad; it proves the current single-GPU SD3.5 OCR profile
does not justify implementing it as the next performance lever.

Future multi-GPU can change the inequality. The foundation should therefore
measure the terms above and keep payload boundaries explicit, but it should not
pre-build a full stage runtime before the inequality can turn positive.

### 4.1 Timing vocabulary

Use both timings; they answer different questions.

```text
wall time:
  Host-observed elapsed time around a code region. In current runtime_debug this
  comes from time.perf_counter() around encode / denoise / decode blocks.
  It includes Python overhead, scheduling, CUDA launch overhead, implicit waits,
  allocator stalls, and any CPU-side work. Because CUDA launches are async,
  wall timing only becomes stage-accurate when the region naturally synchronizes
  or when the profiler inserts synchronization.

CUDA total:
  GPU time attributed by torch profiler to kernels under a record_function range.
  It answers how much GPU work happened under that range. It does not include
  Python overhead, queue wait, Ray overhead, or CPU-side scheduling cost.
```

For physical pipeline decisions:

```text
wall time tells us visible rollout latency and queue/relay impact.
CUDA total tells us whether there is actual GPU work worth placing on another GPU.
```

Do not use only one. A stage with high wall time but low CUDA total may be CPU /
launch / sync bound; moving it to another GPU may not help. A stage with high
CUDA total is a real GPU placement candidate, but wall time still decides whether
relay and queues erase the gain.

## 5. Foundation work to do now

These changes prepare the future pipeline without committing to physical actors:

```text
typed payload schemas for denoise output, decode input/output, and reward input
serial_staged mode that runs the same payloads inside one process
per-payload tensor byte accounting
per-stage wall time and CUDA time in runtime_debug
explicit policy_version / sample identity fields on stage payloads
placement schema draft with validation but no default physical execution path
```

This is the correct foundation because it preserves today’s fused chunk fallback
while making future physical workers a transport/scheduler substitution, not a
semantic rewrite of RL rollout payloads.

### 5.1 Ray actor / scheduler feasibility

这是 doable，但不要一跳做到完整 SGLang-Omni runtime。正确顺序是三层：

```text
Level 1: driver-routed Ray stage actors
  One actor per physical stage.
  Driver routes PipelineStagePayload through PipelineTopology.
  Purpose: validate actor boundary, placement metadata, payload serialization,
  stage metrics, and error envelopes.

Level 2: per-stage bounded queues
  Each stage owns an inbox/outbox and max_inflight policy.
  Driver submits SampleChunk payloads; stage workers pull/drain with backpressure.
  Purpose: prove overlap opportunity and catch decode/reward drain bottlenecks.

Level 3: real stage scheduler + relay
  Stage-local batching, cost functions, queue depth, cancellation, drain, and
  non-CPU tensor relay.
  Purpose: production physical pipeline.
```

Level 1 is already represented by the current contract layer:

```text
PipelineTopology
PipelineStagePayload
PipelineStageWorkerCore
RayPipelineStageWorker
RayPipelineRunner
```

Level 2 is the next useful scheduler boundary. It should schedule `SampleChunk`
payloads through stage queues, not replace `SampleChunk` itself. The scheduler
should see:

```text
request_id
policy_version
sample identity / chunk key
current stage
stage runtime policy
payload byte size
terminal / error state
```

Level 3 only starts after Level 2 proves the inequality in section 4. Otherwise
the CPU/Ray queue and relay cost can be larger than the overlap benefit.

First scheduler shape:

```text
StageScheduler
  inbox: bounded queue[PipelineStagePayload]
  outbox: bounded queue[PipelineStageResult]
  batch policy: PipelineStageRuntimePolicy
  admission: policy_version barrier + queue credits
  drain: explicit shutdown / weight-sync barrier
```

This stays close to SGLang-Omni's `stage` concept, but VRL keeps different
payload semantics because diffusion RL carries replay tensors, sample identity,
and policy version constraints.

## 6. Proposed phases（打开 gate 后）

### P0. Measurement

在现有 runtime 上补足证据：

```text
per-stage wall time and GPU utilization
Ray object store / tensor transfer cost
in-flight reward drain time
OOM retry frequency and split depth
policy sync drain time
```

输出是一份 measured breakdown，不是新 runtime。

### P1. Payload schema

先冻结跨 stage contract：

```text
latent payload
replay tensor payload
decoded artifact payload
reward artifact payload
sample identity / group identity / policy_version
debug metadata
```

没有 schema，不写 actor。

### P2. Single-node prototype

只做单节点、显式 GPU placement、bounded queues：

```text
denoise actor
decode actor
reward artifact actor if needed
bounded queue / backpressure
no cross-node relay
```

目标是验证调度收益，不追求最终 transport。

### P3. Relay replacement

如果 P2 证明 stage 拆分有效，再把 validation transport 换成 CUDA IPC / NCCL / shared
memory 等长期 relay。没有 P2 收益，不做 P3。

### P4. Continuous rollout integration

最后再接入现有 continuous producer / consumer / weight sync barrier。不能让 trainer
消费未打分 batch，也不能绕过现有 staleness contract。

## 7. Non-Goals

```text
不改当前 cross-model performance sprint 的 GPM scale bump 目标
不把 ExecutionStage 扩成 Ray actor / placement / queue 配置
不照搬 SGLang-Omni 的 AR serving scheduler、paged KV、prefix cache、tree cache
不把 CPU Ray object store relay 当作长期解法
不在没有 profiling gate 前预建空 pipeline 框架
不让 frozen VAE/text encoder 跟随 policy weight sync
```

## 8. References

- `docs/sprints/info/SPRINT_cross_model_performance.md`
- `docs/sprints/parked/SPRINT_diffusion_rollout_stage_pipeline.md`
- `docs/sprints/reading/SPRINT_diffusion_rollout_system.md`
- `docs/sprints/parked/SPRINT_generation_scheduler.md`
- `docs/sprints/planned/SPRINT_slime_overlap_strategy.md`
- `vrl/generation/execution/planner.py`
- `vrl/generation/execution/chunks.py`
- `vrl/generation/protocols.py`
- `vrl/generation/pipeline/topology.py`
- `vrl/generation/pipeline/payload.py`
- `vrl/generation/pipeline/runner.py`
- `vrl/generation/diffusion/executor.py`
- `vrl/generation/ray/stage_worker.py`
- `vrl/generation/ray/pipeline_runner.py`
- `vrl/rollouts/orchestration/continuous/producer.py`
- `vrl/rollouts/orchestration/continuous/consumer.py`
- `vrl/generation/ray/weight_sync.py`
