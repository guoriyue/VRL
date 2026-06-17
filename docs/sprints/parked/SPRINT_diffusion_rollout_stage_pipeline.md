# SPRINT: Diffusion rollout stage pipeline

状态：部分落地（设计 + 已建基座）。T1（DiffusionExecutor 内 typed stage payloads + 方法对齐）已随 b224383 "Add generation stage pipeline foundation" 落地——forward_chunk_plan 现以 DiffusionPromptStageInput→DiffusionPromptStageOutput→DiffusionPreparedStageOutput→DiffusionDenoisedStageOutput 串接 build_prompt_stage_input/run_prompt_encode_stage/run_prepare_stage/run_denoise_stage/run_decode_stage（executor.py:437-454），且同 commit 落地 vrl/generation/pipeline/ 契约层（PipelineTopology/PipelineStage/SerialPipelineRunner，tests/generation/pipeline 12 passed）；但 T0（唯一 immediate 项：model.memory.vae_decode.batch_size YAML 旋钮）仍未做——VaeDecodeMemory 仅有 tiling/slicing，batch_size 会被当 unknown key 拒绝，decode_batch_size 只走 getattr(pipe,...)（sd3_5/model.py:404）；T2–T6（stage_pipeline config / serial_staged / ray_staged 物理管线）未接入任何生产路径，仍待 profiling gate（§4，内部测量决策，非外部事件）。

Status: discussion; gated by profiling. Only T0 is an immediate implementation
candidate.

This document records the stage-pipeline direction for diffusion rollout. It is
not an approved implementation sprint yet. The goal is not to migrate to
SGLang-Omni. The useful part to study is its stage-runtime shape, but VRL should
only build a physical stage pipeline after profiling proves that the single
physical chunk boundary is a real bottleneck.

Immediate work should go through
`docs/sprints/parked/SPRINT_runtime_block_policies.md`: it covers per-block batch,
memory, and compile policies while keeping execution serial inside the existing
executor.

## 0. Core Decision

Do not build the full physical stage pipeline now.

VRL already has the high-level rollout system:

```text
continuous producer/consumer
Ray generation workers
sample chunk planning
OOM retry
policy_version / weight sync barriers
trajectory gather
```

Today `DiffusionExecutor.forward_chunk_plan()` is already a coordinator with
logical method boundaries and per-phase timings. It still runs one sample chunk
as a single physical unit:

```text
prompt encode -> prepare latent -> denoise loop -> VAE decode -> chunk result
```

That shape could become limiting in the future because it prevents
stage-specific batch sizing and placement. In particular, VAE decode may need a
smaller mini-batch or a separate GPU while denoise keeps the largest stable
batch size possible.

But the current SD3.5 OCR profile does not justify building a full stage
pipeline yet:

```text
encode:   0.499s total
decode:   0.571s total
denoise: 78.664s total
```

`encode + decode` is less than 1.5% of denoise. That means the current rollout
does not have a meaningful encode/decode bubble to hide. The main bottleneck is
still `generation.denoise_forward`, so the active performance path remains the
denoise transformer path from `SPRINT_rollout_performance.md`.

Future target shape, only if the gate opens:

```text
prompt_encode -> denoise -> vae_decode -> reward -> gather
```

Each stage gets its own:

```text
batch size
max inflight count
placement
memory budget
profiling label
payload contract
```

Expected benefit if the gate opens:

```text
reduce visible stage bubbles
let memory-heavy stages stop blocking denoise batch size
separate denoise / VAE / reward placement on multi-GPU rollout
```

Current immediate work:

```text
T0 only: expose VAE decode mini-batch config.
```

T0 is still worth doing because it is a small memory-control fix. It can unblock
larger denoise batches or avoid VAE decode OOM. It is not evidence that the full
stage pipeline should start now.

## 1. What To Borrow From SGLang-Omni If The Gate Opens

Borrow the stage-runtime shape, not the model/runtime implementation.

### 1.1 Pipeline worker schema

SGLang-Omni has a useful stage declaration shape:

```python
class StageConfig(BaseModel):
    name: str
    factory: str
    next: str | list[str] | None = None
    terminal: bool = False
    gpu: int | list[int] | None = None
    runtime: StageRuntimeConfig = Field(default_factory=StageRuntimeConfig)
    wait_for: list[str] | None = None
    stream_to: list[str] = Field(default_factory=list)
    relay: RelayConfig | None = None
```

VRL should use a diffusion-specific version of this idea, not this exact class.
It should also avoid introducing a generic `StageSpec` name because VRL already
uses `ExecutionStage` for planner-visible execution labels:

```python
class ExecutionStage:
    """One planner-visible execution stage and profiler label."""
```

Naming decision:

```text
ExecutionStage
  Existing planner/profiler concept.
  Keep it as request-plan metadata and profiler labeling.
  Do not add placement, queues, or worker lifecycle to it.

PipelineWorkerSpec / PipelineTask / PipelinePayload
  Future physical pipeline runtime concepts.
  These own placement, batch sizing, max inflight, queueing, and payload flow.
```

`ExecutionStage` can keep recording profiler labels for the future pipeline, but
the physical worker concept needs a separate name and boundary.

Future config surface:

```yaml
rollout:
  stage_pipeline:
    enabled: true
    stages:
      denoise:
        gpu: [0, 1]
        batch_size: 16
        max_inflight: 2
      vae_decode:
        gpu: [2]
        batch_size: 2
        max_inflight: 2
      reward:
        gpu: [3]
        batch_size: 8
        max_inflight: 2
```

### 1.2 Simple scheduler contract

SGLang-Omni's `SimpleScheduler` has the right knobs for non-AR pipeline
workers:

```text
batch_compute_fn
max_batch_size
max_batch_wait_ms
request_cost_fn
max_batch_cost
max_concurrency
```

VRL should mirror this at the pipeline worker level:

```text
denoise cost    = samples * num_steps * latent_tokens
vae_decode cost = samples * frames * height * width
reward cost     = decoded artifacts
```

This lets each pipeline worker batch according to its own memory and throughput
curve.

### 1.3 Placement validation

SGLang-Omni validates per-worker GPU memory fractions before starting workers.
VRL needs the same idea because denoise, VAE, and reward can share or split GPUs.

Minimal VRL version:

```text
PipelineWorkerPlacement(worker_name, gpu_ids, max_memory_fraction)
GpuPipelinePlacement(gpu_id, worker_names, total_memory_fraction)
```

Reject invalid placement before launching Ray actors.

### 1.4 Relay/backpressure idea

SGLang-Omni uses relay credits to prevent upstream workers from producing
unbounded payloads. VRL needs the same backpressure behavior, but the first
implementation should use bounded Ray tasks/queues instead of a custom relay.

Initial rule:

```text
denoise must not produce more latent payloads than decode can drain
decode must not produce more decoded artifacts than reward can drain
```

Only add CUDA IPC / NCCL / NIXL-style transport after profiling shows Ray object
or CPU transfer is the bottleneck.

## 2. What Not To Borrow

Do not copy these SGLang-Omni parts:

```text
OmniScheduler
AR KV cache logic
prefill/decode token scheduling
tree/prefix cache
SGLang server args
model registry
ZMQ control plane
NIXL/NCCL relay as the first transport
```

Those solve AR/multimodal serving problems. SD3.5/Wan/Cosmos diffusion rollout
needs dense denoise batching, pipeline-worker placement, and backpressure.

## 3. Current VRL Boundaries

Current `forward_chunk_plan()` is not an undifferentiated monolith. It is
already a coordinator over existing logical boundaries:

```python
def forward_chunk_plan(...):
    encoded = self.encode_prompt_for_chunk(...)
    stage_durations["encode"] = ...

    chunk_encoded = self.build_chunk_encoded(...)
    prepare_kwargs = self.build_prepare_kwargs(...)
    config = self.build_denoise_config(...)
    state = self.prepare_denoise_state(...)
    stage_durations["prepare_latent"] = ...

    denoise_result = self.run_denoise_steps(...)
    stage_durations["denoise"] = ...

    chunk_result = self.decode_denoise_result(...)
    chunk_result.stage_durations["decode"] = ...
    return chunk_result
```

So the existing serial logical boundaries are:

```text
encode_prompt_for_chunk
build_chunk_encoded / build_prepare_kwargs / build_denoise_config
prepare_denoise_state
run_denoise_steps
decode_denoise_result
```

The missing part is not "split the method into stages." That is mostly already
done. The missing part is typed payloads plus optional physical pipeline workers:

```text
typed payload/result contracts
pipeline-step name alignment with the existing methods
bounded queues
placement
worker lifecycle
```

Current denoise output already exists:

```python
@dataclass(slots=True)
class DiffusionDenoiseResult:
    state: Any
    observations: Any
    actions: Any
    log_probs: Any
    timesteps: Any
    kl: Any
    peak_memory_mb: float | None = None
    engine_counters: dict[str, Any] = field(default_factory=dict)
```

This is the first useful split point. It already separates trainable denoise
from VAE artifact decode.

Current final chunk contract:

```python
@dataclass(slots=True)
class DiffusionChunkResult:
    prompt_index: int
    sample_start: int
    sample_count: int
    observations: Any
    actions: Any
    log_probs: Any
    timesteps: Any
    kl: Any
    video: Any
    replay_tensors: dict[str, Any]
    context: dict[str, Any]
```

This final contract should stay stable at first so `DiffusionChunkGatherer` and
the trainer do not change.

## 4. Profiling Gate

The full stage pipeline stays blocked until a new profile shows at least one
real bottleneck outside denoise forward.

Gate opens if one of these is true after T0 and the current denoise-forward
optimization path:

```text
decode, reward, text encode, or queue wait is >= 10% of rollout wall time
VAE decode memory still prevents the target denoise batch size
multi-GPU rollout shows denoise workers idle while decode/reward drains
Ray transfer or inter-stage payload movement is a measured top bottleneck
```

Gate stays closed while this remains true:

```text
generation.denoise_forward is still the global dominant rollout cost
encode/decode together remain around 1-2% of denoise
no larger denoise batch is blocked by VAE decode memory
```

This mirrors `SPRINT_rollout_performance.md`: optimize denoise first; only then
consider batch-level staged rollout.

## 5. Future Target Architecture

This section is a design reference for the gate-open path, not a current build
plan.

### 5.1 Pipeline specs

Add a diffusion pipeline spec layer. Keep it narrow and typed.

```text
vrl/generation/pipeline/
  specs.py          # PipelineWorkerSpec, PipelineRuntimeSpec, PipelinePlacementSpec
  placement.py      # validate pipeline worker -> GPU/resource mapping
  scheduler.py      # bounded batching scheduler for non-AR workers
  runner.py         # in-process pipeline worker contract

vrl/generation/diffusion/pipeline.py
  DiffusionEncodePayload
  DiffusionDenoisePayload
  DiffusionDecodePayload
  DiffusionRewardPayload
  diffusion_pipeline_graph(...)
```

Do not add an abstract framework wider than the first diffusion use case needs.

### 5.2 Pipeline payloads

Suggested payload flow:

```text
DiffusionEncodePayload
  request
  chunk
  params
  video_request

DiffusionPreparePayload
  request
  chunk
  params
  video_request
  encoded

DiffusionDenoiseOutput
  request_id
  prompt_index
  sample_start
  sample_count
  denoise_result
  config
  stage_durations

DiffusionDecodeOutput
  request_id
  prompt_index
  sample_start
  sample_count
  decoded_video
  replay_tensors
  context
  denoise_result
  stage_durations
```

The final adapter converts `DiffusionDecodeOutput` back into the existing
`DiffusionChunkResult`.

### 5.3 Execution modes

Support two execution modes behind one config flag:

```text
serial_staged
  same process, same worker, explicit pipeline payload boundaries
  used to validate contracts and metrics first

ray_staged
  physical Ray pipeline workers with per-worker placement
  used for multi-GPU rollout
```

Keep the current fused path:

```text
fused_chunk
  current forward_chunk_plan behavior
  fallback path for debugging and model-family bring-up
```

## 6. Gated Implementation Plan

### T0: Add explicit VAE decode mini-batch config

Status: immediate candidate.

Goal: make the immediate memory lever official without building a stage runtime.

Current latent decode already supports `decode_batch_size`, but SD3.5 does not
expose a canonical YAML knob for it.

Add:

```yaml
model:
  memory:
    vae_decode:
      batch_size: 2
```

Extend the existing VAE memory policy parser so it owns:

```text
tiling
slicing
batch_size
```

Acceptance:

```text
SD3.5 decode uses model.memory.vae_decode.batch_size
Wan/Cosmos existing tiling/slicing behavior is unchanged
config tests reject unknown keys
latent decode tests cover batch_size
```

### T1: Add typed payloads and align naming inside `DiffusionExecutor`

Status: gated by profiling.

Goal: keep behavior identical while making the existing logical boundaries
explicit enough to feed a future physical pipeline.

Do not add a parallel set of `run_encode_stage()` / `run_denoise_stage()`
wrappers on top of the existing methods. The current methods are already the
logical boundaries. If names change, treat it as a rename/alignment of existing
methods, not as duplicate wrappers.

Current logical boundary methods:

```python
def encode_prompt_for_chunk(...)
def build_chunk_encoded(...)
def build_prepare_kwargs(...)
def prepare_denoise_state(...)
def run_denoise_steps(...)
def decode_denoise_result(...)
```

Future T1 work is therefore:

```text
add typed payload/result objects around the existing method boundaries
align naming with future PipelineTask/PipelinePayload concepts
keep `forward_chunk_plan()` as the serial coordinator
avoid introducing new methods that duplicate existing boundaries
```

Acceptance:

```text
existing diffusion generation tests pass
stage_durations preserve encode/prepare_latent/denoise/decode
no trainer or gather contract changes
new payload names do not conflict with ExecutionStage
```

### T2: Add serial pipeline executor mode

Status: gated by profiling.

Goal: validate the pipeline payload API without Ray placement complexity.

Add config:

```yaml
rollout:
  stage_pipeline:
    enabled: true
    mode: serial
```

In serial mode, the executor still runs in one worker, but the code path uses
pipeline payload/result classes and per-worker metrics.

Acceptance:

```text
fused_chunk and serial_staged outputs match for fixed seed
policy_version behavior is unchanged
precision drift guard still passes
pipeline metrics report per-worker wall time and tensor bytes
```

### T3: Add pipeline-worker-aware Ray planning

Status: gated by profiling.

Goal: make physical placement possible.

Extend planning from:

```text
chunk -> worker
```

to:

```text
chunk pipeline task -> pipeline worker pool
```

Initial physical pipeline workers:

```text
denoise
vae_decode
reward
```

Do not split every denoise timestep. Diffusion timestep dependency is serial:

```text
x_t -> transformer -> scheduler -> x_{t-1}
```

The useful pipeline is across chunks, not inside one denoise chain.

Acceptance:

```text
pipeline worker placement validates configured GPU ids
denoise worker can feed decode worker through bounded queue
decode worker can feed reward path without changing final trajectory shape
policy_version mismatch drains or rejects stale pipeline payloads
```

### T4: Add bounded pipeline queues and backpressure

Status: gated by profiling.

Goal: prevent the physical pipeline from creating new memory pressure.

Rules:

```text
max_inflight per pipeline worker is enforced
upstream worker blocks when downstream queue is full
request abort or policy_version change cleans queued payloads
OOM in one pipeline worker reports worker name and payload identity
```

Acceptance:

```text
denoise cannot enqueue unlimited latent payloads
decode cannot enqueue unlimited videos
OOM retry still splits at the sample-chunk boundary
metrics include queue wait time by pipeline worker
```

### T5: Multi-GPU SD3.5 OCR rollout validation

Status: gated by profiling.

Goal: prove the pipeline improves capacity or throughput on the real target.

Minimum scenarios:

```text
baseline fused_chunk on one GPU
serial_staged on one GPU
ray_staged with denoise and VAE decode separated
ray_staged with denoise, VAE decode, and reward separated
```

Metrics:

```text
images/sec
GPU active time by pipeline worker
queue wait time by pipeline worker
peak memory by pipeline worker
OOM rate
policy staleness
reward curve parity
precision drift guard result
```

Acceptance:

```text
stage pipeline does not regress reward correctness
stage pipeline enables at least one larger denoise batch that fused_chunk cannot run
or stage pipeline improves end-to-end rollout throughput at the same batch size
```

### T6: Transport upgrade gate

Status: gated by profiling.

Goal: avoid premature relay complexity.

Only implement CUDA IPC / NCCL / NIXL-style tensor relay if profiling shows:

```text
inter-stage transfer is a top rollout bottleneck
or Ray object transfer creates measurable GPU idle bubbles
```

Until then:

```text
use Ray object store / CPU transfer
keep tensor payloads explicit and measured
```

## 7. Throughput Model

After warmup, stage pipeline throughput is limited by the slowest normalized
stage:

```text
throughput ~= 1 / max(
  T_denoise / num_denoise_workers,
  T_vae_decode / num_decode_workers,
  T_reward / num_reward_workers
)
```

The stage pipeline is useful when it lets us tune each stage independently:

```text
denoise: large batch, trainable transformer, compile/low precision
vae_decode: smaller batch, memory-heavy artifact decode
reward: OCR/reward batch, possibly separate device
```

Do not evaluate this sprint by asking whether every kernel reaches 100% SM
occupancy. Evaluate it by rollout throughput, GPU idle bubbles, OOM rate, and
training correctness.

## 8. Non-goals

```text
Do not migrate to SGLang-Omni.
Do not copy OmniScheduler or AR KV-cache logic.
Do not split individual denoise timesteps into separate pipeline stages.
Do not mark this as an implementation sprint while denoise forward dominates.
Do not replace diffusers transformer forward in this sprint.
Do not replace VAE implementation in this sprint.
Do not change trainer trajectory semantics.
Do not add custom relay transport until transfer is proven to bottleneck.
Do not remove the current fused_chunk fallback.
```

## 9. Source References To Follow

VRL:

```text
/home/mingfeiguo/Desktop/wm-infra/vrl/generation/diffusion/executor.py
/home/mingfeiguo/Desktop/wm-infra/vrl/generation/diffusion/gather.py
/home/mingfeiguo/Desktop/wm-infra/vrl/generation/execution/planner.py
/home/mingfeiguo/Desktop/wm-infra/vrl/generation/execution/scheduler.py
/home/mingfeiguo/Desktop/wm-infra/vrl/generation/ray/executor.py
/home/mingfeiguo/Desktop/wm-infra/vrl/models/diffusion/common/latent_decode.py
/home/mingfeiguo/Desktop/wm-infra/vrl/models/diffusion/common/vae_decode_memory.py
/home/mingfeiguo/Desktop/wm-infra/vrl/models/diffusion/sd3_5/model.py
/home/mingfeiguo/Desktop/wm-infra/docs/sprints/info/SPRINT_rollout_performance.md
/home/mingfeiguo/Desktop/wm-infra/docs/sprints/reading/SPRINT_diffusion_rollout_system.md
```

SGLang-Omni:

```text
/home/mingfeiguo/Desktop/sglang-omni/sglang_omni/config/schema.py
/home/mingfeiguo/Desktop/sglang-omni/sglang_omni/config/placement.py
/home/mingfeiguo/Desktop/sglang-omni/sglang_omni/pipeline/stage/runtime.py
/home/mingfeiguo/Desktop/sglang-omni/sglang_omni/scheduling/simple_scheduler.py
/home/mingfeiguo/Desktop/sglang-omni/sglang_omni/relay/base.py
/home/mingfeiguo/Desktop/sglang-omni/sglang_omni/pipeline/relay_io.py
```
