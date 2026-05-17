# SPRINT：Generation Runtime Cache / Pipeline 能力建设

状态：proposed。

启动顺序：先完成并提交 `SPRINT_generation_rollout_boundary_cleanup.md`，再开始本
sprint。原因是本 sprint 要在 `vrl.generation` 的稳定边界上新增 vLLM backend /
cache / diffusion stage 能力；如果边界迁移还没稳定，会同时改包名、import 路径、
Ray worker、AR executor，风险会被放大。

## 核心结论

`SPRINT_generation_rollout_boundary_cleanup.md` 负责把 `engine` / `rollout`
命名边界清干净。本 sprint 不继续做命名迁移，而是定义迁移完成后
`vrl.generation` 必须能承载的 runtime 能力：

```text
AR:
  vLLM-backed paged KV cache
  vLLM-backed prefix cache
  legacy HF DynamicCache fallback

Diffusion:
  staged pipeline
  pipelined execution
```

这不是 full serving engine sprint。当前不做：

```text
HTTP server
OpenAI-compatible API
multi-tenant public request router
streaming response protocol
tokenizer manager subprocess
```

但是 `vrl.generation` 不能只是文件重命名。它必须成为我们自己的
generation runtime，内部允许有 cache manager、stage scheduler、Ray worker
pipeline。`vrl.rollouts` 只能消费 `GenerationRuntime.generate(...)`，不能知道
KV page、prefix cache、diffusion stage placement。

## 架构决策：选择 vLLM AR backend

AR paged KV cache 不走自研优先路线。我们优先做一个 optional vLLM AR backend：

```text
GenerationRuntime
  -> AR executor
  -> VllmARBackend
  -> vLLM model runner / attention / block table / paged KV
```

原因：

```text
只复用 vLLM KVCacheManager 不够。
paged KV 需要 attention forward 同时使用 block table / slot mapping。
HF past_key_values path 不能直接消费 vLLM block allocation。
```

所以本 sprint 的 AR 目标不是把 `ARCacheRows` 换成自研 `PagedKVCacheManager`，而是：

```text
1. 定义我们自己的 AR backend/cache contract。
2. 保留 legacy HF DynamicCache backend。
3. 新增 vLLM backend，把 attention forward 也切到 vLLM 路径。
4. 用一个 AR family 做最小可运行 spike，再决定是否推广。
```

vLLM backend 必须是 optional backend。`vrl.generation` 顶层、rollout collector、
trajectory、reward、trainer 不能因为 vLLM import / ABI 失败而整体不可用。

## 依赖关系

本 sprint 依赖 boundary cleanup sprint 至少完成这些目标：

```text
vrl/generation/
  types.py
  protocols.py
  execution/
  ar/
  diffusion/
  ray/

vrl/rollouts/collector -> vrl.generation
vrl/engine/* 不再作为新代码入口
```

如果 boundary cleanup 还没完成，本 sprint 只能先写接口设计，不能把新实现塞回
旧 runtime 路径。

## 启动门槛

开始本 sprint 前，必须先满足：

```text
1. boundary cleanup sprint 已经合并或至少在当前分支通过完整验证。
2. repo 内新代码主要从 vrl.generation / vrl.trajectory import。
3. 新代码不再依赖 vrl.engine / vrl.distributed.ray.rollout / vrl.rollouts.runtime。
4. Ray generation worker / runtime 能通过现有 distributed tests。
5. AR legacy backend 仍能通过 Janus / NextStep KV decode tests。
6. architecture boundary test 已经保护 generation 不能反向 import rollouts/rewards/trainers。
```

本 sprint 的第一步不能是直接改 Janus model forward。必须先把 backend contract 和
vLLM import gate 建起来，让 legacy path 保持可运行。

允许提前做的旁路工作只有一个：

```text
VLLM feasibility spike:
  在 scratch/test-only 文件里确认当前环境能 import vLLM runner/attention 相关模块，
  并记录 PyTorch/CUDA/vLLM ABI 是否匹配。
```

这个 spike 不能改 `RolloutCollector`，不能把 vLLM import 放进 `vrl.generation`
顶层，也不能替换 legacy AR runtime。

## 当前问题

### AR cache 仍是 row-wise tensor/cache 搬运

当前 AR token loop 通过 `ARCacheRows` 管理 cache lane：

```text
batched cache -> split rows -> gather scheduled rows -> scatter updated rows
```

这个适合现在的 correctness-first rollout，但不是长期 runtime 形态。随着：

```text
samples_per_prompt 变大
AR image token 数变大
Ray worker batch 变大
CFG cond/uncond 双分支
多个 prompt 共享 prefix
```

反复 `DynamicCache.batch_split` / `DynamicCache.from_batch_splits` / tensor cat 会变成
明显 overhead。

### Prefix prefill 没有成为 runtime contract

同一个 prompt 在 RL 中会反复出现：

```text
same prompt
same policy_version
same tokenizer/model family
multiple samples per prompt
many rollout iterations
```

如果每个 sample 都重复 prefill，AR rollout 会浪费大量前缀计算。prefix cache 必须
进入 generation runtime，但不能泄漏到 collector / reward / trajectory。

### Diffusion execution 现在更像 monolithic chunk

当前 diffusion executor 已经有 microbatch / gather，但 execution stage 还不够明确。
未来需要把 diffusion 拆成可调度 stage：

```text
encode prompt / condition
prepare latent
denoise
decode VAE
postprocess / artifact packing
```

这样才能支持：

```text
encoder / denoiser / decoder 分开放置
多 GPU pipeline
Ray worker staged execution
denoise-heavy worker 独立扩缩
```

## 目标目录

在 boundary cleanup 完成后，目标结构是：

```text
vrl/generation/
  ar/
    executor.py
    layout.py
    cache.py
    prefix_cache.py
    vllm_backend.py
    vllm_runner.py
    cache_backends/
      legacy.py
      vllm.py
    token_loop/

  diffusion/
    executor.py
    gather.py
    layout.py
    stages.py
    pipeline.py

  ray/
    runtime.py
    executor.py
    planner.py
    worker.py
    stage_worker.py

  execution/
    ids.py
    microbatching.py
    planner.py
    request_batch.py
    stage_plan.py
```

不要新增：

```text
vrl/generation/serving/
vrl/generation/server/
vrl/generation/http/
```

因为本 sprint 关注 internal runtime，不做 public serving surface。

## 设计原则

### Cache 是 generation 内部能力

这些类型只能存在于 `vrl.generation` 内部：

```text
CacheHandle
ARCacheBackend
VllmARBackend
PrefixCache
PrefixCacheKey
PrefixCacheEntry
```

禁止：

```text
vrl.rollouts imports vrl.generation.ar.cache
vrl.rollouts imports vrl.generation.ar.vllm_backend
vrl.trajectory records CacheHandle
reward scorer receives cache objects
trainer batch stores live KV cache
```

trajectory 只能记录 replay 需要的 tensor / metadata，不能记录 runtime-only cache handle。

### Policy version 必须参与 cache 正确性

prefix cache key 必须包含：

```text
family
task
policy_version
tokenizer_key
prompt token ids / prompt hash
model dtype
cache dtype
CFG branch kind
runtime cache layout version
backend name
vLLM cache config hash when backend=vllm
```

只要 rollout worker 收到新权重：

```text
GenerationWeightSync.push_to_generation_workers(...)
```

必须失效旧 policy version 的 prefix cache。默认策略是按 `policy_version` 整体隔离，
不做跨版本复用。

### vLLM backend 是 AR runtime backend，不是简单 cache helper

vLLM paged KV 必须和 vLLM attention/model-runner path 一起接入。只 import
`vllm.v1.core.KVCacheManager` 不算完成目标，因为 HF model forward 仍然不会使用
vLLM block table / slot mapping。

第一版建立 backend contract：

```text
Legacy backend:
  HF language_model(..., past_key_values=..., use_cache=True)

vLLM backend:
  vLLM runner / attention metadata / block table / slot mapping
  image-token logits extracted from hidden state
```

必须保持同一个上层 contract：

```text
ARDecodeLoop / AR executor
  -> backend.init(...)
  -> backend.step(...)
  -> backend.finalize(...)
```

`ARCacheRows` 保留为 legacy adapter。vLLM backend 不应该依赖 `ARCacheRows` 的
split/cat 行为。

### Diffusion stage 是 runtime stage，不是 rollout stage

这些名字属于 generation：

```text
DiffusionStageKind.ENCODE
DiffusionStageKind.PREPARE_LATENT
DiffusionStageKind.DENOISE
DiffusionStageKind.DECODE
DiffusionStageKind.POSTPROCESS
```

禁止把它们叫成 rollout stage。rollout stage 是：

```text
collect generation
score reward
build RolloutBatch
train/replay
```

## Phase 1：AR backend/cache contract

新增：

```text
vrl/generation/ar/cache.py
vrl/generation/ar/cache_backends/legacy.py
```

核心类型：

```text
CacheHandle
ARCacheBackend
ARBackendInit
ARBackendStep
ARBackendOutput
LegacyARCacheAdapter
```

目标：

```text
AR token loop 不再假设 cache_lanes 一定能 split/cat tensor。
legacy DynamicCache 继续可用。
backend handle 可以被 scheduler 传递，但不进入 trajectory。
```

验收：

- `ARDecodeLoop` 可以继续跑 legacy cache backend。
- 当前 Janus / NextStep 测试不改语义。
- backend/cache handle 不能被 `TrajectoryBatch` 序列化进去。

## Phase 2：vLLM backend import / environment gate

新增：

```text
vrl/generation/ar/vllm_backend.py
vrl/generation/ar/vllm_runner.py
vrl/generation/ar/cache_backends/vllm.py
```

目标：

```text
vLLM backend 只能 lazy import vLLM。
vLLM 不存在或 ABI 不匹配时，legacy backend 仍可用。
错误信息必须明确说明 backend=vllm 初始化失败。
```

验收：

- `import vrl.generation.ar` 不触发 vLLM import。
- `backend="legacy"` 不需要 vLLM。
- `backend="vllm"` 在 vLLM import 失败时 fail fast，并给出清楚错误。
- tests 可以 monkeypatch vLLM 缺失场景。

## Phase 3：Janus vLLM AR runner MVP

先选 Janus-Pro 做 spike，因为它是离散 image-token AR，验证目标最清楚。

目标：

```text
prefill prompt through vLLM attention path
decode one image token through vLLM paged KV path
extract hidden state
compute image-token logits
sample token
return token/logprob with existing GenerationOutput contract
```

验收：

- 单 prompt / 单 sample 可以 decode 至少 1 个 image token。
- logits shape 和 legacy Janus path 一致。
- logprob 可以进入现有 trajectory/replay path。
- legacy Janus path 仍可用。
- vLLM backend 不 import rollout/reward/trainer。

## Phase 4：vLLM prefix cache MVP

新增：

```text
vrl/generation/ar/prefix_cache.py
```

核心类型：

```text
PrefixCacheKey
PrefixCacheStats
VllmPrefixCachePolicy
```

MVP 行为：

```text
use vLLM prefix cache when backend=vllm
separate cond/uncond CFG branch
invalidate by policy_version
expose prefix hit/miss stats
```

非目标：

```text
跨 worker page sharing
CPU/NVMe offload
prefix cache radix tree
自研 paged attention kernel
```

验收：

- 相同 prompt + 相同 policy version 可以命中 prefix cache。
- policy version 改变后不命中旧 cache。
- CFG cond/uncond 分支不能互相复用。
- 默认可以通过 config 关闭。

## Phase 5：Ray worker vLLM backend lifecycle

扩展：

```text
vrl/generation/ray/worker.py
vrl/generation/ray/weight_sync.py
```

目标：

```text
load_policy() 初始化 legacy 或 vLLM backend
update_weights(...) 失效旧 vLLM prefix cache
release_policy() 释放 vLLM runner / KV blocks / prefix cache
shutdown() 不留下 CUDA memory
```

验收：

- weight update 后旧 policy prefix cache 不再命中。
- actor shutdown 后 vLLM runner memory 被释放。
- runtime debug 输出包含 backend name、KV usage、prefix stats。
- vLLM backend 失败时可以回退 legacy backend，除非 config 显式要求 strict。

## Phase 6：Diffusion stage contract

新增：

```text
vrl/generation/diffusion/stages.py
vrl/generation/execution/stage_plan.py
```

核心类型：

```text
DiffusionStageKind
DiffusionStageInput
DiffusionStageOutput
GenerationStagePlan
StagePlacement
```

目标：

```text
把 encode / denoise / decode 的输入输出变成显式 contract。
现有 DiffusionPipelineExecutorBase 可以先用 monolithic stage adapter。
```

验收：

- 单 worker 路径输出和原来一致。
- stage output 只包含下一 stage 需要的数据。
- stage contract 不 import reward / trainer / rollout collector。

## Phase 7：Staged diffusion pipeline MVP

新增：

```text
vrl/generation/diffusion/pipeline.py
```

目标执行形态：

```text
GenerationRequest
  -> encode stage
  -> prepare_latent stage
  -> denoise stage
  -> decode stage
  -> gather GenerationOutput
```

第一版可以在同一个 worker 内顺序执行，重点是建立 stage contract。

验收：

- stage metrics 能分别记录 encode / denoise / decode 时间。
- denoise stage 可以独立设置 microbatch size。
- decode stage 可以独立开关是否返回 decoded artifact。

## Phase 8：Pipelined diffusion Ray execution

扩展：

```text
vrl/generation/ray/planner.py
vrl/generation/ray/stage_worker.py
vrl/generation/ray/executor.py
```

目标：

```text
允许不同 stage 放到不同 worker role：

encoder workers
denoiser workers
decoder workers
```

第一版不要求动态负载均衡。可以先做静态 role placement。

验收：

- monolithic worker path 和 staged worker path 都可用。
- worker role 不匹配时 fail fast。
- denoiser worker 不需要加载完整 decoder-only state，除非 family runtime 要求。
- stage transfer payload 明确，不传 reward/trainer object。

## Phase 9：测试和边界检查

新增测试：

```text
tests/generation/ar/test_backend_contract.py
tests/generation/ar/test_vllm_backend_import_gate.py
tests/generation/ar/test_janus_vllm_backend.py
tests/generation/ar/test_vllm_prefix_cache_policy.py
tests/generation/diffusion/test_stage_contracts.py
tests/generation/diffusion/test_staged_pipeline.py
tests/distributed/ray/test_generation_stage_workers.py
```

架构测试新增规则：

```text
vrl/generation/ar/cache.py cannot import:
  vrl.rollouts
  vrl.rewards
  vrl.trainers
  vrl.algorithms
  vrl.trajectory

vrl/generation/ar/vllm_backend.py cannot import:
  vrl.rollouts
  vrl.rewards
  vrl.trainers
  vrl.algorithms

vrl/generation/diffusion/stages.py cannot import:
  vrl.rollouts
  vrl.rewards
  vrl.trainers
  vrl.algorithms
```

## 非目标

本 sprint 不做：

```text
不做 HTTP serving
不做 OpenAI image API
不做 vLLM OpenAI server / public serving stack
不做跨 worker prefix cache
不做 KV cache CPU/NVMe offload
不做自研 radix prefix tree
不要求一次性支持所有 AR family
不改 reward / advantage / GRPO / DPO 语义
不把 cache handle 写入 RolloutBatch / TrajectoryBatch
```

## 推荐顺序

```text
1. 完成 generation / rollout boundary cleanup。
2. 加 AR backend/cache contract，让 legacy cache 继续跑。
3. 加 vLLM backend lazy import / environment gate。
4. 做 Janus vLLM AR runner MVP：prefill + decode 1 token + logprob。
5. 接 vLLM prefix cache policy 和 policy_version invalidation。
6. 把 Ray worker lifecycle 接入 vLLM backend release / weight sync。
7. 加 diffusion stage contract。
8. 用同 worker staged diffusion path 验证输出不变。
9. 再做 Ray staged / pipelined diffusion。
```

## 验证命令

基础验证：

```bash
ruff check vrl tests
pytest tests/generation tests/engine/ar tests/engine/diffusion tests/models
pytest tests/generation/ar/test_vllm_backend_import_gate.py
```

Ray cache / stage worker 验证：

```bash
pytest tests/distributed/ray
```

Rollout 边界回归：

```bash
pytest tests/rollouts tests/trainers
```

## 完成标准

完成后应该能一句话描述：

```text
vrl.generation 是 internal generation runtime。
AR runtime 有 legacy backend 和 optional vLLM paged-KV backend。
Diffusion runtime 内部支持 staged pipeline。
RL rollout 仍然只消费 GenerationRuntime.generate(...) 和 GenerationOutput。
```

代码上必须满足：

- `RolloutCollector` 不 import AR cache / diffusion stage。
- `TrajectoryBatch` 不保存 live cache handle。
- `GenerationRuntime` 可以暴露 cache/stage debug metrics，但不改变 rollout API。
- weight sync 会触发 prefix cache policy invalidation。
- vLLM backend 失败不会破坏 legacy backend import。
- Janus vLLM backend 至少完成 1-token decode spike。
- staged diffusion 可以退回 monolithic path。

## 参考路径

```text
/home/mingfeiguo/Desktop/wm-infra/SPRINT_generation_rollout_boundary_cleanup.md
/home/mingfeiguo/Desktop/wm-infra/vrl/engine/ar/token_loop/row_cache.py
/home/mingfeiguo/Desktop/wm-infra/vrl/engine/ar/token_loop/state.py
/home/mingfeiguo/Desktop/wm-infra/vrl/engine/ar/token_loop/loop.py
/home/mingfeiguo/Desktop/wm-infra/vrl/models/ar/janus_pro/runner.py
/home/mingfeiguo/Desktop/wm-infra/vrl/models/ar/nextstep_1/runner.py
/home/mingfeiguo/Desktop/wm-infra/vrl/engine/diffusion/executor.py
/home/mingfeiguo/Desktop/wm-infra/vrl/engine/diffusion/gather.py
/home/mingfeiguo/Desktop/wm-infra/vrl/distributed/ray/rollout/worker.py
/home/mingfeiguo/miniconda3/lib/python3.12/site-packages/vllm/v1/core/kv_cache_manager.py
/home/mingfeiguo/miniconda3/lib/python3.12/site-packages/vllm/v1/core/block_pool.py
/home/mingfeiguo/miniconda3/lib/python3.12/site-packages/vllm/v1/attention/backends/flash_attn.py
/home/mingfeiguo/miniconda3/lib/python3.12/site-packages/vllm/v1/attention/ops/paged_attn.py
```
