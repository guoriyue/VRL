# SPRINT：vLLM attention-backed repo-owned model executor

## 核心结论

这个 sprint 不再定义“自研 AR serving engine”。它定义的是：

```text
repo-owned family model executor
  -> uses vLLM attention / KV / metadata primitives where feasible
  -> keeps family-specific visual embedding / logits / replay semantics in repo
```

也就是说，我们不是把整个 Janus/NextStep model 交给 vLLM 现成 engine 跑，也不是自己从零写一套 vLLM 替代品。

正确边界是：

```text
VRL owns:
  Janus / NextStep / SD3 family semantics
  visual token embedding
  image-token logits projection
  replay tensors and parity hooks
  CFG / reward-facing artifact semantics
  policy_version correctness
  model weight mapping and adapter/reference control

vLLM provides:
  attention primitive
  paged attention metadata shape
  KV cache layout conventions where usable
  block table / slot mapping interface where usable
```

这和 generation runtime sprint 的关系是：

```text
SPRINT_generation_runtime_cache_pipeline.md:
  owns GenerationRequest -> GenerationOutput runtime boundary
  owns rollout runtime lifecycle / metrics / cache invalidation policy
  owns vrl.generation package shape

SPRINT_generation_runtime_future_backends_pipeline.md:
  owns future full-engine adoption experiments
  evaluates whether whole vLLM / vLLM-Omni engine can be used

this sprint:
  owns model-family executor internals
  builds Janus executor that can call vLLM attention primitives
  builds replay/parity hooks that external engines may not expose
```

因此，本 sprint 不能再写成：

```text
vLLM owns whole Janus model
repo owns a second AR generation engine
repo owns production paged KV scheduler
```

它应该写成：

```text
repo owns model forward and family semantics
vLLM attention/KV primitives are implementation dependencies inside that forward
generation runtime decides how rollout requests are scheduled
```

## 为什么还需要 model layers / kernels

如果只接完整 vLLM engine，repo 仍然需要回答这些问题：

```text
Janus visual token embedding 怎么进 decoder path
image-token logits projection 怎么和 HF wrapper 对齐
CFG cond/uncond hidden/logits 怎么拿
RL replay 需要的 logprob / old_logprob / hidden / token artifacts 怎么稳定导出
LoRA disabled/reference path 怎么保证 parity
SD3/vLLM-Omni 如果不给 replay hook，repo 怎么自己验证 denoise path
```

这些都不是 rollout collector 或 reward scorer 应该知道的东西，也不是 generic `vrl.generation` runtime 应该硬编码的 family 细节。

所以 model layers / kernels 的定位是：

```text
model-family executor support
replay parity support
external-engine integration support
explicit fallback support when external engine cannot expose RL hooks
```

不是：

```text
self-owned public serving backend
self-owned AR scheduler
self-owned prefix-cache radix tree
self-owned vLLM replacement
```

## 当前问题

Janus runner 仍然会访问 model private internals：

```text
self.model._lm_trunk()
self.model._base()
self.model._last_token_hidden()
```

NextStep runner 也类似：

```text
self.model._init_kv()
self.model._image_in_projector()
self.model._step_llm()
```

这说明当前 `model.py` 同时承担了太多角色：

```text
model loading
architecture wrapper
private LM/VQ helper
training replay
R1 workflow
adapter/reference control
scheduled AR decode path
```

如果继续在这个形状上加 cache 或 attention helper，只会让边界更乱。

我们需要的是一层明确的 model executor：

```text
JanusProModel facade
  -> JanusProExecutor
     -> visual embedding primitive
     -> vLLM-backed attention primitive when available
     -> image-token logits primitive
     -> replay / decode parity outputs
```

## vLLM 对照

vLLM 的模型看起来只是：

```text
runner -> model.forward(...)
```

但它能做到这一点，是因为它拥有整套 execution context：

```text
gpu_model_runner
  -> prepares attention metadata
  -> prepares slot mapping
  -> prepares block tables
  -> calls model.forward(...)

model_executor/layers/attention
  -> consumes forward context
  -> reads/writes KV cache through expected layout
```

所以本 repo 不能把 vLLM attention 当普通 `q, k, v -> out` helper 用。正确目标是：

```text
repo-owned executor context
  -> translates Janus replay/decode state into vLLM-compatible attention metadata
  -> calls vLLM attention primitive when API/ABI allows
  -> falls back to torch reference only for parity/debug
```

## 范围

范围内：

- 定义 repo-owned model executor contract。
- 定义 executor forward context / attention metadata / KV cache layout 边界。
- 新增 vLLM attention adapter，必须 lazy import。
- 新增 torch reference attention path，只用于 parity/debug。
- 新增 generic executor ops/layers/kernel dispatch 目录。
- 新增 Janus executor 目录和 architecture audit。
- 新增 Janus replay forward parity path。
- 新增 Janus visual token embedding / image-token logits projection parity hooks。
- 保留当前 HF wrapper path 作为 oracle，直到 parity 通过。
- 为 SD3.5 replay-denoise hook 留出 executor/fallback 边界，但不把 SD3 full rollout 切到 owned path。

范围外：

- 不自研 public AR serving backend。
- 不自研 production paged KV scheduler。
- 不实现 prefix-cache radix tree。
- 不新增 `backend=hf` 或 `engine=auto`。
- 不把 owned executor 直接接成默认 rollout runtime。
- 不在 Janus parity 前迁移 NextStep。
- 不在 vLLM attention API/ABI 确认前写 Triton attention kernel。
- 不替换 SD3/Wan/Cosmos 的完整 diffusion rollout sampling path。

## 目标目录

```text
vrl/models/executor/
  context.py              # executor forward context / mode / metadata carrier
  cache.py                # KV / activation cache contracts, no family imports
  dispatch.py             # torch | vllm | triton dispatch policy
  weights.py              # shared weight mapping helpers

  ops/
    attention.py          # attention op wrapper: torch reference or vLLM adapter
    logits.py             # projection/logprob helpers
    mlp.py                # gated MLP / fused MLP op wrapper
    norm.py               # RMSNorm / LayerNorm op wrapper
    rotary.py             # rotary / position embedding helper
    patch.py              # patchify / token-latent reshape helpers
    flow.py               # flow / denoise math helpers

  vllm/
    attention.py          # lazy vLLM attention primitive adapter
    metadata.py           # slot mapping / block table / seq metadata adapter
    cache.py              # vLLM-compatible KV layout view, not a scheduler

  kernels/
    torch/
      attention.py
      mlp.py
      norm.py
      rotary.py
    triton/
      README.md           # future extension point; no production dependency yet

  layers/
    attention.py          # nn.Module wrapper around ops + context/cache
    linear.py             # packed/fused mapping boundary
    mlp.py
    norm.py
    transformer_block.py

vrl/models/ar/janus_pro/
  executor/
    audit.py              # architecture / module tree / weight name inspection
    config.py             # resolved Janus executor config
    model.py              # repo-owned Janus executor boundary
    layers.py             # Janus decoder layer composition where owned
    weights.py            # upstream checkpoint -> owned executor mapping
    parity.py             # fixed-input parity helpers

vrl/models/diffusion/sd3_5/
  executor/
    config.py             # optional fallback/replay config
    model.py              # replay-denoise executor boundary if needed
    weights.py            # diffusers -> owned mapping helper
    parity.py             # denoise parity helper
```

Rules:

- `vrl/models/executor/` must not import Janus / NextStep / SD3 / Wan / Cosmos.
- `vrl/models/executor/` must not import rollout / reward / trainer.
- `vrl/models/executor/vllm/*` must lazy import vLLM.
- `vrl/models/ar/janus_pro/executor/` may import generic executor layers.
- `vrl/models/diffusion/sd3_5/executor/` may import generic executor layers.
- family code must not import Triton directly.
- `vrl/generation` must not import Janus executor internals.
- `TrajectoryBatch` / `RolloutBatch` must never store live KV/cache handles.

## Layer / kernel design rule

Generic layer 的边界服务三个目标：

```text
1. torch reference gives deterministic parity.
2. vLLM adapter can provide real attention/KV primitive.
3. Triton/custom kernels can be added later without changing family model forward.
```

调用链应该是：

```text
family executor
  -> vrl.models.executor.layers.*
  -> vrl.models.executor.ops.*
  -> dispatch selects torch | vllm | triton
```

允许进入 generic executor 的内容：

- causal attention / block attention wrapper。
- vLLM attention metadata adapter。
- KV cache layout view。
- rotary / position embedding。
- RMSNorm / LayerNorm。
- gated MLP / packed linear。
- image-token logits projection helper。
- patchify / unpatchify / token-latent reshape。
- flow / denoise math 中可以跨 family 复用的 tensor op。

不允许进入 generic executor 的内容：

- Janus CFG policy。
- Janus VQ decoder workflow。
- Janus R1 multi-stage workflow。
- NextStep flow-head semantics。
- SD3 scheduler semantics。
- Wan/Cosmos conditioning workflow。
- reward / rollout / trainer business logic。

判断规则：

```text
需要知道 family name / prompt format / reward artifact / scheduler name -> 不 generic。
只需要 tensor shape / dtype / mask / slot mapping / cache metadata -> 可以 generic。
```

## Phase 0：冻结 current oracle

目标：保留当前能跑路径，作为新 executor 的 truth source。

工作：

- 明确当前 Janus HF-wrapper path 的固定输入输出。
- 固定 deterministic fixture：
  - prompt token ids
  - image token ids
  - attention mask
  - LoRA enabled/disabled state
  - policy/reference path
- 记录这些对照点：
  - text embedding shape
  - prompt prefill hidden
  - `past_key_values` shape
  - image-token embedding
  - image-token logits
  - replay `forward_image_logits`
  - one-step decode logits

完成标准：

- 当前 HF path 没有被删除。
- 有一个清楚的 parity oracle。
- 文档写清楚哪些行为要求 bitwise parity，哪些只要求 tolerance parity。

## Phase 1：Janus architecture audit

目标：弄清楚 Janus-Pro-1B 真实 module tree，避免凭感觉重写。

工作：

- 列出真实 module tree。
- 记录：
  - hidden size
  - number of layers
  - attention heads / KV heads
  - head dim
  - rotary config
  - vocab / image vocab / image token offset
  - attention mask semantics
  - BOS/EOS/pad behavior
  - LoRA target module names
  - `gen_head` / image token projection path
  - `prepare_gen_img_embeds` path
- 明确 weight names：
  - upstream checkpoint names
  - current wrapper module names
  - owned executor target names

完成标准：

- 不再依赖“Janus 应该像 Llama”这种假设。
- 可以写出 explicit weight mapping 表。
- 能解释当前 `forward_image_logits` 和 rollout prefill/step 用的是哪条 trunk path。

## Phase 2：executor context / dispatch contract

新增：

```text
ExecutorForwardContext
  mode: replay | prefill | decode
  attention metadata
  slot mapping
  block table
  cache layout view
  dtype/device policy

ExecutorCacheHandle
  layer cache lookup
  layer cache update
  materialize torch view for parity

ExecutorKernelConfig
  backend: torch | vllm | triton
  allow_fallback: bool
  debug_parity: bool
```

完成标准：

- generic executor contract 不 import family code。
- `backend=torch` reference path 可以独立测试。
- `backend=vllm` adapter lazy import vLLM。
- vLLM 不存在或 ABI 不匹配时，非 vLLM tests 不受影响。
- dispatch error 必须包含 op name、shape、dtype、backend。

## Phase 3：vLLM attention adapter spike

目标：验证当前环境能否把 repo-owned executor context 转成 vLLM attention primitive 所需输入。

工作：

- 新增 `vrl/models/executor/vllm/attention.py`。
- 新增 `vrl/models/executor/vllm/metadata.py`。
- lazy import vLLM attention 相关模块。
- 记录 vLLM attention API 需要的：
  - forward context
  - attention metadata
  - slot mapping
  - block table
  - KV cache tensor layout
- 用 toy tensor 做 smoke test。

完成标准：

- `import vrl.models.executor` 不触发 vLLM import。
- vLLM import failure 有清楚错误，不破坏普通 tests。
- 有 test 覆盖 vLLM unavailable path。
- torch reference path 和 vLLM path 的输入输出 contract 对齐。

## Phase 4：Janus-owned replay forward

目标：先拥有 training/replay forward，不先碰 rollout sampling。

为什么先 replay：

- replay 输入是完整 token sequence，调度复杂度低。
- 可以直接对齐当前 `forward_image_logits`。
- 如果 replay logits 不一致，decode/KV path 没有意义。

工作：

- 新增 `JanusOwnedExecutor.forward_image_logits(...)`。
- 支持 full-sequence causal forward。
- 接入 Janus image-token embedding。
- 接入 image-token logits projection。
- 加 weight mapping，让 owned executor 能加载 upstream/checkpoint 权重。
- 首版可以用 torch reference attention，vLLM attention adapter 作为可选 backend。

完成标准：

- fixed prompt + image tokens 上，新 executor logits 和当前 HF wrapper logits 对齐。
- LoRA disabled/reference path 行为明确。
- parity report 记录 `max_abs_error` / `mean_abs_error`。
- 不改 rollout runtime 默认路径。

## Phase 5：Janus prefill/decode primitive parity

目标：让 owned executor 支持 prefill/decode primitive，但不把它直接变成 public runtime。

工作：

```text
prefill:
  text embeddings
  -> executor context
  -> layer KV/cache layout
  -> last hidden

decode step:
  previous image token embedding
  -> executor context
  -> attention/KV update
  -> last hidden
  -> image-token logits
```

对齐：

```text
old HF past_key_values path
new owned executor cache/context path
full replay forward
```

完成标准：

- same prompt / same sampled token prefix 下，next-token logits 和 HF path 对齐。
- prefill + N decode step 与 full forward replay 对齐。
- vLLM attention backend 可用时通过同样 interface。
- cache handle 不进入 `GenerationOutput` / `TrajectoryBatch` / `RolloutBatch`。
- 不新增 `backend=hf` / `engine=auto` public selector。

## Phase 6：runner integration gate

这不是默认实现阶段，而是进入生产路径前的 gate。

只有这些条件满足后，才允许让 `JanusProARModelRunner` 调 owned executor primitive：

```text
replay logits parity passed
prefill/decode parity passed
adapter disable/reference parity passed
generation runtime sprint has settled AR scheduling ownership
vLLM attention adapter API is stable enough
```

允许的集成形态：

```text
runner.init_ar
  -> owned_executor.prefill(...)

runner.step_ar
  -> owned_executor.decode_step(...)
  -> family CFG sampling
  -> state token/logprob recording

runner.finalize_ar
  -> returns trainer/replay artifacts
```

不允许：

```text
runner creates public backend selector
runner stores live cache in trajectory
runner exposes vLLM objects to rollout collector
generation runtime imports Janus executor internals
```

## Phase 7：R1 workflow split

目标：把 workflow 从 `model.py` 移出去，但不和 executor parity 混在一起。

工作：

- 将 `generate_with_refine(...)` 迁到 family workflow module。
- `model.py` 只保留 facade / primitive。
- R1 workflow 通过 runner/runtime 注入 image sampler。
- R1 workflow 不直接拥有 AR token schedule。

完成标准：

- `JanusProModel` 不再是 workflow dumping ground。
- R1 tests 仍通过。
- R1 workflow 可以使用 HF oracle 或 owned executor primitive，但不绕过 parity gate。

## Phase 8：SD3.5 replay hook / fallback executor

SD3.5 这条线不是为了替换完整 diffusion rollout runtime。它的目的只有两个：

```text
1. 当 vLLM-Omni 暴露 replay hooks 不足时，repo 有 fallback/parity primitive。
2. 让 generic executor ops 不只服务 AR。
```

新增可选目录：

```text
vrl/models/diffusion/sd3_5/executor/
  config.py
  model.py
  weights.py
  parity.py
```

工作：

- 定义 replay-denoise boundary：

```text
forward_denoise(
  latents,
  timesteps,
  prompt_embeds,
  prompt_attention_mask,
) -> prediction
```

- diffusers transformer 作为 oracle。
- 先实现 parity helper 和 weight mapping skeleton。
- 不替换完整 sampling loop。

完成标准：

- SD3.5 executor 不 import rollout/reward/trainer。
- fixed latent / timestep / prompt embeds 有 parity fixture。
- 是否继续做 owned DiT layers 取决于 vLLM-Omni hook 评估结果。

## Phase 9：NextStep migration queue

NextStep 等 Janus executor boundary 稳定后再迁移。

工作方向：

- 复用 executor context / cache layout。
- continuous-token + flow head 留在 family executor。
- flow tensor ops 可以复用 `vrl.models.executor.ops.flow`。
- 对齐 replay logprob / saved noise / flow old_log_prob。

完成标准：

- NextStep runner 不再访问 model private internals。
- NextStep replay/rollout artifacts 和旧 path parity。

## Boundary gates

Generic executor 不能 import family：

```bash
rg "janus|nextstep|sd3|wan|cosmos|vq|cfg|reward|trainer|rollout" vrl/models/executor
```

Family code 不能直接 import Triton：

```bash
rg "import triton|from triton" vrl/models/ar vrl/models/diffusion
```

vLLM import 只能在 adapter 内 lazy import：

```bash
rg "import vllm|from vllm" vrl/models vrl/generation
```

Runner 不应访问 model private internals after integration gate：

```bash
rg "model\\._|self\\.model\\._" vrl/models/ar/*/runner.py
```

Model 不应 own rollout schedule：

```bash
rg "ARDecodeLoop|TokenScheduler|sample_rows|GenerationRequest" vrl/models/ar/*/model.py
```

Runtime-only cache handle 不应进入 trajectory / rollout batch：

```bash
rg "CacheHandle|KVCache|block_table|slot_mapping" vrl/trajectory vrl/rollouts
```

## Risk

最大风险不是 attention math，而是 parity 和 vLLM primitive API。

高风险点：

- vLLM attention API/ABI 变化。
- rotary / position id 对齐。
- attention mask 语义。
- KV cache layout。
- slot mapping / block table 语义。
- image-token embedding path。
- `gen_head` 使用 hidden 的位置。
- LoRA target module 和 adapter disable。
- checkpoint weight name mapping。
- replay full forward 与 incremental decode 不一致。
- torch reference 与 vLLM attention path 不一致。

规避规则：

- 不一次性替换 rollout path。
- 先 replay parity，再 decode primitive parity，再 runner integration gate。
- 不在 parity 前删除 HF wrapper oracle。
- 不把 performance optimization 和 architecture ownership 混在一个 phase。
- Triton kernel 永远在 torch/vLLM parity 之后。

## 最终完成标准

这个 sprint 完成时应满足：

```text
vrl/models/executor/ exists and has no family imports.
vrl/models/executor/vllm/ contains lazy vLLM attention adapter boundary.
Janus executor has replay parity against HF wrapper.
Janus executor has prefill/decode primitive parity.
HF wrapper remains as oracle until production integration gate passes.
No public self-owned AR serving backend was added.
No runtime cache handle enters TrajectoryBatch or RolloutBatch.
```

如果后续确认完整 vLLM engine 可以直接承载 Janus visual-token generation，本 sprint 的 executor 产物仍有价值：

```text
it becomes the parity oracle and wrapper reference for vLLM integration.
```

如果完整 vLLM engine 不能承载 Janus visual-token generation，本 sprint 的 executor 产物就是 fallback path 的基础。
