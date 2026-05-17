# SPRINT：repo-owned visual model executor

## 结论

这个 sprint 的目标不是“把 `Attention` 抽成一个 helper”，而是逐步拥有一套 repo-owned visual model executor：

```text
engine schedule / diffusion step
  -> family runner / runtime
  -> repo-owned model executor
  -> repo-owned layers / kernels / cache / logits primitive
```

核心判断：

- 可以做 generic layers，而且应该给 Triton kernels 留出明确位置；但前提是 repo 真正拥有调用这些 layers 的 model forward。
- 只写一个 `vrl/models/layers/attention.py` 没有意义；如果 Janus / NextStep 仍然调用上游 HF/remote model trunk，它不会接管真实 KV cache。
- 这个 sprint 第一条线先做 Janus，因为 Janus 是离散 image-token AR，最接近 vLLM text decoder，可以最快验证 layer / KV / runner 边界。
- diffusion / DiT / video transformer 也应该进入同一套 executor/layers 方向，但不要和 Janus 第一条 parity path 混在同一个 phase。
- NextStep 是 AR transformer + continuous-token flow head，应该在 Janus executor 边界稳定后再迁移。
- 当前 HF-style wrapper path 必须保留到 parity 通过；不能为了架构洁癖先删除可工作的路径。

最终目标：

```text
Engine 不懂 Janus / NextStep。
Runner 不摸 model private internals。
Model 不调度 rollout loop。
Attention / KV cache / DiT block / flow head kernels 由 repo-owned executor 管。
Replay / rollout forward 在固定输入上有 parity gate。
```

## 当前问题

当前 AR 边界已经比之前干净：

```text
vrl/engine/ar/token_loop/
  -> ARDecodeLoop
  -> ARTokenLoopEnvelope
  -> cache lane gather/scatter

vrl/models/ar/*/runner.py
  -> init_ar / step_ar / finalize_ar
```

但 model 侧还不是 vLLM-style executor。

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
```

如果继续在这个形状上加 generic attention，只会让边界更乱。

## vLLM 对照

vLLM 的模型看起来只是：

```text
runner -> model.forward(...)
```

但它能做到这一点，是因为它拥有整套 executor：

```text
vllm/v1/worker/gpu_model_runner.py
  -> prepares attn metadata / slot mapping / cudagraph mode
  -> calls model.forward(...)

vllm/model_executor/models/llama.py
  -> defines LlamaAttention / LlamaDecoderLayer / LlamaForCausalLM

vllm/model_executor/layers/attention/attention.py
  -> owns Attention layer
  -> reads forward context
  -> updates layer KV cache
```

也就是说，vLLM 的 generic `Attention` 不是孤立 helper。它被完整 model architecture 调用，并且和 runner 准备的 forward context 对齐。

本 repo 如果要达到类似效果，必须先让至少一个真实 model family 的 trunk 变成 repo-owned architecture，而不是继续把上游 trunk 当黑盒。

Janus 是第一条验证线；diffusion/DiT 是第二条复用线。两者应该共享底层 executor/layers/kernels，但不能共享 family-specific workflow。

## 范围

范围内：

- 新增 repo-owned visual executor contract。
- 新增最小 generic executor layers / ops / kernel dispatch 边界。
- 新增 Janus-owned replay forward path。
- 新增 Janus-owned KV decode path。
- 让 Janus runner 调 public executor primitive，不再摸 private model internals。
- 保留旧 HF wrapper path 作为 parity oracle，直到新 path 验证通过。
- 新增 SD3.5-owned DiT replay denoise executor 目录、layer composition、weight mapping、parity helper。
- SD3.5 replay denoise parity 是这个 sprint 的正式交付，不是后续预留。

范围外：

- 不在这个 sprint 重写所有 model family。
- 不先迁移 NextStep rollout path。
- 不迁移 Wan/Cosmos owned executor。
- 不在 SD3.5 replay denoise parity 之前替换完整 SD3 rollout sampling path。
- 不改 trainer / reward / Ray resource 架构。
- 不追求 vLLM 级 CUDA graph / paged KV 性能。
- 不一开始支持所有 quantization / tensor parallel / pipeline parallel。

## 目标目录

第一版目录应该明确区分 generic executor、kernel extension point、family-owned architecture：

```text
vrl/models/executor/
  context.py              # forward context / attention metadata
  cache.py                # owned KV / activation / latent cache contract
  dispatch.py             # torch/triton/custom-op implementation selection
  weights.py              # shared weight-loading helpers
  ops/
    attention.py          # functional attention op wrapper
    mlp.py                # gated MLP / fused MLP op wrapper
    norm.py               # RMSNorm / LayerNorm op wrapper
    rotary.py             # rotary / position embedding op wrapper
    patch.py              # patchify / unpatchify / token-latent reshape ops
    flow.py               # reusable flow / denoise math op wrapper
  kernels/
    torch/
      attention.py
      mlp.py
      norm.py
      rotary.py
    triton/
      attention.py        # optional; only after torch parity exists
      mlp.py
      norm.py
      rotary.py
  layers/
    attention.py          # nn.Module wrapper around ops + cache/context
    linear.py             # packed/fused mapping boundary
    mlp.py                # gated MLP composition
    norm.py               # norm module wrapper
    transformer_block.py  # minimal decoder/DiT block composition

vrl/models/ar/janus_pro/
  executor/
    config.py             # resolved Janus executor config
    model.py              # repo-owned Janus language/image-token executor
    layers.py             # Janus decoder layer / attention / MLP composition
    weights.py            # upstream checkpoint -> owned module mapping
    parity.py             # small helpers for parity fixtures
  model.py                # facade + legacy wrapper path during migration
  runner.py               # ARDecodeLoop-facing runner
  runtime.py              # rollout runtime packing

vrl/models/diffusion/sd3_5/
  executor/
    config.py             # resolved SD3 executor config
    model.py              # repo-owned SD3/DiT replay denoise executor
    layers.py             # SD3/DiT block composition
    weights.py            # diffusers checkpoint -> owned module mapping
    parity.py             # denoise-output parity helpers
```

Rules:

- `vrl/models/executor/` must not import `janus_pro` or `nextstep_1`.
- `vrl/models/executor/` must not import `sd3_5`, `wan_2_1`, or `cosmos`.
- `vrl/models/ar/janus_pro/executor/` may import generic executor layers.
- `vrl/models/diffusion/*/executor/` may import generic executor layers.
- `vrl/engine/ar/` must not import model family code.
- `runner.py` may import model-family executor types.
- `model.py` may expose facade methods, but should not own token schedule.

## Layer / kernel design rule

Generic layer 的边界应该服务两个目标：

```text
1. 先用 torch implementation 建立 parity。
2. 后续可以把单个 op 切到 Triton kernel，而不改 family model forward。
```

所以不要让 family model 直接调用 Triton kernel。调用链应该是：

```text
family executor layer
  -> vrl.models.executor.ops.*
  -> dispatch selects torch | triton | custom op
```

允许进入 generic executor 的内容：

- causal / bidirectional / block attention op。
- rotary / position embedding。
- RMSNorm / LayerNorm。
- gated MLP / fused MLP。
- patchify / unpatchify / token-latent reshape。
- DiT block 中通用的 adaLN / modulation primitive。
- flow / denoise math 中可以跨 NextStep 和 diffusion 复用的 tensor op。

不允许进入 generic executor 的内容：

- Janus CFG policy。
- Janus VQ decoder family wrapper。
- NextStep flow-head semantics。
- SD3 scheduler semantics。
- Wan/Cosmos video conditioning workflow。
- reward / rollout / trainer business logic。

判断规则：

```text
如果一个模块需要知道 family name、prompt format、reward artifact、CFG policy、scheduler name，它不是 generic layer。
如果一个模块只需要 tensor shape、dtype、mask/cache metadata、kernel config，它可以进入 generic executor。
```

## Phase 0：冻结 current oracle

目标：保留当前能跑路径，作为新 executor 的 truth source。

工作：

- 明确当前 Janus HF-wrapper path 的固定输入输出。
- 记录这些对照点：
  - text embedding shape
  - prompt prefill hidden
  - `past_key_values` shape
  - image-token embedding
  - image-token logits
  - replay `forward_image_logits`
  - decode one token step
- 为 Janus replay 和 rollout 建立 deterministic fixture。

完成标准：

- 当前 HF path 没有被删除。
- 有一个清楚的 parity oracle：新 executor 每一步都能和旧 path 对齐。
- 文档里写清楚哪些行为必须 bitwise parity，哪些只要求 tolerance parity。

## Phase 1：Janus architecture audit

目标：弄清楚 Janus LM trunk 具体是什么，避免凭感觉重写。

工作：

- 列出 Janus-Pro-1B 真实 module tree。
- 记录以下信息：
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

## Phase 2：generic visual executor contract

目标：先定义最小 executor contract，不急着追求性能。

新增 contract：

```text
ExecutorForwardContext
  attention metadata
  cache handle
  slot / row / latent mapping
  mode: prefill | decode | replay

ModelCacheHandle
  read layer cache
  write layer cache
  gather rows
  scatter rows
  materialize full tensor view when needed

CausalAttention / BlockAttention
  query/key/value -> hidden
  updates owned KV cache through context

DiffusionBlockContext
  timestep
  text/context embedding metadata
  latent grid metadata
  attention mask metadata
```

注意：

- 第一版可以是 eager PyTorch attention。
- 第一版不用 paged KV。
- 第一版不用 CUDA graph。
- 第一版不支持 quantization。
- 第一版只要 shape / parity / boundary 正确。
- SD3.5 侧第一版必须落地 executor 目录和 replay-denoise contract；Wan/Cosmos 只保留后续扩展边界。

完成标准：

- generic executor layer 不 import Janus。
- generic executor layer 不 import SD3/Wan/Cosmos。
- attention layer 可以在 prefill 和 one-step decode 两种模式下工作。
- KV cache 的 ownership 在 executor/context，不在 model runner 临时 dict 里。
- layer op 可以通过 dispatch 选择 `torch` 实现；Triton hook 存在但默认不开。
- SD3.5 executor 可以 import generic executor layers，但不能把 SD3 scheduler 语义塞进 generic executor。

## Phase 2.5：kernel extension point

目标：给 Triton kernels 留出位置，但不让 kernel design 污染 family model。

工作：

- 定义 `ExecutorKernelConfig`：
  - backend: `torch | triton`
  - allow_fallback: bool
  - debug_parity: bool
- 每个 op 必须有 torch reference implementation。
- Triton implementation 只能挂在 `vrl/models/executor/kernels/triton/`。
- family executor 只能调用 `vrl.models.executor.ops.*`，不能直接 import Triton。
- kernel fallback 失败时要能明确报出 op name、shape、dtype、backend。

完成标准：

- `ops.attention` 可以选择 torch backend。
- Triton backend 缺失时不会影响 torch parity path。
- 结构门能证明 family code 没有直接 import `triton`。

## Phase 3：Janus-owned replay forward

目标：先拥有 training/replay forward，不先碰 rollout sampling。

为什么先 replay：

- replay 输入是完整 token sequence，调度复杂度低。
- 可以直接对齐当前 `forward_image_logits`。
- 如果 replay logits 不一致，rollout KV step 没有意义。

工作：

- 实现 `JanusOwnedExecutor.forward_image_logits(...)`。
- 先支持 full-sequence causal forward。
- 接入 Janus image-token embedding 和 image-token logits projection。
- 加 weight mapping，让 owned executor 能加载 upstream/checkpoint 权重。

完成标准：

- fixed prompt + image tokens 上，新 executor logits 和当前 wrapper logits 对齐。
- LoRA disabled/reference path 行为明确。
- `Diff / max_abs_error / mean_abs_error` 有稳定记录。

## Phase 4：Janus-owned KV prefill/decode

目标：让 owned executor 支持 AR one-step decode。

工作：

- 实现 owned prefill：

```text
text embeddings -> layer KV cache -> last hidden
```

- 实现 owned decode step：

```text
previous image token embedding
  -> layer KV cache update
  -> last hidden
  -> image-token logits
```

- 对齐旧 path：

```text
old HF past_key_values path
new owned KV cache path
```

完成标准：

- same prompt / same sampled token prefix 下，next-token logits 和 old path 对齐。
- prefill + N decode step 与 full forward replay 对齐。
- KV cache row update 不依赖 family-specific hack。

## Phase 5：runner integration

目标：让 `JanusProARModelRunner` 调 owned executor primitive。

工作：

- 新 runner path：

```text
runner.init_ar
  -> owned_executor.prefill(...)

runner.step_ar
  -> owned_executor.decode_step(...)
  -> CFG sampling
  -> state token/logprob recording

runner.finalize_ar
  -> returns old_log_prob artifacts
```

- runner 不再访问：

```text
_lm_trunk()
_base()
_last_token_hidden()
```

- 保留 config 开关：

```yaml
model_executor: hf_wrapper | owned
```

完成标准：

- `hf_wrapper` 仍可运行。
- `owned` 可运行同样 deterministic rollout fixture。
- `vrl/engine/ar` 不 import Janus executor。
- runner 仍然是 family boundary，不把 Janus 细节泄漏到 engine。

## Phase 6：R1 / workflow 拆分

目标：把 workflow 从 `model.py` 移出去。

工作：

- 将 `generate_with_refine(...)` 迁到：

```text
vrl/models/ar/janus_pro/r1_generation.py
```

- `model.py` 只保留 primitive / facade。
- R1 workflow 通过 runner/runtime 注入 image sampler。
- R1 workflow 不直接拥有 AR token schedule。

完成标准：

- `JanusProModel` 不再包含 multi-stage rollout workflow。
- R1 tests 仍通过。
- R1 workflow 可以选择 `hf_wrapper` 或 `owned` sampler。

## Phase 7：删除旧黑盒 path

目标：只有 parity 和 integration 都过了，才删除旧 path。

删除条件：

- replay logits parity 通过。
- KV prefill/decode parity 通过。
- deterministic rollout artifact parity 通过。
- old_log_prob 对齐。
- adapter disable/reference path 对齐。
- current training smoke 路径不回退。

删除内容：

- runner 对 HF private internals 的访问。
- `model_executor: hf_wrapper` legacy path。
- 只为旧 path 存在的 helper。

完成标准：

- Janus rollout/replay 默认走 owned executor。
- 没有 production code 依赖旧黑盒 LM trunk helper。

## Phase 8：SD3.5-owned DiT replay executor

目标：在这个 sprint 内让 generic layers 真正服务 diffusion / DiT，而不是 AR-only executor。

当前 diffusion 事实：

```text
SD3 / Wan / Cosmos 现在主要调用 diffusers / official pipeline transformer。
replay path 已经围绕 transformer、scheduler、latents、prompt embeds 工作。
```

所以 diffusion 不应该一开始替换完整 rollout sampling，但 SD3.5 replay denoise executor 必须在这个 sprint 内落地。SD3.5 是第一条 diffusion 迁移线，因为它的 replay denoise transformer 边界最清楚。

工作：

- 新增目录：

```text
vrl/models/diffusion/sd3_5/executor/
  config.py
  model.py
  layers.py
  weights.py
  parity.py
```

- 定义 `DiffusionOwnedExecutor` contract：

```text
forward_denoise(
  latents,
  timesteps,
  prompt_embeds,
  prompt_attention_mask,
) -> prediction
```

- 实现 SD3.5-owned replay denoise torch reference path。
- 实现 SD3/DiT layer composition skeleton，不只是空文件：
  - patch / latent projection boundary
  - transformer block boundary
  - attention boundary
  - MLP boundary
  - norm / modulation boundary
  - timestep/context embedding boundary
- 实现 diffusers checkpoint -> owned executor 的 weight mapping skeleton。
- 实现 parity helper，固定输入包括：
  - latents
  - timesteps
  - prompt embeds
  - prompt attention mask
- 先对齐 replay denoise output，不先替换完整 sampling loop。
- 将可复用 pieces 接到 generic executor：
  - patch / unpatchify
  - transformer block
  - attention
  - MLP
  - norm / modulation
  - timestep embedding / adaLN primitive
- 保留 diffusers transformer 作为 parity oracle。

完成标准：

- `vrl/models/diffusion/sd3_5/executor/{config,model,layers,weights,parity}.py` 存在且不是占位空壳。
- fixed latent / timestep / prompt embeds 上，owned SD3 executor 和 diffusers transformer 输出对齐。
- generic executor 的 `attention/mlp/norm/patch` 不是 AR-only。
- SD3 runtime 仍可以继续走旧 diffusers path，直到 parity 完成。
- SD3.5 executor 不 import rollout/reward/trainer 代码。

## Phase 9：NextStep migration

目标：等 Janus executor 边界稳定后，再迁移 NextStep。

工作：

- 复用 generic executor context / model cache。
- NextStep 保留 continuous-token + flow head 特性。
- flow sampling 仍然是 model-family primitive，不进 generic attention。
- flow tensor ops 可以逐步复用 `vrl.models.executor.ops.flow`，但 family semantic 留在 NextStep executor。
- 对齐：
  - continuous token replay logprob
  - saved noise
  - flow old_log_prob
  - decode image path

完成标准：

- NextStep runner 不再访问 model private internals。
- NextStep replay/rollout artifacts 和旧 path parity。

## Boundary gates

每个 phase 都要跑结构门，不只跑 tests。

Engine 不能反向 import model family：

```bash
rg "from vrl.models.ar.janus|from vrl.models.ar.nextstep|JanusPro|NextStep" vrl/engine/ar
```

Generic executor 不能 import family：

```bash
rg "janus|nextstep|sd3|wan|cosmos|vq|cfg|reward" vrl/models/executor
```

Family code 不能直接 import Triton：

```bash
rg "import triton|from triton" vrl/models/ar vrl/models/diffusion
```

Runner 不应访问 model private internals：

```bash
rg "model\\._|self\\.model\\._" vrl/models/ar/*/runner.py
```

Model 不应 own rollout schedule：

```bash
rg "ARDecodeLoop|TokenScheduler|sample_rows|GenerationRequest" vrl/models/ar/*/model.py
```

旧 wrapper 删除前必须有明确例外；删除后这些 gate 应无命中。

## Risk

最大风险不是 attention math，而是 parity。

高风险点：

- rotary / position id 对齐。
- attention mask 语义。
- KV cache layout。
- diffusion latent grid / patch order。
- timestep embedding / modulation。
- image-token embedding path。
- `gen_head` 使用 hidden 的位置。
- LoRA target module 和 adapter disable。
- checkpoint weight name mapping。
- replay full forward 与 rollout incremental decode 不一致。
- Triton kernel 与 torch reference 不一致。

规避规则：

- 不一次性替换 rollout path。
- 先 replay parity，再 KV parity，再 runner integration。
- 不在 parity 之前删除旧 HF wrapper。
- 不把 performance optimization 和 architecture ownership 混在一个 phase。
- Triton kernel 永远在 torch parity 之后接入。

## 最终完成标准

这个 sprint 完成时，Janus 应满足：

```text
ARDecodeLoop owns schedule.
JanusProARModelRunner owns rollout step semantics.
Janus owned executor owns transformer forward and KV cache.
JanusProModel is a small facade, not a workflow dumping ground.
Replay and rollout both have parity gates.
Engine has zero family import.
Runner has zero private model access.
```

Generic executor 应满足：

```text
executor ops have torch reference implementations.
kernel dispatch can later select Triton without changing family model code.
generic executor has zero family import.
SD3.5 DiT replay executor is part of this sprint, not an afterthought.
```

SD3.5 应满足：

```text
vrl/models/diffusion/sd3_5/executor/ exists.
model.py defines the owned replay denoise executor boundary.
layers.py defines SD3/DiT block composition boundaries.
weights.py defines diffusers-to-owned mapping boundaries.
parity.py owns fixed-input denoise parity helpers.
owned SD3 path does not replace full rollout sampling until replay parity passes.
```

Wan/Cosmos 和 NextStep 可以仍在迁移队列里，但必须有清楚的 follow-up phase，不能继续让 current architecture 漂着。
