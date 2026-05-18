# SPRINT：NN Modules / Layers / Kernels

状态：implemented。

## 核心结论

这个 sprint 的目标是把模型 forward 里的可复用计算组件收敛成类似 vLLM `model_executor/layers` 的结构：

```text
family runner / runtime
  -> NN module
  -> NN layer
  -> kernel / backend
```

也就是说：

```text
vrl.generation:
  owns request / batch / rollout execution / Ray lifecycle

vrl.models:
  owns family semantics / checkpoints / visual tokenizer / replay outputs

vrl.nn:
  owns reusable neural-network building blocks
  owns layer contracts
  owns kernel dispatch
  hides vLLM / Triton / torch backend details
```

当前 `vrl/generation/ar/paged_attention.py` 和 `vrl/generation/ar/vllm_paged_attention.py` 的问题不是名字，而是边界错了：

```text
generation/ar should not own model attention layers
family runner should not import vLLM internal APIs
kernel code should not know Janus / NextStep semantics
```

正确方向是：

```text
JanusProARModelRunner
  -> ARDecoderModule
  -> PagedSelfAttentionLayer
  -> VllmPagedAttentionKernel

SD3 / Cosmos / Wan executor
  -> DiffusionTransformerModule
  -> DiTBlock / AttentionLayer / MLP layer
  -> torch or Triton kernels
```

## 和 vLLM 的对照

vLLM 的 `model_executor/layers` 不是单纯的 kernel 目录。它的核心是稳定 layer API：

```text
attention/
linear.py
layernorm.py
rotary_embedding.py
logits_processor.py
fused_moe/
quantization/
```

底层 backend 可以很复杂，但 model forward 看到的是 layer。

VRL 应该借这个边界，而不是照搬 vLLM 的所有目录：

```text
borrow:
  layer as model-composition boundary
  backend/kernel hidden behind layer
  attention cache metadata owned by attention layer contract
  quantization / Triton / vLLM as backend details

do not borrow yet:
  full vLLM scheduler
  full quantization stack
  fused MoE stack
  public serving engine lifecycle
  model registry shape
```

## 目标目录

本 sprint 只新增当前迁移必须用到的目录和文件：

```text
vrl/nn/
  __init__.py

  layers/
    __init__.py
    attention/
      __init__.py
      base.py
      paged.py

  kernels/
    __init__.py
    attention/
      __init__.py
      torch.py
      vllm_paged.py

  modules/
    __init__.py
    ar_decoder.py
```

后续真正有迁移需求时再新增：

```text
vrl/nn/layers/attention/dense.py
vrl/nn/layers/attention/cross.py
vrl/nn/layers/linear.py
vrl/nn/layers/norm.py
vrl/nn/layers/rotary_embedding.py
vrl/nn/layers/mlp.py
vrl/nn/layers/transformer_block.py
vrl/nn/layers/dit_block.py
vrl/nn/layers/logits_processor.py
vrl/nn/kernels/attention/triton.py
vrl/nn/kernels/norm/*
vrl/nn/kernels/linear/*
vrl/nn/modules/diffusion_transformer.py
vrl/nn/modules/vae.py
```

不要新增：

```text
vrl/generation/ar/*_kernel.py
vrl/models/ar/*/vllm_*.py
vrl/models/executor/ops/*
vrl/models/executor/kernels/*
```

`vrl/models/executor` 这个方向会把 family model ownership 和 reusable NN building blocks 混在一起。新的 reusable NN 层统一放到 `vrl/nn`。

## 分层规则

### Module

`module` 是模型推理图的一段稳定子图，知道输入输出语义，但不直接依赖具体 kernel。

例子：

```text
ARDecoderModule
DiffusionTransformerModule
VAEModule
```

允许知道：

```text
decode mode
prefill mode
hidden states
attention mask
position ids
replay tensors needed by trainer
```

不允许知道：

```text
Ray actor lifecycle
RolloutBatch merge/split
reward artifact storage
trainer step semantics
```

### Layer

`layer` 是 `torch.nn.Module` 风格的可组合计算边界，类似 vLLM `Attention` / `Linear` / `LayerNorm`。

例子：

```text
PagedSelfAttentionLayer
DenseAttentionLayer
CrossAttentionLayer
RMSNormLayer
RotaryEmbeddingLayer
DecoderTransformerBlock
DiTBlock
LogitsProcessorLayer
```

允许知道：

```text
q / k / v shape
head count
head dim
slot mapping
block table
KV cache view
dtype / device policy
```

不允许知道：

```text
Janus image-token offset
NextStep flow-head semantics
SD3 scheduler semantics
Cosmos conditioning format
reward component names
```

### Kernel

`kernel` 是后端实现，负责调用 torch / vLLM / Triton。

例子：

```text
TorchAttentionKernel
VllmPagedAttentionKernel
TritonAttentionKernel
TorchRMSNormKernel
TritonRMSNormKernel
```

要求：

```text
vLLM import 只能出现在 vrl/nn/kernels/attention/vllm_paged.py
Triton import 只能出现在 vrl/nn/kernels/*
family runner 不直接 import kernel
generation runtime 不直接 import kernel
kernel error 必须带 op name / backend / shape / dtype / device
```

## 当前迁移目标

当前：

```text
vrl/generation/ar/paged_attention.py
  holds AR paged attention contract and types

vrl/generation/ar/vllm_paged_attention.py
  holds vLLM import gate
  holds vLLM kernel wrapper
  holds decoder trunk stepping logic
```

目标：

```text
vrl/nn/layers/attention/base.py
  AttentionLayerBase
  AttentionMetadata
  AttentionCacheView

vrl/nn/layers/attention/paged.py
  PagedSelfAttentionLayer
  PagedAttentionConfig
  PagedAttentionPrefillInput
  PagedAttentionStepInput
  PagedAttentionOutput

vrl/nn/kernels/attention/vllm_paged.py
  VllmPagedAttentionKernel
  lazy vLLM import gate
  vLLM ABI validation
  block table / slot mapping conversion helpers

vrl/nn/kernels/attention/torch.py
  TorchAttentionKernel
  parity/debug implementation

vrl/nn/modules/ar_decoder.py
  ARDecoderModule
  prefill / step / finalize contract
```

`decode_loop.py` 不迁移到 `vrl/nn`：

```text
vrl/generation/ar/decode_loop.py
  owns token-level scheduling
  owns ActiveSequence / TokenBatch grouping
  owns request_id / sample_id row routing
  owns per-row opaque decode state storage

vrl/nn/modules/ar_decoder.py
  owns model primitive contract
  owns prefill / step / finalize tensor IO
  owns cache metadata shape seen by attention layers
  does not own request scheduling
```

两者并存，但职责不能重叠：

```text
decode_loop.py:
  decides which rows run next
  gathers/scatters per-row state
  calls family runner/runtime

ARDecoderModule:
  executes one model prefill or decode step
  calls PagedSelfAttentionLayer
  returns updated opaque state to caller
```

如果后续 `ARDecoderModule` 替换当前 runner 内部 cache path，`decode_loop.py` 也只持有 module 返回的 opaque state，不能持有 vLLM block table / slot mapping / kernel object。

Janus / NextStep runner 只能依赖：

```text
vrl.nn.modules.ar_decoder
vrl.nn.layers.attention.paged
```

不能依赖：

```text
vrl.nn.kernels.attention.vllm_paged
vllm.*
vrl.generation.ar.vllm_paged_attention
```

## 非目标

本 sprint 不做：

- 不实现 public serving engine。
- 不实现 rollout scheduler。
- 不实现 prefix-cache radix tree。
- 不把 vLLM 整个 engine 接进 Janus。
- 不复制 vLLM quantization / fused MoE 目录。
- 不重写所有 SD3 / Wan / Cosmos diffusion block。
- 不给每个小 tensor op 都单独建一个文件。
- 不保留旧 import shim。
- 不新增 legacy compatibility wrapper。
- 不让 `generation` 依赖具体 kernel。

## Phase 0：现状审计

目标：找出所有 model compute code 当前落错位置的地方。

检查：

```text
rg "vllm" vrl/generation vrl/models
rg "PagedAttention" vrl
rg "DynamicCache|past_key_values|block_table|slot_mapping" vrl
rg "torch\\.amp|flash|triton|attention" vrl/models vrl/generation
```

输出：

```text
current AR attention files
current Janus runner direct vLLM / cache dependencies
current NextStep runner direct vLLM / cache dependencies
current diffusion transformer / DiT compute boundaries
```

完成标准：

- 明确哪些代码移动到 `vrl/nn/layers`。
- 明确哪些代码移动到 `vrl/nn/kernels`。
- 明确哪些 family-specific 代码留在 `vrl/models/*`。

## Phase 1：建立 nn package skeleton

只新增当前 AR paged attention 迁移实际需要的文件：

```text
vrl/nn/layers/attention/base.py
vrl/nn/layers/attention/paged.py
vrl/nn/kernels/attention/torch.py
vrl/nn/kernels/attention/vllm_paged.py
vrl/nn/modules/ar_decoder.py
```

要求：

- `vrl/nn` 不 import `vrl.generation`。
- `vrl/nn` 不 import `vrl.rollouts`。
- `vrl/nn` 不 import trainer / reward。
- vLLM 只能 lazy import。
- 没有旧路径 shim。
- 不创建空的 `linear.py` / `norm.py` / `mlp.py` / `dit_block.py` / `diffusion_transformer.py`。
- 不迁移 `vrl/generation/ar/decode_loop.py`。

完成标准：

- package import 通过。
- 非 vLLM 环境可以 import `vrl.nn`。
- `ruff check vrl/nn` 通过。
- `decode_loop.py` 仍然只负责 scheduling / row routing，不新增 layer/kernel import。

## Phase 2：迁移 AR paged attention contract

移动：

```text
from vrl/generation/ar/paged_attention.py
to vrl/nn/layers/attention/paged.py
```

更新所有内部 import。

删除：

```text
vrl/generation/ar/paged_attention.py
```

不保留：

```text
from vrl.generation.ar.paged_attention import ...
```

完成标准：

- `rg "generation\\.ar\\.paged_attention" vrl tests` 无结果。
- Janus / NextStep tests 仍然通过。
- 没有 compatibility shim。

## Phase 3：拆分 vLLM paged attention implementation

当前 `vrl/generation/ar/vllm_paged_attention.py` 要拆成两层：

```text
VllmPagedAttentionKernel
  -> vrl/nn/kernels/attention/vllm_paged.py

PagedSelfAttentionLayer / ARDecoderModule glue
  -> vrl/nn/layers/attention/paged.py
  -> vrl/nn/modules/ar_decoder.py
```

要求：

- vLLM internal API import 只在 kernel 文件。
- `PagedSelfAttentionLayer` 不知道 Janus / NextStep。
- `ARDecoderModule` 不知道 Ray / rollout / trainer。
- family runner 不直接构造 vLLM kernel。
- `decode_loop.py` 不 import `vrl.nn.kernels`，也不 import vLLM。

完成标准：

- `rg "vllm\\.v1|from vllm|import vllm" vrl/models vrl/generation` 无结果。
- vLLM import 只出现在 `vrl/nn/kernels` 和 tests。
- CUDA vLLM parity test 仍然比较真实 vLLM paged path 和 HF eager path。

## Phase 4：Janus / NextStep runner 接 module，不接 kernel

目标调用链：

```text
JanusProARModelRunner
  -> ARDecoderModule
  -> PagedSelfAttentionLayer
  -> VllmPagedAttentionKernel

NextStepARModelRunner
  -> ARDecoderModule
  -> PagedSelfAttentionLayer
  -> VllmPagedAttentionKernel
```

runner 可以知道：

```text
family sampling config
visual token embedding
image-token logits projection
replay output schema
```

runner 不可以知道：

```text
vLLM block table class location
vLLM attention backend class location
vLLM paged attention op module path
Triton kernel path
```

完成标准：

- Janus / NextStep runner 没有 direct vLLM import。
- runner 中没有 generic block table / slot mapping conversion helper。
- old HF parity path 只作为 oracle 测试依赖，不作为 production backend 名字暴露。

## Phase 5：暂缓 diffusion NN module

本 sprint 不新增 diffusion module。原因是当前如果只包装现有 family model call：

```text
SD3 / Cosmos / Wan family forward
  -> DiffusionTransformerModule
  -> same family forward
```

这不构成真实 contract 迁移，只会多一层透传。

diffusion 进入 `vrl/nn` 的条件是至少满足一个：

```text
DiT block is split into reusable layer
dense/cross attention layer is shared by multiple diffusion families
diffusion executor can collect shape/latency/device stats through a stable module without importing family code
Triton kernel work needs a stable DiT layer boundary
```

后续 diffusion sprint 再新增：

```text
vrl/nn/modules/diffusion_transformer.py
vrl/nn/layers/dit_block.py
vrl/nn/layers/attention/dense.py
vrl/nn/layers/attention/cross.py
```

本 sprint 完成标准：

- 不新增 diffusion-only empty files。
- SD3 / Cosmos / Wan path 不改。
- 后续 DiT layer sprint 继续消费 `vrl/nn` 的 module/layer/kernel 边界。

## Phase 6：dispatch policy 和错误边界

新增：

```text
InferenceBackend
  torch
  vllm
  triton

InferenceDispatchPolicy
  preferred_backend
  allow_debug_fallback
  require_real_kernel
```

规则：

```text
production path:
  no silent fallback

parity/debug path:
  explicit torch fallback allowed

test path:
  skip if backend unavailable
```

错误信息必须包含：

```text
layer name
kernel name
backend
input shape
dtype
device
missing module or ABI mismatch
```

完成标准：

- vLLM 不存在时，非 vLLM tests 不失败。
- 请求 vLLM backend 但 vLLM 不可用时，错误清楚，不 silent fallback。
- parity test 明确标注 torch path 是 oracle/debug，不是默认 runtime。

## Phase 7：测试和验证

必须保留或新增：

```text
tests/nn/layers/test_paged_attention_contract.py
tests/nn/kernels/test_vllm_paged_attention_import_gate.py
tests/nn/modules/test_ar_decoder_module_contract.py
tests/generation/ar/test_janus_vllm_paged_attention_backend.py
tests/generation/ar/test_nextstep_vllm_paged_attention_backend.py
```

验证命令：

```text
ruff check vrl tests
pytest tests/nn tests/generation/ar tests/models/ar -q
```

CUDA / vLLM tests：

```text
skip when CUDA unavailable
skip when vLLM unavailable
must compare real vLLM paged attention path against HF eager oracle
```

完成标准：

- `vrl/generation/ar/paged_attention.py` 删除。
- `vrl/generation/ar/vllm_paged_attention.py` 删除。
- `vrl/models/ar/*/runner.py` 不直接 import vLLM。
- `vrl/nn/kernels/attention/vllm_paged.py` 是唯一 vLLM internal attention import 位置。
- Janus / NextStep AR tests 通过。
- diffusion tests 不因新 package 引入额外依赖。

## 后续 sprint

这些不属于本 sprint，但要按本 sprint 的边界继续：

```text
SPRINT_ar_model_executor.md
  consumes vrl.nn layers/modules for Janus/NextStep executor cleanup

future DiT layer sprint
  decomposes SD3/Cosmos/Wan transformer blocks into DiTBlock / AttentionLayer / MLP layer

future Triton kernel sprint
  adds Triton attention / RMSNorm / MLP kernels behind vrl.nn.kernels

future prefix-cache sprint
  uses ARDecoderModule and PagedSelfAttentionLayer metadata, not generation runtime internals
```

## 最终完成标准

完成后应该满足：

```text
model compute code has one home: vrl/nn
generation runtime does not own model layers
family runner does not own backend kernels
vLLM is hidden behind kernel import gates
torch reference path is explicit parity/debug only
no legacy import shim remains
```

这时新的调用关系应该能清楚表达为：

```text
vrl.generation
  -> requests rollout output from family runtime

vrl.models.<family>
  -> owns model-specific semantics
  -> calls vrl.nn modules/layers

vrl.nn.layers
  -> owns reusable model layer contracts
  -> calls vrl.nn.kernels

vrl.nn.kernels
  -> owns torch / vLLM / Triton backend-specific code
```
