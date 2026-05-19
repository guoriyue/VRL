# SPRINT：Diffusion native transformer executor

状态：proposed。

## 核心结论

当前 SD3.5 只有 attention processor 进入了 `vrl.nn`：

```text
SD3.5 diffusers transformer
  -> SD3JointAttentionProcessor
  -> TorchSDPAAttentionKernel
```

Wan / Cosmos 现在没有真正使用 `vrl.nn` layer/kernel。它们只是通过：

```text
vrl.models.diffusion.common.DiffusionBackboneCaller
```

复用了 CFG、timestep、transformer call orchestration。这是 model-call helper，不是 native model executor，也不是 vLLM `model_executor` 风格的 layer ownership。

这个 sprint 的目标是把 **Wan 2.1** 先迁移成 repo-owned native transformer executor，然后再做 **Cosmos Predict2 / Predict2.5**。真正完成后，VRL 需要拥有 transformer forward 里的 block / attention / MLP / norm / modulation 组合，而不是只调用 diffusers transformer。

## KV cache 结论

repo 现在已经有 AR KV / paged attention 路径：

```text
vrl.nn.layers.attention.paged
vrl.nn.kernels.attention.vllm_paged
vrl.nn.modules.ar_decoder.VllmDecoderPagedAttentionBackend
vrl.models.ar.nextstep_1.runner
```

这条路径服务的是 autoregressive decode：

```text
prefill prompt once
one-token decode repeatedly
past K/V grows with generated token length
block table / slot mapping / prefix cache can improve scheduling and memory
```

diffusion denoise 不是这个形态：

```text
each denoise step runs a full transformer forward
latent hidden_states changes every step
self-attention K/V depends on current latent tokens
cross-step self-attention KV reuse would change denoise semantics
```

所以 diffusion 不应该优先做 AR-style paged KV cache。

可能有价值的是 **context projection cache**，不是 AR KV cache：

```text
Wan cross-attention text K/V may be projected from constant encoder_hidden_states
Cosmos text/image context K/V may be partly constant depending on family path
cache target is projected context tensors, not latent self-attention KV
no block table
no prefix-cache scheduler
no growing decode history
```

这件事必须通过 profiling 决定。如果 context K/V projection 不是瓶颈，就不应该放到第一阶段。

## 为什么先 Wan，后 Cosmos

Wan 更适合作为第一条 native executor：

```text
WanTransformerBlock:
  norm1 + AdaLN scale/shift/gate
  self attention with RoPE
  norm2
  cross attention
  norm3
  FeedForward
```

它的结构比 Cosmos 更直，且 diffusers `WanTransformer3DModel` 顶层支持 `set_attn_processor`，说明 attention processor 边界比较明确。

Cosmos 更复杂：

```text
CosmosAdaLayerNormZero
CosmosAttnProcessor2_0 / CosmosAttnProcessor2_5
text context + image context tuple
attention_mask / condition_mask / padding_mask
GQA repeat_interleave
ControlNet residual hooks
Predict2 和 Predict2.5 forward 形态不同
```

所以 Cosmos 不应该作为第一条 native ownership path。先用 Wan 把 layer/kernel/executor/weight mapping/parity 方法跑通，再迁移 Cosmos。

## 当前问题

Wan 当前仍然由 diffusers 拥有 transformer：

```python
from diffusers import WanPipeline

pipeline = WanPipeline.from_pretrained(...)
self.transformer = pipeline.transformer
```

rollout/replay forward 只是调用：

```python
DiffusionBackboneCaller(
    self.transformer,
    WanDiffusionBackboneRunner(),
)
```

这意味着：

```text
VRL owns:
  CFG branch packing
  timestep shape
  VAE decode chunking
  replay state wrapper

diffusers still owns:
  WanTransformer3DModel.forward
  WanTransformerBlock
  WanAttention
  FeedForward
  FP32LayerNorm / RMSNorm behavior
  timestep embedding / rotary embedding
  transformer state_dict key layout
```

这不是最终想要的 model executor。

## 目标边界

目标调用链：

```text
vrl.models.diffusion.wan_2_1.model.WanT2VDiffusersModel
  -> WanNativeTransformerExecutor
     -> WanTransformerBlock
        -> WanSelfAttention / WanCrossAttention
           -> vrl.nn.kernels.attention.*
        -> WanFeedForward
        -> WanAdaLayerNormModulation
```

`DiffusionBackboneCaller` 可以继续负责 CFG branch orchestration，但它调用的 transformer 应该可以是 VRL native executor，而不是只能是 diffusers transformer。

## 目录设计

只创建被生产路径或 parity path 真实调用的文件。不要预建空 skeleton。

```text
vrl/nn/layers/
  attention/
    dense.py              # dense self/cross attention helper used by native diffusion blocks
    wan.py                # WanSelfAttention / WanCrossAttention if the block needs named layers
    joint.py              # existing SD3 joint text/image processor
  mlp.py                  # FeedForward-compatible MLP once Wan block uses it
  norm.py                 # RMSNorm / FP32LayerNorm-compatible wrappers when needed
  modulation.py           # AdaLN scale/shift/gate helpers when Wan block uses it

vrl/nn/modules/diffusion/
  wan.py                  # WanTransformerBlock only if block parity needs a named nn.Module
  cosmos.py               # CosmosTransformerBlock only after attention parity is real

vrl/models/diffusion/wan_2_1/
  executor.py             # WanNativeTransformerExecutor boundary
  weights.py              # diffusers WanTransformer3DModel -> native executor state mapping
  parity.py               # fixed-input parity helpers against diffusers oracle
  model.py                # selects native transformer only after parity passes

vrl/models/diffusion/cosmos/
  executor_common.py      # only if Predict2 and Predict2.5 actually share executor code

vrl/models/diffusion/cosmos/predict2/
  executor.py
  weights.py
  parity.py

vrl/models/diffusion/cosmos/predict2_5/
  executor.py
  weights.py
  parity.py
```

注意：`vrl/nn/modules/diffusion/` 这次可以重新出现，但含义必须不同。它只能放真正被 native forward 调用的 `torch.nn.Module`，不能放 CFG、timestep、decode policy 这种 model-call helper。

## Layer / module / dataclass 边界

这个 sprint 的默认落点应该是 `vrl.nn.layers`，不是 `vrl.nn.modules`。先按下面规则拆：

```text
kernel  = raw backend op, no model semantics
layer   = reusable math / weights, no request lifecycle
module  = composed model primitive, owns multiple layers plus state or execution contract
executor = family-specific full transformer forward boundary under vrl.models.*
```

Wan native path 的主要工作应该是 layers：

```text
WanSelfAttention / WanCrossAttention -> layer
WanFeedForward-compatible MLP        -> layer
FP32LayerNorm / RMSNorm wrappers     -> layer
AdaLN scale / shift / gate helpers   -> layer
```

只有这类对象才应该进 `vrl.nn.modules`：

```text
WanTransformerBlock
CosmosTransformerBlock
```

因为 block 组合了 norm、attention、MLP、modulation，并且需要做 copied-weight parity。`WanNativeTransformerExecutor` 不属于 `vrl.nn.modules`；它是 family model executor，应该在：

```text
vrl/models/diffusion/wan_2_1/executor.py
```

### 为什么 AR paged-attention backend 不是 Wan 的模板

`vrl/nn/modules/ar_decoder.py` 是 AR paged-attention 的特殊模块，不是 diffusion native executor 的模板。它有合理的 module 边界，因为它拥有：

```text
prefill / one-token step protocol
sequence state returned to family runners
physical vLLM KV page ownership
block table / slot mapping construction
KV cache allocation and growth
HF decoder trunk traversal
```

这些是 AR decoder lifecycle，不是单个 layer 的数学。diffusion denoise 没有 growing decode history，也没有跨 token 的 paged KV cache，所以不要把 AR paged-attention backend 的结构照搬到 Wan/Cosmos。

### Dataclass 使用规则

dataclass 只用于边界 payload 或多字段不变量，不用于普通局部变量打包。

允许：

```text
public API input/output payload
state object returned across calls
cache identity or ownership record
weight-mapping report with missing/unexpected keys
forward context that must keep several tensor fields shape-aligned
```

避免：

```text
只在一个 helper 里创建、立刻解包、没有验证逻辑的 private dataclass
为了减少函数参数数量而创建的临时容器
没有跨函数语义、不参与测试、不表达 invariant 的 wrapper
不增加空的中间 contract class；直接继承真正的接口
```

`ar_decoder.py` 当前只应该保留真正跨调用的状态对象：

```text
VllmDecoderPagedSequenceState
  合理：这是跨 prefill/step 返回给 family runner 的物理 KV page ownership。
```

这个 sprint 不把“一次性 private dataclass / 空 contract subclass / 临时上下文 wrapper”这种风格复制到 diffusion。Wan 第一版最多只需要少量明确 contract，例如 native executor config 或 weight mapping report；能直接来自 diffusers config 的字段，不要另建 dataclass。

## Non-goals

本 sprint 不做：

```text
不替换 VAE
不替换 scheduler
不替换 prompt/text encoder
不替换 pipeline loading
不把 Ray rollout scheduler 放进 model executor
不做 fake native backend selector
不保留 legacy compatibility alias
不创建没有 production/parity 调用的 empty layer files
不让 tests 只验证 wrapper 存在
不把 AR paged KV cache 套到 diffusion denoise
```

短期仍然可以从 diffusers 加载 pipeline/checkpoint。native executor 的第一步是替换 transformer forward，不是删除所有 diffusers import。

## Phase 0：Wan architecture audit

目标：把 diffusers Wan transformer 的真实结构固定下来，避免边写边猜。

产出：

```text
tests/models/diffusion/wan/test_wan_architecture_audit.py
```

验证内容：

```text
WanTransformer3DModel has transformer_blocks
each block has norm1 / attn1 / norm2 / attn2 / norm3 / ffn / scale_shift_table
attn1 is self attention
attn2 is cross attention
state_dict contains expected q/k/v/out/ffn/norm keys
LoRA target names currently used by configs still resolve after mapping plan
```

验收标准：

```text
pytest -q tests/models/diffusion/wan/test_wan_architecture_audit.py
```

这个 phase 不创建 native files，只确认 contract。

## Phase 0.5：Diffusion KV cache feasibility audit

目标：先确认 diffusion 是否存在值得缓存的 K/V projection，不把 AR KV cache 错搬进 denoise。

新增：

```text
tests/models/diffusion/wan/test_wan_context_kv_cache_audit.py
tests/models/diffusion/cosmos/test_cosmos_context_kv_cache_audit.py
```

验证内容：

```text
Wan self-attention K/V depends on current hidden_states and cannot be reused across denoise steps
Wan cross-attention K/V depends on encoder_hidden_states and can be considered for per-prompt context projection cache
Cosmos self-attention K/V depends on current hidden_states and cannot be reused across denoise steps
Cosmos text/image context K/V cacheability is documented per Predict2 / Predict2.5 path
```

产出：

```text
WanContextProjectionCache is either implemented with parity+benchmark or explicitly deferred
CosmosContextProjectionCache is either implemented with parity+benchmark or explicitly deferred
no diffusion ARPagedAttentionBackend usage is added
```

验收标准：

```text
No diffusion code imports ARPagedAttentionBackend
No diffusion code allocates vLLM block tables or slot mappings
Any context projection cache has exact parity against uncached attention
Benchmark shows projection cache is material before making it production default
```

## Phase 1：Shared dense attention helper

目标：抽一个真正可复用的 dense attention helper，而不是 family-specific wrapper。

新增：

```text
vrl/nn/layers/attention/dense.py
tests/nn/layers/attention/test_dense_attention.py
```

职责：

```text
projected q/k/v reshape helpers
head split / merge
optional q/k norm application
optional attention mask pass-through
optional RoPE hook input
call TorchSDPAAttentionKernel
```

不包含：

```text
Wan-specific image context split
Cosmos-specific GQA repeat
SD3-specific joint text/image concat
diffusers Attention object duck typing
```

验收：

```text
dense self attention matches direct torch.nn.functional.scaled_dot_product_attention
dense cross attention matches direct torch.nn.functional.scaled_dot_product_attention
mask shape errors are explicit
dtype/device are preserved
```

命令：

```bash
ruff check vrl tests
pytest -q tests/nn/layers/attention/test_dense_attention.py
```

## Phase 2：Wan attention layer

目标：实现 native Wan attention，并用 diffusers `WanAttention + WanAttnProcessor` 做 oracle。

新增：

```text
vrl/nn/layers/attention/wan.py
tests/nn/layers/attention/test_wan_attention.py
```

先实现：

```python
WanSelfAttention
WanCrossAttention
```

需要支持：

```text
to_q / to_k / to_v / to_out
norm_q / norm_k
self-attention RoPE
cross-attention encoder_hidden_states
added_kv_proj_dim path for Wan I2V if current model config needs it
```

不做：

```text
不先实现 whole Wan block
不做 fused QKV
不做 FlashAttention/Triton
不做 vLLM dense attention
```

验收：

```text
WanSelfAttention output matches diffusers WanAttention attn1 on fixed tensors
WanCrossAttention output matches diffusers WanAttention attn2 on fixed tensors
state_dict copy from diffusers attention module to VRL attention module is exact
```

## Phase 3：Wan MLP / norm / modulation

目标：实现 Wan block 所需的最小 layer 组合。

新增或扩展：

```text
vrl/nn/layers/mlp.py
vrl/nn/layers/norm.py
vrl/nn/layers/modulation.py
tests/nn/layers/test_diffusion_mlp_norm_modulation.py
```

需要覆盖：

```text
FP32LayerNorm-compatible behavior
Wan scale_shift_table AdaLN chunking
gate_msa / c_gate_msa residual modulation
FeedForward gelu-approximate path
```

验收：

```text
layernorm output matches diffusers block layernorm path
feed-forward output matches diffusers FeedForward on copied weights
Wan modulation chunks match diffusers block forward intermediate shapes
```

不要抽全量“通用 DiTBlock”。先只实现 Wan block 真实需要的 pieces。

## Phase 4：WanTransformerBlock parity

目标：VRL 拥有单个 Wan block forward。

扩展：

```text
vrl/nn/modules/diffusion/wan.py
tests/nn/modules/diffusion/test_wan_block.py
```

实现：

```python
WanTransformerBlock
```

输入 contract 必须对齐 diffusers：

```text
hidden_states
encoder_hidden_states
temb
rotary_emb
```

验收：

```text
copy one diffusers WanTransformerBlock state_dict into VRL block
fixed hidden_states / encoder_hidden_states / temb / rotary_emb output matches
train/eval mode behavior matches for dropout=0
dtype float32 and fp16/bf16 on CUDA when available
```

命令：

```bash
pytest -q tests/nn/modules/diffusion/test_wan_block.py
```

## Phase 5：WanNativeTransformerExecutor

目标：VRL 拥有 Wan transformer forward，不再只调用 diffusers transformer。

新增：

```text
vrl/models/diffusion/wan_2_1/executor.py
vrl/models/diffusion/wan_2_1/weights.py
tests/models/diffusion/wan/test_wan_native_executor.py
```

实现：

```python
WanNativeTransformerExecutor
load_wan_diffusers_transformer_weights(...)
```

必须对齐：

```text
patch embedding / latent flattening
timestep embedding
rotary embedding creation / use
transformer block sequence
final norm / projection
return_dict=False behavior
output shape equals diffusers WanTransformer3DModel
```

验收：

```text
state_dict mapping has no missing trainable weights
state_dict mapping has no unexpected trainable weights
one-step forward output matches diffusers transformer within tolerance
gradient path exists through trainable transformer params
```

## Phase 6：Wan model production switch

目标：Wan rollout/replay 默认调用 native executor。

改：

```text
vrl/models/diffusion/wan_2_1/model.py
vrl/models/diffusion/wan_2_1/runtime.py
vrl/models/diffusion/wan_2_1/runner.py
```

规则：

```text
WanT2VDiffusersModel.from_spec may still load WanPipeline for VAE/text/scheduler
pipeline.transformer weights are copied into WanNativeTransformerExecutor
self.transformer becomes WanNativeTransformerExecutor
DiffusionBackboneCaller calls native executor
diffusers transformer remains only in tests/parity helpers, not production fallback
```

LoRA 验收：

```text
existing Wan LoRA target names either map to native module names or config validation fails clearly
apply_lora works on native executor
replay runtime loads transformer weights into native executor without text/VAE pipeline modules
```

验证：

```bash
ruff check vrl tests
pytest -q tests/models/diffusion/wan tests/models/test_wan_diffusion_backbone_parity.py
pytest -q tests/models/test_diffusion_decode_layout_parity.py
```

## Phase 7：Wan benchmark gate

目标：确认 native executor 没有明显变慢，并为后续 Flash/Triton 提供基线。

新增：

```text
tests/models/diffusion/wan/test_wan_native_executor_benchmark.py
```

规则：

```text
benchmark test 默认 skip unless VRL_RUN_BENCHMARKS=1
记录 diffusers transformer vs native executor latency
记录 peak memory when CUDA is available
不把 benchmark 数字写死成稳定单测
```

通过标准：

```text
native executor correctness tests pass
benchmark report prints latency/memory
native path does not introduce obvious allocation explosion
context projection cache remains disabled unless benchmark proves value
```

## Phase 8：Cosmos Predict2 attention parity

目标：先迁移 Cosmos attention processors，不直接迁移 whole transformer。

新增：

```text
vrl/nn/modules/diffusion/cosmos.py
tests/nn/modules/diffusion/test_cosmos_attention.py
```

实现：

```python
CosmosAttention2_0
CosmosAttention2_5
```

需要支持：

```text
q/k/v projection
q/k norm
image_rotary_emb
GQA repeat_interleave
text attention mask
image context branch for 2.5
img_mask when used by upstream
```

验收：

```text
CosmosAttention2_0 matches diffusers CosmosAttnProcessor2_0
CosmosAttention2_5 matches diffusers CosmosAttnProcessor2_5
Predict2 and Predict2.5 tensor shapes are both covered
```

## Phase 9：Cosmos block parity

目标：VRL 拥有 Cosmos block forward。

扩展：

```text
vrl/nn/modules/diffusion/cosmos.py
tests/nn/modules/diffusion/test_cosmos_block.py
```

需要覆盖：

```text
CosmosAdaLayerNormZero
attn1
attn2
FeedForward
before_proj / after_proj if config enables them
controlnet_residual hook if current forward uses it
extra_pos_emb / image_rotary_emb pass-through
```

验收：

```text
one CosmosTransformerBlock copied from diffusers matches output
Predict2 block config covered
Predict2.5 img_context block config covered
```

## Phase 10：Cosmos native executors

目标：Predict2 和 Predict2.5 拥有 native transformer executor。

新增：

```text
vrl/models/diffusion/cosmos/predict2/executor.py
vrl/models/diffusion/cosmos/predict2/weights.py
vrl/models/diffusion/cosmos/predict2_5/executor.py
vrl/models/diffusion/cosmos/predict2_5/weights.py
tests/models/diffusion/cosmos/test_predict2_native_executor.py
tests/models/diffusion/cosmos/test_predict25_native_executor.py
```

规则：

```text
Predict2 and Predict2.5 may share low-level blocks
Predict2 and Predict2.5 executor files stay separate if forward signatures differ
do not create a fake all-Cosmos executor if it only adds conditionals
```

验收：

```text
state_dict mapping for Predict2 has no missing trainable weights
state_dict mapping for Predict2.5 has no missing trainable weights
one-step forward parity against diffusers transformer
existing replay/runtime minimal module loading still avoids text/VAE/pipeline on trainer side
```

## Phase 11：Kernel acceleration follow-up

这个 sprint 的 first success 是 native ownership，不是马上加速。

native executor 通过后，才做：

```text
vrl/nn/kernels/attention/flash.py
vrl/nn/kernels/attention/triton_dense.py
```

前置条件：

```text
Wan native executor is default and parity-tested
Cosmos native executor is parity-tested or has attention-level parity
benchmark shows attention is material runtime/memory bottleneck
context projection cache audit shows projection cost is worth caching, if cache is pursued
```

不满足这些条件时，不写 Triton kernel。否则又会变成看起来很酷但没有 production path 的代码。

## Testing matrix

每个 phase 都至少跑：

```bash
ruff check vrl tests
pytest -q tests/nn tests/models
git diff --check
```

Wan native switch 前必须额外跑：

```bash
pytest -q tests/models/test_wan_diffusion_backbone_parity.py
pytest -q tests/models/test_diffusion_decode_layout_parity.py
pytest -q tests/engine/diffusion tests/engine/generation
```

Cosmos native switch 前必须额外跑：

```bash
pytest -q tests/models/test_cosmos_predict2_diffusion_backbone_parity.py
pytest -q tests/models/test_cosmos_predict25_diffusion_backbone_parity.py
pytest -q tests/models/test_diffusion_decode_layout_parity.py
pytest -q tests/engine/diffusion tests/engine/generation
```

## Completion criteria

完成本 sprint 时必须满足：

```text
Wan production forward uses WanNativeTransformerExecutor by default
Wan native executor has block-level and full-transformer parity against diffusers
Wan LoRA path works or fails with explicit unsupported-target validation
Cosmos attention/block/native executor parity is implemented according to phases completed
vrl.nn contains only real layers/modules/kernels called by native forward
no legacy import aliases
no fake backend selector
no empty skeleton files
diffusers remains only for pipeline/checkpoint oracle where still intentionally needed
diffusion does not use AR-style paged KV cache
context projection cache, if added, has exact parity and benchmark evidence
```

## Rollback rule

如果 native executor parity fails, rollback the production switch, not the tested layer code:

```text
keep passing WanAttention / WanTransformerBlock parity modules
do not keep a broken default native executor
do not add runtime flag that silently falls back to diffusers
fix mapping/parity before switching production path
```
