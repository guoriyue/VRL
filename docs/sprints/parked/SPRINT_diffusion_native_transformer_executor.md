# SPRINT：Diffusion native transformer executor

状态：未开始 / parked。Phase 0-11 全部未落地（grep WanNativeTransformerExecutor / WanSelfAttention / load_wan_diffusers_transformer_weights / CosmosAttention2_0 在 vrl/ + tests/ 零命中；vrl/nn/layers/{dense,wan,mlp,norm,modulation}.py、vrl/nn/modules/diffusion/、vrl/models/diffusion/*/executor.py+weights.py 均不存在；backbone.py:88,153 仍 self.transformer = diffusers object）。唯一相关提交 0abb5fd 只是把本 doc 砍成 forward-looking plan，未写任何实现。触发事件：bf16 rollout dtype + torch.compile（31f6843）启用后，需重新 profiling 确认 attention 仍是瓶颈（见 SPRINT_rollout_performance：attention 34% 但全 fp32）才启动。

## 核心结论

Wan / Cosmos 的 transformer forward 仍然由 diffusers 拥有。VRL 只做了 CFG branch orchestration（`DiffusionBackboneCaller`），没有 layer ownership。

本 sprint 的目标：把 **Wan 2.1** 先迁移成 repo-owned native transformer executor，再做 **Cosmos Predict2 / Predict2.5**。完成后 VRL 拥有 transformer forward 里的 block / attention / MLP / norm / modulation 组合，而不是只调用 diffusers transformer。

## KV cache 结论

diffusion denoise 不适用 AR-style paged KV cache。每步 denoise 跑完整 transformer forward，latent hidden_states 每步变化，self-attention K/V 依赖当前 latent tokens，跨步复用会改变 denoise 语义。

可能有价值的是 **context projection cache**（constant encoder output 的 K/V projection），但必须通过 profiling 决定，不在第一阶段做。

## 为什么先 Wan，后 Cosmos

Wan 结构更直：

```text
WanTransformerBlock:
  norm1 + AdaLN scale/shift/gate
  self attention with RoPE
  norm2
  cross attention
  norm3
  FeedForward
```

Cosmos 更复杂（CosmosAdaLayerNormZero、GQA repeat_interleave、text+image context tuple、ControlNet residual hooks、Predict2 和 Predict2.5 forward 形态不同）。先用 Wan 把 layer/kernel/executor/weight mapping/parity 方法跑通，再迁移 Cosmos。

## 目标调用链

```text
WanT2VDiffusersModel
  -> DiffusionBackboneCaller (CFG orchestration, unchanged)
     -> WanNativeTransformerExecutor
        -> WanTransformerBlock
           -> WanSelfAttention / WanCrossAttention
              -> TorchSDPAAttentionKernel
           -> WanFeedForward
           -> WanAdaLayerNormModulation
```

`DiffusionBackboneCaller` 继续负责 CFG branch orchestration，不变。它调用的 transformer 从 diffusers object 替换成 VRL native executor。

## 目录设计

只创建被生产路径或 parity path 真实调用的文件。不预建空 skeleton。

```text
vrl/nn/layers/
  attention/
    dense.py              # dense self/cross attention helper (shared by Wan, future Cosmos)
    wan.py                # WanSelfAttention / WanCrossAttention
  mlp.py                  # FeedForward-compatible MLP
  norm.py                 # FP32LayerNorm / RMSNorm wrappers
  modulation.py           # AdaLN scale/shift/gate helpers

vrl/nn/modules/diffusion/
  wan.py                  # WanTransformerBlock (组合 norm + attention + MLP + modulation)

vrl/models/diffusion/wan_2_1/
  executor.py             # WanNativeTransformerExecutor
  weights.py              # diffusers -> native state_dict mapping
```

Cosmos 后续阶段：

```text
vrl/nn/modules/diffusion/
  cosmos.py               # CosmosTransformerBlock

vrl/models/diffusion/cosmos/predict2/
  executor.py
  weights.py

vrl/models/diffusion/cosmos/predict2_5/
  executor.py
  weights.py
```

## Layer / module / executor 边界

```text
kernel  = raw backend op, no model semantics         (vrl/nn/kernels/)
layer   = reusable math + weights, no request lifecycle  (vrl/nn/layers/)
module  = composed model primitive, owns multiple layers  (vrl/nn/modules/)
executor = family-specific full transformer forward      (vrl/models/diffusion/*/executor.py)
```

Wan native path 的主要工作在 layers。只有 WanTransformerBlock 进 `vrl.nn.modules`（因为它组合了多个 layers 并需要 copied-weight parity）。WanNativeTransformerExecutor 是 family model executor，属于 `vrl/models/diffusion/wan_2_1/executor.py`。

AR paged-attention backend 不是 Wan 的模板。diffusion denoise 没有 growing decode history 和 paged KV cache，不照搬。

Dataclass 只用于边界 payload 或多字段不变量。能直接来自 diffusers config 的字段，不另建 dataclass。

## Non-goals

```text
不替换 VAE
不替换 scheduler
不替换 prompt/text encoder
不替换 pipeline loading（仍从 diffusers 加载 pipeline 获取权重）
不改 DiffusionBackboneCaller / DiffusionBackboneRunner 接口
不把 Ray rollout scheduler 放进 model executor
不做 fake native backend selector
不保留 legacy compatibility alias
不创建没有 production/parity 调用的 empty layer files
不把 AR paged KV cache 套到 diffusion denoise
```

## Phase 0：Wan architecture audit

目标：把 diffusers Wan transformer 的真实内部结构固定下来。

产出：

```text
tests/models/diffusion/wan_2_1/test_wan_architecture_audit.py
```

验证内容：

```text
WanTransformer3DModel has transformer_blocks
each block has norm1 / attn1 / norm2 / attn2 / norm3 / ffn / scale_shift_table
attn1 is self attention (no cross_attention_dim)
attn2 is cross attention (has cross_attention_dim)
state_dict contains expected q/k/v/out/ffn/norm keys
LoRA target names currently used by configs resolve correctly
```

验收：

```bash
pytest -q tests/models/diffusion/wan_2_1/test_wan_architecture_audit.py
```

不创建 native files，只确认 contract。

## Phase 1：Shared dense attention helper

目标：一个可复用的 dense attention helper（不同于 SD3 joint attention 和 AR paged attention）。

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

不包含 Wan-specific image context split、Cosmos GQA repeat、SD3 joint concat、diffusers Attention duck typing。

验收：

```text
dense self attention matches direct F.scaled_dot_product_attention
dense cross attention matches direct F.scaled_dot_product_attention
mask shape errors are explicit
dtype/device are preserved
```

## Phase 2：Wan attention layer

目标：native Wan attention，用 diffusers `WanAttention` 做 oracle parity。

新增：

```text
vrl/nn/layers/attention/wan.py
tests/nn/layers/attention/test_wan_attention.py
```

实现 `WanSelfAttention` 和 `WanCrossAttention`。需要支持 to_q/to_k/to_v/to_out、norm_q/norm_k、self-attention RoPE、cross-attention encoder_hidden_states、added_kv_proj_dim path for I2V。

不做 fused QKV、FlashAttention/Triton。

验收：

```text
WanSelfAttention output matches diffusers WanAttention attn1 on fixed tensors
WanCrossAttention output matches diffusers WanAttention attn2 on fixed tensors
state_dict copy from diffusers attention module to VRL attention module is exact
```

## Phase 3：Wan MLP / norm / modulation

目标：Wan block 所需的最小 layer 组合。

新增：

```text
vrl/nn/layers/mlp.py
vrl/nn/layers/norm.py
vrl/nn/layers/modulation.py
tests/nn/layers/test_diffusion_mlp_norm_modulation.py
```

覆盖 FP32LayerNorm-compatible behavior、Wan scale_shift_table AdaLN chunking、gate_msa / c_gate_msa residual modulation、FeedForward gelu-approximate path。

不抽全量"通用 DiTBlock"。先只实现 Wan block 真实需要的 pieces。

## Phase 4：WanTransformerBlock parity

目标：VRL 拥有单个 Wan block forward。

新增：

```text
vrl/nn/modules/diffusion/wan.py
tests/nn/modules/diffusion/test_wan_block.py
```

输入 contract 对齐 diffusers：hidden_states、encoder_hidden_states、temb、rotary_emb。

验收：

```text
copy one diffusers WanTransformerBlock state_dict into VRL block
fixed input output matches within tolerance
train/eval mode behavior matches for dropout=0
dtype float32 and fp16/bf16 on CUDA when available
```

## Phase 5：WanNativeTransformerExecutor

目标：VRL 拥有 Wan transformer forward。

新增：

```text
vrl/models/diffusion/wan_2_1/executor.py
vrl/models/diffusion/wan_2_1/weights.py
tests/models/diffusion/wan_2_1/test_wan_native_executor.py
```

实现 `WanNativeTransformerExecutor` 和 `load_wan_diffusers_transformer_weights`。

必须对齐 patch embedding / latent flattening、timestep embedding、rotary embedding、transformer block sequence、final norm / projection、return_dict=False behavior、output shape。

验收：

```text
state_dict mapping has no missing trainable weights
state_dict mapping has no unexpected trainable weights
one-step forward output matches diffusers transformer within tolerance
gradient path exists through trainable transformer params
```

## Phase 6：Wan model production switch

目标：Wan rollout/replay 默认调用 native executor。

修改 `vrl/models/diffusion/wan_2_1/model.py` 和 `runtime.py`。

规则：

```text
from_build may still load WanPipeline for VAE/text/scheduler
pipeline.transformer weights are copied into WanNativeTransformerExecutor
self.transformer becomes WanNativeTransformerExecutor
DiffusionBackboneCaller calls native executor (runner unchanged)
diffusers transformer remains only in tests/parity helpers, not production fallback
```

LoRA 验收：

```text
existing Wan LoRA target names either map to native module names or config validation fails clearly
apply_lora works on native executor
replay runtime loads transformer weights into native executor
```

验证：

```bash
ruff check vrl tests
pytest -q tests/models/diffusion/wan_2_1/
pytest -q tests/models/diffusion/common/test_decode_layout_parity.py
pytest -q tests/models/diffusion/test_tiny_pipeline_wiring.py
```

## Phase 7：Wan benchmark gate

目标：确认 native executor 没有明显变慢，为后续 Flash/Triton 提供基线。

新增：

```text
tests/models/diffusion/wan_2_1/test_wan_native_executor_benchmark.py
```

默认 skip unless `VRL_RUN_BENCHMARKS=1`。记录 diffusers vs native latency + peak memory。不把 benchmark 数字写死成稳定单测。

## Phase 8：Cosmos Predict2 attention parity

目标：先迁移 Cosmos attention processors。

新增：

```text
vrl/nn/modules/diffusion/cosmos.py
tests/nn/modules/diffusion/test_cosmos_attention.py
```

实现 `CosmosAttention2_0` 和 `CosmosAttention2_5`。需要支持 q/k/v projection、q/k norm、image_rotary_emb、GQA repeat_interleave、text attention mask、image context branch for 2.5。

验收：output matches diffusers CosmosAttnProcessor2_0 / 2_5，Predict2 和 Predict2.5 tensor shapes covered。

## Phase 9：Cosmos block parity

目标：VRL 拥有 Cosmos block forward。

扩展 `vrl/nn/modules/diffusion/cosmos.py`。覆盖 CosmosAdaLayerNormZero、attn1、attn2、FeedForward、before_proj / after_proj、controlnet_residual hook、extra_pos_emb / image_rotary_emb。

验收：Predict2 block config 和 Predict2.5 img_context block config 均 covered。

## Phase 10：Cosmos native executors

目标：Predict2 和 Predict2.5 拥有 native transformer executor。

新增：

```text
vrl/models/diffusion/cosmos/predict2/executor.py
vrl/models/diffusion/cosmos/predict2/weights.py
vrl/models/diffusion/cosmos/predict2_5/executor.py
vrl/models/diffusion/cosmos/predict2_5/weights.py
tests/models/diffusion/cosmos/predict2/test_native_executor.py
tests/models/diffusion/cosmos/predict2_5/test_native_executor.py
```

Predict2 和 Predict2.5 may share blocks but keep separate executors if forward signatures differ。

验证：

```bash
ruff check vrl tests
pytest -q tests/models/diffusion/cosmos/predict2/
pytest -q tests/models/diffusion/cosmos/predict2_5/
pytest -q tests/models/diffusion/common/test_decode_layout_parity.py
```

## Phase 11：Kernel acceleration follow-up

本 sprint 的 first success 是 native ownership，不是马上加速。

native executor 通过后才做 `vrl/nn/kernels/attention/flash.py` 和 `triton_dense.py`。前置条件：Wan native executor is default and parity-tested，benchmark shows attention is material bottleneck。不满足时不写 Triton kernel。

## Completion criteria

```text
Wan production forward uses WanNativeTransformerExecutor by default
Wan native executor has block-level and full-transformer parity against diffusers
Wan LoRA path works or fails with explicit unsupported-target validation
Cosmos parity is implemented according to phases completed
vrl.nn contains only real layers/modules/kernels called by native forward
no legacy import aliases, no fake backend selector, no empty skeleton files
diffusers remains only for pipeline/checkpoint loading where still needed
diffusion does not use AR-style paged KV cache
```

## Rollback rule

如果 native executor parity fails，rollback the production switch，not the tested layer code。不加 runtime flag silently falling back to diffusers。fix mapping/parity before switching production path。
