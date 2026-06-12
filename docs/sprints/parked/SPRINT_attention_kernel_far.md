# SPRINT: Own AR decoder definitions and weight loading

状态: proposed, far / aspirational. **Do not start this sprint by default.**

前置依赖: `SPRINT_attention_kernel_medium.md` 必须完成并稳定:

- `ARDecoderDriver` exists and matches HF/vLLM parity within documented tolerances.
- `ARAttentionKernel` exists and has at least one real backend using it.
- The remaining bottleneck is proven to be HF submodule ownership, not runner wiring.

本 sprint 的目标是从 “owned driver + HF submodules” 进一步升级到
**owned decoder `nn.Module` definitions + explicit HF weight loader + shared attention kernels**.
这是 vLLM/SGLang 维护成本级别的工作, 不是 cleanup.

## 0. Start Criteria

Only start this sprint if at least one of these is true and measured:

1. A concrete kernel optimization is blocked by HF module structure:
   - fused QKV projection;
   - KV cache quantization;
   - custom RoPE layout;
   - chunked prefill;
   - fused attention + MLP scheduling;
   - backend-specific tensor layout that HF modules cannot expose cleanly.

2. Model count × backend count is large enough that HF-submodule adapters are now the main maintenance cost.

3. A production performance target cannot be reached with the medium sprint architecture.

If none of these is true, stop at medium. Owning model definitions creates a long-term maintenance burden.

## 1. Core Decision

Medium sprint still uses HF/PEFT modules as the weight and compute containers:

```text
HF decoder layer modules -> ARDecoderDriver -> ARAttentionKernel
```

This far sprint replaces the HF decoder layer modules with our own decoder modules:

```text
HF checkpoint / PEFT checkpoint
  -> explicit weight loader
  -> owned AR decoder nn.Module
     -> ARAttentionKernel
```

The kernel stays shared. The decoder definition becomes ours.

## 2. Scope Boundary

Be precise about what “own the model” means. This sprint should own only the AR decoder unless a separate
decision expands scope.

### In Scope

- Decoder-only transformer definitions for supported architectures:
  - `vrl/models/ar/decoders/llama.py`
  - `vrl/models/ar/decoders/qwen2.py`
- Attention layers that call `ARAttentionKernel`.
- Explicit weight loaders from HF state dicts into owned decoder parameters.
- Parity tests against HF/PEFT outputs.
- Minimal integration into Janus/NextStep runners.

### Out of Scope Unless Explicitly Reopened

- Janus VQ decoder.
- Janus `gen_head`, multimodal wrapper, processor, image-token embedding.
- NextStep VAE.
- NextStep `image_head`, `image_in_projector`, `image_out_projector`.
- Tokenizers/processors.
- Diffusion models.

The first implementation should keep those non-decoder pieces in the existing family model wrappers and
replace only the decoder trunk.

## 3. Architecture

```text
vrl/models/ar/decoders/
  base.py             # owned decoder protocol / common small helpers
  llama.py            # Llama-compatible decoder module
  qwen2.py            # Qwen2-compatible decoder module
  weights.py          # HF/PEFT weight loader helpers and remap tables

OwnedARDecoder
  layers
    self_attn -> ARAttentionKernel
    mlp
    norms
  rotary
  final norm
```

Runtime shape:

```text
JanusProModel / NextStep1Model
  keeps non-decoder pieces
  owns or references OwnedARDecoder
  runner calls decoder through ARDecoderDriver or a thin compatible interface
```

Do not create a generic “all AR model” base class unless it removes real duplication. Per-architecture modules
are expected.

## 4. Weight Loading

This is the highest-risk part of the far sprint.

The loader must cover:

- base HF decoder weights;
- PEFT/LoRA adapter weights;
- tied or shared embeddings if the owned decoder touches them;
- dtype/device placement;
- strict missing/extra key reporting;
- per-architecture key remapping.

Suggested structure:

```text
vrl/models/ar/decoders/weights.py
  load_hf_decoder_weights_into(decoder, state_dict, *, architecture)
  load_peft_lora_weights_into(decoder, state_dict, *, architecture)
  validate_loaded_decoder_against_hf(decoder, hf_trunk)
```

Keep remap tables isolated and named by architecture:

```text
LLAMA_DECODER_WEIGHT_MAP
QWEN2_DECODER_WEIGHT_MAP
```

These are acceptable ALL_CAPS constants because they are deliberately isolated checkpoint/key mapping tables.
Do not mix them into runtime workflow code.

## 5. LoRA / PEFT Boundary

Do not assume LoRA is identical across current AR families.

Current behavior differs:

- Janus wraps only `mmgpt.language_model`.
- NextStep wraps the whole upstream `language_model` / pipeline model.

This sprint must choose one of these policies and document it:

1. **Load merged weights only**: require LoRA to be merged into base decoder before constructing the owned decoder.
2. **Own LoRA modules**: represent LoRA adapters inside the owned decoder and load adapter state separately.
3. **Hybrid transition**: keep HF/PEFT decoder for LoRA path, use owned decoder only for full/base inference
   until LoRA parity exists.

Recommended first policy: hybrid transition. It keeps the far sprint from breaking training-time LoRA paths
while proving owned decoder inference first.

## 6. Implementation Steps

### T1. Pick One Architecture First

Start with one architecture, not both.

Recommended first target: Qwen2/NextStep if the goal is continuous-token AR kernel work; Llama/Janus if the
goal is discrete VQ-token AR.

Criteria:

- smallest parity test surface;
- easiest checkpoint/key layout;
- clearest performance target.

### T2. Build Owned Decoder Module

Implement the minimal decoder needed for inference parity:

- q/k/v/o projection;
- GQA;
- RoPE;
- MLP;
- RMSNorm/layernorm;
- residual order;
- final norm;
- cache-state interface to `ARAttentionKernel`.

No optimization yet. First version should be boring and parity-focused.

### T3. Build Weight Loader

Load one tiny HF trunk and one real checkpoint path if available.

Tests must check:

- all expected keys loaded;
- all loaded shapes match;
- unexpected keys are either rejected or explicitly ignored with documented reason;
- decoder output matches HF trunk output before any optimization.

### T4. Integrate Behind a Feature Flag

Add a new internal runtime option, not a public default:

```text
sampling.decoder_impl = "hf_submodules" | "owned"
```

Default remains `hf_submodules` until parity and performance are proven.

### T5. Optimize Only After Parity

Only after T1-T4 pass:

- fused QKV;
- KV-cache quantization;
- chunked prefill;
- FlashInfer/Triton kernels;
- layout-specific cache allocation.

Each optimization must have parity against the unoptimized owned decoder.

## 7. Numerical Contract

Use explicit tolerance levels:

- owned decoder vs HF trunk, same dtype/device: tight `assert_close`; exact bitwise is not required unless
  proven stable.
- owned decoder with LoRA vs HF/PEFT LoRA: separate gate; do not infer from base parity.
- optimized kernel vs unoptimized owned decoder: tolerance depends on backend and dtype, must be documented per test.

Minimum test matrix:

```text
architecture x dtype x cache path
  llama or qwen2
  fp32 CPU for deterministic smoke if supported
  fp16/bf16 GPU for real kernel parity
  prefill + one decode step
  left-padded prompt masks
  LoRA path when enabled
```

## 8. Verification

Example commands after implementation:

```bash
python -m pytest tests/models/ar tests/generation/ar -q
python -m pytest -m gpu tests/generation/ar -q
python -m pytest tests/models/ar/decoders -q
ruff check vrl/models/ar/decoders vrl/nn/layers/attention vrl/nn/modules
```

The far sprint is not complete until owned decoder parity passes and the HF-submodule path still works.

## 9. Non-Goals

- Do not remove the HF-submodule driver path in the first far implementation.
- Do not replace all AR families at once.
- Do not move non-decoder Janus/NextStep components unless separately justified.
- Do not make owned decoder the default before parity and perf gates are recorded.
- Do not implement custom kernels before the unoptimized owned decoder is correct.
- Do not touch diffusion.

## 10. Stop Criteria

Stop or defer this sprint if:

- weight-loader complexity exceeds the performance benefit;
- LoRA parity is not clear;
- only one or two AR families remain and medium architecture is sufficient;
- the target optimization can be implemented with HF submodules through the medium driver.

## 11. References

```text
docs/sprints/planned/SPRINT_attention_kernel_medium.md
  required precursor: owned driver + HF submodules

vrl/nn/layers/attention/kernel.py
  expected medium output: ARAttentionKernel

vrl/nn/modules/ar_decoder_driver.py
  expected medium output: shared decoder driver

vrl/models/ar/{janus_pro,nextstep_1}/model.py
  current family models and non-decoder ownership

vrl/models/ar/{janus_pro,nextstep_1}/runner.py
  current runtime use of attention backend
```

External references:

- vLLM owned model definitions and weight loaders: `vllm/model_executor/models/{llama,qwen2}.py`
- vLLM attention layers: `vllm/model_executor/layers/attention/*`
- SGLang model definitions: `python/sglang/srt/models/*`
- SGLang attention backends: `python/sglang/srt/layers/attention/*`
