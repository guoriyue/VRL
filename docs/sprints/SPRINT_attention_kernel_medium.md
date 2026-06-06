# SPRINT: Own the AR decoder driver while reusing HF submodules

状态: proposed. 接续 `SPRINT_backend.md` 已落地的 name-based attention-backend dispatch.

本 sprint 的真实目标不是“仍走 HF forward”。它是:
**自己维护一个 Llama/Qwen2-compatible AR decoder forward driver,**
**但继续复用 HF trunk 的 layer/module 作为权重和计算单元**。
也就是我们自己跑 decoder loop, attention 这一步交给 pluggable kernel, 但暂时不声明自有 `nn.Module`
decoder、不写 HF checkpoint weight-loader.

这一步风险中等偏高, 不是纯低风险抽取。它会把 mask、position ids、RoPE、GQA、
sliding window、KV cache 状态、LoRA unwrap 等行为纳入我们自己的维护范围。
收益是把“forward driver”和“attention kernel”拆开, 为后续 vLLM/SGLang 式
kernel backend 打基础。

## 0. Core Decision

当前 `ARAttentionBackend.prefill/step` 把两件事揉在一起:

1. **Decoder forward driver**: 遍历 decoder layers, 跑 layernorm、self-attention、MLP、residual、final norm。
2. **Attention kernel**: 已投影并已 RoPE 的 `q/k/v -> attention output`, 以及 KV cache 更新。

本 sprint 拆成三层:

```text
family runner
  -> ARAttentionBackend              # existing prefill/step boundary
     -> ARDecoderDriver              # shared decoder loop, owned by us
        -> HFDecoderView             # thin adapter over HF trunk submodules
        -> ARAttentionKernel         # pluggable attention math/cache backend
```

`ARDecoderDriver` 不是 generic “all AR models” driver. 它只承诺支持当前 repo 需要的
**Llama/Qwen2-compatible decoder view**:

```text
layers
  layer.input_layernorm
  layer.self_attn.{q,k,v,o}_proj
  layer.post_attention_layernorm
  layer.mlp
rotary_emb
norm
config.{num_attention_heads,num_key_value_heads,hidden_size}
```

Janus-Pro 和 NextStep-1 目前都能暴露成这个 view, 但 adapter 必须显式验证,
不能靠“看起来命名一样”。

## 1. What Stays Unchanged

- `attention_backend` 对外名字保持: `vllm_paged` / `torch_native`.
- Runtime selection 继续使用 `resolve_attention_backend(family, name, model, **kwargs)`.
- Backend selector 已经在 `vrl.nn.modules.ar_attention_backends`; `family` 由 runtime 传给 builder.
- HF/PEFT submodules 仍是权重容器; medium sprint 不写自有 decoder `nn.Module`.
- Janus/NextStep 非 decoder 组件仍保留在 family model 中:
  - Janus: VQ decoder, image-token embedding, `gen_head`, multimodal wrapper.
  - NextStep: image head/projectors, VAE, upstream pipeline-specific helpers.

## 2. What Changes

### T1. Add `HFDecoderView`

新增一个小 adapter 层, 把 family model 的 `_lm_trunk()` 暴露成 driver 需要的标准 view.

建议路径:

```text
vrl/nn/modules/ar_decoder_view.py
```

职责:

- unwrap Janus PEFT / base model path through existing `model._lm_trunk()`.
- unwrap NextStep PEFT / Qwen2-style trunk through existing `model._lm_trunk()`.
- expose `layers`, `rotary_emb`, `norm`, and per-layer attention/norm/mlp modules.
- validate required attributes at construction time with actionable errors.

Non-goal: 不做 generic reflection framework. 这里只服务 Llama/Qwen2-compatible decoder view.

### T2. Add `ARAttentionKernel`

建议路径:

```text
vrl/nn/layers/attention/kernel.py
```

Kernel protocol should own only attention math/cache, not the full decoder loop:

```python
class ARAttentionKernel:
    def init_state(self, *, config: ARAttentionConfig, num_layers: int, ...) -> Any: ...

    def attention(
        self,
        *,
        layer_idx: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer: Any,
        kv_state: Any,
        metadata: Mapping[str, Any],
    ) -> tuple[torch.Tensor, Any]: ...
```

`kv_state` is opaque:

- `VllmPagedAttentionKernel`: block ids, physical KV cache, slot mapping, vLLM metadata.
- `TorchSDPAAttentionKernel`: continuous per-sequence K/V tensors or HF-compatible cache equivalent.

The protocol must not assume `name -> kernel` is enough to build the full backend. Family adapters still
decide how to expose the trunk and non-attention model pieces.

### T3. Extract `ARDecoderDriver`

建议路径:

```text
vrl/nn/modules/ar_decoder_driver.py
```

Move the loop currently embedded in `VllmDecoderPagedAttentionBackend._forward_paged_trunk` into the driver:

```text
input_layernorm
q/k/v projection
RoPE
kernel.attention(...)
o_proj
residual
post_attention_layernorm
mlp
residual
final norm
```

The driver owns:

- position id handling and rotary embedding calls.
- attention head / KV head / head dim derivation.
- residual and MLP ordering.
- shape normalization between `[B,T,H]`, packed token layout, and last-hidden output.

The kernel owns:

- KV state allocation/update.
- paged or continuous cache layout.
- actual attention operator call.
- backend-specific metadata.

### T4. Refactor Existing Backends Onto Driver

`VllmDecoderPagedAttentionBackend` becomes:

```text
pack prefill/step inputs
manage vLLM sequence state identity
call ARDecoderDriver(view, VllmPagedAttentionKernel)
return ARAttention{Prefill,Step}Output
```

`TorchNativeDecoderAttentionBackend` has two valid paths:

1. Keep it as HF-forward fallback for CPU/no-vLLM while vLLM path uses the new driver.
2. Move it to the same driver with `TorchSDPAAttentionKernel`.

Do not force path 2 unless parity is proven. Keeping HF-forward `torch_native` as a safety oracle is useful
while the owned driver is new.

## 3. Backend Selector Boundary

Do **not** confuse backend selection with a kernel registry.

Current shape:

```python
vrl.nn.modules.ar_attention_backends.resolve_attention_backend(family, name, model, **kwargs)
```

Keep backend names global. It is a real runtime selector boundary because adding a model family
should not require new register lines:

- `vllm_paged` maps to `build_vllm_attention_backend`.
- `torch_native` maps to `build_torch_native_backend`.
- `resolve_attention_backend(family, name, model, **kwargs)` still passes `family` into the builder.
- Shared builders read the model's `_lm_trunk()` and keep backend-specific wiring out of family runners.
- This selector belongs in `vrl/nn/modules/ar_attention_backends.py`, not `vrl/nn/layers/attention`.

This is still **not** `name -> kernel`. A backend builder creates an `ARAttentionBackend`.
A future lower-level kernel registry is separate and only worth adding if it removes real duplication:

```text
backend selector: backend name -> shared backend builder
kernel registry: backend/kernel name -> attention kernel implementation
```

## 4. Numerical Contract

Replace “bitwise identical” with explicit parity levels:

- Same implementation moved between files: expect exact or very tight `assert_close`.
- HF forward vs owned driver using the same HF submodules: expect tight `assert_close`; do not require bitwise
  unless tests prove it.
- HF SDPA vs vLLM FlashAttention: use existing GPU tolerances:
  - prefill max error `<= 3e-3`
  - step max error `<= 5e-3`

Current tests already use tolerance for vLLM parity, so the sprint should not claim all outputs are bitwise identical.

## 5. Tests

Add/keep these gates before touching production call sites:

1. `HFDecoderView` construction tests for Janus and NextStep:
   - validates layer count, q/k/v/o projections, norms, MLP, rotary, final norm.
   - validates PEFT unwrap path when LoRA is attached if feasible with a lightweight fake.

2. Driver parity tests:
   - Janus tiny/stub trunk: owned driver vs HF trunk forward on CPU.
   - NextStep/Qwen2 tiny trunk: owned driver vs HF Qwen2 forward on CPU.
   - include left-padded prompt masks because current vLLM packer trims contiguous valid spans.

3. Kernel parity tests:
   - `VllmPagedAttentionKernel` vs current `_forward_vllm_attention` behavior before deleting old path.
   - `TorchSDPAAttentionKernel` vs HF forward only if we choose to replace `torch_native`.

4. End-to-end AR regression:

```bash
python -m pytest tests/models/ar tests/generation/ar tests/nn/layers tests/nn/modules -q
python -m pytest -m gpu tests/generation/ar -q
```

## 6. Non-Goals

- Do not write self-owned AR decoder `nn.Module` definitions.
- Do not write HF checkpoint weight-loaders.
- Do not move Janus/NextStep image heads, VQ/VAE, token projectors, or processors.
- Do not reintroduce per-family backend register lines.
- Do not start FlashInfer/Triton/custom kernel work in this sprint.
- Do not touch diffusion attention.

## 7. Stop Criteria

Stop at this sprint if:

- the shared driver matches HF/vLLM parity within the documented tolerances;
- backend selection still works through `attention_backend`;
- adding a new attention kernel only requires a kernel implementation plus backend wiring, not runner surgery.

Do **not** proceed to `SPRINT_attention_kernel_far.md` unless a concrete optimization is blocked by HF
submodule ownership.

## 8. References

```text
vrl/nn/modules/ar_decoder.py
  current VllmDecoderPagedAttentionBackend, _forward_paged_trunk, _forward_vllm_attention

vrl/nn/kernels/attention/vllm_paged.py
  existing vLLM paged-attention wrapper

vrl/nn/layers/attention/paged.py
  current ARAttentionBackend protocol and input/output data classes

vrl/nn/modules/ar_attention_backends.py
  shared AR backend selector plus vLLM and torch_native builders

vrl/nn/modules/torch_attention.py
  current HF-forward torch_native backend

vrl/models/ar/{janus_pro,nextstep_1}/runner.py
  current family runners that consume ARAttentionBackend
```

vLLM / SGLang references:

- vLLM model layer and attention separation: `vllm/model_executor/models/*`, `vllm/model_executor/layers/attention/*`
- SGLang attention backends: `python/sglang/srt/layers/attention/*`
