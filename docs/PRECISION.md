# Precision configuration

VRL separates each role's ordinary dtype from selective low-precision kernels.
The complete public shape is:

```yaml
precision:
  float32_precision: tf32

  training:
    dtype: bf16

  rollout:
    dtype: bf16
    quantization:
      format: fp8
      recipe: rowwise
    prompt_encoders:
      dtype: fp16

  diffusion_math:
    dtype: fp32
```

`precision.float32_precision` and `precision.training.dtype` are required.
`float32_precision` accepts `ieee` or `tf32`. `rollout.dtype` inherits the
training dtype,
`rollout.prompt_encoders.dtype` inherits the resolved rollout dtype, and
`diffusion_math.dtype` defaults to FP32. Set prompt encoders to FP16 explicitly
when that memory/accuracy trade-off is desired; quantization never changes their
dtype implicitly.

## Parameter storage and outer forward

`training.dtype` is the base dtype for the trainer/replay model.
`rollout.dtype` is the base dtype for the generation policy. A base dtype governs
parameter loading. The runtime separately resolves whether the transformer
forward enters outer autocast: diffusion FP16/BF16 roles normally use matching
autocast, while diffusion FP32 roles run without it. AR families execute
natively at their loaded parameter dtype and therefore resolve outer autocast
to `off`.

A model-family requirement may further constrain the resolved runtime contract.
SANA, for example, requires native FP16 transformer parameters with outer
autocast off. This requirement lives in the family registry and is not a public
knob: a user cannot select an unsupported SANA forward mode through YAML.

An aligned BF16 run only needs:

```yaml
precision:
  float32_precision: tf32
  training:
    dtype: bf16
```

The allowed base dtypes are `fp32`, `bf16`, and `fp16`. FP8 and NVFP4 are not
ordinary model or autocast dtypes and are rejected in a `dtype` field.

## FP32 matmul backend

```yaml
precision:
  float32_precision: tf32
  training:
    dtype: bf16
```

`float32_precision` controls FP32 matrix multiplication kernels in every trainer
and rollout process. `tf32` permits TensorFloat-32 acceleration; `ieee` requires
strict IEEE FP32 behavior. This axis is independent of parameter storage and
outer autocast: FP16/BF16 models may still contain FP32 attention or protected
sub-operations whose matmuls consume this policy.

The setting is explicit because process defaults can differ. A family requirement
may constrain it: SANA's canonical preset selects `ieee`, and the registry rejects
a conflicting public value because its linear-attention processor promotes Q/K/V
to FP32 and is numerically sensitive to TF32. That family requirement is not
another user-facing precision setting.

## Selective rollout quantization

Quantization is layered on `rollout.dtype`; it does not replace that dtype.
Normalization, residuals, excluded or unsupported Linears, and quantized GEMM
inputs/outputs still need the rollout base dtype.

### FP8

```yaml
precision:
  float32_precision: tf32
  training:
    dtype: bf16
  rollout:
    dtype: bf16
    quantization:
      format: fp8
      recipe: rowwise
```

FP8 replaces eligible attention projections and MLP Linears. Its recipes are:

- `rowwise` (default)
- `tensorwise`
- `blockwise`

`blockwise` is incompatible with `model.torch_compile`; the runtime rejects that
combination before executing a model.

### NVFP4

```yaml
precision:
  float32_precision: tf32
  training:
    dtype: bf16
  rollout:
    dtype: bf16
    quantization:
      format: nvfp4
```

NVFP4 is an experimental Blackwell-only rollout path. It conservatively replaces
eligible MLP Linears; attention projections remain at `rollout.dtype` pending the
real rollout-to-BF16-replay drift and reward gate. NVFP4 is the complete scaling
scheme and does not accept a `recipe` key. The old generic `format: fp4` spelling
is rejected; use `format: nvfp4`.

A quantized rollout differs from an unquantized replay even when both base dtypes
are BF16. VRL therefore treats the full role policy as different and enables its
rollout/replay precision correction and drift guard.

## Training quantization

The training role has the same structural shape for future expansion:

```yaml
precision:
  float32_precision: tf32
  training:
    dtype: bf16
    quantization:
      format: fp8
      recipe: rowwise
```

This currently fails during policy resolution. The rollout quantized Linears do
not provide an autograd-capable forward/backward path, so accepting the block
would create a silent no-op knob. The shape can become active when a real
training quantization runtime consumes it.

## Prompt encoders

```yaml
precision:
  float32_precision: tf32
  training:
    dtype: bf16
  rollout:
    prompt_encoders:
      dtype: fp16
```

This controls generation-only text/prompt encoders for model families whose
rollout builder exposes them. It defaults to `rollout.dtype` and is independent
of policy quantization.

It does not control the VAE. VAE precision is a model-family fidelity invariant;
current diffusion families keep VAE decode in FP32. The previous
`frozen_components` name incorrectly implied that one public dtype controlled
every frozen pipeline module and has been removed.

## Protected diffusion math

```yaml
precision:
  float32_precision: tf32
  training:
    dtype: bf16
  diffusion_math:
    dtype: fp32
```

This controls custom diffusion SDE, scheduler, and log-probability math outside
the transformer. It defaults to FP32 and is independent of training/rollout base
dtypes, outer autocast, quantization, and `float32_precision`. Selecting FP32
protected math chooses the tensor dtype; `float32_precision` still chooses which
FP32 matmul backend executes any matrix multiplications inside that boundary.
Non-diffusion objectives reject a non-FP32 override.

## Separate precision boundaries

`rollout.trajectory_storage.dtype` controls trajectory transfer and host-memory
storage after generation. It does not select model compute kernels.

`distributed.training.fsdp.precision_policy` controls parameter sharding and
gradient-reduction behavior. It consumes the resolved model parameter dtype and
does not replace the public precision policy.

## Removed forms

Scalar precision and the overloaded legacy fields fail with a migration error:

```yaml
precision: bf16  # removed

actor:
  optim:
    allow_tf32: true  # use precision.float32_precision: tf32

precision:
  train: bf16                 # use training.dtype
  rollout: fp8                # use rollout.dtype + rollout.quantization
  math: fp32                  # use diffusion_math.dtype
  frozen: fp16                # use rollout.prompt_encoders.dtype
  frozen_components: fp16     # removed misleading name
  rollout_recipe: rowwise     # move under rollout.quantization for FP8
```

The nested legacy spelling is also removed:

```yaml
precision:
  float32_precision: tf32
  training:
    dtype: bf16
  rollout:
    frozen_components:
      dtype: fp16
```

Use `precision.rollout.prompt_encoders.dtype` instead.
