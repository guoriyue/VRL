# NVFP4 implementation boundaries

Reviewed complete nn/quantization/fp4.py and fp4_kernels.py, loader hardware guard,
performance-tool call sites, CPU reference decoder, and hardware/parity test
definitions. Previous audit commit: 8414e193c. Disposition: retain implementation.

## Retain and why

- quantize_nvfp4 is shared by activation and weight packing, CPU reference tests
  and the actual CUDA path. It returns the same three-part numeric contract on
  both devices. Moving it into Fp4Linear would couple standalone format encoding
  to a module instance without reducing state or code.
- to_blocked_scale_layout is a public format operation and independent test entry
  point. _ceil_div names the repeated integer padding operation without importing
  Triton into CPU reference code. Neither needs a new utility class or file.
- fp4_kernels.py is a genuine lazy dependency boundary. CUDA dispatch imports
  Triton only when needed, then launches a JIT kernel. Kernel entry parameters and
  constexpr dimensions are framework requirements, not arbitrary helper sprawl.
- _E2M1_BOUNDS caches a tensor representation of the protocol midpoints, rather
  than a second independently maintained table. Imported ALL_CAPS layout and
  encoding constants represent hardware/numeric contracts. Local launch settings
  blocks_per_program and num_warps belong with the launch, not global config.
- _alignment_error is shared by direct construction and traversal eligibility.
  One rejects invalid explicit construction; the other skips ineligible Linears.
  Keeping both callers on one predicate prevents contradictory shape policies.
- Quantization's K % 16 check protects complete encoding blocks. The Linear's
  K % 32 and N % 16 checks protect packed scaled-mm eligibility. These constraints
  are not redundant checks of the same boundary. The exported CUDA wrapper also
  checks matrix/device/contiguity before launching pointer arithmetic that assumes
  those properties; this is not replaceable with Tensor annotations.
- Fp4Linear owns persistent master versus non-persistent packed cache through the
  common QuantizedLinear lifecycle. Forward requests bf16 GEMM output, restores
  the two tensor scales on device and restores the activation dtype before bias.
  There is no detached module-lifecycle helper to consolidate.
- nvfp4_available is a shared device capability query used before loader mutation
  and by performance tools. It has no stateful receiver. Its separate existence
  lets callers reject unsupported targets before loading a large model.

Non-goals: modify packing, rounding, scale precision, launch geometry, production
MLP scope or public helper signatures just to reduce free-function count. No
Python change or new test was needed for this disposition.

## Evidence and limitations

The immediately preceding 8414e193c run executed the unchanged NVFP4 CPU tests as
part of the quantization suite: 45 passed, 18 skipped overall. CPU tests check
exact grid values, tie rounding, reference dequantization error, blocked layout,
constructor alignment, target selection and loader policy. No redundant rerun
was performed for this documentation-only review.

GPU tests compare packed values and scales byte-for-byte for random matrices,
padded/partial launch chunks, sparse blocks and midpoints. Separate GPU tests cover
actual scaled-mm, cache moves and compilation. Those tests were skipped with CUDA
hidden; their existence is not evidence that this environment passed them.

Deferred hardware verification: nvfp4_available checks CUDA, dtype presence and
compute-capability major >= 10, not installed Triton/cuBLAS kernel execution. The
GPU hardware test deliberately compares that declaration with an empirical kernel
probe without using the probe itself as its skip condition. A true gate result
therefore remains a hardware eligibility claim, not a complete software-stack
health guarantee. Do not add import/launch fallback to a different quantizer.

Supported-input limits: the CUDA wrapper assumes its tensor_scale is the scalar
produced by quantize_nvfp4; it does not independently validate arbitrary external
scale shapes/devices. Normal production calls satisfy that ownership contract.
The common cache movement and incremental-swap failure limits remain documented
in quantization_ownership.md rather than being duplicated as new recovery logic.
