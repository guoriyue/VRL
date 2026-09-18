# Generation and replay names

Names distinguish the data being handled from the order in which it is generated.

| Term | Meaning | Examples |
| --- | --- | --- |
| `Denoise` | Shared denoising execution and replay operations, including flow matching | `DenoiseModelBase`, `DenoiseSamplingParams`, `DenoiseBackboneCaller` |
| `Diffusers` | An adapter to the Diffusers library | `DiffusersPipelineModelBase` |

The generation regime describes execution order: `full_sequence` or
`chunk_autoregressive`. The policy step kind describes the operation, and every
remaining family's step is `denoise`. A chunk-autoregressive model denoises each
chunk, so regime and step kind are separate dimensions.

The token-autoregressive regime and its `Token` / `DiscreteToken` /
`DecoderAttention` vocabulary were removed with the token-AR families
(`3e340ed23`); the names below that came from that side no longer exist.

## Python API rename

The internal Python API uses these names in place of the former abbreviations:

| Previous | Current |
| --- | --- |
| `DiffusionBatchExecutorBase`, `DiffusionBatchGatherer` | `DenoiseBatchExecutorBase`, `DenoiseBatchGatherer` |
| `DiffusionSamplingParams`, `DiffusionRequestLayout` | `DenoiseSamplingParams`, `DenoiseRequestLayout` |
| `DiffusionSDELogProbEvaluator` | `DenoiseSDELogProbEvaluator` |

`Diffusion` is kept where it names an algorithm or a recipe kind rather than
the step: `DiffusionNFT`, `algorithm.kind: diffusion_dpo`.

Related family state, runner, and backbone classes follow the same vocabulary.
Callers import the new names directly; there are no old-name forwarding aliases.
Driver and workers must run the same code revision when exchanging Python objects.

This rename does not change generation regime values, checkpoint state keys, or
algorithm names. One YAML key did follow: `precision.diffusion_math` is now
`precision.denoise_math` (`DenoiseMathPrecisionConfig`), and a config that still
spells the old key is rejected as unknown. Historical audit and sprint documents
retain the names used at the time.
