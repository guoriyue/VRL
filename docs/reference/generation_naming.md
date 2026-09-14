# Generation and replay names

Names distinguish the data being handled from the order in which it is generated.

| Term | Meaning | Examples |
| --- | --- | --- |
| `Token` | Token inputs, outputs, state, or one token step; tokens may be discrete or continuous | `TokenRequestLayout`, `TokenBatchInputs`, `NextStep1TokenRunner` |
| `DiscreteToken` | Vocabulary token IDs and their categorical log-probabilities | `DiscreteTokenState`, `DiscreteTokenBatchResult` |
| `Autoregressive` | Model/replay or sampling behavior conditioned on earlier outputs | `AutoregressiveModelBase`, `AutoregressiveReplayCore`, `AutoregressiveSamplingSection` |
| `DecoderAttention` | Attention prefill and incremental decode, independent of the selected backend | `DecoderAttentionBackend`, `DecoderAttentionStepInput` |
| `DecoderCacheRows` | Per-row decoder cache storage used when batching requests | `DecoderCacheRows` |
| `Denoise` | Shared denoising execution and replay operations, including flow matching | `DenoiseModelBase`, `DenoiseSamplingParams`, `DenoiseBackboneCaller` |
| `Diffusers` | An adapter to the Diffusers library | `DiffusersPipelineModelBase` |

The generation regime describes execution order: `token_autoregressive`,
`full_sequence`, or `chunk_autoregressive`. The policy step kind describes the
operation: `token` or `denoise`. A chunk-autoregressive model can denoise each
chunk, so these are separate dimensions.

## Python API rename

The internal Python API uses these names in place of the former abbreviations:

| Previous | Current |
| --- | --- |
| `ARModelBase`, `ARReplayCore` | `AutoregressiveModelBase`, `AutoregressiveReplayCore` |
| `ARRequestLayout`, `ARBatchInputs` | `TokenRequestLayout`, `TokenBatchInputs` |
| `ARDiscreteTokenRunner` | `DiscreteTokenRunner` |
| `ARAttentionBackend`, `ARCacheRows` | `DecoderAttentionBackend`, `DecoderCacheRows` |
| `_sample_ar_step`, `_build_ar_runner` | `_sample_token_step`, `_build_token_runner` |
| `DiffusionBatchExecutorBase`, `DiffusionBatchGatherer` | `DenoiseBatchExecutorBase`, `DenoiseBatchGatherer` |
| `DiffusionSamplingParams`, `DiffusionRequestLayout` | `DenoiseSamplingParams`, `DenoiseRequestLayout` |
| `DiffusionSDELogProbEvaluator` | `DenoiseSDELogProbEvaluator` |

Related family state, runner, and backbone classes follow the same vocabulary.
Callers import the new names directly; there are no old-name forwarding aliases.
Driver and workers must run the same code revision when exchanging Python objects.

This rename does not change YAML keys, task identifiers such as `ar_t2i`,
generation regime values, checkpoint state keys, or algorithm names such as
`DiffusionNFT`. The public `precision.diffusion_math` section and its
`DiffusionMathPrecisionConfig` retain their matching names. Historical audit and
sprint documents retain the names used at the time.
