# Denoise evaluator replay ownership

Reviewed complete denoise/sde_logprob.py, chunk_autoregressive_logprob.py and
denoise/__init__.py, trajectory-reader slicing, signal assembly and existing
reference-cache, slice-access and chunk evaluator tests. Previous commit: 8f7636862.

## Change

SDE reference evaluation now selects the cached prediction or computes the
reference prediction before one shared sde_step_with_logprob call. Remove the
duplicated math invocation and result-field assignments without introducing a
helper class or callback. Cache priority, no_grad scope, identical-model adapter
disable, scheduler arguments and optional output fields remain unchanged.

The shared call is gated on a reference source being selected, not merely a
non-None prediction. This preserves failure behavior if a reference model returns
an invalid None payload: require_value checks key presence, not value validity.
It must not silently turn a malformed reference result into absent KL signals.

## Retain and why

- Step SDE and complete causal-chunk replay have distinct granularity. Flattening
  chunk policy axes only for algorithm signals preserves model-owned temporal
  replay order. Combining these evaluators would obscure that execution contract.
- _flatten_actions is shared by current, old, mask and reference chunk outputs.
  Its sample-alignment check and reshape are one consistent adapter, not four
  independent tensor conversions. Keep the original semantic axes in trajectories.
- SDE stores scheduler/noise/math options on its evaluator. Device movement uses
  the shared utility after selecting the needed replay step. Cached prediction
  and proposal-mean extras retain their existing step indexing contract.
- The denoise reference convention requires an explicit reference model when no
  cache exists; identical objects disable adapters. Token reference handling also
  supports an implicit adapter-off model when ref_model is None. Do not merge
  those policies by reusing the token context manager unchanged.
- The denoise package initializer is a public facade for both evaluator classes;
  there is no domain vocabulary table requiring relocation.

Non-goals: change training math, infer missing reference values, modify cache
formats, add generic validation wrappers or unify distinct replay granularities.

## Validation and open limits

21 CPU tests passed across reference-noise cache, denoise slice access, chunk
logprob and evaluator contract tests. Existing cache tests compare cached versus
fresh reference signals and count forward calls; slice tests prevent full tensor
movement before step selection. Ruff check and format check pass for the changed
source. No GPU/pretrained parity is claimed.

The chunk reference gradient-isolation gap found here is addressed by the
follow-up below. The original constant-output fake did not exercise autograd.

Both denoise evaluators may return absent reference signals when need_ref is true
but no source exists. This is existing behavior; deciding whether the request
must fail belongs with algorithm/reference setup validation, not an incidental
cleanup of the duplicated math call.

## Follow-up: scope chunk reference autograd

Following 03b328e69, wrap the chunk reference forward in torch.no_grad while
preserving the identical-model adapter-disable context. Current-policy replay
remains differentiable. No new context helper or shared reference-selection class
is introduced: denoise's explicit reference convention stays intact.

CausVid replay_forward calls differentiable replay_log_probs; its base
disable_adapter method controls adapters, not the global gradient mode. The
online factory currently supplies the same model for LoRA KL, where frozen base
parameters may already make the reference graph-free. This audit does not claim
every current production run retained a graph. The evaluator also accepts distinct
reference models and must own its frozen reference execution consistently with
the SDE/token evaluators rather than depending on incidental parameter settings.

Two regressions exercise real tensor autograd through a differentiable replay
double with shared and distinct reference objects. Both failed on the prior code.
They verify reference forward runs with grad disabled, current output remains
differentiable, reference output has no graph, and a log-prob difference backward
produces exactly the current branch's gradient without reference accumulation.
The existing test continues to verify reference values, adapter disable count and
one ordered request per branch.

16 chunk evaluator, evaluator contract and reference cache tests passed on CPU.
Ruff check and formatting pass on the changed files. No CUDA/pretrained execution
was required to establish the autograd boundary. Compatibility: callers relying on
gradients through reference signals now receive detached reference results, as the
frozen-reference training contract requires; forward values and policy gradients
are otherwise preserved.
