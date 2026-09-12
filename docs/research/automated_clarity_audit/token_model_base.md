# Token model and replay bases

Reviewed complete models/steps/token/base.py, state/capability tests, checkpoint
adapter and vocab payload tests, scoped weight/PEFT helpers and family consumers
covered in the preceding token reviews. Previous audit commit: 2b82b1d67.

## Change

Remove the unused _paged_state_kwargs helper from test_runtime_state.py. Repository
search found only its definition; it was not a fixture and no test consumed it.
Production behavior remains unchanged.

## Retain and why

- ARModelBase binds common behavior to model-owned state. load/verify_trainable_state
  share the explicit model prefix with sender/checkpoint roots. They are protocol
  adapters, not unnecessary forwarding functions. They use the common strict
  weight helpers rather than duplicate key validation in every family.
- _lm_trunk and disable_adapter bind PEFT behavior to the family's language_model.
  Janus overrides the trunk hop for its CausalLM wrapper. No family-name dispatch
  or speculative model-tree traversal belongs in the shared base.
- policy_cores and adapter_roots describe different owners: optimization walks the
  language trunk, while checkpoint exports map the model wrapper's root name to
  its inner adapter. Collapsing them would confuse namespaces and include decoders.
- LM_EXCLUDE is imported from the isolated quantization targeting taxonomy;
  it keeps vocabulary heads/embeddings out of selective quantization. __all__ is
  an API list. No new hardcoded per-family workflow facts are needed.
- _head_replay_values delays eager logits until an unsplittable head requires them.
  _resolve_image_token_replay shares request validation and trajectory reading;
  families retain their forward/embedding math. ReplayRolloutStubs explicitly
  rejects unavailable decoding instead of each minimal model reimplementing it.
- ARReplayCore owns construction/loading of checkpoint-named minimal modules;
  token.loader owns file formats and shard selection. Optional imports and the
  family-declared checkpoint_subfolder preserve the upstream loader boundary.
- ARDiscreteTokenState derives sequence length from token storage, without another
  mutable count. The runner checks upper row/position bounds before family sampling;
  basic index validation remains the TokenStepBatch constructor's responsibility.

Non-goals: making all family surfaces abstract, merging distinct checkpoint and
adapter root mappings, weakening external checkpoint checks, or adding validation
to every tensor read.

## Validation and limits

31 state, training-capability, checkpoint-loader and vocab-head tests passed on CPU.
Capability tests instantiate small replay models to compare actual requires_grad
state with descriptor support. Checkpoint tests cover real tensor copies and a
real safetensors shard; they do not load full pretrained models. Ruff check and
format check passed for the changed test file.

The base does not independently prove all output buffers were filled before
finalize_token, nor make family state immutable. Normal sequence composition owns
complete scheduling; manually calling finalize early is outside that guarantee.
ARReplayCore loads config and weights separately, so callers still need immutable
source identity for exact replay. No additional filename/version inference was added.
