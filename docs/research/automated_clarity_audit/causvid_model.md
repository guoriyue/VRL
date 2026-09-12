# CausVid model completion

Following 11d398379, read the remaining cache allocation, cached/full-prefix
prediction, VAE decode, sigma conversion, policy/replay model and generation
methods. Together with causvid_loading.md this completes model.py's source review.

Changes:

- Fix the obsolete CausVidChunkExecutor name in the unsupported single-step
  sampling error to the registered CausVidBatchExecutor.
- Preserve transitive ModuleNotFoundError details in FlashAttention preflight.
  A genuinely absent candidate is skipped; a candidate missing an internal module
  joins the unusable-install diagnostics. Either working candidate still succeeds.
  This closes the focused diagnostic follow-up in causvid_loading.md without
  changing backend selection or introducing a new gate.

Retain and why:

- Cached rollout and full-prefix differentiable replay are separate backend
  methods. Replay re-evaluates earlier clean chunks with kv_cache=None rather than
  mutating the inference K/V cache during autograd. The quadratic prefix work is
  an explicit current cost; no performance improvement is claimed here.
- Replay detaches recorded prefix chunks, conditioning and scored actions where
  required; gradients pass through current predictions. Autocast scopes model
  execution while density math remains in the shared re-noise implementation.
- _validate_replay_tensors checks the family's runtime recipe and full latent
  geometry, while transport gather checks wire axes/presence. Keep this public
  replay method's boundary rather than assuming all callers used the gatherer.
- _sigma_for_timesteps mirrors the pinned shifted schedule lookup, including
  float32 construction and float64 conversion math. _flow_to_x0 is shared by
  cached and full-prefix prediction. Do not replace either with guessed sigma
  division or remove dtype conversions based on their count.
- VAE mean/std values are the pretrained 16-channel normalization contract,
  not workflow vocabulary. They remain architecture-owned; no global magic-number
  table or generic media wrapper is introduced.
- The backend retains the causal core while updating its PEFT/compile forward
  shell. Gradient-checkpointing setup and mask caching need that core identity.
  Upstream causal_model constructs per-forward kwargs containing block_mask;
  the inspected path does not merely re-read a global mask inside each block call.
- Full generation owns prompt/VAE work; replay intentionally rejects those APIs.
  Unsupported single-step hooks preserve DiffusionModelBase's interface while
  directing callers to the grouped executor. Thin methods are interface adapters.
- Build-time restrictions apply before weight resolution; request-time
  restrictions apply to per-request sampling overrides. Keep both boundaries,
  although their similar literals should not be mistaken for interchangeable
  call sites. Geometry derives from the existing release owner.
- Source/checkpoint constants and named result/trajectory fields are external
  integrity/schema boundaries. No new ALL_CAPS registry or validator owner added.

Limits: cache geometry/device ownership follows one backend lifetime; replay's
mask cache key assumes the fixed family chunk geometry. The review does not
establish concurrent backend calls, arbitrary device migration or a general
differentiable cache. Full CUDA model parity remains outside CPU evidence.

Validation: 44 existing runner, replay/loading and chunk-binding tests passed.
A one-off import failure probe confirmed a missing flash_attn_2_cuda detail appears
in the final error while an absent top-level candidate is skipped. No repository
tests were added. Ruff check and format check passed. Coverage: 222 reviewed,
277 pending baseline modules. All five CausVid family modules are now reviewed.
