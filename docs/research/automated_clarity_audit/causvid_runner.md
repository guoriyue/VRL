# CausVid configuration, runtime and causal runner

Following 55cff84b4, reviewed CausVid config.py, runtime.py, runner.py and package
__init__.py. Followed registry construction, model generator creation and result
projection, checkpoint identity tests and runner/replay tests. model.py itself
remains pending its full review.

Change: make the runner's generators tuple required and remove _transition_noise's
None branch. Previously omitting generators returned noise=None, which the shared
renoise sampler rejects because neither noise nor recorded next_sample was given.
All repository calls already supply the tuple. Model._sample_generators creates a
generator per sample even without a configured seed, using generator.seed().
The runner therefore exposes its actual supported contract instead of a broken
optional path. Shape/count checks remain; no new gate or fallback was added.

Compatibility: external callers omitting generators now fail at the run signature,
before cache allocation/model work, rather than failing during the first re-noise
transition. Explicit None is unsupported. Valid seeded and unseeded model calls
keep their exact draws and numerical behavior.

Retain and why:

- Source revision and checkpoint member constants are pinned source/artifact
  boundaries shared by schema defaults and identity consumers. The existing
  identity test proves omitted and explicit defaults resolve equally.
- OFFICIAL_CAUSVID_GEOMETRY and OFFICIAL_CAUSVID_SCHEDULE are architecture/release
  constants, not workflow business vocabulary. The pinned upstream YAML contains
  the 1000/757/522 sequence and three-frame blocks. Other upstream recipes are not
  implicitly supported by changing this family's selected schedule.
- CausVidGeometry derives chunk count/latent shape; CausVidSchedule describes the
  fixed three-prediction/two-transition release. Keep these existing owners and
  shape/partition constraints, without a new general geometry validator.
- CausVidCausalBackend separates cached rollout prediction from differentiable
  full-prefix replay. The protocol permits CPU ordering tests without importing
  the upstream causal transformer and its optional kernels.
- The runner owns one request-local cache and the explicit temporal loop. The
  timestep-zero cache commit consumes the terminal clean chunk but is not a third
  stochastic policy action. Keep it visible instead of merging it into a generic
  denoise helper that would obscure the sequence.
- _transition_noise centralizes per-row RNG in checkpoint dtype; re-noise math
  then promotes for scoring. Do not replace it with a global random draw.
- CausVidRunResult.trajectory_mapping projects family facts into the shared wire
  schema and declares replay axes. Prompt embeddings carry only a sample axis;
  next_sigmas additionally carries chunk/transition axes. Keep the mapping method
  and TypedDict instead of inferring axes from tensor dimensions.
- The runtime builder is the registry's lazy, transformer-only replay adapter;
  the thin executor subclass supplies family identity with the shared shape.
  Empty root exports preserve config discovery's import boundary.

Non-goals: support arbitrary upstream schedules, change sampler math, add seed
fallbacks, generalize cache management or introduce hypothetical validation tests.

Validation: 55 existing CausVid runner/replay/loading and checkpoint identity tests
passed on CPU. Tests check call ordering, no rollout autograd, row-local repeatable
noise, saved-action replay and explicit/default identity agreement. Ruff check and
format check passed. No full GPU model generation or latency claim. Coverage:
221 reviewed, 278 pending baseline modules.
