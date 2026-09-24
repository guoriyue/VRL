# Post-advantage rollout admission

The one-shot online trainer records sample selection in
`<output_dir>/admission/<attempt-id>.rank-<rank>.jsonl`. This is independent of raw
reward scoring archives and the offline reward evaluation service.

Algorithms still compute advantages, including global normalization and
component-aware objectives. Admission then applies the previous exact-zero
`nonzero_advantage_mask`. It may retain a whole group, drop it, or retain only
some rows. Disabling `actor.drop_zero_advantage` retains the original batches
and advantages. The `adv_zero_rate` diagnostic uses a small-value threshold;
it must not be confused with this exact-zero selection rule.

Each record retains request/sample identities, original prompt, an input-derived
prompt key, normalized input metadata, rollout policy version, raw rewards and all reward axes, prepared
advantages, the exact selected-row mask and a reason. The prompt key includes
reward metadata such as reference paths, but is not a file-content attestation;
use the run's dataset provenance and reward archives for asset identity. `input_metadata`
retains the same JSON-normalized values used to build the key, including task and
source-group labels; it is a detached snapshot, so later metadata mutation cannot
rewrite an earlier decision. `prompt_id` accepts explicit `prompt_id`, `id`, then
`task_id`. Older records may omit `input_metadata`; do not infer missing labels
from the digest alone. Synthetic
test batches without trajectories explicitly report missing identity.

Selection records precede backward. `optimizer_applied` is therefore unknown;
they are not proof that the optimizer eventually stepped. Join trainer/global
steps with update metrics and checkpoint evidence. Rollout policy versions are
owned by weight synchronization and need not numerically equal trainer steps.
Zero advantage does not establish task difficulty, reward correctness or model
failure; attribution remains undetermined.

Each process attempt owns a new file, including after resume. Previous audit
records are retained instead of being truncated to the checkpoint. Rank-local
writers do not race one shared file; distributed audit failures are agreed across
ranks before entering backward. A crashed process may leave an incomplete final
line, which remains evidence of the interrupted attempt. Context-parallel ranks
can record the same sample IDs: do not count these as independent rollouts.

The change moves sample-selection ownership into `vrl.rollouts.admission` and
adds audit persistence. Existing algorithm normalization, loss denominators,
cross-rank group balancing, streaming global-std spooling and reward scoring
stay unchanged. The shared exact-zero helper remains in the algorithm module
because streaming preflight uses the same rule to calculate surviving groups.
No automatic retries, difficulty thresholds or adaptive sampling are enabled.
