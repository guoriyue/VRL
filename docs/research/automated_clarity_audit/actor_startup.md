# Actor startup and shared failure identity

Reviewed complete `ray/actor_group.py` and `runtime_errors.py`, actor-group
startup/metadata timeout tests, the real CPU launch test, and caller excerpts in
the generation launcher, weight-delivery probe, continuous producer and verdict
writer. Those larger caller files remain pending.

## Change and open ownership issue

Correct actor_group's module description: normal shutdown retains failed actor
handles, but startup failure only attempts kill and annotates the original
exception when kill fails. The old description promised retention unconditionally.

Initial finding: if launch creates actors, startup/metadata then fails, and kill
also fails, no structured cleanup owner is returned. In generation launcher,
`actor_group` is still None when RayActorGroup.launch raises. The launcher cannot
retry those failed handles; it only adds another note when a group had already
been returned. The acceptance probe likewise assigns its group only on return.
Traceback locals may incidentally retain references, but that is not a cleanup
API or a reliable ownership transfer.

Follow-up in `launcher_ownership.md` narrows this finding: online launch already
has the outer placement owner, and a real CPU actor test verifies group removal
reclaims an actor even when direct kill failed. The owned-cluster acceptance
probe also shuts its local Ray session down in finally. Standalone actor-group
launch with neither outer owner still lacks a structured handle-return path on
double failure; defer that public-API change explicitly. Do not add an unused
exception attribute, assume shared-cluster shutdown is authorized, or add
unbounded kill retries. The original production-wide ownership claim was too broad.

## Retain and why

- RayActorGroup.launch owns created actor handles immediately; its exception
  sweep covers failures during creation, startup RPC submission, metadata reads
  and handle construction. Keep the sweep around the whole launch sequence.
  Separate startup and metadata waits receive separate bounded budgets.
- RayActorHandle holds transport metadata; RayActorGroup holds mutable ownership.
  Their existing class split has a real lifetime purpose. Shared shutdown uses
  kill_and_retain rather than another actor-specific cleanup implementation.
- Worker/config/bundle length checks protect positional correspondence before
  actor creation. Framework metadata conversion remains at this boundary. No
  new float/token tests or checker classes are justified by this review.
- `find_error_cause` locates a requested error type for live control flow;
  `root_failure_cause` chooses the stable classification used by restart/verdict
  logic. The first terminal domain error must take precedence over a dependency
  TimeoutError beneath it. Keep both APIs rather than exposing one ambiguous
  'root' operation.
- `_error_chain` is shared traversal, preferring explicit root_cause wrappers
  and detecting cycles by identity. This is not guessing a checkpoint field from
  several unrelated representations; it follows the explicit exception protocol.
  The tiny dependency-neutral module prevents imports between runtime, rollout
  control and verdict ownership. No wrapper class would improve that boundary.
- Export lists are API declarations. Method names, metadata keys and Ray resource
  option keys belong to framework/protocol boundaries; there are no mixed-in
  ALL_CAPS business tables in these modules.

Non-goals: alter restart categories, flatten causal chains into strings, move
actor dispatch into actor-group construction, or claim the startup cleanup
ownership gap has been solved by documentation.

## Validation

8 deadline/actor-launch tests passed, including real two-actor CPU startup,
metadata/config propagation and shutdown; one upstream Ray environment warning.
3 shared exception-chain tests passed (explicit root, terminal precedence,
cycles). Ruff check and format check passed for the docstring-only Python change.
No production execution behavior changed in this batch.
Previous isolated audit commit: `33f53c6de`.
