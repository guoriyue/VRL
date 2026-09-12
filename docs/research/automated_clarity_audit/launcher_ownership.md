# Launcher and outer cleanup ownership

Reviewed complete `generation/ray/launcher.py`, eager/deferred session creation,
actor-group failure handling and the online recipe/placement-owner cleanup
chain. Only launcher receives complete-module coverage here; other large files
were read at the relevant boundaries.

## Correct the earlier scope

The actor-group API cannot hand failed startup handles back through its normal
return value, but online generation has an outer resource owner. The launcher
uses a supplied RolePlacement; _OnlineRecipeLifecycle always attempts
placement_owner.shutdown after role cleanup, including rollout launch failure.
GlobalRayPlacementOwner retains its group handle until removal succeeds.
The acceptance weight-delivery probe instead owns its local Ray connection and
calls ray.shutdown in finally even when group assignment never completed.

Do not add another owner or change launch exceptions for those paths merely
because the inner constructor cannot return a group on failure. Keep the
remaining limitation precise: standalone RayActorGroup.launch on a shared
cluster, without a placement/cluster owner, has no structured recovery handle
if both startup and direct kill fail. A public API for that use would need a
caller-owned acquisition object or explicit failure resource transfer and real
consumers. That API change is deferred; no production caller reviewed here needs
it to complete outer teardown.

## Evidence added

A real CPU Ray actor deliberately raises during startup. Only direct kill is
injected to fail; the actor remains callable afterward. The test removes the
real placement group, then observes ActorDiedError from the actor handle. It
retains and cleans its own handles in finally and never shuts down the shared
test cluster. This closes a gap in the earlier probe-kill test, which actually
killed its actor before reporting a synthetic failure.

The new test plus existing online rollout-launch-failure and cleanup-retry tests
pass: 3 passed, 25 deselected, one upstream Ray environment warning. These prove
the component paths and actor fate on a healthy local CPU cluster; they are not
a complete real training startup or a multi-node control-plane outage experiment.

## Retain and why

- `_launch_session` owns actor assembly, placement validation, dispatcher and
  weight-sync wiring, then returns the session. `_launch_session_async` keeps
  Ray initialization on the caller thread and moves blocking actor startup to
  a worker thread. The outer shielded activation task is the lifetime owner;
  don't delete this adapter because its body is short.
- `create_runtime` selects eager construction or a bound deferred factory and
  enables parking for on-demand GPU launches. Keep this composition separate
  from the blocking fleet constructor; no second launcher class is needed.
- `_validate_rank_gpu_ids` binds the generic placement check to this role and
  uses node identity for cross-node launch. `_all_ranks_support_versioned_slots`
  probes all ranks under a deadline; a failed RPC must propagate, not silently
  become unsupported. These are actual capability/placement boundaries.
- `_find_rendezvous_port` is a local socket adapter. Its docstring correctly says
  closing the socket does not reserve the port. Do not promise race-free remote
  rendezvous or solve allocation by speculative string fallback.
- Engine IDs/rank IDs are launch protocol identities. HEALTH_CONCURRENCY_GROUP
  is shared Ray configuration; there are no local ALL_CAPS business tables to
  move. Consistent framework signatures are more useful than fewer methods.

Non-goals: add unconsumed cleanup callbacks, global Ray shutdown, new launch
configuration knobs, or broad topology changes. No production behavior changed
in this batch. Ruff check and format check passed for the added test.
Previous isolated audit commit: `e5884894c`.
