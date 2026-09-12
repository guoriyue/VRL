# GPU bundle reassignment and CPU capacity

Reviewed all of placement.py, resource-plan fields consumed by BundleLayout,
launcher actor CPU requests, shared metadata consumers, and placement lifecycle
and mapping tests. resources.py is not yet fully reviewed by this entry.
Previous isolated audit commit: 50eb6ca52.

## Change

Single-node placement learns physical GPU assignments by probing, then remaps
roles to bundles. Previously CPU capacity was sized using the original role
indices. With planned trainer/rollout/reward GPUs 0/1/2 and probed bundle GPUs
2/0/1, rollout moves to bundle 2, which reserved only 0.001 CPU although its
actor requests 1 CPU. The existing permutation test checked GPU selection but
did not check the selected bundle's ability to host that worker. Extending it
with this contract failed before the fix (0.001 >= 1.0).

Size all interchangeable single-node GPU bundles for rollout's CPU request
when a GPU rollout exists. Keep CPU-only and cross-node positional sizing.
Fold the sole-use _bundle_cpu helper into _bundle_requirements so the bundle's
GPU identity and role-remapping capacity rule are expressed together.

Compatibility: single-node plans with separate trainer/reward reservations
now reserve additional CPU capacity. For two GPU bundles and a 2-CPU rollout
worker, the plan now needs 4 CPUs rather than ceil(2.001)=3. The existing local
cluster CPU calculation derives the new total automatically; an external Ray
cluster must have enough capacity. No extra configuration knob is introduced.
This fixes the requested placement contract rather than dropping GPU pinning.

## Retain and why

- BundleLayout owns the intended role/device plan. Its local gpu_bundle helper
  coalesces shared devices into one reservation, while cpu_bundles allocates
  separate worker slots. These share construction-local state; no extra builder
  class is warranted. RolePlacement owns per-engine grouping and borrowed PG
  access, without taking group teardown ownership.
- GlobalRayPlacementOwner retains the raw group before readiness and keeps it
  after failed cleanup so shutdown can retry. Probe actors have their own
  cleanup, backed by group removal on failure. Do not simplify to a handle that
  is only stored after successful startup.
- Single-node validation compares local ordinals; cross-node validation compares
  node/GPU pairs and excludes the trainer node. Their separate methods express
  different namespaces. Live cluster preflight checks capacity, while actor
  metadata checks actual placement; one does not replace the other.
- actor_meta_get adapts wire mappings and typed actor handles and is also used
  by actor_group.py. Keep this real shared adapter. Keep lazy scheduling and
  placement-group creation/removal adapters rather than forcing Ray imports
  into module import or inventing a stateless utility class.
- _ProbeActor is a Ray framework adapter. _match_gpu_bundles shares role mapping
  logic between rollout and reward. _PLACEMENT_READY_TIMEOUT_S is a transport
  readiness budget, CPU/GPU and role names are resource/protocol keys, and
  __all__ declares the facade. No mixed-in ALL_CAPS business taxonomy exists.

Non-goals: changing GPU ordering, cross-node topology, trainer process ownership,
actor CPU requests, or replacing the runtime's cleanup owner. No real GPU
scheduling/performance claim: this regression verifies the planner/launcher
resource relation with an injected valid GPU permutation.

## Follow-up findings

- Tiny positive cpus_per_worker values below 0.001 are accepted by the worker
  config, but _ProbeActor requests 0.001 CPU. Such GPU bundles still cannot host
  the probe. Resolve this with the CPU resource-domain review: either reserve
  the probe minimum for GPU bundles or define the supported minimum at the
  configuration boundary. Do not scatter another numeric validator here.
- create currently labels every readiness exception as inability to satisfy
  capacity after 600 seconds. The chained cause is retained, but immediate Ray
  failures may receive misleading text. Separate timeout diagnostics when
  reviewing readiness failure contracts; this capacity fix leaves that behavior.

## Validation

The extended GPU permutation regression failed before the production change.
90 global-placement, actor-pool and runtime-config tests passed afterward; one
GPU test was deselected. Existing checks cover CPU-only capacity, cross-node
mapping, shared GPU roles, cleanup retry and launcher CPU propagation. Ruff
check and format check passed on the two changed Python files.
