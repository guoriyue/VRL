# Resource resolution ownership and discovery failures

Reviewed the complete 1,000-line resources.py, the schema's consuming dataclass
boundary, placement/launcher consumers, collector lifecycle properties, and
resource-resolution tests. Previous audit commit: c95d2374d.

## Changes

- Remove unused _MISSING. Unlike the separately used sentinels in checkpoint
  identity and PEFT code, this module's object had no consumer.
- Rename _dedupe_ints to _validate_device_ids at all five callers. It rejects
  duplicates rather than removing them. Keep behavior and error messages; the
  name now reflects the operation. It is private and has no other repository
  callers; external private imports must migrate.
- Correct visible_devices documentation: it is also used by torch/plan ordinal
  translation, not merely printed provenance.
- Auto discovery no longer converts arbitrary import/CUDA exceptions into an
  empty GPU plan. Only a missing torch package or an explicit unavailable CUDA
  result returns no devices. A broken torch dependency/import or failed query
  surfaces its original exception. Remove int() around torch's integer count.
  This deliberately changes failure behavior so setup errors cannot silently
  choose a CPU/no-rollout topology. Optional torch import remains lazy.

## Retain and why

Resource request dataclasses are also the schema field definitions. Their role
ClassVars provide diagnostic paths without becoming YAML keys. Rollout count
and engine arithmetic belongs to RolloutResourceConfig; reward has one local
reservation and no fleet, so sharing a fleet base class would obscure its API.

RayLifecyclePlan stores three independent GPU-sharing relations. Its short
properties are phase views over those facts, consumed by collector and launcher.
They prevent independently stored release flags from drifting. Preserve them
and the resource plan properties instead of copying calculations into callers.

Resolution helpers read multiple role pools or distinguish static cross-node
budget tokens from local device ordinals. These are useful shared calculations,
not a reason to introduce another resolver instance or move all functions into
one namespace class. The FSDP guards compare strategy, world size and role sets;
they cannot be replaced by one device-list validator. The local reward helper
applies one shared overlap rule across explicit and automatic selection paths.

Parsing helpers implement the existing public auto/null/list grammar and retain
ordered device validation. Subset and uniqueness are different contracts.
format_distributed_resource_plan is shared presentation with no state. Keep it
free-standing. Lazy torch discovery preserves config imports without Ray startup.
The only ALL_CAPS object removed was dead; __all__ is the public facade. Role,
pool, strategy and lease names are protocol/schema vocabulary deliberately
located in the resource model rather than algorithm-name lists in workflow code.

Non-goals: changing auto placement policy, accepted YAML coercions, FSDP/DDP
topology, trainer/reward ownership, or making resource dataclasses deeply frozen.
Do not merge short helpers merely to reduce the function count.

## Follow-up evidence and limits

- Ordinal conversion currently infers a narrowed process view from matching
  CUDA device count and visible_devices length. Equal counts alone do not prove
  equal device ordering. Existing tests cover a rank explicitly narrowed to GPU
  2; a stronger mapping contract needs launch_environment and trainer-device
  ownership reviewed together. Do not replace it with a second heuristic here.
- With both rollout counts auto and no spare GPU, requested_gpu_count receives
  zero available devices and returns zero before the sharing fallback. The
  existing single-GPU fallback test instead requests one GPU/engine explicitly.
  Decide the all-auto policy with preset/launcher requirements; do not silently
  change no-fleet semantics during naming cleanup.
- Mutable/prebuilt dataclass and string/list coercion paths retain historical
  int conversions. This review adds no blanket integer/type validation suite.

## Validation

The regression injects a CUDA query failure through the public from_root path.
Before the change it did not raise; afterward it preserves the same exception
object. 90 resource, global-placement and config-resource tests passed. This
includes actual CPU Ray placement tests with logical GPUs, not GPU training.
Ruff check and format check passed for both changed Python files.
