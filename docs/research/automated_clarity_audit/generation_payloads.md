# Generation payload and protocol boundaries

Reviewed complete generation `types.py`, `protocols.py` and package `__init__.py`,
collector runtime attachment, gather range validation and existing request,
sample-identity, actor-protocol and reference-conditioning tests.

## Change

Batch range validation reads len(self.inputs) directly instead of constructing
self.prompts just to count it. Both counts are identical for GenerationRequest;
the new expression directly names the indexed collection and avoids rebuilding
a prompt list for each checked batch. No new validation or performance test is
needed for this local change. External subclasses overriding prompts to change
cardinality no longer redefine the input-range boundary implicitly.

Correct protocol documentation: the current collector attaches the runtime
without an isinstance check. The rank protocol lists the core actor methods;
bucket-transfer and acceptance endpoints are additional actor APIs. The existing
conformance test checks listed methods, not every string used in every RPC.
Do not add blanket structural checks to make the old prose true.

## Retain and why

- GenerationRequest's explicit constructor accepts strings and conditioning
  objects, normalizes only that boundary, and preserves a typed inputs list.
  prompts is a useful text view for actual text consumers; remove neither it
  nor structured inputs. Input/task defaults remain with family composition.
- Sample IDs derive deterministically from request/prompt/sample indices and
  join generation, trajectory and rewards. Keep sample_rows request-owned.
  Batch range checks compare received batch coordinates with the source request;
  their constructor alone cannot prove that cross-object relation.
- DenoiseRequest geometry checks also cover direct eval construction. Runtime
  type annotations do not enforce those external values. Retain this boundary;
  do not add internal repeated float/token checks.
- GenerationOutput's identity/sample properties delegate to its trajectory,
  preventing two competing records. Its lazy trajectory import preserves the
  public package's lightweight import. Reward/advantage math stays outside.
- Runtime, executor, model-free gatherer and optional probe capability represent
  different consumers and ownership. The typed protocols and BatchPayload=Any
  preserve family-specific payloads without inventing a false universal tensor
  layout. The rank protocol is a framework boundary, not a wrapper to flatten.
- Package exports are a public facade. __all__ and protocol method names are API
  declarations; no ALL_CAPS business vocabulary occurs here. Mutable requests
  are not deep-validated forever after construction; arbitrary mutation is not
  a reason to add a second validator framework.

Non-goals: collapse protocols into one object, change wire fields, eliminate
family adapters for line count, alter SDE seed derivation, or extend the core
rank protocol solely to mirror every optional endpoint.

## Validation

69 request/sample-batch/engine/reference-conditioning tests passed. A fresh
interpreter importing vrl.generation reported torch absent from sys.modules.
Ruff check and format check passed for both changed Python files. No GPU work
or end-to-end training-speed claim is involved.
Previous isolated audit commit: `ea7ecabf1`.
