# Model and sampling schema boundaries

Reviewed five complete modules: `config/model_schema.py`,
`config/sampling_schema.py`, `models/dtypes.py`, `models/families/names.py`, and
`rollouts/collector/requests.py`. Inspected schema selection, registry capability
validation and runtime dtype-construction consumers, plus schema, precision,
family registry and request-builder regression coverage. Those larger consumer
modules are not marked reviewed based on these excerpts.

## Decisions: retain the existing boundaries

- Model sections declare YAML vocabulary and checkpoint identity metadata.
  `MODEL_MEMORY_SECTIONS` derives from the schema; runtime capabilities validate
  their declared target names against it. Keep this schema boundary, not another
  hand-maintained memory-target allow-list. LoRA identity declarations are schema
  data, not workflow-local business vocabulary.
- Sampling classes represent actual family differences: native-cache LlamaGen
  does not accept the shared attention backend; Janus-R1 alone accepts reflection
  length; Echo fixes guidance to the distilled value. Empty subclasses or short
  field-only classes are meaningful schema specialization, not redundant runtime
  wrappers. Collapsing them would make unsupported knobs appear configurable.
- `SamplingSection.require_overrides` belongs to the selected schema class. Its
  classmethod uses that class's field definitions, validates per-prompt overrides
  and preserves explicit field presence with `exclude_unset`. YAML validation
  cannot replace it because prompt overrides arrive separately from the recipe.
- `GenerationRequestBuilder` owns family/config state. Keep its build method and
  `_apply_input_defaults`: strings are prompt shorthand; `replace` adds a missing
  task type without mutating supplied conditioning records. `_DENOISE_FIELDS`
  derives the separate denoise-options schema. Unknown remaining overrides must
  fail against the family's sampling vocabulary before submission.
- Dtype parsing and public precision policy are distinct: external/checkpoint
  aliases normalize to torch attribute names; role policy accepts plain parameter
  dtypes and treats quantization separately. Keep `_DTYPE_SPELLINGS` and its
  derived index in this deliberately isolated taxonomy. The four public helpers
  serve different return/error contracts. `require_plain_dtype` is a shared
  runtime construction boundary used for both parameters and prompt encoders.
- Family aliases live in a torch-free naming module. Keep `_ALIASES_BY_FAMILY`
  and derived `_FAMILY_BY_ALIAS`; `_index_aliases` checks uniqueness at table
  construction, and registry validation checks targets/canonical collisions.
  Unknown strings pass through normalization and are rejected by registry lookup;
  the naming helper does not guess another family.

No production changes were justified by this inspection. In particular, do not
introduce classes merely to namespace dtype conversion or flatten schema classes
merely because they are short.

## Constraints for later reviews

Many sampling fields still use `Any`. A successful schema parse proves their key
is allowed, not that every value is a valid runtime shape/count. Removing runtime
range/type checks requires tracing that field's full construction path first.
Likewise, mutable Pydantic sections can be supplied directly; annotations are not
an immutable validation certificate.

`ModelSection.use_lora` and path values are deliberately broad at this layer.
Their runtime/identity interpretation remains part of the pending model-building
review. No input tightening or default changes were made here.

## Validation

221 tests passed across `tests/config/test_schema.py`, `test_precision.py`, and
`tests/rollouts/runtime/{test_engine_requests,test_family_registry}.py` with CUDA
disabled. This covers family-selected keys, alias resolution, falsy value
projection, prompt overrides, and precision parsing; it does not establish real
model generation or all checkpoint identity behavior.

Previous implementation commit: `b5badc4d4` (lint placeholder error filtering).
