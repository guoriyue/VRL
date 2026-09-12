# Runtime configuration policies

Reviewed `config/builders.py`, `precision.py`, `reward_inference.py`, and `data.py`.
Inspected their schema entry points, trainer precision-correction consumer,
HTTP client constructor, prompt manifest loader, and relevant tests. Other parts
of those consumer modules remain pending.

## Changes

- `build_configs` no longer checks `root.algorithm is not None` again after the
  mandatory-section guard has already established it.
- The HTTP client trusts `RewardInferenceConfig.timeout_s` on its typed path.
  This frozen config validates and normalizes the timeout at construction. The
  URL-plus-keyword path still calls the shared `require_timeout` boundary.
  Both public construction forms retain the same timeout semantics and errors
  for invalid input; no wire, cancellation or session logic changed.
- Correct the precision module's outdated claim that the base preset forces
  fp16 prompt encoders. The actual default inherits the rollout dtype, which
  the bundled SD3.5 recipe test verifies as bf16.

## Retained boundaries

- `build_configs` aggregates several owners and normalizes resume state in both
  raw and typed config. Keep its public entry and lazy package facade; simply
  adding `BuiltConfigs.from_cfg` plus a forwarding function would add a hop.
- `RewardRuntimeConfig.from_cfg` keeps unknown kwargs checks and positive-online
  reward enforcement. The raw schema accepts original weight representations,
  so producing actual float weights here is conversion, not just revalidation.
  Missing inference entries deliberately select in-process execution.
- `build_precision_split_safety_configs` creates a related pair consumed both by
  training and the hardware drift probe. Keep one shared policy source; neither
  individual returned config owns the pair.
- Precision constructors remain usable directly, independently of Pydantic.
  Keep normalized dtype/recipe checks there and schema adapters at YAML intake.
  `_PLAIN_DTYPES`, `_QUANTIZATION_FORMAT_RULES`, and derived tokens are an isolated
  protocol taxonomy. `_QuantizationFormatRules` describes that table's values;
  it is not a new workflow state object. `QuantizationPolicy.from_section` already
  owns optional construction. No reintroduction of the old free constructor.
- `_normalize_plain_dtype` and `_normalize_float32_precision` are shared by
  multiple schema and direct-policy constructors. Role inheritance in
  `PrecisionPolicy.from_section` is an explicit documented default, not inferred
  checkpoint state. Keep role equality including quantization/autocast.
- `RewardInferenceConfig` owns deployment only; HTTP model identity and origin
  shape are real operator-facing constraints. `require_http_origin` is shared
  with direct client construction. `_INFERENCE_FIELDS` derives from the dataclass
  schema, preserving unknown-key diagnostics without a second hand-maintained
  field list.
- `resolve_data_loader` keeps the existing loader/format compatibility contract;
  `manifest_sources` keeps the two supported manifest spellings. Their outputs
  are shared by schema, loader and data bootstrap; no stateful loader wrapper is
  warranted merely to replace these functions.

## Validation

158 tests passed: config builders, precision, prompt dataset configs, reward
deployment config and the reward service test module. This includes actual local
HTTP round trips and identity checks, invalid direct timeouts, typed-config
construction, recipe precision/default resolution and offline/online builds.
CUDA was disabled for this test process. Ruff passed on touched files.

Previous implementation commit: `1f56e2807` (YAML defaults duplicate check).
