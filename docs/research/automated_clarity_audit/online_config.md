# Online trainer configuration

Reviewed complete trainers/online/config.py, ActorSection scalar declarations,
config builder entry point, batch-plan/config tests and experiment-load coverage.
Previous audit commit: b0ccbd802.

Change: simplify the local actor count reader to return zero for an omitted
setting and the checked integer otherwise. Remove zero-to-None-to-zero round trips
and repeated int conversions. No new checker or validation behavior is introduced.

Retain and why:

- OnlineBatchPlan owns optimizer batch geometry, derives accumulation from an
  explicit microbatch size and exposes a single streaming interpretation. Schema
  StrictInt fields do not enforce non-negative ranges or divisibility, so the
  plan's existing checks are not all redundant with YAML parsing.
- actor_count is a small closure over the selected actor section used for both
  alternative geometry inputs. Promoting it into a general checker class would
  obscure this configuration-specific omission convention.
- required_public_paths serves both direct plan construction and TrainerConfig's
  aggregated missing-field report. Preserve the shared declaration rather than
  duplicating the two required rollout names in both owners.
- TrainerConfig.from_root derives ownership and requiredness from dataclass and
  public section fields. Its bridged field set is a deliberately isolated schema
  mapping; it does not encode per-algorithm facts. The ownership assertion catches
  a real mismatch between independently maintained public/internal declarations.
- Precision labels come from one resolved policy. Split-precision defaults respect
  explicitly supplied correction/guard sections; deleting that distinction would
  overwrite experiment decisions. Existing precision guard tests cover it.
- Streaming permits one PPO epoch because released microbatches cannot be replayed
  again; sde_window derives its own step coverage. These are actual training
  constraints, not checks for imagined input types.

Non-goals: remove remaining range checks by pattern, introduce nested config
wrapper classes, change batch geometry, precision defaults or direct dataclass
construction semantics. Shared field projection is preferable to per-field
handwritten parsing even though it uses reflection. No ALL_CAPS workflow table
needs extraction in this module.

76 existing tests passed across online config, all-experiment loading and rollout
drift guard suites. Ruff check and format check passed for config.py. No tests or
examples were added. This validates configuration behavior and bundled recipes,
not actual GPU accumulation or memory-budget performance.
