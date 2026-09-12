# Config lint error filtering

Reviewed `vrl/config/lint.py` and its composition/schema/error-formatting
dependencies and tests. This is a source-level config check, not training launch.

## Reproduced problem and fix

This input was incorrectly reported clean:

```yaml
actor:
  optim:
    lr: ???
distributed:
  training:
    strategy: bogus
```

Before the change, `experiment_parse_error` returned `None`; supplying `lr=0.001`
made the unsupported strategy appear. The lint adapter only saw the formatted
first error from `parse_config`; when that error concerned a mandatory placeholder,
it returned success and lost all later Pydantic diagnostics.

Lint now passes its already-composed plain mapping to the same `RootConfig`
validation boundary and retains the full structured `ValidationError`. It filters
individual placeholder diagnostics, then formats the retained errors through the
existing shared formatter, preserving unknown-key priority and dotted paths.
No guessed placeholder values or additional validator classes are introduced.
Runtime `parse_config` and its required-value rejection are unchanged.

## Retained functions and limitations

Keep mandatory-value projection as a lint-specific pure transform, the public
single-config/sweep functions as tool/CI interfaces, and `main` as the CLI adapter.
The nested `strip` function owns recursion over that one projection; it is not
a stateful loader needing a wrapper class. This module has no domain ALL_CAPS
table to relocate.

An unresolved mandatory field can prevent a section or root post-validator from
running. Lint reports all diagnostics available from that validation pass; it
does not prove cross-section compatibility for unspecified launch values. A
fully supplied configuration still needs normal parse and launch gates.

## Validation

29 tests passed across unknown-key/structural lint, loading composition and
resource loading. The new regression checks that the invalid strategy remains
visible beside a mandatory learning rate, and that correcting the strategy
allows the template to lint without supplying that rate. Existing bundled
experiment sweep remains green. Ruff and formatting checks passed.

Previous implementation commit: `a0ff2314b` (normalized deployment timeout).
