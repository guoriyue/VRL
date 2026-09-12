# Project instructions

- Chinese for conversation, explanations, reviews, and plan files; English for code, comments, docs, commit messages.
- Ruff is the formatter and linter; run it only on the files a task touched, never repository-wide.
- Commit work in reviewable groups; never push unless the user asks for a push in that message.
- Read the actual source, call sites, configs, and tests before changing code or recommending a deletion; cite the path that supports the claim.

## Python function signatures

- Keep primary data inputs positional-or-keyword, before a bare `*`. Put behavioral options (flags, limits, precision, placement, timeouts, and output settings) after it, so calls name their intent. Required options can also be keyword-only; defaults do not decide parameter kind.
- Use all-keyword-only signatures for configuration-only APIs with no primary data input. Do not add `*` to every function, introduce positional-only `/`, or move existing keyword-only parameters forward just for visual uniformity.
- Keep parameter kinds consistent across a protocol, its implementations, and same-purpose family adapters. Preserve signatures imposed by frameworks, callbacks, and vendored upstream code, including PyTorch and Diffusers overrides.
- When changing parameter kinds, inspect and update callers, including tests and RPC forwarding. Avoid unrelated argument reordering. Run the existing behavioral tests; do not add runtime signature checkers or tests that only assert punctuation.
