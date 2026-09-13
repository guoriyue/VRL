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

## Local helper functions

- Keep inner functions when repeated calls, recursion, or captured per-operation state make the parent operation clearer. Do not expand them into duplicated logic or manual traversal stacks merely to remove nesting.
- Inline single-use helpers when that improves readability. Keep framework callbacks and native-API simplifications based on their actual role, not invocation count alone.

## Cleanup scope

- Do not count name-only changes as structural cleanup. Rename only when the user explicitly requests clearer naming; describe such work as a rename, not as architectural completion.
- Cleanup must identify the concrete duplication, state, ownership, dependency, or control-flow complexity it removes. Moving the same branches into a classmethod alone does not prove simplification.

## Tests

- Tests are written for the reader: prefer self-contained tests; a shared builder/fixture is allowed only when the same construction is genuinely reused by 3+ test files. Delete on sight: tests that echo a constructor's fields or a default back, and parametrized matrices whose rows all exercise the same code path.
- A test outside the `gpu` lane never sees CUDA: `tests/conftest.py` pins `torch.cuda.is_available()` to False for every unmarked test. A test that models a GPU host patches `is_available`/`device_count` itself (or uses `cuda_devices`); a test that needs the card carries `@pytest.mark.gpu`.
- Constructor bypasses (`object.__new__(Cls)` plus attribute pokes) belong in one named package helper (`bare_trainer` in `tests/trainers/online/_helpers.py`), never inline in a test: the attribute names a method reads should appear in one place.
- A test module carries one theorem family and says so in its name (`test_checkpoint_cli.py`, `test_checkpoint_schema_restore.py`), not one production module; shared doubles for a family live in a `_<family>_helpers.py` beside them.
- Run tests only from a venv synced to the lock (`uv sync --frozen --group test --group lint --extra cosmos ...`; `uv sync --dry-run` is the drift check). Never `pip install` into `.venv` by hand.
