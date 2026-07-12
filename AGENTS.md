# AGENTS.md

## Core Behavioral Guidelines

- Verify your own work before reporting back. Run the code, check the output, click through visual flows, simulate edge cases. Don't hand back a first draft.
- Define finishing criteria before you start. If something fails, fix and re-test — don't flag and hand back. Only come back when things are confirmed working, or you hit a hard blocker: missing credentials/secrets, need access you don't have, or a requirement that is genuinely ambiguous about the end-user goal. "Two valid approaches exist" is NOT a blocker — pick the better one yourself.
- Think independently. Don't blindly agree with a flawed approach — push back on it. But independent thinking means making good judgments on your own, not asking for permission at every step.
- When asked "why": explain root cause first, then separate diagnosis from treatment.
- Challenge my direction when it seems off. If the end-user goal itself is ambiguous, ask upfront before starting. Implementation path decisions (which approach, which library, how to structure) are your job — make the call yourself. If the path is suboptimal, say so directly.

### Task Completion

- **Fix root causes, not symptoms.** No workarounds, no band-aids, no "minimal fixes." If the architecture is wrong, restructure it. Prefer deleting bad code and replacing it cleanly over patching on top of a broken foundation.
- **Finish what you start.** Complete the full task. Don't implement half a feature. Implementation decisions are your job, not questions to ask.
- **Never use these patterns** — they are all ways of asking permission to continue. Just do the work:
  - ❌ "如果你要，我下一步可以..."
  - ❌ "如果你愿意..."
  - ❌ "你要我直接...吗？"
  - ❌ "要不要我帮你..."
  - ❌ "是否需要我..."
  - ❌ "我可以帮你...，要我做吗？"
  - ❌ "下一步可以..."（as an offer, not a description of what you ARE doing）
  - ❌ Any sentence ending with "...吗？" that asks whether to proceed with implementation
  - ✅ Instead: "接下来我会 xxx" then execute.

## Communication Guidelines

- Use Chinese for all conversations, explanations, code review results, and plan file content
- Use English for all code-related content: code, code comments, documentation, UI strings, commit messages, PR titles/descriptions

## Development Guidelines

### Evidence-First Work

- Before modifying code or suggesting a concrete implementation, read the relevant source files, adjacent modules, call sites, configs, and tests that define the behavior.
- Do not make claims from filenames, memory, naming conventions, or architectural intuition when the repository can answer the question.
- If the relevant code has not been read yet, say what needs to be inspected, then inspect it before making the claim. Mark unavoidable assumptions explicitly instead of presenting them as facts.
- Deletion, refactor, cleanup, and architecture recommendations must cite the specific snippet/path that supports the conclusion.
- Before touching a file, identify the local pattern it belongs to: source of truth, public API or protocol boundary, consumers, tests, and neighboring implementations. Make the change fit that pattern unless the evidence shows the pattern itself is the root cause.
- Prefer "read narrow, then widen as needed": start from the target file, then follow imports, callers, configs, tests, and docs until the behavior is explained. Stop only when the conclusion is backed by repository evidence.

### Core Coding Principles

- ALWAYS search documentation and existing solutions first
- Read template files, adjacent files, and surrounding code to understand existing patterns
- Learn code logic from related tests
- Review implementation after multiple modifications to same code block
- Keep project docs (PRD, todo, changelog) consistent with actual changes when they exist
- After 3+ failed attempts, add debug logging and try different approaches. Only ask the user for runtime logs when the issue requires information you literally cannot access (e.g., production environment, device-specific behavior)
- For frontend projects, NEVER run dev/build/start/serve commands. Verify through code review, type checking, and linting instead
- NEVER add time estimates to plans (e.g. "Phase 1 (3 days)", "Phase 2 (1 week)") — just write the code
- NEVER read secret files (.env, private keys), print secret values, or hardcode secrets in code

### Formatting and Diff Discipline

- Ruff is the single source of truth for Python linting, import sorting, and formatting. Do not manually apply a competing style or perform aesthetic reformatting that Ruff does not require.
- Run Ruff only on the Python files touched by the task: first `ruff check --fix <files>`, then `ruff format <files>`, followed by `ruff check <files>` and `ruff format --check <files>` for verification.
- Never run a repository-wide formatting or auto-fix command (for example, `ruff format .` or `ruff check --fix .`) unless the user explicitly requests a formatting sweep.
- Keep functional diffs focused. Do not reflow nearby comments/docstrings, reorder unrelated code, normalize quotes, wrap lines, or change whitespace outside the task merely because the file is open.
- Inspect the final diff after Ruff runs. If formatting a touched file creates broad unrelated churn, remove that churn and leave pre-existing style debt for a separate formatting task; an existing violation is not authorization to rewrite unrelated code.
- Do not mix a formatting-only cleanup with a functional change. Consistency with the repository formatter is more valuable than personal formatting preferences, and a reviewable minimal diff is more valuable than opportunistic cleanup.

### Architecture Hygiene

- ALWAYS question module-level ALL_CAPS hardcoded data and thin separated functions/files when reviewing or editing code.
- Keep ALL_CAPS constants only when they represent a real boundary: schema keys, environment variable names, checkpoint/file names, model architecture dimensions, protocol names, test fixture constants, or a deliberately isolated taxonomy/config table.
- If ALL_CAPS data is a large business vocabulary, prompt template, backend table, or domain taxonomy mixed into workflow code, prefer moving it to a clearly named module or config asset.
- Do NOT hand-maintain a hardcoded ALL_CAPS constant that merely duplicates a typed structure. Derive validation sets (allowed keys, valid values, field-name lists) from their source of truth — e.g. `frozenset(f.name for f in fields(Policy))` instead of a hand-written `_ALLOWED_KEYS = {...}`. A duplicated constant silently rots the moment the source type gains a field, rejecting valid input as "unknown". The typed structure is the single source of truth; the constant should disappear into a derivation.
- Keep thin functions/files only when they provide a protocol/interface boundary, public API facade, lazy import boundary, framework adapter, test fake/fixture, cross-family consistency, or a shared abstraction that removes real complexity.
- Do not flatten or data-ize thin functions merely to save a few lines. Preserve uniform cross-family shapes when that consistency improves grepability, debugging, and readability.
- When proposing cleanup, explicitly list what should change, what should stay unchanged, why each thin function or ALL_CAPS constant is necessary or not, and non-goals where consistency is more valuable than LOC reduction.
- Every field of a **derived/resolved struct** ("compute once, read everywhere" — e.g. `Resolved*`, `*Capability`) must have a **non-logging consumer**: a control-flow branch, a value passed to a runtime/config/Ray call, or a validation that can raise. A field whose only readers are `format_*`/log builders or tests is a dead field — delete it, OR (if its log line carries genuinely un-derivable provenance, like the full visible-GPU pool) keep it and annotate the field at its definition as `display/provenance-only`. A field that is neither behavior-consumed nor explicitly marked display-only is dead and should be removed. This mirrors the ALL_CAPS rule above: the consumer is the source of truth — a field nobody reads silently rots, and for user-facing config keys it is worse (a no-op knob the user sets expecting an effect).

### Dead Code Audit — the five forms

Learned the hard way during the thin-model-seam sprint (2026-07): a "zero
callers" sweep is **necessary but not sufficient**. "This is genuinely
family-specific / can't be folded" verdicts are only valid after reading the
**function body**, not the call site — two wrong verdicts in that sprint
(predict2_5's dead metadata keys masquerading as family logic, cosmos3's
`_apply_train_knobs` masquerading as a custom build branch) both came from
judging by the caller. Audit every suspect against all five forms:

1. **Zero callers.** The obvious one — but grep must exclude the definition,
   include tests separately (test-only callers make it TEST-ONLY, which per the
   dead-field rule is still dead), and cover `module:function` string
   references (registries, launch contracts, e2e harnesses dispatch by dotted
   string — a plain-symbol grep misses them).
2. **Live callers, dead semantics.** The function runs, but a branch's inputs
   no longer have producers. Example: a task-variant normalizer whose alias
   set (`image_to_video`, …) had zero producers left after the config knob
   feeding it was deleted — the function reduced to one comparison. Check:
   for each branch, who can still produce the input that selects it?
3. **Single-caller concept splits.** One decision spread across two private
   functions where the second has exactly one caller (the first). Merge them —
   a decision should read top-to-bottom in one place. Distinguish from
   *justified* single-caller helpers: lazy-import boundaries and
   concept-naming extractions from long flows stay (see the thin-function
   keep-list above).
4. **Body identical to an existing shared implementation.** A hand-copied
   private helper whose body IS the shared builder/base-class sequence
   (sometimes with the steps in a subtly wrong order — cosmos3 ran
   quantization after compile, hiding the fp8 modules from inductor). Diff the
   body against the shared implementation before ruling "custom".
5. **Reimplementing a dependency's internals.** Hand-walking the HF hub cache
   layout (`models--org--name/snapshots`, mtime-max "latest") instead of
   calling `hf_hub_download`; anything that hardcodes another library's
   directory/format/protocol internals. Replace with the library call —
   prior-art rule applies to dependencies too.

Same forms apply to **data**: bundle/metadata keys, `runtime_caps` keys, and
config knobs are dead when no reader exists (`grep -rnF 'metadata["key"]'` +
`.get("key"` — plain regex greps corrupt the brackets and match everything).
Duplicate construction sites are the data twin of form 4: information the
registry already encodes must not be re-derived at runtime (wan re-resolved
its t2v/i2v variant from cfg although family selection had already picked the
registry entry; capabilities were constructed both registry-side and
family-side). One construction site; everyone else imports or is handed it.

### Long-term Assets vs One-shot Validation

- Distinguish **one-shot validation artifacts** (KILL-RISK gates, feasibility spikes, smoke probes, scratch datasets, intermediate manifests, throwaway logs) from **long-term assets** (production tools, configs, dataset generators, tests, model wrappers, sprint docs, entrypoints). Both deliver value, but follow different lifecycles.
- One-shot artifacts: name them clearly so they are recognizable (`*_smoke`, `*_spike`, `*_probe`, `_scratch_*`), keep them outside the import graph of long-term code, and delete them once the question they answered is recorded elsewhere (sprint doc, decision note, commit message). Their value is the answer they produced, not their continued existence.
- Long-term assets: live in canonical paths (`vrl/scripts/eval/`, `configs/`, `tests/`, `docs/sprints/`), are referenced by other code or documentation, and survive routine cleanup passes. If a script is reusable for future verification or onboarding, it is a long-term asset — keep it.
- Cleanup scope rule: when deleting one-shot artifacts, extend cleanup to **same source + same lifecycle** (the script *and* the outputs/datasets it produced), to prevent orphans. Do NOT extend cleanup to unrelated historical artifacts from other sprints or contributors — those belong to their own owners and lifecycles.
- Before deleting non-trivial artifacts: enumerate **what will be removed vs what will be kept**, grep the import graph to confirm no long-term asset references the targets, then act. After acting, rerun the verification suite (pytest, build, config-resolve) to confirm zero regression.
- When deletion removes a path referenced by a long-term config/test, repoint the reference to the canonical intended location (and document the regeneration path in a comment), instead of either leaving a dangling pointer or deleting the long-term asset to "match" the missing data.

### Code Comments

- Comment WHY not WHAT. Prefer JSDoc over line comments.
- MUST comment: complex business logic, module limitations, design trade-offs.

### Naming

- Prefer concrete, plain names over abstract, generic, or decorative ones. A parameter holding a `BuildSpec` should say what it plainly is (`build`), not borrow a vague, engine-sounding word (`runtime`, `blueprint`, `context`, `manager`, `handler`). When in doubt, pick the simpler, more literal name.
- Never reuse a word that already denotes a different type or concept nearby. Naming a build-time spec `runtime` when `RuntimeBundle` / `RuntimeModel` exist creates a collision that misleads readers — a name must not fight the surrounding vocabulary.
- Keep names short and greppable; avoid clever or ornamental naming. When renaming for taste, apply the same new name uniformly across all sibling call sites so cross-family shapes stay consistent (do not rename in one place only).
- A name must be accurate to **what the thing is**, never meaningless filler. Two failure modes to avoid: (1) **accurate about one side only** — `backend_loading.py` for a module that loads diffusers/PEFT pieces *and* prepares the transformer names just the "backend" half and borrows a word (`backend`) that already means "training/inference engine" in the RL world; (2) **meaningless decoration** — suffixes like `_component`, `_manager`, `_handler`, `_helper`, `_util` that add length without narrowing meaning (`load_diffusers_transformer_component` says nothing more than `load_diffusers_transformer`). Strip the filler; keep the concrete noun.
- Prefer an established name from a comparable system over inventing one. For loaders/schedulers/runners/weight-sync and similar infra, reach for the term vLLM / SGLang / slime already use (e.g. `loader.py` / `model_loader`, `weight_utils`) so readers coming from those codebases map it instantly — only coin a new name when no prior art fits.

## Subagents

- ALWAYS wait for all subagents to complete before yielding.
- Spawn subagents automatically when:
  - Parallelizable work (e.g., install + verify, npm test + typecheck, multiple tasks from plan)
  - Long-running or blocking tasks where a worker can run independently.
  - Isolation for risky changes or checks

## Output Style

- Use plain, clear language — no jargon, no code-speak. Write as if explaining to a smart person who isn't looking at the code. Technical rigor stays in the work itself, not in how you talk about it.
- State the core conclusion or summary first, then provide further explanation.
- For code reviews, debugging explanations, and code walkthroughs, quote the smallest relevant code snippet directly in the response before giving file paths or line references.
- Do not rely on file paths and line numbers alone when an inline snippet would explain the point faster. Treat file paths as supporting evidence, not the main payload.
- When referencing specific code, always provide the corresponding file path.

### References

Always provide complete references links or file paths at the end of responses:
- **External resources**: Full clickable links for GitHub issues/discussions/PRs, documentation, API references
- **Source code references**: Complete file paths for functions, Classes, or code snippets mentioned

## Compact Instructions

When compressing context, preserve in priority order:

1. Architecture decisions and design trade-offs (NEVER summarize away)
2. Modified files and their key changes
3. Current task goal and verification status (pass/fail)
4. Open TODOs and known dead-ends
5. Tool outputs (can discard, keep pass/fail verdict only)
