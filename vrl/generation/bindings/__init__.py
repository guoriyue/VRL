"""Concrete composition-by-step generation bindings.

One subpackage per generation regime — ``full_sequence_denoise``,
``chunk_autoregressive_denoise``, ``token_autoregressive`` — because the
regime, not the model family, fixes the trajectory schema and the per-batch
control flow; model families subclass their regime's executor base and are
wired in by registry dotted strings. Each binding adapts the family-neutral
step/loop machinery (``vrl/generation/steps``, ``vrl/generation/composition``)
to ``execution``'s batch-executor contract. Within a binding, executors run
worker-side next to the model, while gatherers reassemble batch payloads
driver-side where no model is loaded (see ``vrl/generation/protocols.py``).
"""
