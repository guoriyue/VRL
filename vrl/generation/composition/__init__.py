"""Generation-regime composition shared across policy implementations.

The layer between ``steps`` (the contract for one scheduled policy step) and
``bindings`` (regime executors bound to the chunk-executor contract): a
composition owns the loop that drives many steps over a family runner —
scheduling and row-state routing — while staying family-neutral. It is
separate from ``bindings`` so family runtimes can reuse a loop directly
(janus_pro and nextstep_1 do) without adopting request/chunk transport.
"""
