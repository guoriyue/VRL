"""Unit tests for the shared vrl logging helpers."""

from __future__ import annotations

import logging

from vrl.utils.logging import init_logger, kv


def test_init_logger_installs_single_handler() -> None:
    """Repeated init_logger calls must not stack duplicate handlers.

    propagate=False is not asserted here: the autouse conftest fixture
    re-enables propagation under pytest so caplog keeps working.
    """
    init_logger("vrl.test_logging.a")
    init_logger("vrl.test_logging.b")

    root = logging.getLogger("vrl")
    stream_handlers = [
        h for h in root.handlers if isinstance(h, logging.StreamHandler)
    ]
    assert len(stream_handlers) == 1


def test_logger_emits_once_to_stdout(capsys) -> None:
    """INFO records reach stdout exactly once even with basicConfig set."""
    logging.basicConfig(level=logging.INFO)
    logger = init_logger("vrl.test_logging.emit")

    logger.info("hello %s", kv(run_id=7))

    captured = capsys.readouterr()
    assert captured.out.count("hello run_id=7") == 1
    # propagate=False keeps the record off the root handler (stderr).
    assert "hello run_id=7" not in captured.err


def test_kv_formats_floats_and_passthrough() -> None:
    """Floats render as .3f; other values render verbatim."""
    assert kv(elapsed_s=1.23456, n=3, name="kling") == (
        "elapsed_s=1.235 n=3 name=kling"
    )
    assert kv() == ""
