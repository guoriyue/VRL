"""Compatibility entry point for the quantized rollout drift probe."""

from __future__ import annotations

from vrl.scripts.perf.quantized_rollout_drift_probe import main as quantized_main


def main() -> None:
    """Preserve the legacy command's FP8-only probe semantics."""

    quantized_main(scheme="fp8")


if __name__ == "__main__":
    main()
