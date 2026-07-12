"""Compatibility entry point for the renamed quantized-linear benchmark."""

from __future__ import annotations

from vrl.scripts.perf.quantized_linear_benchmark import main as quantized_main


def main() -> None:
    """Preserve the legacy command's FP8-only gate semantics."""

    quantized_main(schemes=("fp8",))


if __name__ == "__main__":
    main()
