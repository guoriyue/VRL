"""Train the editor on declared edit chains through vrl's online recipe.

The config is an ordinary vrl training config (the same ``--config`` and
overrides as ``vrl-train``); its ``data`` section still declares the sampler.
The chains replace the config's prompt rows: each chain is one prompt item that
collects one group per step, with each step conditioned on a drawn sample of
the previous one. Drawn parent images are written under
``<trainer.output_dir>/edit_chains``.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from agentic.chains import load_edit_chains
from vrl.config.loading import load_config
from vrl.config.schema import parse_config
from vrl.scripts.common.online import run_online_recipe


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Bundled config name or YAML path")
    parser.add_argument("--chains", required=True, type=Path, help="Edit-chain JSONL manifest")
    parser.add_argument("overrides", nargs="*", help="Preset overlays and dotlist overrides")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, overrides=args.overrides)
    root = parse_config(cfg)
    if root.trainer is None or not root.trainer.output_dir:
        raise ValueError("config missing required field: trainer.output_dir")
    chains = load_edit_chains(args.chains, media_dir=Path(root.trainer.output_dir) / "edit_chains")
    asyncio.run(run_online_recipe(cfg, prompt_examples=chains))


if __name__ == "__main__":
    main()
