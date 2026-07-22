"""Merge disjoint ``encode_targets`` outputs into one SFT latent shard."""

from __future__ import annotations

import argparse

from vrl.trainers.data.sft_latents import merge_sft_latent_shards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    merge_sft_latent_shards(args.input, args.out)


if __name__ == "__main__":
    main()
