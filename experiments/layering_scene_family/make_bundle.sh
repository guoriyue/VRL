#!/bin/bash
# Gather everything training needs into one directory of symlinks (about 11 GB of real files),
# so a single `rsync -aL BUNDLE/ host:DATA/` copies it to another machine.
# Usage: ./make_bundle.sh /path/to/bundle [scene_family_dir] [layered_rl_pipeline_dir]
set -eu
B=${1:?bundle dir}
S=${2:-$(cd "$(dirname "$0")/../.." && pwd)/outputs/layered_rl_pipeline_20261007/scene_family}
P=${3:-$(dirname "$S")}
U=$(cd "$P/.." && pwd)/stable_layers_reproduction_20261005/upstream
mkdir -p "$B"
for d in dataset dataset_v2 rord_layers rord_layers_matte real_rooms; do ln -sfn "$S/$d" "$B/$d"; done
ln -sf "$S/empty_frame_latent.pt" "$B/empty_frame_latent.pt"
ln -sf "$(readlink -f "$P/known4_candidate_probe/initial_adapter.pt")" "$B/initial_adapter.pt"
ln -sf "$P/known4_candidate_probe/plan.json" "$B/plan.source.json"
mkdir -p "$B/upstream"; ln -sf "$U/decompose.py" "$B/upstream/decompose.py"; ln -sf "$U/LICENSE.md" "$B/upstream/LICENSE.md"
du -shL "$B"/* | sort -k2
