# Danbooru Dataset Area

This directory groups anime prompt datasets by task and keeps shared
Danbooru2023 analysis at the top level.

Subdirectories:

- `anatomy/`: quota-balanced anatomy prompt manifests for Anima RL.
- `safety/`: anime safety prompt manifests and baseline eval prompts. These
  prompts are grouped here for dataset organization; the primary build path
  derives explicit/questionable prompt tags from Danbooru metadata.

Shared files:

- `dataset_analysis_report.md`: Danbooru2023 metadata analysis used to size and
  balance the anatomy prompt dataset.

The anatomy prompt build uses `metadata/posts.tar.gz` from
`nyanko7/danbooru2023`. It does not require downloading the image tarballs.
Image files are only needed for later positive-image manifests, crops,
hard-negative mining, or reward calibration.

The committed safety manifests are frozen and their builder has been removed
from the tree. To regenerate them, check out `archive/main-20261022` and run the
builder there (it reads `manifests/danbooru/config.yaml`):

```bash
git switch --detach archive/main-20261022
python -m vrl.scripts.data.setup anime-safety-prompts
```

The builder is `archive/main-20261022:vrl/scripts/data/danbooru.py`.
