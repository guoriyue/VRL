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
