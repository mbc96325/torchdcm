# TorchDCM Benchmark Datasets

TorchDCM uses public datasets with explicit provenance and reproducible release
rules.

## Storage Policy

| Class | Location | Rule |
| --- | --- | --- |
| Small canonical datasets | `datasets/small/<dataset_id>/data.csv` | Can be committed to GitHub. |
| Large processed datasets | Google Drive release zip | Only processed choice sets and attributes are uploaded. |
| Raw validation mirrors | `validation/datasets/raw/` | Local/remote validation workspace, not committed by default. |

## Current Release

- 28 public datasets are downloaded or exported from Biogeme, Apollo, and
  `mlogit`.
- 7 large travel-survey sources are tracked as processed-data candidates.
- LPMC London is treated as a large processed release: raw data stays in
  validation, while processed wide/long choice-set files are packaged for Google
  Drive.

See:

- `dataset_index.csv` for the complete catalog.
- `small/small_datasets.csv` for GitHub-hosted small datasets.
- `large/google_drive_links.csv` for large processed-release status and links.

## Rebuilding

```bash
python scripts/materialize_dataset_release.py
python scripts/process_lpmc_london.py --zip
```

Run these after refreshing `validation/datasets/dataset_manifest.json`.
