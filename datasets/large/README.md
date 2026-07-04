# Large Processed Datasets

Large datasets are not committed to GitHub. They should be published as
processed release archives on Google Drive.

Rules:

1. Do not upload raw household/person/trip survey dumps.
2. Upload only processed files with defined alternatives, choice indicators,
   availability, attributes, IDs, weights, and schema.
3. Keep a `schema.json` in every release archive.
4. Record the final Google Drive link in `google_drive_links.csv`.

## Current Processed Candidate

`lpmc_london` is processable from the Biogeme official data page into:

- `choice_wide.csv`
- `choice_long.csv`
- `schema.json`

Build the upload artifact:

```bash
python scripts/process_lpmc_london.py --zip
```

The zip is created under `datasets/large/releases/` and should be uploaded to
Google Drive manually or with an authenticated Drive uploader.
