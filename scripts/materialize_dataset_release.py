from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "validation" / "datasets" / "dataset_manifest.json"
DATASETS_DIR = ROOT / "datasets"
SMALL_DIR = DATASETS_DIR / "small"
LARGE_DIR = DATASETS_DIR / "large"
SMALL_SIZE_LIMIT = 10 * 1024 * 1024
LARGE_IDS = {"lpmc_london"}


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def dataset_bytes(dataset: dict) -> int:
    return sum(int(item.get("bytes", 0)) for item in dataset.get("files", []))


def choose_csv_file(dataset: dict) -> Path | None:
    for item in dataset.get("files", []):
        path = Path(item["path"])
        if path.name == "data.csv":
            return ROOT / "validation" / "datasets" / path
    return None


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def materialize_small(manifest: dict) -> list[dict]:
    SMALL_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for dataset in manifest["datasets"]:
        dataset_id = dataset["dataset_id"]
        if dataset["status"] != "downloaded":
            continue
        if dataset_id in LARGE_IDS or dataset_bytes(dataset) > SMALL_SIZE_LIMIT:
            continue
        csv_file = choose_csv_file(dataset)
        if csv_file is None or not csv_file.exists():
            continue
        out_dir = SMALL_DIR / dataset_id
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(csv_file, out_dir / "data.csv")
        metadata = {
            "dataset_id": dataset_id,
            "source_family": dataset.get("source_family", ""),
            "dataset_name": dataset.get("dataset_name", ""),
            "upstream_url": dataset.get("upstream_url", ""),
            "license_or_terms": dataset.get("license_or_terms", ""),
            "rows": dataset.get("rows"),
            "columns": dataset.get("columns"),
            "sha256": next((f.get("sha256") for f in dataset["files"] if f["path"].endswith("/data.csv")), ""),
            "notes": dataset.get("notes", {}),
        }
        write_text(out_dir / "metadata.json", json.dumps(metadata, indent=2, sort_keys=True))
        rows.append(
            {
                "dataset_id": dataset_id,
                "rows": dataset.get("rows"),
                "columns": dataset.get("columns"),
                "source_family": dataset.get("source_family", ""),
                "path": f"datasets/small/{dataset_id}/data.csv",
                "upstream_url": dataset.get("upstream_url", ""),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    lines = [",".join(columns)]
    for row in rows:
        values = []
        for column in columns:
            value = "" if row.get(column) is None else str(row.get(column, ""))
            values.append('"' + value.replace('"', '""') + '"')
        lines.append(",".join(values))
    write_text(path, "\n".join(lines) + "\n")


def write_indexes(manifest: dict, small_rows: list[dict]) -> None:
    large_rows = []
    for dataset in manifest["datasets"]:
        if dataset["status"] == "manual_large_pending" or dataset["dataset_id"] in LARGE_IDS:
            large_rows.append(
                {
                    "dataset_id": dataset["dataset_id"],
                    "release_status": "processed_pending" if dataset["dataset_id"] != "lpmc_london" else "processed_locally_not_uploaded",
                    "processed_artifact": f"datasets/large/releases/{dataset['dataset_id']}.zip",
                    "google_drive_url": "TODO",
                    "source_url": dataset.get("upstream_url", ""),
                    "notes": dataset.get("message", ""),
                }
            )
    write_csv(
        DATASETS_DIR / "dataset_index.csv",
        [
            {
                "dataset_id": item["dataset_id"],
                "status": item["status"],
                "rows": item["rows"],
                "columns": item["columns"],
                "storage": "github_small"
                if any(row["dataset_id"] == item["dataset_id"] for row in small_rows)
                else ("google_drive_processed" if item["dataset_id"] in LARGE_IDS or item["status"] == "manual_large_pending" else item["status"]),
                "upstream_url": item.get("upstream_url", ""),
            }
            for item in manifest["datasets"]
        ],
        ["dataset_id", "status", "rows", "columns", "storage", "upstream_url"],
    )
    write_csv(SMALL_DIR / "small_datasets.csv", small_rows, ["dataset_id", "rows", "columns", "source_family", "path", "upstream_url"])
    write_csv(
        LARGE_DIR / "google_drive_links.csv",
        large_rows,
        ["dataset_id", "release_status", "processed_artifact", "google_drive_url", "source_url", "notes"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize GitHub-small and Google-Drive-large dataset release metadata.")
    parser.parse_args()
    manifest = load_manifest()
    small_rows = materialize_small(manifest)
    write_indexes(manifest, small_rows)
    print(f"small_datasets={len(small_rows)}")
    print(f"index={DATASETS_DIR / 'dataset_index.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
