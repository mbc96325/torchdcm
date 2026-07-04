from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "validation" / "datasets" / "raw" / "lpmc_london" / "data.csv"
OUT_DIR = ROOT / "datasets" / "large" / "processed" / "lpmc_london"
RELEASE_DIR = ROOT / "datasets" / "large" / "releases"

ALT_MAP = {1: "walk", 2: "cycle", 3: "pt", 4: "drive"}
ALTERNATIVES = ["walk", "cycle", "pt", "drive"]


def build_wide(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["obs_id"] = df["trip_id"].astype(int)
    out["household_id"] = df["household_id"].astype(int)
    out["person_id"] = df["household_id"].astype(str) + "_" + df["person_n"].astype(str)
    out["choice"] = df["travel_mode"].map(ALT_MAP)
    out["purpose"] = df["purpose"]
    out["survey_year"] = df["survey_year"]
    out["travel_year"] = df["travel_year"]
    out["day_of_week"] = df["day_of_week"]
    out["start_time"] = df["start_time"]
    out["age"] = df["age"]
    out["female"] = df["female"]
    out["driving_license"] = df["driving_license"]
    out["car_ownership"] = df["car_ownership"]
    out["distance_m"] = df["distance"]

    out["time_walk_min"] = df["dur_walking"] * 60.0
    out["time_cycle_min"] = df["dur_cycling"] * 60.0
    out["time_pt_min"] = (df["dur_pt_access"] + df["dur_pt_rail"] + df["dur_pt_bus"] + df["dur_pt_int"]) * 60.0
    out["time_drive_min"] = df["dur_driving"] * 60.0
    out["cost_walk"] = 0.0
    out["cost_cycle"] = 0.0
    out["cost_pt"] = df["cost_transit"]
    out["cost_drive"] = df["cost_driving_fuel"] + df["cost_driving_ccharge"]
    out["pt_interchanges"] = df["pt_interchanges"]
    out["driving_traffic_percent"] = df["driving_traffic_percent"]

    out["avail_walk"] = out["time_walk_min"].notna() & (out["time_walk_min"] > 0)
    out["avail_cycle"] = out["time_cycle_min"].notna() & (out["time_cycle_min"] > 0)
    out["avail_pt"] = out["time_pt_min"].notna() & (out["time_pt_min"] > 0)
    out["avail_drive"] = (
        out["time_drive_min"].notna()
        & (out["time_drive_min"] > 0)
        & (out["driving_license"] == 1)
        & (out["car_ownership"] > 0)
    )
    for alt in ALTERNATIVES:
        out.loc[out["choice"] == alt, f"avail_{alt}"] = True
    return out


def build_long(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    shared_columns = [
        "obs_id",
        "household_id",
        "person_id",
        "purpose",
        "survey_year",
        "travel_year",
        "day_of_week",
        "start_time",
        "age",
        "female",
        "driving_license",
        "car_ownership",
        "distance_m",
    ]
    for alt in ALTERNATIVES:
        part = wide[shared_columns].copy()
        part["alt"] = alt
        part["choice"] = wide["choice"].eq(alt)
        part["available"] = wide[f"avail_{alt}"].astype(bool)
        part["time_min"] = wide[f"time_{alt}_min"]
        part["cost"] = wide[f"cost_{alt}"]
        part["pt_interchanges"] = np.where(alt == "pt", wide["pt_interchanges"], 0)
        part["driving_traffic_percent"] = np.where(alt == "drive", wide["driving_traffic_percent"], 0.0)
        rows.append(part)
    long = pd.concat(rows, ignore_index=True)
    return long.sort_values(["obs_id", "alt"]).reset_index(drop=True)


def write_schema(out_dir: Path, wide: pd.DataFrame, long: pd.DataFrame) -> None:
    schema = {
        "dataset_id": "lpmc_london",
        "source": "Biogeme official data page / London Passenger Mode Choice",
        "alternatives": ALTERNATIVES,
        "choice_mapping": {str(key): value for key, value in ALT_MAP.items()},
        "files": {
            "choice_wide.csv": {"rows": int(len(wide)), "columns": list(wide.columns)},
            "choice_long.csv": {"rows": int(len(long)), "columns": list(long.columns)},
        },
        "attributes": {
            "time_min": "Alternative-specific travel time in minutes.",
            "cost": "Alternative-specific monetary cost; walk and cycle are zero-cost by construction.",
            "available": "Derived from positive alternative time and car ownership/license constraints, with chosen alternatives forced available.",
        },
    }
    (out_dir / "schema.json").write_text(json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Process Biogeme LPMC raw data into TorchDCM long/wide choice-set files.")
    parser.add_argument("--raw", type=Path, default=RAW_PATH)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--zip", action="store_true", help="Create datasets/large/releases/lpmc_london.zip for Google Drive upload.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.raw)
    wide = build_wide(raw)
    long = build_long(wide)
    wide.to_csv(args.out / "choice_wide.csv", index=False)
    long.to_csv(args.out / "choice_long.csv", index=False)
    write_schema(args.out, wide, long)
    if args.zip:
        RELEASE_DIR.mkdir(parents=True, exist_ok=True)
        archive = shutil.make_archive(str(RELEASE_DIR / "lpmc_london"), "zip", root_dir=args.out)
        print(f"release={archive}")
    print(f"wide_rows={len(wide)} long_rows={len(long)} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
